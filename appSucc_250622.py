import os
import hmac
import base64
import hashlib
import requests
import json
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, abort
import logging
import pandas as pd
import threading

# =============================================================================
# 1. INITIAL SETUP
# =============================================================================

# Initialize the Flask application
app = Flask(__name__)

# Configure basic logging to see outputs in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 2. CONFIGURATION & API CREDENTIALS
# =============================================================================

# Load API credentials from environment variables
API_KEY = os.getenv("OKXCW_API_KEY")
API_SECRET = os.getenv("OKXCW_API_SECRET")
API_PASSPHRASE = os.getenv("OKXCW_API_PASSPHRASE")

# Validate that OKX API credentials are set
if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logger.critical("CRITICAL ERROR: OKX API Key, Secret, or Passphrase is not set. Exiting.")
    exit(1)

# OKX API base URL and trading constants
BASE_URL = "https://www.okx.com"
TRAILING_STOP_PERCENT = 15.0

# --- Gemini API Settings ---
# Load GEMINI_API_KEY from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate that Gemini API key is set
if not GEMINI_API_KEY:
    logger.critical("CRITICAL ERROR: GEMINI_API_KEY is not set. Please set it as an environment variable.")
    exit(1)

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
AI_RESPONSE_TIMEOUT_SECONDS = 15  # AI 응답 대기 시간 (초) - Increased for structured response


# =============================================================================
# 3. API HELPER FUNCTIONS
# =============================================================================

def get_iso_timestamp():
    """Generates a UTC timestamp in the format required by the OKX API."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def generate_headers(method, request_path, body=""):
    """Generates the required authentication headers for an OKX API request."""
    timestamp = get_iso_timestamp()
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    mac = hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    sign = base64.b64encode(mac.digest()).decode('utf-8')

    return {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": API_PASSPHRASE,
        "Content-Type": "application/json"
    }


def get_balance(currency="USDT"):
    """Fetches the available balance for the trading account."""
    url_path = "/api/v5/account/balance"  # Query all balances for the trading account
    headers = generate_headers("GET", url_path)

    try:
        response = requests.get(BASE_URL + url_path, headers=headers)
        response.raise_for_status()
        res_data = response.json()

        if res_data.get('code') != '0':
            logger.error(f"Failed to get balance, OKX API error: {res_data}")
            return None

        # Unified account has one account object in the 'data' list
        if res_data.get('data') and res_data['data'][0]:
            account_details = res_data['data'][0].get('details', [])
            for asset in account_details:
                if asset.get('ccy') == currency:
                    avail_bal = asset.get('availBal')
                    if avail_bal:
                        logger.info(f"Successfully fetched trading account balance: {avail_bal} {currency}")
                        return float(avail_bal)
            logger.warning(f"'{currency}' not found in trading account details.")
            return 0.0
        else:
            logger.warning("No 'data' field in balance response from OKX.")
            return None

    except Exception as e:
        logger.error(f"An exception occurred while fetching balance: {e}")
        return None


def get_open_positions(symbol):
    """Fetches open positions for a specific symbol."""
    url_path = f"/api/v5/account/positions?instId={symbol}"
    headers = generate_headers("GET", url_path)
    try:
        response = requests.get(BASE_URL + url_path, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get('code') == '0':
            return data.get('data', [])
        else:
            logger.error(f"Error fetching positions: {data}")
            return []
    except Exception as e:
        logger.error(f"Failed to get positions for {symbol}: {e}")
        return []


def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR) for volatility assessment."""
    if df is None or df.empty:
        return None, None
    df_copy = df.copy()
    df_copy['high_low'] = df_copy['high'] - df_copy['low']
    df_copy['high_close_prev'] = abs(df_copy['high'] - df_copy['close'].shift(1))
    df_copy['low_close_prev'] = abs(df_copy['low'] - df_copy['close'].shift(1))
    df_copy['tr'] = df_copy[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df_copy['atr'] = df_copy['tr'].rolling(window=period).mean()
    # Return last ATR value and the average of all calculated ATR values
    return df_copy['atr'].iloc[-1], df_copy['atr'].mean()


def get_market_data(symbol, timeframe='5m', limit=100):
    """Fetches recent market candle data from OKX."""
    url_path = f"/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
    headers = generate_headers("GET", url_path)

    try:
        response = requests.get(BASE_URL + url_path, headers=headers)
        response.raise_for_status()
        data = response.json().get('data', [])
        if not data:
            logger.warning(f"No market data returned for {symbol}")
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote',
                                         'confirm'])
        # Add .astype(int) to address the FutureWarning if timestamp comes as string that looks like number
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
        return df.sort_values(by='timestamp').reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get market data for {symbol}: {e}")
        return None


def get_ai_decision(data_frame, target_symbol):
    """
    Gets a structured trading decision and reason from the Gemini model.
    Returns (decision, reason) or (None, None) if an error occurs or invalid response.
    """
    if data_frame is None or data_frame.empty:
        logger.warning("No market data to send to AI.")
        return None, "시장 데이터 없음"

    # Drop timestamp for a cleaner JSON payload for the AI
    ai_data = data_frame.drop(columns=['timestamp']).to_json(orient='records')

    prompt_content = (
        "You are a highly skilled cryptocurrency trading expert. "
        "Analyze the provided 5-minute market data (OHLCV) for trends, support and resistance levels, and momentum. "
        "Based on your analysis, provide a clear trading decision and a brief reason for it. "
        "The decision MUST be 'long' or 'short'. "
        "The reason should be a concise explanation in Korean, no more than 2-3 sentences. "
        "Respond ONLY with a JSON object containing two fields: 'decision' (string) and 'reason' (string). "
        "Example: {\"decision\": \"long\", \"reason\": \"상승 추세가 강하며 지지선이 확고합니다.\"}. "
        f"Market data for {target_symbol}: {ai_data}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "decision": {"type": "STRING"},
                    "reason": {"type": "STRING"}
                },
                "required": ["decision", "reason"]
            }
        }
    }

    result_container = {"decision": None, "reason": "AI 응답 오류"}

    def call_gemini():
        try:
            logger.info("Requesting decision from AI...")
            response = requests.post(GEMINI_API_URL, json=payload, timeout=AI_RESPONSE_TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()

            # Ensure the structure is as expected
            if (result.get('candidates') and len(result['candidates']) > 0 and
                    result['candidates'][0].get('content') and
                    result['candidates'][0]['content'].get('parts') and
                    len(result['candidates'][0]['content']['parts']) > 0):

                json_string = result['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(json_string)

                decision = parsed_json.get('decision', '').strip().lower()
                reason = parsed_json.get('reason', '이유 설명 없음').strip()

                if decision in ['long', 'short']:
                    result_container['decision'] = decision
                    result_container['reason'] = reason
                else:
                    logger.warning(f"AI returned invalid decision: '{decision}'. Reason: '{reason}'.")
                    result_container['decision'] = None
                    result_container['reason'] = "AI가 유효한 결정 (long/short)을 반환하지 않았습니다."
            else:
                logger.error(f"Unexpected AI response structure: {result}")
                result_container['decision'] = None
                result_container['reason'] = "AI 응답 구조 오류"

        except requests.exceptions.Timeout:
            logger.error("AI request timed out.")
            result_container['decision'] = None
            result_container['reason'] = "AI 요청 시간 초과"
        except json.JSONDecodeError:
            logger.error(f"AI response was not valid JSON: {response.text}")
            result_container['decision'] = None
            result_container['reason'] = "AI 응답이 유효한 JSON이 아닙니다."
        except Exception as e:
            logger.error(f"An error occurred during AI call: {e}")
            result_container['decision'] = None
            result_container['reason'] = f"AI 호출 중 예외 발생: {e}"

    ai_thread = threading.Thread(target=call_gemini)
    ai_thread.start()
    ai_thread.join(timeout=AI_RESPONSE_TIMEOUT_SECONDS + 5)  # Give a little extra time for thread cleanup

    final_decision = result_container.get('decision')
    final_reason = result_container.get('reason')
    logger.info(f"Final AI decision: {final_decision}, Reason: {final_reason}")
    return final_decision, final_reason


def place_order(symbol, amount, side):
    """Places a market order on OKX."""
    url_path = "/api/v5/trade/order"
    body = {"instId": symbol, "tdMode": "cross", "side": side.lower(), "ordType": "market", "sz": str(amount)}
    json_body = json.dumps(body)
    headers = generate_headers("POST", url_path, body=json_body)

    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body)
        response.raise_for_status()
        logger.info(f"Successfully placed order: {side} {amount} {symbol}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Order placement failed: {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"error": "Order failed", "detail": str(e)}


def place_trailing_stop_order(symbol, entry_side):
    """Places a trailing stop loss order to protect a position."""
    url_path = "/api/v5/trade/order-algo"
    close_side = "sell" if entry_side.lower() == "buy" else "buy"
    body = {"instId": symbol, "tdMode": "cross", "side": close_side, "ordType": "move_order_stop",
            "callbackRatio": str(TRAILING_STOP_PERCENT / 100)}
    json_body = json.dumps(body)
    headers = generate_headers("POST", url_path, body=json_body)

    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body)
        response.raise_for_status()
        logger.info(f"Successfully placed trailing stop order for {symbol}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Trailing stop order failed: {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"error": "Trailing stop failed", "detail": str(e)}


def log_trade_status(message, balance, positions, okx_symbol, tradingview_direction=None, atr_data=None,
                     ai_decision=None, ai_reason=None):
    """Logs the current status in the requested format."""
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n==================== {message}: {timestamp} ====================")
    print("\n--- 현재 상태 정보 갱신 중... ---")

    if balance is not None:
        print(f"--- 선물 지갑 잔고: {balance:.2f} USDT ---")
    else:
        print("--- 선물 지갑 잔고: 조회 실패 ---")

    open_positions = [p for p in positions if p.get('pos') and float(p.get('pos')) != 0]
    if open_positions:
        print("--- 현재 열려있는 포지션 내역: ---")
        total_pnl = 0
        for pos in open_positions:
            pnl = float(pos.get('upl', '0'))
            pnl_ratio = float(pos.get('uplRatio', '0')) * 100
            total_pnl += pnl
            print(f"  - {pos.get('instId')} | {pos.get('posSide')} | PNL: {pnl:.2f} ({pnl_ratio:.2f}%)")
        print(f"현재 미실현은 PNL : {total_pnl:.2f}")
    else:
        print("--- 현재 열려있는 포지션 내역: 없음 ---")
    print("---------------------------------")

    if atr_data and atr_data[0] is not None and atr_data[1] is not None:
        current_atr, avg_atr = atr_data
        volatility = "높은 변동성" if current_atr > avg_atr else "보통 변동성"
        print(f"시장 상황: {volatility} (현재 ATR: {current_atr:.4f} ~ 평균 ATR: {avg_atr:.4f})")

    if tradingview_direction:
        print(" '얼러트 신호 우선, AI 확인' 로직으로 신규 진입을 검토합니다.")
        print(f"통합 전략 얼러트의 1차 판단: {tradingview_direction.upper()}")

    if ai_decision:
        print(f"AI의 최종 판단: {ai_decision.upper()}")
        print(f"AI 판단 이유: {ai_reason}")
    else:
        print("AI가 유효한 트레이딩 결정을 내리지 못했습니다.")


# =============================================================================
# 4. FLASK WEB SERVER ROUTES
# =============================================================================

@app.route('/', methods=['GET'])
def status_check():
    """A simple endpoint to check if the server is running."""
    logger.info("Status check endpoint '/' was hit.")
    return jsonify({"status": "Flask server with AI-check is running"}), 200


@app.route('/webhook', methods=['POST'])
def webhook_receiver():
    """Receives signals from TradingView, gets AI decision, and trades if they match."""
    if not request.is_json:
        logger.warning("Received non-JSON request to webhook.")
        abort(415)

    data = request.get_json()
    logger.info(f"Received data from TradingView: {json.dumps(data)}")

    symbol = data.get('symbol')
    action = data.get('action')
    amount = data.get('amount')

    if not all([symbol, action, amount]):
        logger.error(f"Webhook failed: Missing 'symbol', 'action', or 'amount'.")
        return jsonify({"error": "Missing required fields"}), 400

    okx_symbol = symbol.replace('USDT.P', '-USDT-SWAP')
    tradingview_direction = 'long' if action.lower() == 'buy' else 'short'

    # --- Initial Status Logging ---
    balance = get_balance()
    positions = get_open_positions(okx_symbol)
    # log_trade_status will be updated at the end with AI decision and reason
    log_trade_status("실행 시작", balance, positions, okx_symbol, tradingview_direction)

    # --- Main Logic ---
    market_data_df = get_market_data(okx_symbol)
    atr_data = calculate_atr(market_data_df)
    ai_decision, ai_reason = get_ai_decision(market_data_df, okx_symbol)  # Get both decision and reason

    if ai_decision and ai_decision == tradingview_direction:  # Proceed only if AI made a valid decision and it matches
        logger.info(
            f"MATCH: AI decision '{ai_decision}' matches TradingView signal '{tradingview_direction}'. Proceeding with trade.")
        order_result = place_order(okx_symbol, amount, action)
        if "error" not in order_result:
            time.sleep(1)  # Wait a moment for the position to update
            place_trailing_stop_order(okx_symbol, action)
    else:
        if not ai_decision:
            logger.warning(f"NO AI DECISION: AI failed to provide a valid decision. Order skipped.")
        else:
            logger.warning(
                f"NO MATCH: AI decision '{ai_decision}' != TradingView signal '{tradingview_direction}'. Order skipped.")

    # --- Final Status Logging ---
    final_balance = get_balance()
    final_positions = get_open_positions(okx_symbol)
    log_trade_status("실행 종료", final_balance, final_positions, okx_symbol, atr_data=atr_data, ai_decision=ai_decision,
                     ai_reason=ai_reason)

    # Return a simple OK to TradingView
    return jsonify({"status": "processed"}), 200


# =============================================================================
# 5. RUN THE FLASK APPLICATION
# =============================================================================

if __name__ == '__main__':
    # Make sure to install pandas: pip install pandas requests
    app.run(host='0.0.0.0', port=80)