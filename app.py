# -*- coding: utf-8 -*-
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
API_KEY = os.getenv("OKXYH_API_KEY")
API_SECRET = os.getenv("OKXYH_API_SECRET")
API_PASSPHRASE = os.getenv("OKXYH_API_PASSPHRASE")

# Validate that OKX API credentials are set
if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logger.critical("CRITICAL ERROR: OKX API Key, Secret, or Passphrase is not set. Exiting.")
    exit(1)

# OKX API base URL
BASE_URL = "https://www.okx.com"

# --- Gemini API Settings ---
# Load GEMINI_API_KEY from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate that Gemini API key is set
if not GEMINI_API_KEY:
    logger.critical("CRITICAL ERROR: GEMINI_API_KEY is not set. Please set it as an environment variable.")
    exit(1)

# Changed to Gemini 1.5 Flash
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
AI_RESPONSE_TIMEOUT_SECONDS = 50  # AI 응답을 위해 타임아웃을 넉넉하게 설정

# --- Trading Settings (Global Variables) ---
TRADING_TIMEFRAME = "15m"  # ✅ 전역 거래 시간 주기 (예: '5m', '15m', '1h')
CANDLE_LIMIT = 200  # ✅ 기술적 지표 계산을 위한 캔들 수

# --- User Confirmation Settings ---
DEFAULT_TP_SL_CONFIRM_TO_YES = False
USER_CONFIRM_TIMEOUT_SECONDS = 10

# --- ATR Stage Boundaries ---
ATR_STAGE_BOUNDARIES = {
    "STAGE1_MAX": 100.0,
    "STAGE2_MAX": 200.0
}


# =============================================================================
# 3. API & DATA PROCESSING HELPER FUNCTIONS
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
    url_path = "/api/v5/account/balance"
    headers = generate_headers("GET", url_path)

    try:
        response = requests.get(BASE_URL + url_path, headers=headers, timeout=10)
        response.raise_for_status()
        res_data = response.json()

        if res_data.get('code') != '0':
            logger.error(f"Failed to get balance, OKX API error: {res_data}")
            return None

        if res_data.get('data') and res_data['data'][0]:
            account_details = res_data['data'][0].get('details', [])
            for asset in account_details:
                if asset.get('ccy') == currency:
                    avail_bal = asset.get('availBal')
                    if avail_bal:
                        logger.info(f"Successfully fetched trading account balance: {float(avail_bal):.1f} {currency}")
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
    """
    Fetches open positions for a specific symbol and determines side (long/short)
    based on pos value.
    """
    url_path = f"/api/v5/account/positions?instId={symbol}"
    headers = generate_headers("GET", url_path)
    try:
        response = requests.get(BASE_URL + url_path, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('code') == '0':
            positions = data.get('data', [])
            for p in positions:
                if p.get('pos') is not None:
                    pos_val = float(p['pos'])
                    p['pos'] = pos_val
                    if pos_val > 0:
                        p['posSide'] = 'long'
                    elif pos_val < 0:
                        p['posSide'] = 'short'
                    else:
                        p['posSide'] = 'none'
            return positions
        else:
            logger.error(f"Error fetching positions: {data}")
            return []
    except Exception as e:
        logger.error(f"Failed to get positions for {symbol}: {e}")
        return []


def get_market_data(symbol, timeframe, limit, retries=3, delay=1):
    """
    Fetches recent market candle data from OKX with retry mechanism.
    """
    url_path = f"/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
    headers = generate_headers("GET", url_path)

    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL + url_path, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json().get('data', [])

            if not data:
                logger.warning(
                    f"No market data returned for {symbol} ({timeframe}, limit {limit}) on attempt {attempt + 1}/{retries}.")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                return None

            df = pd.DataFrame(data,
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote',
                                       'confirm'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Successfully fetched {len(df)} candles for {symbol} ({timeframe}, limit {limit})")
            return df.sort_values(by='timestamp').reset_index(drop=True)
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to get market data for {symbol} ({timeframe}, limit {limit}) on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return None
    return None


def calculate_technical_indicators(df):
    """
    Calculates key technical indicators and returns the most recent values.
    """
    if df is None or len(df) < 50:
        logger.warning("Not enough market data to calculate all technical indicators.")
        return None

    indicators = {}

    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['atr'] = df['tr'].ewm(span=14, adjust=False).mean()
    indicators['current_atr'] = df['atr'].iloc[-1]
    indicators['average_atr'] = df['atr'].mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    indicators['rsi'] = df['rsi'].iloc[-1]

    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    indicators['sma20'] = df['sma20'].iloc[-1]
    indicators['sma50'] = df['sma50'].iloc[-1]

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    indicators['macd'] = df['macd'].iloc[-1]
    indicators['macd_signal'] = df['macd_signal'].iloc[-1]

    logger.info(f"Calculated Indicators: { {k: f'{v:.4f}' for k, v in indicators.items()} }")
    return indicators


def get_btc_trend_and_data():
    """
    Fetches BTC-USDT data for 1-hour and 4-hour timeframes and determines the market trend.
    """
    logger.info("Fetching BTC-USDT data for market trend analysis...")
    btc_data = {}
    btc_df_1h = get_market_data("BTC-USDT-SWAP", timeframe='1h', limit=50)
    btc_df_4h = get_market_data("BTC-USDT-SWAP", timeframe='4h', limit=20)

    if btc_df_1h is not None and not btc_df_1h.empty:
        btc_data['1h'] = btc_df_1h
    if btc_df_4h is not None and not btc_df_4h.empty:
        btc_data['4h'] = btc_df_4h

    trend_1h, trend_4h = "Uncertain", "Uncertain"
    MIN_CANDLES_1H, MIN_CANDLES_4H = 50, 20

    if btc_df_1h is not None and len(btc_df_1h) >= MIN_CANDLES_1H:
        btc_df_1h['sma50'] = btc_df_1h['close'].rolling(window=50).mean()
        last_close_1h, last_sma50_1h = btc_df_1h['close'].iloc[-1], btc_df_1h['sma50'].iloc[-1]
        if last_close_1h > last_sma50_1h * 1.005:
            trend_1h = "Bullish"
        elif last_close_1h < last_sma50_1h * 0.995:
            trend_1h = "Bearish"
        else:
            trend_1h = "Sideways/Consolidating"
        logger.info(f"BTC 1-hour Trend: {trend_1h} (Close: {last_close_1h:.2f}, SMA50: {last_sma50_1h:.2f})")
    else:
        logger.warning(f"Not enough 1-hour BTC data for trend analysis.")

    if btc_df_4h is not None and len(btc_df_4h) >= MIN_CANDLES_4H:
        btc_df_4h['sma20'] = btc_df_4h['close'].rolling(window=20).mean()
        last_close_4h, last_sma20_4h = btc_df_4h['close'].iloc[-1], btc_df_4h['sma20'].iloc[-1]
        if last_close_4h > last_sma20_4h * 1.01:
            trend_4h = "Strongly Bullish"
        elif last_close_4h < last_sma20_4h * 0.99:
            trend_4h = "Strongly Bearish"
        else:
            trend_4h = "Sideways"
        logger.info(f"BTC 4-hour Trend: {trend_4h} (Close: {last_close_4h:.2f}, SMA20: {last_sma20_4h:.2f})")
    else:
        logger.warning(f"Not enough 4-hour BTC data for trend analysis.")

    if "Bullish" in trend_1h and "Bullish" in trend_4h:
        final_trend = "Strongly Bullish"
    elif "Bearish" in trend_1h and "Bearish" in trend_4h:
        final_trend = "Strongly Bearish"
    elif "Bullish" in trend_1h and "Sideways" in trend_4h:
        final_trend = "Weakly Bullish"
    elif "Bearish" in trend_1h and "Sideways" in trend_4h:
        final_trend = "Weakly Bearish"
    elif "Strongly Bullish" in trend_4h:
        final_trend = "Strongly Bullish"
    elif "Strongly Bearish" in trend_4h:
        final_trend = "Strongly Bearish"
    else:
        final_trend = "Uncertain/Mixed"

    return final_trend, btc_data


def calculate_atr_levels(current_price, current_atr, position_side):
    """Calculates TP/SL levels based on ATR in 3 stages."""
    TP_MULTIPLIERS, SL_MULTIPLIERS = [2.0, 3.5, 5.0], [1.5, 2.0, 2.5]

    # ✅ 수정된 부분: 딕셔너리 초기화 방식 변경
    levels = {}
    for i in range(3):
        levels[f"TP{i + 1}"] = None
        levels[f"SL{i + 1}"] = None

    if current_atr is None or current_atr <= 0:
        logger.warning("Cannot calculate ATR levels, ATR is invalid.")
        return levels

    if position_side == 'long':
        for i in range(3):
            levels[f"TP{i + 1}"] = current_price + (TP_MULTIPLIERS[i] * current_atr)
            levels[f"SL{i + 1}"] = current_price - (SL_MULTIPLIERS[i] * current_atr)
    elif position_side == 'short':
        for i in range(3):
            levels[f"TP{i + 1}"] = current_price - (TP_MULTIPLIERS[i] * current_atr)
            levels[f"SL{i + 1}"] = current_price + (SL_MULTIPLIERS[i] * current_atr)
    else:
        logger.warning(f"Invalid position side '{position_side}' for ATR level calculation.")
        return levels

    for key in levels:
        if levels[key] is not None:
            levels[key] = round(levels[key], 1)
    logger.info(f"Calculated ATR Levels for {position_side}: {levels}")
    return levels


def calculate_atr_stage(current_atr):
    """Determines the ATR volatility stage (1, 2, or 3)."""
    if current_atr is None: return 1
    if current_atr <= ATR_STAGE_BOUNDARIES["STAGE1_MAX"]:
        return 1
    elif current_atr <= ATR_STAGE_BOUNDARIES["STAGE2_MAX"]:
        return 2
    else:
        return 3


def get_ai_decision(data_frame, tech_indicators, btc_trend, target_symbol, btc_raw_data=None):
    """Gets a structured trading decision from the Gemini model."""
    if data_frame is None or data_frame.empty or tech_indicators is None:
        logger.warning("No market data or indicators for AI decision.")
        return None, "시장 데이터 또는 기술 지표 없음"

    ai_data = data_frame.drop(columns=['timestamp']).tail(100).to_json(orient='records')
    indicators_json = json.dumps({k: f'{v:.4f}' for k, v in tech_indicators.items()})

    btc_raw_data_json = ""
    if btc_raw_data:
        for tf, df in btc_raw_data.items():
            if df is not None and not df.empty:
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: {df.drop(columns=['timestamp']).tail(50).to_json(orient='records')}"
            else:
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: 데이터 없음"

    prompt_content = (
        "You are a master cryptocurrency trading analyst. Your task is to provide a trading recommendation "
        f"based on a comprehensive dataset. I will provide you with raw **{TRADING_TIMEFRAME}** OHLCV data for the target symbol, "  # ✅ 전역 변수 사용
        "a summary of key technical indicators, the general trend of the BTC-USDT market, and recent raw OHLCV data for BTC-USDT-SWAP."
        "\n\n**Analysis Instructions:**"
        f"\n1.  **Primary Data & Price Action (Target Symbol - {TRADING_TIMEFRAME}):** Analyze the provided **{TRADING_TIMEFRAME}** OHLCV data for **candlestick patterns, momentum, volume trends, and significant price action.**"  # ✅ 전역 변수 사용
        "\n2.  **Technical Indicators:** Use the provided indicators (RSI, MACD, SMAs, ATR) to confirm your analysis. Check for **divergences, crossovers, and overbought/oversold conditions.**"
        "\n3.  **Market Context (BTC Trend & Data):** This is crucial. A **long position is significantly safer in a bullish BTC market**, and a **short position is safer in a bearish one.** If BTC trend is 'Strongly Bullish', prioritize long. If 'Strongly Bearish', prioritize short. If 'Uncertain/Mixed', be cautious."
        "\n\n**Your Response:**"
        "\nProvide a clear decision ('long' or 'short') and a concise reason in Korean (3-4 sentences). "
        "Respond ONLY with a JSON object: {\"decision\": \"string\", \"reason\": \"string\"}."
        f"\n\n--- DATA FOR ANALYSIS ---"
        f"\n**Target Symbol:** {target_symbol}"
        f"\n**Overall BTC Trend:** {btc_trend}"
        f"\n**Calculated Technical Indicators (Current):** {indicators_json}"
        f"\n**Recent Market Data (OHLCV) - Target Symbol ({TRADING_TIMEFRAME}):** {ai_data}"  # ✅ 전역 변수 사용
        f"\n**Recent BTC-USDT-SWAP Market Data (OHLCV):** {btc_raw_data_json}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT", "properties": {"decision": {"type": "STRING"}, "reason": {"type": "STRING"}},
                "required": ["decision", "reason"]
            }
        }
    }

    result_container = {"decision": None, "reason": "AI 응답 오류"}

    def call_gemini():
        try:
            logger.info("Requesting enhanced decision from AI...")
            response = requests.post(GEMINI_API_URL, json=payload, timeout=AI_RESPONSE_TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()
            if result.get('candidates'):
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(json_string)
                decision, reason = parsed_json.get('decision', '').strip().lower(), parsed_json.get('reason',
                                                                                                    '이유 설명 없음').strip()
                if decision in ['long', 'short']:
                    result_container['decision'], result_container['reason'] = decision, reason
                else:
                    result_container['reason'] = "AI가 유효한 결정 (long/short)을 반환하지 않았습니다."
            else:
                result_container['reason'] = "AI 응답 구조 오류"
        except Exception as e:
            logger.error(f"An error occurred during AI call: {e}")
            result_container['reason'] = f"AI 호출 중 예외 발생: {e}"

    ai_thread = threading.Thread(target=call_gemini)
    ai_thread.start()
    ai_thread.join(timeout=AI_RESPONSE_TIMEOUT_SECONDS + 5)

    final_decision, final_reason = result_container.get('decision'), result_container.get('reason')
    logger.info(f"Final AI decision: {final_decision}, Reason: {final_reason}")
    return final_decision, final_reason


def get_ai_tp_sl_recommendation(position_data, market_data_df, tech_indicators, btc_trend, target_symbol, atr_levels,
                                btc_raw_data=None):
    """Gets TP/SL and trailing stop recommendations from the Gemini model."""
    if not position_data or market_data_df is None or tech_indicators is None:
        return None, None, None, "TP/SL 추천을 위한 데이터 부족"

    current_position = next(
        (p for p in position_data if p.get('instId') == target_symbol and float(p.get('pos', 0)) != 0), None)
    if not current_position:
        return None, None, None, "현재 오픈 포지션 없음"

    position_side, avg_entry_price = current_position.get('posSide'), float(current_position.get('avgPx'))
    position_size, last_close_price = abs(float(current_position.get('pos'))), market_data_df['close'].iloc[-1]
    current_atr = tech_indicators.get('current_atr')

    ai_data = market_data_df.drop(columns=['timestamp']).tail(100).to_json(orient='records')
    indicators_json = json.dumps({k: f'{v:.4f}' for k, v in tech_indicators.items()})
    atr_levels_json = json.dumps(atr_levels)

    btc_raw_data_json = ""
    if btc_raw_data:
        for tf, df in btc_raw_data.items():
            if df is not None and not df.empty:
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: {df.drop(columns=['timestamp']).tail(50).to_json(orient='records')}"
            else:
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: 데이터 없음"

    prompt_content = (
        "You are a master cryptocurrency trading analyst specializing in risk management. "
        "Recommend optimal Take Profit (TP), Stop Loss (SL) prices, and a Trailing Stop callback ratio."
        f"\n\n**Given Position Details:**"
        f"\n- **Symbol:** {target_symbol}, **Side:** {position_side.upper()}, **Entry Price:** {avg_entry_price:.4f}"
        f"\n- **Current Price:** {last_close_price:.4f}, **Current ATR:** {current_atr:.4f}"
        "\n\n**Analysis Instructions:**"
        "\n1.  **Market Context (BTC Trend & Data):** Consider the overall BTC trend and raw data to influence TP/SL aggressiveness."
        f"\n2.  **Primary Data & Price Action (Target Symbol - {TRADING_TIMEFRAME}):** Analyze recent volatility and key support/resistance levels from the {TRADING_TIMEFRAME} data."  # ✅ 전역 변수 사용
        "\n3.  **Technical Indicators & ATR Levels:** Use indicators and the provided ATR-based levels as a guide. Higher ATR suggests wider TP/SL and a larger trailing stop callback ratio."
        "\n\n**Your Response:**"
        "\nProvide precise TP/SL prices and a trailing callback ratio (%). The reason should be a concise explanation in Korean. "
        "Respond ONLY with a JSON object: {\"take_profit_price\": float, \"stop_loss_price\": float, \"recommended_trailing_callback_ratio\": float, \"reason\": \"string\"}."
        f"\n\n--- DATA FOR ANALYSIS ---"
        f"\n**Overall BTC Trend:** {btc_trend}"
        f"\n**Calculated Technical Indicators:** {indicators_json}"
        f"\n**ATR Based TP/SL Levels (Reference):** {atr_levels_json}"
        f"\n**Recent Market Data (OHLCV) - Target Symbol ({TRADING_TIMEFRAME}):** {ai_data}"  # ✅ 전역 변수 사용
        f"\n**Recent BTC-USDT-SWAP Market Data (OHLCV):** {btc_raw_data_json}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "take_profit_price": {"type": "NUMBER"}, "stop_loss_price": {"type": "NUMBER"},
                    "recommended_trailing_callback_ratio": {"type": "NUMBER"}, "reason": {"type": "STRING"}
                },
                "required": ["take_profit_price", "stop_loss_price", "recommended_trailing_callback_ratio", "reason"]
            }
        }
    }

    result_container = {"take_profit_price": None, "stop_loss_price": None, "recommended_trailing_callback_ratio": None,
                        "reason": "AI 응답 오류"}

    def call_gemini_tp_sl():
        try:
            logger.info("Requesting TP/SL and Trailing Stop recommendations from AI...")
            response = requests.post(GEMINI_API_URL, json=payload, timeout=AI_RESPONSE_TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()
            if result.get('candidates'):
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(json_string)
                tp, sl, trail, reason = parsed_json.get('take_profit_price'), parsed_json.get(
                    'stop_loss_price'), parsed_json.get('recommended_trailing_callback_ratio'), parsed_json.get(
                    'reason', '이유 없음')
                if isinstance(tp, (int, float)) and isinstance(sl, (int, float)) and isinstance(trail, (int, float)):
                    result_container['take_profit_price'], result_container['stop_loss_price'] = float(tp), float(sl)
                    result_container['recommended_trailing_callback_ratio'], result_container['reason'] = float(
                        trail), reason
                else:
                    result_container['reason'] = "AI가 유효한 TP/SL/트레일링 값을 반환하지 않았습니다."
            else:
                result_container['reason'] = "AI 응답 구조 오류 (TP/SL/Trailing)"
        except Exception as e:
            logger.error(f"An error occurred during AI TP/SL/Trailing call: {e}")
            result_container['reason'] = f"AI 호출 중 예외 발생: {e}"

    ai_thread = threading.Thread(target=call_gemini_tp_sl)
    ai_thread.start()
    ai_thread.join(timeout=AI_RESPONSE_TIMEOUT_SECONDS + 5)

    final_tp, final_sl, final_trailing, final_reason = result_container.get('take_profit_price'), result_container.get(
        'stop_loss_price'), result_container.get('recommended_trailing_callback_ratio'), result_container.get('reason')
    logger.info(f"Final AI TP={final_tp}, SL={final_sl}, Trailing={final_trailing}%, Reason: {final_reason}")
    return final_tp, final_sl, final_trailing, final_reason


# =============================================================================
# 4. TRADING EXECUTION FUNCTIONS
# =============================================================================

def place_order(symbol, amount, side):
    """Places a market order on OKX."""
    url_path = "/api/v5/trade/order"
    body = {"instId": symbol, "tdMode": "cross", "side": side.lower(), "ordType": "market", "sz": str(amount)}
    json_body = json.dumps(body)
    headers = generate_headers("POST", url_path, body=json_body)
    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get('code') == '0':
            logger.info(
                f"Successfully placed market order: {side} {amount} {symbol}. Order ID: {res_data['data'][0]['ordId']}")
            return res_data
        else:
            logger.error(f"Market order failed: {res_data}. Request body: {json_body}")
            return {"error": "Order failed", "detail": res_data}
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return {"error": "Order failed", "detail": str(e)}


def place_algo_order_tpsl(symbol, pos_side, tp_price, sl_price, position_size):
    """Places a Take Profit and Stop Loss algorithm order."""
    url_path = "/api/v5/trade/order-algo"
    close_side = 'sell' if pos_side == 'long' else 'buy'
    body = {
        "instId": symbol, "tdMode": "cross", "side": close_side, "ordType": "conditional",
        "sz": str(abs(position_size)), "tpTriggerPx": str(tp_price), "tpOrdPx": "-1",
        "slTriggerPx": str(sl_price), "slOrdPx": "-1"
    }
    json_body = json.dumps(body)
    headers = generate_headers("POST", url_path, body=json_body)
    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get('code') == '0':
            logger.info(f"Successfully placed TP/SL algo order for {symbol}. Algo ID: {res_data['data'][0]['algoId']}")
            return res_data
        else:
            logger.error(f"TP/SL algo order failed: {res_data}. Request body: {json_body}")
            return {"error": "TP/SL failed", "detail": res_data}
    except Exception as e:
        logger.error(f"TP/SL algo order placement failed: {e}")
        return {"error": "TP/SL failed", "detail": str(e)}


def place_trailing_stop_order(symbol, entry_side, callback_ratio, position_size):
    """Places a trailing stop loss order."""
    url_path = "/api/v5/trade/order-algo"
    close_side = "sell" if entry_side.lower() == "long" else "buy"
    body = {
        "instId": symbol, "tdMode": "cross", "side": close_side, "ordType": "move_order_stop",
        "sz": str(abs(position_size)), "callbackRatio": str(callback_ratio / 100)
    }
    json_body = json.dumps(body)
    headers = generate_headers("POST", url_path, body=json_body)
    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get('code') == '0':
            logger.info(
                f"Successfully placed trailing stop order for {symbol}. Algo ID: {res_data['data'][0]['algoId']}")
            return res_data
        else:
            logger.error(f"Trailing stop order failed: {res_data}. Request body: {json_body}")
            return {"error": "Trailing stop failed", "detail": res_data}
    except Exception as e:
        logger.error(f"Trailing stop order failed: {e}")
        return {"error": "Trailing stop failed", "detail": str(e)}


def cancel_okx_algo_order(algo_id, symbol):
    """Cancels an existing algorithm order on OKX."""
    url_path = "/api/v5/trade/cancel-algos"
    body = {"algoId": algo_id, "instId": symbol}
    json_body = json.dumps([body])
    headers = generate_headers("POST", url_path, body=json_body)
    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get('code') == '0':
            logger.info(f"Successfully cancelled algo order {algo_id} for {symbol}.")
            return res_data
        else:
            logger.error(f"Failed to cancel algo order {algo_id}, OKX API error: {res_data}")
            return {"error": "Cancellation failed", "detail": res_data}
    except Exception as e:
        logger.error(f"Algo order cancellation failed: {e}")
        return {"error": "Cancellation failed", "detail": str(e)}


# =============================================================================
# 5. LOGGING & STATUS
# =============================================================================

def log_trade_status(message, balance, positions, okx_symbol, **kwargs):
    """Logs the current status in a comprehensive and readable format."""
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    print(f"\n{'=' * 60}\n       *** {message}: {timestamp} ***\n{'=' * 60}\n")
    print(f"{'─' * 25} 계정 및 포지션 상태 {'─' * 25}")
    print(f"  - 선물 지갑 잔고: {balance:.1f} USDT")
    if 'current_btc_price' in kwargs and kwargs['current_btc_price']:
        print(f"  - 현재 BTC-USDT 가격: {kwargs['current_btc_price']:.1f}")

    open_positions = [p for p in positions if p.get('pos') and float(p.get('pos')) != 0]
    if open_positions:
        print("\n  - 현재 보유 포지션:")
        total_pnl = sum(float(p.get('upl', '0')) for p in open_positions)
        for pos in open_positions:
            pos_side = "롱" if pos.get('posSide') == 'long' else "숏"
            pnl, pnl_ratio = float(pos.get('upl', '0')), float(pos.get('uplRatio', '0')) * 100
            print(
                f"    ▶️ {pos.get('instId')} | {pos_side} | 크기: {abs(pos.get('pos')):.1f} | PNL: {pnl:.1f} USDT ({pnl_ratio:.1f}%)")
        print(f"  - 총 미실현 PNL: {total_pnl:.1f} USDT")
    else:
        print("\n  - 현재 보유 포지션: 없음")
    print(f"{'─' * 60}\n")

    print(f"{'─' * 25} 시장 분석 정보 {'─' * 25}")
    if 'btc_trend' in kwargs: print(f"  - BTC 시장 동향: {kwargs['btc_trend']}")
    if 'tech_indicators' in kwargs and kwargs['tech_indicators']:
        tech = kwargs['tech_indicators']
        volatility = "높은 변동성" if tech['current_atr'] > tech['average_atr'] else "보통 변동성"
        print(f"  - {okx_symbol} 변동성: {volatility} (현재 ATR: {tech['current_atr']:.4f})")
        print(f"  - ATR 변동성 단계: {kwargs.get('atr_stage')}단계")
        print(
            f"  - 기술 지표: RSI {tech['rsi']:.1f}, MACD {tech['macd']:.4f}, SMA20/50 {tech['sma20']:.1f}/{tech['sma50']:.1f}")
    if 'atr_levels' in kwargs and kwargs['atr_levels']:
        levels = kwargs['atr_levels']
        print("  - ATR 기반 TP/SL 레벨 (AI 참조용):")
        print(
            f"    - 1단계: TP {levels['TP1']:.1f}, SL {levels['SL1']:.1f} | 2단계: TP {levels['TP2']:.1f}, SL {levels['SL2']:.1f} | 3단계: TP {levels['TP3']:.1f}, SL {levels['SL3']:.1f}")
    print(f"{'─' * 60}\n")

    print(f"{'─' * 25} 트레이딩 결정 과정 {'─' * 25}")
    if 'tradingview_direction' in kwargs: print(f"  - TradingView 1차 신호: {kwargs['tradingview_direction'].upper()}")
    if 'ai_decision' in kwargs and kwargs['ai_decision']:
        print(f"  - 🤖 AI 최종 판단: {kwargs['ai_decision'].upper()}")
        print(f"  - 💬 AI 판단 근거: {kwargs['ai_reason']}")
    else:
        print(f"  - 🤖 AI 최종 판단: 보류 또는 오류")
        if 'ai_reason' in kwargs: print(f"  - 💬 AI 판단 근거: {kwargs['ai_reason']}")

    if 'ai_tp_price' in kwargs and kwargs['ai_tp_price']:
        print(f"  - 🎯 AI 추천 TP: {kwargs['ai_tp_price']:.1f} | 🛡️ AI 추천 SL: {kwargs['ai_sl_price']:.1f}")
        print(f"  - 💡 AI 추천 근거: {kwargs['ai_tpsl_reason']}")
        print(f"  - TP/SL 설정 상태: {kwargs.get('tp_sl_confirmation_status')}")
    if 'recommended_trailing_callback_ratio' in kwargs and kwargs['recommended_trailing_callback_ratio']:
        print(f"  - ⛓️ AI 추천 트레일링 스탑: {kwargs['recommended_trailing_callback_ratio']:.1f}%")
        if 'effective_trailing_callback_ratio' in kwargs and kwargs['effective_trailing_callback_ratio']:
            print(
                f"  - 📈 최종 적용 트레일링 스탑 (ATR 단계 {kwargs.get('atr_stage')} 반영): {kwargs['effective_trailing_callback_ratio']:.1f}%")
    print(f"{'─' * 60}\n")


# =============================================================================
# 6. FLASK WEB SERVER ROUTES
# =============================================================================
@app.route('/', methods=['GET'])
def status_check():
    """A simple endpoint to check if the server is running."""
    logger.info("Status check endpoint '/' was hit.")
    return jsonify({"status": "Enhanced Flask server with AI-check is running"}), 200


@app.route('/webhook', methods=['POST'])
def webhook_receiver():
    """Receives signals, refreshes all data, gets AI decision, and then trades."""
    if not request.is_json:
        logger.warning("Received non-JSON request to webhook.")
        abort(415)

    data = request.get_json()
    logger.info(f"Received data from TradingView: {json.dumps(data)}")

    symbol, action, amount = data.get('symbol'), data.get('action'), data.get('amount')
    if not all([symbol, action, amount]):
        logger.error("Webhook failed: Missing 'symbol', 'action', or 'amount'.")
        return jsonify({"error": "Missing required fields"}), 400

    okx_symbol = symbol.replace('USDT.P', '-USDT-SWAP')
    tradingview_direction = 'long' if action.lower() == 'buy' else 'short'

    # --- 1. 웹훅 수신 후 모든 데이터 갱신 ---
    logger.info("Webhook received. Refreshing all market and technical data...")
    market_data_df = get_market_data(okx_symbol, timeframe=TRADING_TIMEFRAME, limit=CANDLE_LIMIT)  # ✅ 전역 변수 사용
    btc_trend, btc_raw_data = get_btc_trend_and_data()
    tech_indicators = calculate_technical_indicators(market_data_df)

    current_market_price = market_data_df['close'].iloc[
        -1] if market_data_df is not None and not market_data_df.empty else None
    atr_levels, atr_stage = None, None
    if tech_indicators and 'current_atr' in tech_indicators and current_market_price is not None:
        atr_levels = calculate_atr_levels(current_market_price, tech_indicators['current_atr'], tradingview_direction)
        atr_stage = calculate_atr_stage(tech_indicators['current_atr'])

    # --- 2. 갱신된 데이터를 기반으로 AI 결정 요청 ---
    ai_decision, ai_reason = get_ai_decision(market_data_df, tech_indicators, btc_trend, okx_symbol, btc_raw_data)

    # --- 3. 주문 실행 및 후속 조치 ---
    ai_tp_price, ai_sl_price, ai_recommended_trailing_callback_ratio, ai_tpsl_reason = None, None, None, None
    effective_trailing_callback_ratio = None
    tp_sl_confirmation_status = "미실행 (주문 조건 불충족)"

    if (ai_decision and ai_decision == tradingview_direction) or (ai_decision is None and tradingview_direction):
        if ai_decision is None:
            logger.warning(f"AI decision unclear. Proceeding with TradingView signal '{tradingview_direction}'.")
        else:
            logger.info(f"MATCH: AI decision '{ai_decision}' matches TV signal. Proceeding with trade.")

        order_result = place_order(okx_symbol, amount, action)

        if "error" not in order_result:
            time.sleep(3)  # 포지션 체결 대기
            current_positions_after_trade = get_open_positions(okx_symbol)
            active_pos = next((p for p in current_positions_after_trade if float(p.get('pos', 0)) != 0), None)

            if active_pos:
                ai_tp_price, ai_sl_price, ai_recommended_trailing_callback_ratio, ai_tpsl_reason = get_ai_tp_sl_recommendation(
                    [active_pos], market_data_df, tech_indicators, btc_trend, okx_symbol, atr_levels, btc_raw_data)

                if ai_recommended_trailing_callback_ratio is not None and atr_stage is not None:
                    effective_trailing_callback_ratio = max(1.0, ai_recommended_trailing_callback_ratio * atr_stage)
                    trailing_stop_result = place_trailing_stop_order(okx_symbol, active_pos['posSide'],
                                                                     effective_trailing_callback_ratio,
                                                                     abs(active_pos['pos']))
                    if "error" in trailing_stop_result:
                        logger.error(f"Failed to place Trailing Stop: {trailing_stop_result['detail']}")
                else:
                    logger.warning("Skipping Trailing Stop: AI recommendation or ATR stage unavailable.")

                if ai_tp_price is not None and ai_sl_price is not None:
                    user_input = None

                    def get_user_input():
                        nonlocal user_input
                        try:
                            auto_confirm = "자동 Y" if DEFAULT_TP_SL_CONFIRM_TO_YES else "자동 N"
                            user_input = input(
                                f"AI 추천 TP/SL ({ai_tp_price:.1f}/{ai_sl_price:.1f}) 주문? (Y/N, {USER_CONFIRM_TIMEOUT_SECONDS}초 후 {auto_confirm}): ").strip().lower()
                        except:
                            user_input = 'y' if DEFAULT_TP_SL_CONFIRM_TO_YES else 'n'

                    input_thread = threading.Thread(target=get_user_input)
                    input_thread.start()
                    input_thread.join(timeout=USER_CONFIRM_TIMEOUT_SECONDS)

                    if user_input == 'y':
                        tpsl_order_result = place_algo_order_tpsl(okx_symbol, active_pos['posSide'], ai_tp_price,
                                                                  ai_sl_price, abs(active_pos['pos']))
                        if "error" not in tpsl_order_result:
                            tp_sl_confirmation_status = "설정됨 (사용자 승인)"
                        else:
                            tp_sl_confirmation_status = f"설정 실패: {tpsl_order_result['detail']}"
                    else:
                        tp_sl_confirmation_status = "미설정 (사용자 거부 또는 타임아웃)"
                else:
                    tp_sl_confirmation_status = "미설정 (AI 추천 실패)"
            else:
                tp_sl_confirmation_status = "미설정 (포지션 확인 실패)"
        else:
            tp_sl_confirmation_status = "미설정 (시장가 주문 실패)"
    else:
        logger.warning(f"NO MATCH: AI decision '{ai_decision}' != TV signal '{tradingview_direction}'. Order skipped.")
        tp_sl_confirmation_status = "미설정 (신호 불일치)"

    # --- 4. 최종 상태 로깅 ---
    final_balance = get_balance()
    final_positions = get_open_positions(okx_symbol)
    final_btc_price = btc_raw_data['1h']['close'].iloc[-1] if '1h' in btc_raw_data and not btc_raw_data[
        '1h'].empty else None

    log_trade_status("실행 종료", final_balance, final_positions, okx_symbol,
                     tradingview_direction=tradingview_direction, tech_indicators=tech_indicators,
                     btc_trend=btc_trend, ai_decision=ai_decision, ai_reason=ai_reason,
                     ai_tp_price=ai_tp_price, ai_sl_price=ai_sl_price, ai_tpsl_reason=ai_tpsl_reason,
                     current_btc_price=final_btc_price,
                     recommended_trailing_callback_ratio=ai_recommended_trailing_callback_ratio,
                     effective_trailing_callback_ratio=effective_trailing_callback_ratio,
                     atr_stage=atr_stage, atr_levels=atr_levels,
                     tp_sl_confirmation_status=tp_sl_confirmation_status)

    return jsonify({"status": "processed"}), 200


# =============================================================================
# 7. RUN THE FLASK APPLICATION
# =============================================================================
if __name__ == '__main__':
    # Make sure to install dependencies: pip install flask pandas requests
    app.run(host='0.0.0.0', port=80)