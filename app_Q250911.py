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

# Changed to Gemini 2.5 Flash
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
AI_RESPONSE_TIMEOUT_SECONDS = 30  # AI 응답을 위해 타임아웃을 넉넉하게 설정

# --- User Confirmation Settings ---
# TP/SL 설정을 기본적으로 'N' (자동 미실행)으로 설정합니다.
# 'Y'를 입력해야만 AI 추천 TP/SL이 설정됩니다.
# True로 바꾸면 10초 후 자동 'Y'로 동작합니다.
DEFAULT_TP_SL_CONFIRM_TO_YES = False
USER_CONFIRM_TIMEOUT_SECONDS = 30  # 사용자 컨펌 대기 시간

# --- ATR Stage Boundaries (ATR 값에 따른 변동성 단계 구분 기준, 중요: 반드시 실제 시장에 맞춰 조정 필요) ---
# 예시: ATR 값이 100 이하면 1단계, 100~200은 2단계, 200 초과는 3단계
# 이 값은 사용하시는 심볼, 시간 프레임, 그리고 ATR의 평균적인 크기에 따라 설정해야 합니다.
# BTC-USDT 5분봉 기준으로 대략적인 값이며, 실제 데이터 확인 후 정교하게 설정 필요.
ATR_STAGE_BOUNDARIES = {
    "STAGE1_MAX": 100.0,  # ATR이 이 값 이하이면 1단계
    "STAGE2_MAX": 200.0  # ATR이 이 값 이하이면 2단계 (이 값을 초과하면 3단계)
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
            # Convert pos to float and determine posSide if pos is non-zero
            for p in positions:
                if p.get('pos') is not None:
                    pos_val = float(p['pos'])
                    p['pos'] = pos_val  # Update to float
                    if pos_val > 0:
                        p['posSide'] = 'long'
                    elif pos_val < 0:
                        p['posSide'] = 'short'
                    else:
                        p['posSide'] = 'none'  # No position
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
                    time.sleep(delay)  # Wait before retrying
                    continue  # Try again
                return None  # All retries failed

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
                time.sleep(delay)  # Wait before retrying
                continue  # Try again
            return None  # All retries failed
    return None  # Should not be reached if retries > 0


def calculate_technical_indicators(df):
    """
    Calculates key technical indicators and returns the most recent values.
    Returns a dictionary with the latest indicator values.
    """
    if df is None or len(df) < 50:  # Need enough data for calculations
        logger.warning("Not enough market data to calculate all technical indicators.")
        return None

    indicators = {}

    # --- ATR ---
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['atr'] = df['tr'].ewm(span=14, adjust=False).mean()
    indicators['current_atr'] = df['atr'].iloc[-1]
    indicators['average_atr'] = df['atr'].mean()

    # --- RSI ---
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    indicators['rsi'] = df['rsi'].iloc[-1]

    # --- Moving Averages ---
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    indicators['sma20'] = df['sma20'].iloc[-1]
    indicators['sma50'] = df['sma50'].iloc[-1]

    # --- MACD ---
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
    Fetches BTC-USDT data for 1-minute, 1-hour, and 4-hour timeframes
    and determines the general market trend based on combined analysis.
    Returns trend string and a dictionary of raw BTC dataframes.
    """
    logger.info("Fetching BTC-USDT data for market trend analysis...")

    btc_data = {}

    # Fetch 1-hour and 4-hour candle data (1m removed as requested)
    btc_df_1h = get_market_data("BTC-USDT-SWAP", timeframe='1h', limit=100)
    btc_df_4h = get_market_data("BTC-USDT-SWAP", timeframe='4h', limit=50)

    if btc_df_1h is not None and not btc_df_1h.empty:
        btc_data['1h'] = btc_df_1h
    if btc_df_4h is not None and not btc_df_4h.empty:
        btc_data['4h'] = btc_df_4h

    trend_1h = "Uncertain"
    trend_4h = "Uncertain"

    # Check minimum required candles for analysis
    MIN_CANDLES_1H = 50
    MIN_CANDLES_4H = 20

    # Analyze 1-hour trend
    if btc_df_1h is not None and len(btc_df_1h) >= MIN_CANDLES_1H:
        btc_df_1h['sma50'] = btc_df_1h['close'].rolling(window=50).mean()
        last_close_1h = btc_df_1h['close'].iloc[-1]
        last_sma50_1h = btc_df_1h['sma50'].iloc[-1]

        if last_close_1h > last_sma50_1h * 1.005:  # Price is 0.5% above 50-period SMA
            trend_1h = "Bullish"
        elif last_close_1h < last_sma50_1h * 0.995:  # Price is 0.5% below 50-period SMA
            trend_1h = "Bearish"
        else:
            trend_1h = "Sideways/Consolidating"
        logger.info(f"BTC 1-hour Trend: {trend_1h} (Last Close: {last_close_1h:.2f}, 50-hr SMA: {last_sma50_1h:.2f})")
    else:
        logger.warning(
            f"Not enough 1-hour BTC data for trend analysis. ({len(btc_df_1h) if btc_df_1h is not None else 0} of min {MIN_CANDLES_1H} candles required)")

    # Analyze 4-hour trend
    if btc_df_4h is not None and len(btc_df_4h) >= MIN_CANDLES_4H:
        btc_df_4h['sma20'] = btc_df_4h['close'].rolling(window=20).mean()
        last_close_4h = btc_df_4h['close'].iloc[-1]
        last_sma20_4h = btc_df_4h['sma20'].iloc[-1]

        if last_close_4h > last_sma20_4h * 1.01:  # Price is 1% above 20-period SMA
            trend_4h = "Strongly Bullish"
        elif last_close_4h < last_sma20_4h * 0.99:  # Price is 1% below 20-period SMA
            trend_4h = "Strongly Bearish"
        else:
            trend_4h = "Sideways"
        logger.info(f"BTC 4-hour Trend: {trend_4h} (Last Close: {last_close_4h:.2f}, 20-4hr SMA: {last_sma20_4h:.2f})")
    else:
        logger.warning(
            f"Not enough 4-hour BTC data for trend analysis. ({len(btc_df_4h) if btc_df_4h is not None else 0} of min {MIN_CANDLES_4H} candles required)")

    # Combine trends for a final verdict
    if "Bullish" in trend_1h and "Bullish" in trend_4h:
        final_trend = "Strongly Bullish"
    elif "Bearish" in trend_1h and "Bearish" in trend_4h:
        final_trend = "Strongly Bearish"
    elif "Bullish" in trend_1h and "Sideways" in trend_4h:
        final_trend = "Weakly Bullish"
    elif "Bearish" in trend_1h and "Sideways" in trend_4h:
        final_trend = "Weakly Bearish"
    elif "Strongly Bullish" in trend_4h:  # If 4-hour trend is strong, prioritize it
        final_trend = "Strongly Bullish"
    elif "Strongly Bearish" in trend_4h:
        final_trend = "Strongly Bearish"
    else:
        final_trend = "Uncertain/Mixed"  # Mixed or uncertain conditions

    return final_trend, btc_data


def calculate_atr_levels(current_price, current_atr, position_side):
    """
    Calculates TP/SL levels based on ATR in 3 stages.
    These multipliers are tuned to provide wider ranges for TP/SL.
    """
    # TP/SL 단계별 ATR 배수 (더 넓은 범위를 위해 조정)
    TP_MULTIPLIERS = [2.0, 3.5, 5.0]
    SL_MULTIPLIERS = [1.5, 2.0, 2.5]

    levels = {
        "TP1": None, "SL1": None,
        "TP2": None, "SL2": None,
        "TP3": None, "SL3": None
    }

    if current_atr is None or current_atr <= 0:
        logger.warning("Cannot calculate ATR levels, ATR is invalid.")
        return levels

    # 롱 포지션
    if position_side == 'long':
        for i in range(3):
            levels[f"TP{i + 1}"] = current_price + (TP_MULTIPLIERS[i] * current_atr)
            levels[f"SL{i + 1}"] = current_price - (SL_MULTIPLIERS[i] * current_atr)
    # 숏 포지션
    elif position_side == 'short':
        for i in range(3):
            levels[f"TP{i + 1}"] = current_price - (TP_MULTIPLIERS[i] * current_atr)
            levels[f"SL{i + 1}"] = current_price + (SL_MULTIPLIERS[i] * current_atr)
    else:
        logger.warning(f"Invalid position side '{position_side}' for ATR level calculation.")
        return levels

    # 소수점 1자리로 반올림 (출력 요구사항에 맞춰)
    for key in levels:
        if levels[key] is not None:
            levels[key] = round(levels[key], 1)

    logger.info(f"Calculated ATR Levels for {position_side}: {levels}")
    return levels


def calculate_atr_stage(current_atr):
    """
    Determines the ATR volatility stage (1, 2, or 3) based on predefined boundaries.
    """
    if current_atr is None:
        return 1  # Default to lowest stage if ATR is unavailable

    if current_atr <= ATR_STAGE_BOUNDARIES["STAGE1_MAX"]:
        return 1
    elif current_atr <= ATR_STAGE_BOUNDARIES["STAGE2_MAX"]:
        return 2
    else:
        return 3


def get_ai_decision(data_frame, tech_indicators, btc_trend, target_symbol, btc_raw_data=None):
    """
    Gets a structured trading decision and reason from the Gemini model using enhanced data.
    """
    if data_frame is None or data_frame.empty or tech_indicators is None:
        logger.warning("No market data or technical indicators to send to AI for decision.")
        return None, "시장 데이터 또는 기술 지표 없음"

    # Send last 100 records for more context
    ai_data = data_frame.drop(columns=['timestamp']).tail(100).to_json(orient='records')
    indicators_json = json.dumps({k: f'{v:.4f}' for k, v in tech_indicators.items()})

    btc_raw_data_json = ""
    if btc_raw_data:
        for tf, df in btc_raw_data.items():
            if df is not None and not df.empty:
                # BTC 데이터를 JSON으로 변환하여 프롬프트에 포함 (각 시간 프레임별 최신 50개 캔들)
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: {df.drop(columns=['timestamp']).tail(50).to_json(orient='records')}"
            else:
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: 데이터 없음"

    prompt_content = (
        "You are a master cryptocurrency trading analyst. Your task is to provide a trading recommendation "
        "based on a comprehensive dataset. I will provide you with raw **15-minute** OHLCV data for the target symbol, "  # 15분봉 명시
        "a summary of key technical indicators, and the general trend of the BTC-USDT market. "
        "Additionally, I will provide recent raw OHLCV data for BTC-USDT-SWAP across different timeframes (1h, 4h) for enhanced market context."  # 1m 제거
        "\n\n**Analysis Instructions:**"
        "\n1.  **Primary Data & Price Action (Target Symbol - 15m):** Analyze the provided **5-minute** OHLCV data for **candlestick patterns, momentum, volume trends, and significant price action.** Identify potential **support/resistance zones, trend lines, and any signs of market structure shifts.** Consider the **recent price history** beyond just the immediate candles."  # 15분봉 명시
        "\n2.  **Technical Indicators:** Use the provided technical indicators (RSI, MACD, SMAs, ATR) to "
        "confirm or challenge your analysis from the primary data. Specifically, check for **divergences, crossovers, overbought/oversold conditions, and how current price relates to key moving averages (SMA20, SMA50).** Use ATR to gauge volatility and potential stop-loss distances."
        "\n3.  **Market Context (BTC Trend & Data):** This is crucial. Consider the **overall BTC-USDT market trend ('Strongly Bullish', 'Weakly Bullish', 'Strongly Bearish', 'Weakly Bearish', 'Uncertain/Mixed').** Furthermore, deeply analyze the **provided BTC 1-hour and 4-hour OHLCV data to gauge broader market sentiment and volatility**, which can significantly impact the target symbol. A **long position is significantly safer in a bullish BTC market**, and a **short position is significantly safer in a bearish one.** If BTC trend is 'Strongly Bullish', prioritize long opportunities on the target symbol. If 'Strongly Bearish', prioritize short opportunities. If 'Uncertain/Mixed', approach with caution and rely more on the target symbol's specific technicals, potentially favoring sideways strategies or no trade."  # BTC 데이터 분석 지시 업데이트 (1m 제거)
        "\n\n**Your Response:**"
        "\nBased on your complete analysis, you must provide a clear, final decision. "
        "The decision MUST be 'long' or 'short'."
        "The reason should be a concise explanation in Korean, no more than 3-4 sentences, integrating insights from the data, "
        "indicators, and the BTC trend, explicitly considering the impact of BTC's overall market direction."  # BTC 단기 변동성 영향 문구 제거
        "\nRespond ONLY with a JSON object containing two fields: 'decision' (string) and 'reason' (string). "
        "Example: {\"decision\": \"long\", \"reason\": \"BTC 시장이 강세이며, 대상 종목이 주요 이동평균선 위에서 지지받고 있습니다. RSI가 과매수 구간이 아니므로 상승 여력이 있습니다.\"}"
        f"\n\n--- DATA FOR ANALYSIS ---"
        f"\n**Target Symbol:** {target_symbol}"
        f"\n**Overall BTC Trend:** {btc_trend}"
        f"\n**Calculated Technical Indicators (Current):** {indicators_json}"
        f"\n**Recent Market Data (OHLCV) - Target Symbol (15m):** {ai_data}"  # 15분봉 명시
        f"\n**Recent BTC-USDT-SWAP Market Data (OHLCV) - Various Timeframes:** {btc_raw_data_json}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {"decision": {"type": "STRING"}, "reason": {"type": "STRING"}},
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

            if (result.get('candidates') and result['candidates'][0].get('content') and
                    result['candidates'][0]['content'].get('parts')):
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(json_string)
                decision = parsed_json.get('decision', '').strip().lower()
                reason = parsed_json.get('reason', '이유 설명 없음').strip()
                if decision in ['long', 'short']:
                    result_container['decision'] = decision
                    result_container['reason'] = reason
                else:
                    logger.warning(f"AI returned invalid decision: '{decision}'. Reason: '{reason}'.")
                    result_container['reason'] = "AI가 유효한 결정 (long/short)을 반환하지 않았습니다."
            else:
                logger.error(f"Unexpected AI response structure: {result}")
                result_container['reason'] = "AI 응답 구조 오류"
        except requests.exceptions.Timeout:
            logger.error("AI request timed out.")
            result_container['reason'] = "AI 요청 시간 초과"
        except json.JSONDecodeError as e:
            logger.error(
                f"AI response was not valid JSON: {e}. Response text: {response.text if response else 'No response body'}")
            result_container['reason'] = "AI 응답이 유효한 JSON이 아닙니다."
        except Exception as e:
            logger.error(f"An error occurred during AI call: {e}")
            result_container['reason'] = f"AI 호출 중 예외 발생: {e}"

    ai_thread = threading.Thread(target=call_gemini)
    ai_thread.start()
    ai_thread.join(timeout=AI_RESPONSE_TIMEOUT_SECONDS + 5)

    final_decision = result_container.get('decision')
    final_reason = result_container.get('reason')
    logger.info(f"Final AI decision: {final_decision}, Reason: {final_reason}")
    return final_decision, final_reason


def get_ai_tp_sl_recommendation(position_data, market_data_df, tech_indicators, btc_trend, target_symbol, atr_levels,
                                btc_raw_data=None):
    """
    Gets TP/SL price and trailing stop callback ratio recommendations from the Gemini model
    based on current position and market data, including ATR-based levels and BTC raw data.
    """
    if position_data is None or not position_data or market_data_df is None or tech_indicators is None:
        logger.warning("No position, market data or technical indicators to send to AI for TP/SL.")
        return None, None, None, "TP/SL 추천을 위한 데이터 부족"

    current_position = next(
        (p for p in position_data if p.get('instId') == target_symbol and float(p.get('pos', 0)) != 0), None)

    if not current_position:
        logger.warning(f"No open position found for {target_symbol} to recommend TP/SL.")
        return None, None, None, "현재 오픈 포지션 없음"

    position_side = current_position.get('posSide')  # 'long' or 'short'
    avg_entry_price = float(current_position.get('avgPx'))
    position_size = abs(float(current_position.get('pos')))
    last_close_price = market_data_df['close'].iloc[-1]
    current_atr = tech_indicators.get('current_atr')

    ai_data = market_data_df.drop(columns=['timestamp']).tail(100).to_json(orient='records')
    indicators_json = json.dumps({k: f'{v:.4f}' for k, v in tech_indicators.items()})
    atr_levels_json = json.dumps(atr_levels)  # ATR 레벨 추가

    btc_raw_data_json = ""
    if btc_raw_data:
        for tf, df in btc_raw_data.items():
            if df is not None and not df.empty:
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: {df.drop(columns=['timestamp']).tail(50).to_json(orient='records')}"
            else:
                btc_raw_data_json += f"\n  - BTC-{tf} OHLCV: 데이터 없음"

    prompt_content = (
        "You are a master cryptocurrency trading analyst specializing in risk management and profit taking. "
        "Your task is to recommend optimal Take Profit (TP) and Stop Loss (SL) prices for an existing position, "
        "and also recommend an appropriate Trailing Stop callback ratio."
        "\n\n**Given Position Details:**"
        f"\n- **Symbol:** {target_symbol}"
        f"\n- **Position Side:** {position_side.upper()}"
        f"\n- **Average Entry Price:** {avg_entry_price:.4f}"
        f"\n- **Position Size:** {position_size}"
        f"\n- **Current Price (Last Close):** {last_close_price:.4f}"
        f"\n- **Current ATR:** {current_atr:.4f}"  # 현재 ATR 값 명시
        "\n\n**Analysis Instructions:**"
        "\n1.  **Market Context (BTC Trend & Data):** Crucially consider the **overall BTC-USDT market trend ('Strongly Bullish', 'Weakly Bullish', 'Strongly Bearish', 'Weakly Bearish', 'Uncertain/Mixed').** Also, examine the **provided raw BTC OHLCV data across timeframes (1h, 4h)** to understand broader market sentiment and volatility. This should strongly influence the aggressiveness of TP and tightness of SL, as well as the trailing stop strategy."  # 1m 제거
        "\n2.  **Primary Data & Price Action (Target Symbol - 15m):** Analyze the provided **15-minute** OHLCV data for **recent volatility, key support/resistance levels, and recent price action.** Look for areas where price might reverse or accelerate."  # 15분봉 명시
        "\n3.  **Technical Indicators & ATR Levels:** Use the provided technical indicators (RSI, MACD, SMAs, ATR) to "
        "identify potential overbought/oversold conditions, momentum shifts, and volatility levels. **Crucially, consider the provided ATR-based TP/SL levels (TP1/SL1, TP2/SL2, TP3/SL3) as potential targets and protection points.** Use these levels as a guide, adjusting based on other market factors. A **higher ATR suggests higher volatility, potentially requiring wider TP/SL and a larger trailing stop callback ratio.** Conversely, **lower ATR suggests tighter settings.**"
        "\n\n**Your Response:**"
        "\nBased on your complete analysis, provide precise TP and SL prices, and a recommended Trailing Stop callback ratio (as a percentage, e.g., 5.0 for 5%). "
        "The prices should be realistic and actionable, based on current market conditions and risk management principles. "
        "Ensure TP is higher than current price for long, lower for short. Ensure SL protects against significant loss. "
        "The **Trailing Stop callback ratio should be dynamically adjusted based on volatility (ATR) and overall market trend.** For instance, a more volatile market (higher ATR) or strong trend might suggest a larger callback ratio to allow for more swing, while a less volatile or range-bound market might require a tighter callback ratio."  # 1m 제거
        "\nThe reason should be a concise explanation in Korean, no more than 3-4 sentences, integrating insights from all provided data."
        "\nRespond ONLY with a JSON object containing four fields: 'take_profit_price' (float), 'stop_loss_price' (float), 'recommended_trailing_callback_ratio' (float), and 'reason' (string)."
        "Example: {\"take_profit_price\": 1234.5, \"stop_loss_price\": 1200.0, \"recommended_trailing_callback_ratio\": 8.0, \"reason\": \"현재 강세 시장에서 대상 종목이 강한 지지선을 형성하고 있으며, 높은 ATR을 고려하여 합리적인 TP/SL 및 트레일링 스탑을 설정했습니다.\"}"
        f"\n\n--- DATA FOR ANALYSIS ---"
        f"\n**Overall BTC Trend:** {btc_trend}"
        f"\n**Calculated Technical Indicators (Current):** {indicators_json}"
        f"\n**ATR Based TP/SL Levels (Reference):** {atr_levels_json}"  # AI에게 ATR 레벨 제공 (참고용으로)
        f"\n**Recent Market Data (OHLCV) - Target Symbol (15m):** {ai_data}"  # 15분봉 명시
        f"\n**Recent BTC-USDT-SWAP Market Data (OHLCV) - Various Timeframes:** {btc_raw_data_json}"  # BTC 원본 데이터 추가
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "take_profit_price": {"type": "NUMBER"},
                    "stop_loss_price": {"type": "NUMBER"},
                    "recommended_trailing_callback_ratio": {"type": "NUMBER"},  # 트레일링 콜백 비율 추가
                    "reason": {"type": "STRING"}
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

            if (result.get('candidates') and result['candidates'][0].get('content') and
                    result['candidates'][0]['content'].get('parts')):
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(json_string)
                tp_price = parsed_json.get('take_profit_price')
                sl_price = parsed_json.get('stop_loss_price')
                # 새로운 필드: 트레일링 콜백 비율
                recommended_trailing_callback_ratio = parsed_json.get('recommended_trailing_callback_ratio')
                reason = parsed_json.get('reason', '이유 설명 없음').strip()

                if isinstance(tp_price, (int, float)) and isinstance(sl_price, (int, float)) and isinstance(
                        recommended_trailing_callback_ratio, (int, float)):
                    result_container['take_profit_price'] = float(tp_price)
                    result_container['stop_loss_price'] = float(sl_price)
                    result_container['recommended_trailing_callback_ratio'] = float(recommended_trailing_callback_ratio)
                    result_container['reason'] = reason
                else:
                    logger.warning(
                        f"AI returned invalid TP/SL/Trailing prices: TP={tp_price}, SL={sl_price}, Trailing={recommended_trailing_callback_ratio}. Reason: {reason}.")
                    result_container['reason'] = "AI가 유효한 TP/SL/트레일링 값을 반환하지 않았습니다."
            else:
                logger.error(f"Unexpected AI response structure for TP/SL/Trailing: {result}")
                result_container['reason'] = "AI 응답 구조 오류 (TP/SL/Trailing)"
        except requests.exceptions.Timeout:
            logger.error("AI TP/SL/Trailing request timed out.")
            result_container['reason'] = "AI TP/SL/트레일링 요청 시간 초과"
        except json.JSONDecodeError as e:
            logger.error(
                f"AI TP/SL/Trailing response was not valid JSON: {e}. Response text: {response.text if response else 'No response body'}")
            result_container['reason'] = "AI 응답이 유효한 JSON이 아닙니다 (TP/SL/Trailing)."
        except Exception as e:
            logger.error(f"An error occurred during AI TP/SL/Trailing call: {e}")
            result_container['reason'] = f"AI 호출 중 예외 발생: {e}"

    ai_thread = threading.Thread(target=call_gemini_tp_sl)
    ai_thread.start()
    ai_thread.join(timeout=AI_RESPONSE_TIMEOUT_SECONDS + 5)  # Give a bit more time for complex AI calc

    final_tp = result_container.get('take_profit_price')
    final_sl = result_container.get('stop_loss_price')
    final_trailing = result_container.get('recommended_trailing_callback_ratio')
    final_reason = result_container.get('reason')
    logger.info(f"Final AI TP={final_tp}, SL={final_sl}, Trailing={final_trailing}, Reason: {final_reason}")
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Order placement failed: {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"error": "Order failed", "detail": str(e)}


def place_algo_order_tpsl(symbol, pos_side, tp_price, sl_price, position_size):
    """
    Places a Take Profit and Stop Loss algorithm order.
    pos_side: 'long' or 'short'
    """
    url_path = "/api/v5/trade/order-algo"

    # 포지션 방향에 따라 주문 방향 설정
    if pos_side == 'long':
        close_side = 'sell'
    elif pos_side == 'short':
        close_side = 'buy'
    else:
        logger.error(f"Invalid position side for TP/SL: {pos_side}")
        return {"error": "Invalid position side"}

    body = {
        "instId": symbol,
        "tdMode": "cross",  # Cross-margin mode
        "side": close_side,
        "ordType": "conditional",  # 조건부 주문 (TP/SL)
        "sz": str(abs(position_size)),  # 포지션 전체 청산 (절대값 사용)
        "tpTriggerPx": str(tp_price),  # Take Profit 트리거 가격
        "tpOrdPx": "-1",  # TP 주문 시장가 (-1)
        "slTriggerPx": str(sl_price),  # Stop Loss 트리거 가격
        "slOrdPx": "-1"  # SL 주문 시장가 (-1)
    }
    json_body = json.dumps(body)
    headers = generate_headers("POST", url_path, body=json_body)

    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get('code') == '0':
            logger.info(
                f"Successfully placed TP/SL algo order for {symbol} ({pos_side}). Algo ID: {res_data['data'][0]['algoId']}")
            return res_data
        else:
            logger.error(f"TP/SL algo order failed: {res_data}. Request body: {json_body}")
            return {"error": "TP/SL failed", "detail": res_data}
    except requests.exceptions.RequestException as e:
        logger.error(
            f"TP/SL algo order placement failed: {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"error": "TP/SL failed", "detail": str(e)}


def place_trailing_stop_order(symbol, entry_side, callback_ratio, position_size):
    """
    Places a trailing stop loss order to protect a position.
    Added position_size to resolve 'sz can't be empty' error.
    """
    url_path = "/api/v5/trade/order-algo"
    close_side = "sell" if entry_side.lower() == "long" else "buy"
    body = {
        "instId": symbol,
        "tdMode": "cross",
        "side": close_side,
        "ordType": "move_order_stop",  # 트레일링 스탑 주문 타입
        "sz": str(abs(position_size)),  # 포지션 크기 (오류 수정)
        "callbackRatio": str(callback_ratio / 100)  # 예: '0.10' for 10%
    }
    json_body = json.dumps(body)
    headers = generate_headers("POST", url_path, body=json_body)

    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get('code') == '0':
            logger.info(
                f"Successfully placed trailing stop order for {symbol} ({entry_side}). Algo ID: {res_data['data'][0]['algoId']}")
            return res_data
        else:
            logger.error(f"Trailing stop order failed: {res_data}. Request body: {json_body}")
            return {"error": "Trailing stop failed", "detail": res_data}
    except requests.exceptions.RequestRequestException as e:
        logger.error(f"Trailing stop order failed: {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"error": "Trailing stop failed", "detail": str(e)}


def cancel_okx_algo_order(algo_id, symbol):
    """
    Cancels an existing algorithm order (e.g., TP/SL or trailing stop order) on OKX.
    """
    url_path = "/api/v5/trade/cancel-algos"
    body = {
        "algoId": algo_id,
        "instId": symbol
    }
    json_body = json.dumps([body])  # OKX는 리스트 형태로 받음
    headers = generate_headers("POST", url_path, body=json_body)

    try:
        response = requests.post(BASE_URL + url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        if res_data.get('code') == '0':
            logger.info(f"Successfully cancelled algo order {algo_id} for {symbol}: {res_data}")
            return res_data
        else:
            logger.error(f"Failed to cancel algo order {algo_id} for {symbol}, OKX API error: {res_data}")
            return {"error": "Cancellation failed", "detail": res_data}
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Algo order cancellation failed: {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"error": "Cancellation failed", "detail": str(e)}


# =============================================================================
# 5. LOGGING & STATUS
# =============================================================================

def log_trade_status(message, balance, positions, okx_symbol, tradingview_direction=None,
                     tech_indicators=None, btc_trend=None, ai_decision=None, ai_reason=None,
                     ai_tp_price=None, ai_sl_price=None, ai_tpsl_reason=None,
                     current_btc_price=None, recommended_trailing_callback_ratio=None,
                     effective_trailing_callback_ratio=None, atr_stage=None, atr_levels=None,
                     tp_sl_confirmation_status=None, btc_raw_data=None):
    """Logs the current status in a comprehensive and readable format."""
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    print(f"\n{'=' * 60}")
    print(f"       *** {message}: {timestamp} ***")
    print(f"{'=' * 60}\n")

    # --- 계정 및 포지션 상태 ---
    print(f"{'─' * 25} 계정 및 포지션 상태 {'─' * 25}")
    print(f"  - 선물 지갑 잔고: {balance:.1f} USDT")

    if current_btc_price is not None:
        print(f"  - 현재 BTC-USDT 가격: {current_btc_price:.1f}")

    open_positions = [p for p in positions if p.get('pos') and float(p.get('pos')) != 0]
    if open_positions:
        print("\n  - 현재 보유 포지션:")
        total_pnl = sum(float(p.get('upl', '0')) for p in open_positions)
        for pos in open_positions:
            pos_size = abs(pos.get('pos'))
            pos_side_text = "롱" if pos.get('posSide') == 'long' else "숏"
            pnl = float(pos.get('upl', '0'))
            pnl_ratio = float(pos.get('uplRatio', '0')) * 100
            print(
                f"    ▶️ 심볼: {pos.get('instId')} | 방향: {pos_side_text} | 크기: {pos_size:.1f} | PNL: {pnl:.1f} USDT ({pnl_ratio:.1f}%)")
        print(f"  - 총 미실현 PNL: {total_pnl:.1f} USDT")
    else:
        print("\n  - 현재 보유 포지션: 없음")
    print(f"{'─' * 60}\n")

    # --- 시장 분석 정보 ---
    print(f"{'─' * 25} 시장 분석 정보 {'─' * 25}")
    if btc_trend:
        print(f"  - BTC 시장 동향: {btc_trend}")
    if tech_indicators:
        volatility = "높은 변동성" if tech_indicators['current_atr'] > tech_indicators['average_atr'] else "보통 변동성"
        print(f"  - {okx_symbol} 변동성: {volatility} (현재 ATR: {tech_indicators['current_atr']:.4f})")
        print(f"  - ATR 변동성 단계: {atr_stage}단계")  # ATR 단계 출력
        print("  - 기술 지표 요약:")
        print(f"    - RSI: {tech_indicators['rsi']:.1f}")
        print(f"    - MACD: {tech_indicators['macd']:.4f} (Signal: {tech_indicators['macd_signal']:.4f})")
        print(f"    - SMA 20/50: {tech_indicators['sma20']:.1f} / {tech_indicators['sma50']:.1f}")
    if atr_levels:
        print("  - ATR 기반 TP/SL 레벨 (AI 참조용):")
        print(f"    - 1단계: TP {atr_levels['TP1']:.1f}, SL {atr_levels['SL1']:.1f}")
        print(f"    - 2단계: TP {atr_levels['TP2']:.1f}, SL {atr_levels['SL2']:.1f}")
        print(f"    - 3단계: TP {atr_levels['TP3']:.1f}, SL {atr_levels['SL3']:.1f}")

    # BTC 원본 데이터 요약 (가독성을 위해 상세 내용은 생략하고 존재 여부만 표시)
    if btc_raw_data:
        print("\n  - BTC-USDT-SWAP 원본 데이터 (추세 및 변동성 분석에 활용):")
        for tf, df in btc_raw_data.items():
            if df is not None and not df.empty:
                print(f"    - {tf}봉: {len(df)}개 캔들 데이터 존재")
            else:
                print(f"    - {tf}봉: 데이터 없음")

    print(f"{'─' * 60}\n")

    # --- 트레이딩 결정 과정 ---
    print(f"{'─' * 25} 트레이딩 결정 과정 {'─' * 25}")
    if tradingview_direction:
        print(f"  - TradingView 1차 신호: {tradingview_direction.upper()}")

    if ai_decision:
        print(f"  - 🤖 AI 최종 판단: {ai_decision.upper()}")
        print(f"  - 💬 AI 판단 근거: {ai_reason}")
    else:  # AI decision is None
        print(f"  - 🤖 AI 최종 판단: 보류 또는 오류 (TradingView 신호 사용)")
        print(f"  - 💬 AI 판단 근거: {ai_reason}")

    if ai_tp_price is not None and ai_sl_price is not None:
        print(f"  - 🎯 AI 추천 TP 가격: {ai_tp_price:.1f}")
        print(f"  - 🛡️ AI 추천 SL 가격: {ai_sl_price:.1f}")
        print(f"  - 💡 AI 추천 TP/SL 근거: {ai_tpsl_reason}")
        print(f"  - TP/SL 설정 상태: {tp_sl_confirmation_status}")
        print(f"    (TP/SL은 기본 '미설정'이며, 수동 'Y' 입력 시에만 설정됩니다.)")

    if recommended_trailing_callback_ratio is not None:
        print(f"  - ⛓️ AI 추천 트레일링 스탑 콜백 비율: {recommended_trailing_callback_ratio:.1f}%")
        if effective_trailing_callback_ratio is not None:
            print(f"  - 📈 최종 적용 트레일링 스탑 콜백 비율 (ATR 단계 {atr_stage} 반영): {effective_trailing_callback_ratio:.1f}%")
        print(f"    (이 비율은 계약 주문 시 항상 자동 반영됩니다.)")
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
    """Receives signals from TradingView, gets enhanced AI decision, and trades if they match."""
    if not request.is_json:
        logger.warning("Received non-JSON request to webhook.")
        abort(415)

    data = request.get_json()
    logger.info(f"Received data from TradingView: {json.dumps(data)}")

    symbol = data.get('symbol')
    action = data.get('action')  # 'buy' or 'sell'
    amount = data.get('amount')

    if not all([symbol, action, amount]):
        logger.error(f"Webhook failed: Missing 'symbol', 'action', or 'amount'.")
        return jsonify({"error": "Missing required fields"}), 400

    okx_symbol = symbol.replace('USDT.P', '-USDT-SWAP')
    tradingview_direction = 'long' if action.lower() == 'buy' else 'short'

    # --- 초기화: UnboundLocalError 방지 ---
    ai_tp_price = None
    ai_sl_price = None
    ai_recommended_trailing_callback_ratio = None
    ai_tpsl_reason = None
    effective_trailing_callback_ratio = None
    tp_sl_confirmation_status = "미설정 (기본값 'N')"  # 기본 상태 명시
    atr_stage = None  # atr_stage도 초기화

    # --- 초기 상태 로깅을 위한 데이터 수집 ---
    balance = get_balance()
    positions = get_open_positions(okx_symbol)

    # 대상 심볼은 15분봉 200개 요청
    market_data_df = get_market_data(okx_symbol, timeframe='15m', limit=200)  # 변경된 부분
    current_market_price = market_data_df['close'].iloc[
        -1] if market_data_df is not None and not market_data_df.empty else None

    # BTC 데이터 및 추세 분석 (1분봉 제거, 1h, 4h 유지)
    btc_trend, btc_raw_data = get_btc_trend_and_data()
    # current_btc_price는 btc_raw_data에서 1m 대신 다른 적절한 시간 프레임의 최신 가격으로 가져올 수 있습니다.
    # 여기서는 1h 최신 가격을 사용하거나, 단순히 None으로 두어 경고 메시지를 피할 수도 있습니다.
    current_btc_price = btc_raw_data['1h']['close'].iloc[-1] if '1h' in btc_raw_data and not btc_raw_data[
        '1h'].empty else None

    # --- 메인 로직: AI를 위한 모든 데이터 포인트 가져오기 ---
    tech_indicators = calculate_technical_indicators(market_data_df)

    # ATR 기반 TP/SL 레벨 계산 (대상 심볼의 현재 가격과 ATR을 사용)
    atr_levels = None
    if market_data_df is not None and tech_indicators and 'current_atr' in tech_indicators and current_market_price is not None:
        atr_levels = calculate_atr_levels(current_market_price, tech_indicators['current_atr'], tradingview_direction)
        atr_stage = calculate_atr_stage(tech_indicators['current_atr'])

    # AI의 최종 판단 얻기 (BTC 원본 데이터 전달)
    ai_decision, ai_reason = get_ai_decision(market_data_df, tech_indicators, btc_trend, okx_symbol, btc_raw_data)

    # --- 거래 실행 로직 ---
    # AI 판단이 유효하고, 트레이딩뷰 신호와 일치하거나, AI 판단이 어려운 경우 (None) 트레이딩뷰 신호 그대로 실행
    if (ai_decision and ai_decision == tradingview_direction) or (ai_decision is None and tradingview_direction):
        if ai_decision is None:
            logger.warning(
                f"AI decision is unclear. Proceeding with TradingView signal '{tradingview_direction}' as fallback.")
        else:
            logger.info(
                f"MATCH: AI decision '{ai_decision}' matches TradingView signal '{tradingview_direction}'. Proceeding with trade.")

        order_result = place_order(okx_symbol, amount, action)

        if "error" not in order_result:
            # 포지션이 체결될 때까지 잠시 대기
            time.sleep(3)
            current_positions_after_trade = get_open_positions(okx_symbol)
            active_pos = next((p for p in current_positions_after_trade if
                               p.get('instId') == okx_symbol and float(p.get('pos', 0)) != 0), None)

            if active_pos:
                # AI로부터 TP/SL 및 트레일링 스탑 콜백 비율 추천 받기
                ai_tp_price, ai_sl_price, ai_recommended_trailing_callback_ratio, ai_tpsl_reason = get_ai_tp_sl_recommendation(
                    [active_pos], market_data_df, tech_indicators, btc_trend, okx_symbol, atr_levels, btc_raw_data
                )

                # 트레일링 스탑 주문은 항상 AI 추천 값 (ATR 단계 승수 적용)으로 자동 실행
                if ai_recommended_trailing_callback_ratio is not None and atr_stage is not None:
                    effective_trailing_callback_ratio = ai_recommended_trailing_callback_ratio * atr_stage
                    # 콜백 비율은 0보다 커야 함 (OKX API 요구사항)
                    if effective_trailing_callback_ratio <= 0:
                        effective_trailing_callback_ratio = 1.0  # 최소값 설정 (예: 1%)
                        logger.warning(
                            f"Calculated effective trailing callback ratio was non-positive. Setting to minimum 1.0%.")

                    trailing_stop_result = place_trailing_stop_order(
                        okx_symbol,
                        active_pos['posSide'],
                        effective_trailing_callback_ratio,  # AI 추천 비율 * ATR 단계 승수 적용
                        abs(active_pos['pos'])  # 포지션 크기 전달
                    )
                    if "error" not in trailing_stop_result:
                        logger.info(
                            f"Trailing Stop order placed successfully with effective ratio {effective_trailing_callback_ratio:.1f}%. Algo ID: {trailing_stop_result['data'][0]['algoId']}")
                    else:
                        logger.error(
                            f"Failed to place Trailing Stop order with effective ratio {effective_trailing_callback_ratio:.1f}%: {trailing_stop_result['detail']}")
                else:
                    logger.warning(
                        f"AI could not recommend Trailing Stop ratio or ATR stage unavailable. Skipping Trailing Stop placement. Reason: {ai_tpsl_reason if ai_tpsl_reason else 'Unknown'}")
                    effective_trailing_callback_ratio = None  # 설정 실패 시 None

                # AI가 유효한 TP/SL 추천을 했다면 사용자 컨펌 진행
                if ai_tp_price is not None and ai_sl_price is not None:
                    user_confirmed_tp_sl = False
                    user_input = None

                    def get_user_input_for_tpsl():
                        nonlocal user_input
                        try:
                            # DEFAULT_TP_SL_CONFIRM_TO_YES 값에 따라 자동 Y/N 결정
                            auto_confirm_text = "자동 Y" if DEFAULT_TP_SL_CONFIRM_TO_YES else "자동 N"
                            prompt_text = (
                                f"AI 추천 TP/SL ({ai_tp_price:.1f}/{ai_sl_price:.1f})로 주문하시겠습니까? (Y/N, {USER_CONFIRM_TIMEOUT_SECONDS}초 후 {auto_confirm_text}): "
                            )
                            user_input = input(prompt_text).strip().lower()
                        except EOFError:  # 콘솔 환경에서 타임아웃 발생 시
                            user_input = 'y' if DEFAULT_TP_SL_CONFIRM_TO_YES else 'n'
                            logger.info(f"User input timed out. Defaulting to '{user_input.upper()}' for TP/SL setup.")
                        except Exception as e:  # 그 외 입력 오류
                            logger.error(f"Error getting user input for TP/SL: {e}")
                            user_input = 'y' if DEFAULT_TP_SL_CONFIRM_TO_YES else 'n'

                    input_thread_tpsl = threading.Thread(target=get_user_input_for_tpsl)
                    input_thread_tpsl.start()
                    input_thread_tpsl.join(timeout=USER_CONFIRM_TIMEOUT_SECONDS)

                    # 실제 컨펌 로직: 'y' 입력 시 True, 그 외 (None, '', 'n', 타임아웃) 시 False
                    if user_input == 'y':
                        user_confirmed_tp_sl = True
                        tp_sl_confirmation_status = "설정됨 (사용자 승인)"
                        logger.info("User confirmed AI recommended TP/SL.")
                    else:  # user_input is None, empty, or 'n'
                        user_confirmed_tp_sl = False
                        tp_sl_confirmation_status = "미설정 (사용자 거부 또는 자동 'N')"
                        logger.info("User declined AI recommended TP/SL OR input timed out. Not placing TP/SL orders.")

                    if user_confirmed_tp_sl:
                        tpsl_order_result = place_algo_order_tpsl(
                            okx_symbol,
                            active_pos['posSide'],
                            ai_tp_price,
                            ai_sl_price,
                            abs(active_pos['pos'])
                        )
                        if "error" not in tpsl_order_result:
                            logger.info(
                                f"AI recommended TP/SL orders placed successfully. Algo ID: {tpsl_order_result['data'][0]['algoId']}")
                        else:
                            logger.error(f"Failed to place AI recommended TP/SL orders: {tpsl_order_result['detail']}")
                            tp_sl_confirmation_status = f"설정 실패: {tpsl_order_result['detail']}"
                    else:
                        logger.info("TP/SL placement skipped by user.")
                else:
                    logger.warning(
                        f"AI could not provide valid TP/SL recommendations: {ai_tpsl_reason if ai_tpsl_reason else 'Unknown'}. Skipping TP/SL placement.")
                    tp_sl_confirmation_status = "미설정 (AI 추천 실패)"

            else:
                logger.warning(
                    f"Could not confirm active position for {okx_symbol} after trade. Skipping TP/SL and Trailing Stop placement.")
                tp_sl_confirmation_status = "미설정 (포지션 확인 실패)"
        else:
            logger.error(f"Market order failed, skipping TP/SL and Trailing Stop placement.")
            tp_sl_confirmation_status = "미설정 (시장가 주문 실패)"
    else:
        # AI 판단이 유효하지만 트레이딩뷰 신호와 불일치하는 경우
        logger.warning(
            f"NO MATCH: AI decision '{ai_decision}' != TradingView signal '{tradingview_direction}'. Order skipped.")
        tp_sl_confirmation_status = "미설정 (거래 신호 불일치로 주문 스킵)"

    # --- 최종 상태 로깅 ---
    final_balance = get_balance()
    final_positions = get_open_positions(okx_symbol)
    final_market_data_df = get_market_data(okx_symbol, timeframe='15m', limit=200)  # 최종 로깅 시에도 15분봉 데이터 사용
    # BTC 현재 가격은 btc_raw_data에서 1h 최신 가격을 사용. 1m이 필요하면 get_btc_trend_and_data에서 1m을 다시 포함해야 함.
    final_btc_price = btc_raw_data['1h']['close'].iloc[-1] if '1h' in btc_raw_data and not btc_raw_data[
        '1h'].empty else None

    log_trade_status(
        "실행 종료",
        final_balance,
        final_positions,
        okx_symbol,
        tradingview_direction=tradingview_direction,
        tech_indicators=tech_indicators,  # 이전에 계산된 지표 사용
        btc_trend=btc_trend,
        ai_decision=ai_decision,
        ai_reason=ai_reason,
        ai_tp_price=ai_tp_price,
        ai_sl_price=ai_sl_price,
        ai_tpsl_reason=ai_tpsl_reason,
        current_btc_price=final_btc_price,
        recommended_trailing_callback_ratio=ai_recommended_trailing_callback_ratio,
        effective_trailing_callback_ratio=effective_trailing_callback_ratio,
        atr_stage=atr_stage,
        atr_levels=atr_levels,
        tp_sl_confirmation_status=tp_sl_confirmation_status,
        btc_raw_data=btc_raw_data
    )

    return jsonify({"status": "processed"}), 200


# =============================================================================
# 7. RUN THE FLASK APPLICATION
# =============================================================================

if __name__ == '__main__':
    # Make sure to install pandas and requests: pip install pandas requests
    # Run the app on port 80, accessible from the network
    app.run(host='0.0.0.0', port=80)