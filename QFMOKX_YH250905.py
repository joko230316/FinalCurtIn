# -*- coding: utf-8 -*-
# 필요한 라이브러리 임포트
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import ccxt
import warnings
import traceback
import talib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import argparse
import requests  # Gemini 연동을 위해 추가
import json  # Gemini 연동을 위해 추가

# Qiskit은 현재 사용하지 않으므로 주석 처리 (필요 시 활성화)
# from qiskit.circuit.library import ZZFeatureMap
# from qiskit_machine_learning.algorithms import QSVC

warnings.filterwarnings("ignore")

# === 기본 전역 설정 ===
SYMBOL = "BTC-USDT-SWAP"
TIMEFRAME = "15m"
CANDLE_LIMIT_FOR_AI = 200  # AI 분석을 위한 캔들 수
CONTRACT_AMOUNT = 0.3  # 계약 수량 (⚠️ 실거래이므로 신중하게 설정)
INTERVAL_NORMAL = 120  # 일반 모니터링 간격 (2분)
INTERVAL_ACTIVE = 30  # 활성 모니터링 간격 (30초)

# === 기본 손익 관리 설정 (AI 자문 실패 시 사용될 기본값) ===
SL_PARTIAL = -10.0
SL_FULL = -20.0
TP_START = 30.0
TP_INCREMENT = 10.0
TP_CLOSE_PERCENT = 0.5

# === 포지션 관리 ===
POSITION_HISTORY = {}
PARTIAL_CLOSE_RECORD = {}

# === OKX 실거래 API 인증 ===
API_KEY = os.getenv("OKXYH_API_KEY")
API_SECRET = os.getenv("OKXYH_API_SECRET")
API_PASSPHRASE = os.getenv("OKXYH_API_PASSPHRASE")

if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    print("❌ 치명적 오류: OKX API 환경변수가 설정되지 않았습니다.")
    exit(1)

exchange = ccxt.okx({
    'apiKey': API_KEY, 'secret': API_SECRET, 'password': API_PASSPHRASE,
    'enableRateLimit': True, 'options': {'defaultType': 'swap'}
})
print("✅ OKX 실거래 모드가 활성화되었습니다.")

# === Gemini AI 설정 ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ 치명적 오류: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    exit(1)

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
AI_RESPONSE_TIMEOUT_SECONDS = 50
print("✅ Gemini AI 설정이 완료되었습니다.")

# === Gemini AI 요청 프롬프트 ===
PROMPT_TEMPLATE = """
당신은 암호화폐 트레이딩의 리스크 관리 전문 AI 어시스턴트입니다.
제공된 시장 데이터를 분석하여, 현재 시장 상황에 가장 적합한 손익 관리 파라미터를 추천해 주십시오.

## 분석 데이터
- 종목: BTC/USDT 무기한 선물
- 데이터: 최근 {candle_count}개의 15분봉 캔들 데이터 (CSV 형식)
- 데이터 내용:
{market_data_csv}

## 요청 사항
위 데이터를 바탕으로 다음 5가지 파라미터에 대한 최적의 값을 제안해 주십시오. 변동성이 낮다면 보수적인 값을, 높다면 좀 더 넓은 범위의 값을 제안해야 합니다.
- `sl_partial`: 부분 손절 기준 (-% 단위, 예: -8.5)
- `sl_full`: 전체 손절 기준 (-% 단위, 예: -15.0)
- `tp_start`: 부분 익절 시작 기준 (+% 단위, 예: 25.0)
- `tp_increment`: 부분 익절 증가폭 (+% 단위, 예: 10.0)
- `tp_close_percent`: 부분 청산 비율 (0.1 ~ 1.0 사이의 소수, 예: 0.5)

## 출력 형식 (매우 중요)
- 반드시 아래와 같은 순수 JSON 형식으로만 응답해야 합니다. 다른 설명은 절대 추가하지 마십시오.
- 모든 값은 숫자(float 또는 int)여야 합니다.

```json
{{
  "sl_partial": -10.0,
  "sl_full": -20.0,
  "tp_start": 30.0,
  "tp_increment": 10.0,
  "tp_close_percent": 0.5
}}
"""


# === OHLCV 데이터 가져오기 ===
def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=CANDLE_LIMIT_FOR_AI):
    print(f"📈 {symbol}의 {timeframe} 캔들 데이터(최근 {limit}개)를 가져옵니다...")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])

        # 기술적 지표 추가
        df["return"] = df["close"].pct_change()
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["rsi"] = talib.RSI(df["close"], timeperiod=14)
        df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(df["close"], timeperiod=20)
        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

        df = df.dropna().reset_index(drop=True)
        print("...데이터 가져오기 및 가공 완료.")
        return df
    except Exception as e:
        print(f"❌ OHLCV 데이터 가져오기 실패: {e}")
        return None


# === 변동성 계산 ===
def calculate_volatility(df, period=20):
    returns = df["close"].pct_change().dropna()
    volatility = returns.tail(period).std() * np.sqrt(365 * 24 * 4)  # 15분봉 기준 연율화
    return volatility


# === 시장 상황 분석 ===
def analyze_market_condition(df):
    volatility = calculate_volatility(df)
    rsi = df["rsi"].iloc[-1]
    print(f"📊 시장 분석 - 변동성: {volatility:.4f}, RSI: {rsi:.2f}")

    if volatility > 0.8:
        return "rsi"  # 변동성 기준값은 시장에 맞게 조정 필요
    elif volatility < 0.3:
        return "random_forest"
    elif 40 <= rsi <= 60:
        return "ma_crossover"
    else:
        return "random_forest"


# === 데이터 전처리 및 학습/예측 ===
def preprocess(df):
    features = ["return", "ma5", "ma20", "rsi", "macd", "macd_signal", "atr"]
    X = df[features].values[:-1]
    y = (df["close"].diff().shift(-1).values[:-1] > 0).astype(int)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False), scaler


# === 예측 방법들 ===
def random_forest_predict(X_train, y_train, X_latest):
    print("🌲 Random Forest 모델 학습 및 예측...")
    try:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model.predict(X_latest) < span

        class ="footnote-wrapper" >[0](0) < / span >
    except Exception as e:
        print(f"❌ Random Forest 예측 실패: {e}")
        return None


def xgboost_predict(X_train, y_train, X_latest):
    print("🚀 XGBoost 모델 학습 및 예측...")
    try:
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        return model.predict(X_latest) < span

        class ="footnote-wrapper" >[0](0) < / span >
    except Exception as e:
        print(f"❌ XGBoost 예측 실패: {e}")
        return None


def moving_average_crossover(df):
    print("📊 이동평균선 교차 신호 확인...")
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['ma5'] <= prev['ma20'] and latest['ma5'] > latest['ma20']:
        return 1
    elif prev['ma5'] >= prev['ma20'] and latest['ma5'] < latest['ma20']:
        return 0
    return 1 if latest['close'] > latest['ma20'] else 0


def rsi_strategy(df):
    print("📈 RSI 전략 신호 확인...")
    latest_rsi = df['rsi'].iloc[-1]
    if latest_rsi < 30:
        return 1
    elif latest_rsi > 70:
        return 0
    else:
        return moving_average_crossover(df)


# === 예측 메서드 선택 ===
def get_prediction(method, X_train, y_train, X_latest, df):
    if method == "random_forest":
        return random_forest_predict(X_train, y_train, X_latest)
    elif method == "xgboost":
        return xgboost_predict(X_train, y_train, X_latest)
    elif method == "ma_crossover":
        return moving_average_crossover(df)
    elif method == "rsi":
        return rsi_strategy(df)
    else:
        return random_forest_predict(X_train, y_train, X_latest)


# === Gemini AI를 통해 손익 관리 파라미터 받기 ===
def get_gemini_risk_parameters(df: pd.DataFrame, api_key: str) -> dict | None:
    print("🤖 Gemini AI에게 최적의 손익 관리 파라미터를 자문합니다...")
    if not api_key:
        print("⚠️ Gemini API 키가 없어 자문을 요청할 수 없습니다.")
        return None
    try:
        recent_df = df.tail(CANDLE_LIMIT_FOR_AI)
        market_data_csv = recent_df.to_csv(index=False)
        prompt = PROMPT_TEMPLATE.format(candle_count=len(recent_df), market_data_csv=market_data_csv)
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}}

        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload),
                                 timeout=AI_RESPONSE_TIMEOUT_SECONDS)
        response.raise_for_status()

        content_text = response.json()['candidates'] < span

        class ="footnote-wrapper" >[0](0) < / span >['content']['parts'] < span class ="footnote-wrapper" >[0](0) < / span >['text']

        if content_text.strip().startswith("```json"):
            content_text = content_text.strip()[7:-3].strip()

        ai_params = json.loads(content_text)
        required_keys = ["sl_partial", "sl_full", "tp_start", "tp_increment", "tp_close_percent"]
        if not all(key in ai_params for key in required_keys):
            print("❌ AI 응답에 필수 키가 누락되었습니다.")
            return None

        print("✅ AI 자문 수신 완료:")
        print(json.dumps(ai_params, indent=2))
        return ai_params
    except Exception as e:
        print(f"❌ AI 자문 중 오류 발생: {e}")
        return None


# === 포지션 및 주문 관련 함수들 ===
def get_position_status():
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['total'] if 'USDT' in balance else 0.0
        positions = exchange.fetch_positions(symbols=[SYMBOL])
        pos = next((p for p in positions if float(p.get('contracts', 0)) != 0), None)
        if not pos:
            return usdt_balance, None

        side = pos.get('side', '').upper()
        size = float(pos.get('contracts', 0))
        entry_price = float(pos.get('entryPrice', 0))
        pnl = float(pos.get('unrealizedPnl', 0))
        roi = (pnl / (entry_price * size)) * 100 if entry_price > 0 and size > 0 else 0

        return usdt_balance, {"side": side, "size": size, "entry_price": entry_price, "pnl": pnl, "roi": roi}
    except Exception as e:
        print(f"❌ 포지션 정보 조회 실패: {e}")
        return 0.0, None


def execute_order(params):
    try:
        return exchange.create_order(**params)
    except ccxt.BaseError as e:
        if "posSide" in str(e):
            print("⚠️ posSide 오류 감지, 파라미터 없이 재시도...")
            del params["params"]["posSide"]
            return exchange.create_order(**params)
        raise e


def place_order(signal, amount, mode="isolated"):
    side = 'buy' if signal == 1 else 'sell'
    posSide = 'long' if signal == 1 else 'short'
    print(f"🚀 신규 주문: {posSide.upper()} / {amount} 계약")
    params = {"symbol": SYMBOL, "type": "market", "side": side, "amount": amount,
              "params": {"tdMode": mode, "posSide": posSide}}
    return execute_order(params)


def close_position(position, amount, description):
    side = "sell" if position["side"] == "LONG" else "buy"
    posSide = "long" if position["side"] == "LONG" else "short"
    print(f"🔒 {description} 실행: {position['side']} / {amount} 계약")
    params = {"symbol": SYMBOL, "type": "market", "side": side, "amount": amount,
              "params": {"tdMode": "isolated", "posSide": posSide}}
    return execute_order(params)


# === 손익 관리 로직 ===
def manage_risk(position, signal):
    if not position:
        return False
    roi = position['roi']

    # 예측 방향과 포지션이 일치하면 손절/익절 보류
    if (position['side'] == 'LONG' and signal == 1) or (position['side'] == 'SHORT' and signal == 0):
        print(f"📈 예측 방향 일치. 포지션 유지 (현재 ROI: {roi:.2f}%)")
        return False

    # 손절 관리
    if roi <= SL_FULL:
        close_position(position, position['size'], f"전체 손절 (ROI: {roi:.2f}%)")
        return True
    if roi <= SL_PARTIAL and SYMBOL not in PARTIAL_CLOSE_RECORD:
        close_position(position, position['size'] * 0.5, f"부분 손절 (ROI: {roi:.2f}%)")
        PARTIAL_CLOSE_RECORD[SYMBOL] = True  # 부분 손절 1회 제한
        return True

    # 익절 관리
    if roi >= TP_START:
        last_level = PARTIAL_CLOSE_RECORD.get(f"{SYMBOL}_tp_level", -1)
        current_level = int((roi - TP_START) // TP_INCREMENT)
        if current_level > last_level:
            close_position(position, position['size'] * TP_CLOSE_PERCENT, f"부분 익절 (Level {current_level})")
            PARTIAL_CLOSE_RECORD[f"{SYMBOL}_tp_level"] = current_level
            return True

    return False


# === 상태 출력 ===
def print_status(usdt, position, next_run_in, signal, method):
    now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    print("\n" + "=" * 80)
    print(f"** 🤖 트레이딩 봇 상태 ({now}) 🤖 ***")
    print(f"| 계정 잔고: {usdt:.2f} USDT | 예측 방법: {method} | 다음 실행: {next_run_in}초 후")
    print("-" * 80)
    if position:
        print(f"| 포지션: {position['side']} | 수량: {position['size']} | 진입가: {position['entry_price']:.4f}")
        print(f"| 미실현 PNL: {position['pnl']:.2f} USDT | 미실현 ROI: {position['roi']:.2f} %")
    else:
        print("| 포지션 없음")
    print(f"| 예측 신호: {'상승(BUY)' if signal == 1 else '하락(SELL)' if signal is not None else '대기'}")
    print("=" * 80 + "\n")


# === 메인 루프 ===
def main():
    global SL_PARTIAL, SL_FULL, TP_START, TP_INCREMENT, TP_CLOSE_PERCENT

    error_count, max_errors = 0, 5
    time_to_wait = 0

    while True:
        try:
            time.sleep(time_to_wait)
            start_time = time.time()

            df = fetch_ohlcv(limit=CANDLE_LIMIT_FOR_AI + 50)
            if df is None or df.empty:
                error_count += 1
                if error_count >= max_errors:
                    break
                time_to_wait = INTERVAL_NORMAL
                continue

            dynamic_method = analyze_market_condition(df)
            (X_train, _, y_train, _), scaler = preprocess(df)
            latest_features = df[["return", "ma5", "ma20", "rsi", "macd", "macd_signal", "atr"]].values[-1].reshape(1,
                                                                                                                    -1)
            latest_data_scaled = scaler.transform(latest_features)
            signal = get_prediction(dynamic_method, X_train, y_train, latest_data_scaled, df)

            if signal is None:
                error_count += 1
                if error_count >= max_errors:
                    break
                time_to_wait = INTERVAL_NORMAL
                continue

            usdt, position = get_position_status()

            if position:
                if manage_risk(position, signal):
                    time.sleep(3)
                    usdt, position = get_position_status()
            else:  # 포지션이 없을 때만 AI 자문 및 신규 진입
                PARTIAL_CLOSE_RECORD.clear()  # 새 포지션을 위해 기록 초기화
                ai_params = get_gemini_risk_parameters(df, GEMINI_API_KEY)
                if ai_params:
                    print("✨ AI 제안으로 손익 관리 설정 업데이트.")
                    SL_PARTIAL, SL_FULL, TP_START, TP_INCREMENT, TP_CLOSE_PERCENT = [ai_params[k] for k in
                                                                                     ["sl_partial", "sl_full",
                                                                                      "tp_start", "tp_increment",
                                                                                      "tp_close_percent"]]
                else:
                    print("⚠️ AI 자문 실패. 기본 설정값으로 거래 계속.")

                if place_order(signal, CONTRACT_AMOUNT):
                    time.sleep(3)
                    usdt, position = get_position_status()

            monitoring_interval = INTERVAL_ACTIVE if position else INTERVAL_NORMAL
            error_count = 0
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, monitoring_interval - elapsed_time)
            print_status(usdt, position, int(time_to_wait), signal, dynamic_method)

        except Exception as e:
            print(f"🔥 메인 루프 오류: {e}")
            traceback.print_exc()
            error_count += 1
            if error_count >= max_errors:
                print("❌ 연속 오류로 프로그램 종료")
                break
            time_to_wait = INTERVAL_NORMAL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='트레이딩 봇')
    # 필요 시 명령줄 인자 추가
    args = parser.parse_args()
    main()