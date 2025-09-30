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
from typing import Optional, Dict, Tuple, Any

# Qiskit은 현재 사용하지 않으므로 주석 처리 (필요 시 활성화)
# from qiskit.circuit.library import ZZFeatureMap
# from qiskit_machine_learning.algorithms import QSVC

warnings.filterwarnings("ignore")

# === 기본 전역 설정 ===
SYMBOL = "BTC-USDT-SWAP"
TIMEFRAME = "15m"
CANDLE_LIMIT_FOR_AI = 200  # AI 분석을 위한 캔들 수
CONTRACT_AMOUNT = 0.0  # 계약 수량 (⚠️ 실거래이므로 신중하게 설정)
INTERVAL_NORMAL = 120  # 일반 모니터링 간격 (2분)
INTERVAL_ACTIVE = 30  # 활성 모니터링 간격 (30초)

# === 수익율 기반 포지션 관리 설정 (Floating PnL% × 10,000 기준) ===
PROFIT_TARGET_MULTIPLIER = 2.0  # 수익 시 추가 주문 배율
LOSS_CLOSE_PERCENT = 0.5  # 손실 시 청산 비율
MONITORING_INTERVAL = 30  # 모니터링 간격 (초)
PRECISION = 3  # 소수점 자리수
PNL_MULTIPLIER = 10000  # Floating PnL%에 곱할 배수

# Floating PnL% × 10,000 관리 기준 (전역변수)
FLOATING_PNL_SCALED_PROFIT_START = 50000.0  # +5.0% × 10,000 = 50000
FLOATING_PNL_SCALED_PROFIT_TARGET = 100000.0  # +10.0% × 10,000 = 100000
FLOATING_PNL_SCALED_LOSS_START = -100000.0  # -10.0% × 10,000 = -100000
FLOATING_PNL_SCALED_LOSS_INCREMENT = -50000.0  # -5.0% × 10,000 = -50000

# === 기본 손익 관리 설정 (AI 자문 실패 시 사용될 기본값) ===
SL_PARTIAL = -0.001  # -10.0% × 10,000
SL_FULL = -0.002  # -20.0% × 10,000
TP_START = 0.003  # +30.0% × 10,000
TP_INCREMENT = 0.01  # +10.0% × 10,000
TP_CLOSE_PERCENT = 0.5

# === 포지션 관리 ===
POSITION_HISTORY = {}
PARTIAL_CLOSE_RECORD = {}
PROFIT_LEVELS = {}  # 수익 레벨 추적
LOSS_LEVELS = {}  # 손실 레벨 추적
PNL_EXTREMES = {}  # Floating PnL% × 10,000 최대값/최소값 저장

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
        print("✅ 데이터 가져오기 및 가공 완료.")
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
        return model.predict(X_latest)[0]
    except Exception as e:
        print(f"❌ Random Forest 예측 실패: {e}")
        return None


def xgboost_predict(X_train, y_train, X_latest):
    print("🚀 XGBoost 모델 학습 및 예측...")
    try:
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        return model.predict(X_latest)[0]
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
def get_gemini_risk_parameters(df: pd.DataFrame, api_key: str) -> Optional[Dict[str, float]]:
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

        response_data = response.json()
        if 'candidates' not in response_data or not response_data['candidates']:
            print("❌ AI 응답 형식이 올바르지 않습니다.")
            return None

        content_text = response_data['candidates'][0]['content']['parts'][0]['text']

        # JSON 추출
        if "```json" in content_text:
            start_idx = content_text.find("```json") + 7
            end_idx = content_text.find("```", start_idx)
            if end_idx == -1:
                end_idx = content_text.find("}", start_idx) + 1
            json_text = content_text[start_idx:end_idx].strip()
        elif "{" in content_text:
            start_idx = content_text.find("{")
            end_idx = content_text.rfind("}") + 1
            json_text = content_text[start_idx:end_idx].strip()
        else:
            json_text = content_text.strip()

        # JSON 파싱
        ai_params = json.loads(json_text)
        required_keys = ["sl_partial", "sl_full", "tp_start", "tp_increment", "tp_close_percent"]

        if not all(key in ai_params for key in required_keys):
            print("❌ AI 응답에 필수 키가 누락되었습니다.")
            return None

        # 값 검증 및 10,000 곱하기
        for key in required_keys:
            if not isinstance(ai_params[key], (int, float)):
                print(f"❌ {key}의 값이 숫자가 아닙니다: {ai_params[key]}")
                return None
            # AI가 제안한 % 값을 10,000배로 변환
            if key in ["sl_partial", "sl_full", "tp_start", "tp_increment"]:
                ai_params[key] = ai_params[key] * PNL_MULTIPLIER

        print("✅ AI 자문 수신 완료 (10,000배 적용):")
        print(json.dumps(ai_params, indent=2))
        return ai_params
    except json.JSONDecodeError as e:
        print(f"❌ AI 응답 JSON 파싱 실패: {e}")
        print(f"원본 응답: {content_text}")
        return None
    except Exception as e:
        print(f"❌ AI 자문 중 오류 발생: {e}")
        traceback.print_exc()
        return None


# === 포지션 및 주문 관련 함수들 ===
def get_position_status() -> Tuple[float, Optional[Dict]]:
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
        current_price = float(pos.get('markPrice', entry_price))

        # Floating PnL% 계산 및 10,000 곱하기
        if entry_price > 0 and size > 0:
            floating_pnl_percent = (pnl / (entry_price * size)) * 100
            floating_pnl_scaled = floating_pnl_percent * PNL_MULTIPLIER
        else:
            floating_pnl_percent = 0.0
            floating_pnl_scaled = 0.0

        return usdt_balance, {
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "pnl": pnl,
            "floating_pnl_percent": floating_pnl_percent,
            "floating_pnl_scaled": floating_pnl_scaled,
            "current_price": current_price
        }
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


def place_order(signal: int, amount: float, mode: str = "isolated") -> bool:
    try:
        side = 'buy' if signal == 1 else 'sell'
        posSide = 'long' if signal == 1 else 'short'
        print(f"🚀 신규 주문: {posSide.upper()} / {amount:.{PRECISION}f} 계약")
        params = {"symbol": SYMBOL, "type": "market", "side": side, "amount": amount,
                  "params": {"tdMode": mode, "posSide": posSide}}
        result = execute_order(params)
        print(f"✅ 주문 성공: {result['id']}")
        return True
    except Exception as e:
        print(f"❌ 주문 실패: {e}")
        return False


def close_position(position: Dict, amount: float, description: str, mode: str = "cross") -> bool:
    try:
        side = "sell" if position["side"] == "LONG" else "buy"
        posSide = "long" if position["side"] == "LONG" else "short"
        print(f"🔒 {description} 실행: {position['side']} / {amount:.{PRECISION}f} 계약 (모드: {mode})")
        params = {"symbol": SYMBOL, "type": "market", "side": side, "amount": amount,
                  "params": {"tdMode": mode, "posSide": posSide}}
        result = execute_order(params)
        print(f"✅ 청산 성공: {result['id']}")
        return True
    except Exception as e:
        print(f"❌ 청산 실패: {e}")
        return False


def add_to_position(position: Dict, amount_multiplier: float, description: str) -> bool:
    """기존 포지션에 추가 진입"""
    try:
        side = 'buy' if position["side"] == "LONG" else 'sell'
        posSide = 'long' if position["side"] == "LONG" else 'short'
        current_size = position["size"]
        add_amount = current_size * amount_multiplier

        print(f"📈 {description}: {posSide.upper()} / {add_amount:.{PRECISION}f} 계약 추가")
        params = {"symbol": SYMBOL, "type": "market", "side": side, "amount": add_amount,
                  "params": {"tdMode": "cross", "posSide": posSide}}
        result = execute_order(params)
        print(f"✅ 추가 주문 성공: {result['id']}")
        return True
    except Exception as e:
        print(f"❌ 추가 주문 실패: {e}")
        return False


# === Floating PnL% × 10,000 기반 포지션 관리 ===
def manage_position_by_floating_pnl(position: Dict) -> bool:
    """Floating PnL% × 10,000 기반 포지션 관리"""
    if not position:
        return False

    floating_pnl_scaled = position['floating_pnl_scaled']
    symbol_key = SYMBOL

    # 최대값/최소값 업데이트
    if symbol_key not in PNL_EXTREMES:
        PNL_EXTREMES[symbol_key] = {"max": floating_pnl_scaled, "min": floating_pnl_scaled}
    else:
        PNL_EXTREMES[symbol_key]["max"] = max(PNL_EXTREMES[symbol_key]["max"], floating_pnl_scaled)
        PNL_EXTREMES[symbol_key]["min"] = min(PNL_EXTREMES[symbol_key]["min"], floating_pnl_scaled)

    print(f"📊 Floating PnL% × 10,000 모니터링: {floating_pnl_scaled:.{PRECISION}f}")
    print(
        f"📈 최대값: {PNL_EXTREMES[symbol_key]['max']:.{PRECISION}f}, 최소값: {PNL_EXTREMES[symbol_key]['min']:.{PRECISION}f}")

    # +50,000 이상에서 -50,000으로 전환 시 전체 청산 (5.0% → -5.0%)
    if (PNL_EXTREMES[symbol_key]["max"] >= FLOATING_PNL_SCALED_PROFIT_START and
            floating_pnl_scaled <= -50000.0):
        print(
            f"⚠️ 급격한 하락 감지: 최대 {PNL_EXTREMES[symbol_key]['max'] / PNL_MULTIPLIER:.4f}%에서 {floating_pnl_scaled / PNL_MULTIPLIER:.4f}%로 전환")
        if close_position(position, position['size'], "급격한 하락 전체 청산", "cross"):
            PROFIT_LEVELS.pop(symbol_key, None)
            LOSS_LEVELS.pop(symbol_key, None)
            PNL_EXTREMES.pop(symbol_key, None)
            return True

    # 수익 관리: +100,000마다 추가 주문 (+10.0%)
    if floating_pnl_scaled > FLOATING_PNL_SCALED_PROFIT_TARGET:
        current_profit_level = int(floating_pnl_scaled // FLOATING_PNL_SCALED_PROFIT_TARGET)
        last_profit_level = PROFIT_LEVELS.get(symbol_key, 0)

        if current_profit_level > last_profit_level:
            target_scaled = current_profit_level * FLOATING_PNL_SCALED_PROFIT_TARGET
            target_percent = target_scaled / PNL_MULTIPLIER
            print(f"🎯 수익 목표 달성: +{target_percent:.4f}% ({target_scaled:.{PRECISION}f})")
            if add_to_position(position, PROFIT_TARGET_MULTIPLIER, f"+{target_percent:.4f}% 수익 추가 진입"):
                PROFIT_LEVELS[symbol_key] = current_profit_level
                print(f"✅ 추가 주문 완료: {PROFIT_TARGET_MULTIPLIER}배")
                return True

    # 손실 관리: -100,000보다 낮아지면 -50,000마다 부분 청산 (-10.0% → -5.0%)
    if floating_pnl_scaled < FLOATING_PNL_SCALED_LOSS_START:
        current_loss_level = int(abs(floating_pnl_scaled) // abs(FLOATING_PNL_SCALED_LOSS_INCREMENT))
        last_loss_level = LOSS_LEVELS.get(symbol_key, 0)

        if current_loss_level > last_loss_level:
            loss_scaled = -current_loss_level * abs(FLOATING_PNL_SCALED_LOSS_INCREMENT)
            loss_percent = loss_scaled / PNL_MULTIPLIER
            print(f"⚠️ 손실 확대: {loss_percent:.4f}% ({loss_scaled:.{PRECISION}f})")
            close_amount = position['size'] * LOSS_CLOSE_PERCENT
            if close_position(position, close_amount, f"{loss_percent:.4f}% 손실 부분 청산", "cross"):
                LOSS_LEVELS[symbol_key] = current_loss_level
                print(f"✅ 부분 청산 완료: {LOSS_CLOSE_PERCENT * 100:.{PRECISION}f}%")
                return True

    return False


# === 기존 손익 관리 로직 (Floating PnL% × 10,000 기준으로 수정) ===
def manage_risk(position: Dict, signal: int) -> bool:
    if not position:
        return False
    floating_pnl_scaled = position['floating_pnl_scaled']

    # 예측 방향과 포지션이 일치하면 손절/익절 보류
    if (position['side'] == 'LONG' and signal == 1) or (position['side'] == 'SHORT' and signal == 0):
        pnl_percent = floating_pnl_scaled / PNL_MULTIPLIER
        print(f"📈 예측 방향 일치. 포지션 유지 (현재 Floating PnL%: {pnl_percent:.4f}%)")
        return False

    # 손절 관리
    if floating_pnl_scaled <= SL_FULL:
        pnl_percent = floating_pnl_scaled / PNL_MULTIPLIER
        return close_position(position, position['size'], f"전체 손절 (Floating PnL%: {pnl_percent:.4f}%)")

    if floating_pnl_scaled <= SL_PARTIAL and SYMBOL not in PARTIAL_CLOSE_RECORD:
        pnl_percent = floating_pnl_scaled / PNL_MULTIPLIER
        if close_position(position, position['size'] * 0.5, f"부분 손절 (Floating PnL%: {pnl_percent:.4f}%)"):
            PARTIAL_CLOSE_RECORD[SYMBOL] = True  # 부분 손절 1회 제한
            return True

    # 익절 관리
    if floating_pnl_scaled >= TP_START:
        last_level = PARTIAL_CLOSE_RECORD.get(f"{SYMBOL}_tp_level", -1)
        current_level = int((floating_pnl_scaled - TP_START) // TP_INCREMENT)
        if current_level > last_level:
            pnl_percent = floating_pnl_scaled / PNL_MULTIPLIER
            if close_position(position, position['size'] * TP_CLOSE_PERCENT,
                              f"부분 익절 (Level {current_level}, Floating PnL%: {pnl_percent:.4f}%)"):
                PARTIAL_CLOSE_RECORD[f"{SYMBOL}_tp_level"] = current_level
                return True

    return False


# === 상태 출력 ===
def print_status(usdt: float, position: Optional[Dict], next_run_in: int, signal: Optional[int], method: str):
    now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    print("\n" + "=" * 80)
    print(f"** 🤖 트레이딩 봇 상태 ({now}) 🤖 ***")
    print(f"| 계정 잔고: {usdt:.{PRECISION}f} USDT | 예측 방법: {method} | 다음 실행: {next_run_in}초 후")
    print("-" * 80)
    if position:
        print(f"| 포지션: {position['side']} | 수량: {position['size']:.{PRECISION}f} | 진입가: {position['entry_price']:.4f}")
        print(
            f"| 미실현 PNL: {position['pnl']:.{PRECISION}f} USDT | 미실현 PnL%(Floating PnL%): {position['floating_pnl_scaled']:.{PRECISION}f}")

        # Floating PnL% × 10,000 기반 관리 상태 출력
        symbol_key = SYMBOL
        current_profit_level = PROFIT_LEVELS.get(symbol_key, 0)
        current_loss_level = LOSS_LEVELS.get(symbol_key, 0)
        print(f"| 수익 레벨: {current_profit_level} | 손실 레벨: {current_loss_level}")

        # 최대값/최소값 출력
        if symbol_key in PNL_EXTREMES:
            extremes = PNL_EXTREMES[symbol_key]
            print(f"| 최대 PnL%×10K: {extremes['max']:.{PRECISION}f} | 최소 PnL%×10K: {extremes['min']:.{PRECISION}f}")

        # 현재 가격 정보
        print(
            f"| 현재 가격: {position['current_price']:.4f} | 변동: {((position['current_price'] - position['entry_price']) / position['entry_price'] * 100):.4f}%")
    else:
        print("| 포지션 없음")

    signal_text = '상승(BUY)' if signal == 1 else '하락(SELL)' if signal is not None else '대기'
    print(f"| 예측 신호: {signal_text}")
    print(f"| 모니터링 간격: {MONITORING_INTERVAL}초")
    print("=" * 80 + "\n")


# === 메인 루프 ===
def main():
    global SL_PARTIAL, SL_FULL, TP_START, TP_INCREMENT, TP_CLOSE_PERCENT

    error_count, max_errors = 0, 5
    time_to_wait = 0

    print("🚀 트레이딩 봇 시작...")
    print(f"📊 설정: {SYMBOL}, {TIMEFRAME}, 계약수량: {CONTRACT_AMOUNT}")
    print(f"🎯 Floating PnL% × {PNL_MULTIPLIER:,} 관리 설정:")
    print(f"   - 급락 감지: +{FLOATING_PNL_SCALED_PROFIT_START / PNL_MULTIPLIER}% → -5% 시 전체 청산")
    print(f"   - 수익 목표: +{FLOATING_PNL_SCALED_PROFIT_TARGET / PNL_MULTIPLIER}%마다 {PROFIT_TARGET_MULTIPLIER}배 추가")
    print(
        f"   - 손실 관리: {FLOATING_PNL_SCALED_LOSS_START / PNL_MULTIPLIER}% 이하 시 {abs(FLOATING_PNL_SCALED_LOSS_INCREMENT / PNL_MULTIPLIER)}%마다 {LOSS_CLOSE_PERCENT * 100}% 청산")
    print(f"⏰ 모니터링 간격: {MONITORING_INTERVAL}초")

    while True:
        try:
            time.sleep(time_to_wait)
            start_time = time.time()

            # 데이터 가져오기
            df = fetch_ohlcv(limit=CANDLE_LIMIT_FOR_AI + 50)
            if df is None or df.empty:
                error_count += 1
                print(f"⚠️ 데이터 없음, 재시도 {error_count}/{max_errors}")
                if error_count >= max_errors:
                    print("❌ 연속 데이터 오류로 프로그램 종료")
                    break
                time_to_wait = INTERVAL_NORMAL
                continue

            # 시장 분석 및 예측
            dynamic_method = analyze_market_condition(df)
            (X_train, _, y_train, _), scaler = preprocess(df)
            latest_features = df[["return", "ma5", "ma20", "rsi", "macd", "macd_signal", "atr"]].values[-1].reshape(1,
                                                                                                                    -1)
            latest_data_scaled = scaler.transform(latest_features)
            signal = get_prediction(dynamic_method, X_train, y_train, latest_data_scaled, df)

            if signal is None:
                error_count += 1
                print(f"⚠️ 예측 실패, 재시도 {error_count}/{max_errors}")
                if error_count >= max_errors:
                    print("❌ 연속 예측 오류로 프로그램 종료")
                    break
                time_to_wait = INTERVAL_NORMAL
                continue

            # 포지션 상태 확인
            usdt, position = get_position_status()

            if position:
                pnl_percent = position['floating_pnl_scaled'] / PNL_MULTIPLIER
                print(
                    f"📦 현재 포지션: {position['side']}, Floating PnL%: {pnl_percent:.4f}% ({position['floating_pnl_scaled']:.{PRECISION}f})")

                # Floating PnL% × 10,000 기반 포지션 관리 실행
                if manage_position_by_floating_pnl(position):
                    print("🔄 포지션 변경 후 상태 업데이트 중...")
                    time.sleep(3)
                    usdt, position = get_position_status()

                # 기존 AI 기반 손익 관리 실행
                elif manage_risk(position, signal):
                    print("🔄 리스크 관리 후 상태 업데이트 중...")
                    time.sleep(3)
                    usdt, position = get_position_status()
                else:
                    print("✅ 현재 포지션 유지")
            else:
                # 포지션이 없을 때만 AI 자문 및 신규 진입
                PARTIAL_CLOSE_RECORD.clear()
                PROFIT_LEVELS.clear()
                LOSS_LEVELS.clear()
                PNL_EXTREMES.clear()
                print("🆕 신규 포지션 진입 검토 중...")

                ai_params = get_gemini_risk_parameters(df, GEMINI_API_KEY)
                if ai_params:
                    print("✨ AI 제안으로 손익 관리 설정 업데이트.")
                    SL_PARTIAL = ai_params["sl_partial"]
                    SL_FULL = ai_params["sl_full"]
                    TP_START = ai_params["tp_start"]
                    TP_INCREMENT = ai_params["tp_increment"]
                    TP_CLOSE_PERCENT = ai_params["tp_close_percent"]
                else:
                    print("⚠️ AI 자문 실패. 기본 설정값으로 거래 계속.")

                # 신규 주문 실행
                if place_order(signal, CONTRACT_AMOUNT):
                    print("🔄 신규 주문 후 상태 업데이트 중...")
                    time.sleep(3)
                    usdt, position = get_position_status()

            # 다음 실행까지 대기 시간 계산
            monitoring_interval = INTERVAL_ACTIVE if position else INTERVAL_NORMAL
            error_count = 0
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, monitoring_interval - elapsed_time)

            # 상태 출력
            print_status(usdt, position, int(time_to_wait), signal, dynamic_method)

        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 프로그램 종료")
            break
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