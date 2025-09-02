# -*- coding: utf-8 -*-
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ccxt
import pytz
import warnings
import traceback
import talib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import argparse

warnings.filterwarnings("ignore")

# === 전역 설정 ===
SYMBOL = "BTC-USDT-SWAP"
CONTRACT_AMOUNT = 0.3  # 계약 수량 (⚠️ 실거래이므로 신중하게 설정)
INTERVAL_NORMAL = 60  # 일반 모니터링 간격 (2분)
INTERVAL_ACTIVE = 30  # 활성 모니터링 간격 (30초)

# === 손익 관리 설정 ===
SL_PARTIAL = -10.0  # 부분 손절 기준 (-10%)
SL_FULL = -20.0  # 전체 손절 기준 (-20%)
TP_START = 30.0  # 부분 익절 시작 기준 (+30%)
TP_INCREMENT = 10.0  # 부분 익절 증가폭 (+10%)
TP_CLOSE_PERCENT = 0.5  # 부분 청산 비율 (50%)

# === 포지션 관리 ===
POSITION_HISTORY = {}  # 포지션 이력 추적
PARTIAL_CLOSE_RECORD = {}  # 부분 청산 기록

# === OKX 실거래 API 인증 ===
API_KEY = os.getenv("OKXYH_API_KEY")
API_SECRET = os.getenv("OKXYH_API_SECRET")
API_PASSPHRASE = os.getenv("OKXYH_API_PASSPHRASE")

# API 키 존재 여부 확인
if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    print("❌ 치명적 오류: OKX 실거래 API 환경변수가 설정되지 않았습니다.")
    exit()

# ccxt 거래소 객체 생성 (실거래 모드)
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSPHRASE,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})
print("✅ OKX 실거래 모드가 활성화되었습니다.")

# === 명령줄 인자 설정 ===
parser = argparse.ArgumentParser(description='양자 트레이딩 봇 - 예측 방법 선택')
parser.add_argument('--method', type=int, default=3,
                    choices=[1, 2, 3, 4, 5],
                    help='예측 방법 선택 (1: RSI, 2: 이동평균선, 3: Random Forest, 4: XGBoost, 5: Quantum)')
args = parser.parse_args()

# 방법 번호를 이름으로 매핑
METHOD_MAPPING = {
    1: "rsi",
    2: "ma_crossover",
    3: "random_forest",
    4: "xgboost",
    5: "quantum"
}

PREDICTION_METHOD = METHOD_MAPPING[args.method]
print(f"🎯 선택된 예측 방법: {PREDICTION_METHOD} ({args.method}번)")


# === OHLCV 데이터 가져오기 ===
def fetch_ohlcv(symbol=SYMBOL, timeframe="1h", limit=100):
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
    """현재 시장 변동성 계산"""
    returns = df["close"].pct_change().dropna()
    volatility = returns.tail(period).std() * np.sqrt(365)  # 연간화 변동성
    return volatility


# === 시장 상황 분석 ===
def analyze_market_condition(df):
    """시장 상황에 따라 적절한 예측 방법 선택"""
    volatility = calculate_volatility(df)
    rsi = df["rsi"].iloc[-1]

    print(f"📊 시장 분석 - 변동성: {volatility:.4f}, RSI: {rsi:.2f}")

    # 변동성에 따른 방법 선택
    if volatility > 0.03:  # 고변동성 시장
        print("🌪️ 고변동성 시장 - RSI 전략 사용")
        return "rsi"
    elif volatility < 0.01:  # 저변동성 시장
        print("🌊 저변동성 시장 - Random Forest 사용")
        return "random_forest"
    elif 40 <= rsi <= 60:  # 중립 RSI
        print("⚖️ 중립 시장 - 이동평균선 전략 사용")
        return "ma_crossover"
    else:  # 일반 상황
        print("📈 일반 시장 - 기본 Random Forest 사용")
        return "random_forest"


# === 데이터 전처리 및 학습/예측 ===
def preprocess(df):
    print("🤖 데이터를 전처리하고 학습/테스트 세트로 분할합니다...")
    # 더 많은 특성 사용
    features = ["return", "ma5", "ma20", "rsi", "macd", "macd_signal", "atr"]
    X = df[features].values[:-1]
    y = (df["close"].diff().shift(-1).values[:-1] > 0).astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("...데이터 전처리 완료.")
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler


# === 대체 예측 방법들 ===
def random_forest_predict(X_train, y_train, X_latest):
    """랜덤 포레스트를 사용한 예측"""
    print("🌲 Random Forest 모델 학습 및 예측...")
    try:
        # 변동성에 따라 하이퍼파라미터 조정
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1  # 모든 CPU 코어 사용
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_latest)
        proba = model.predict_proba(X_latest)[0]
        print(f"📊 예측 확률: {max(proba) * 100:.1f}%")
        return prediction[0]
    except Exception as e:
        print(f"❌ Random Forest 예측 실패: {e}")
        return None


def xgboost_predict(X_train, y_train, X_latest):
    """XGBoost를 사용한 예측"""
    print("🚀 XGBoost 모델 학습 및 예측...")
    try:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_latest)
        proba = model.predict_proba(X_latest)[0]
        print(f"📊 예측 확률: {max(proba) * 100:.1f}%")
        return prediction[0]
    except Exception as e:
        print(f"❌ XGBoost 예측 실패: {e}")
        return None


def moving_average_crossover(df):
    """이동평균선 교차 전략"""
    print("📊 이동평균선 교차 신호 확인...")
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 골든크로스 (상승 신호)
        if prev['ma5'] <= prev['ma20'] and latest['ma5'] > latest['ma20']:
            print("✅ 골든크로스 발생 - BUY 신호")
            return 1
        # 데드크로스 (하락 신호)
        elif prev['ma5'] >= prev['ma20'] and latest['ma5'] < latest['ma20']:
            print("✅ 데드크로스 발생 - SELL 신호")
            return 0

        # 교차가 없으면 최근 추세 따라가기
        signal = 1 if latest['close'] > latest['ma20'] else 0
        trend = "상승" if signal == 1 else "하락"
        print(f"📈 이동평균선 추세: {trend}")
        return signal
    except Exception as e:
        print(f"❌ 이동평균선 전략 실패: {e}")
        return None


def rsi_strategy(df):
    """RSI 과매수/과매도 전략"""
    print("📈 RSI 전략 신호 확인...")
    try:
        latest_rsi = df['rsi'].iloc[-1]

        if latest_rsi < 30:  # 과매도 구간
            print(f"✅ RSI 과매도({latest_rsi:.1f}) - BUY 신호")
            return 1
        elif latest_rsi > 70:  # 과매수 구간
            print(f"✅ RSI 과매수({latest_rsi:.1f}) - SELL 신호")
            return 0
        else:
            # 중립 구간에서는 이동평균선 전략 사용
            print(f"📊 RSI 중립({latest_rsi:.1f}) - 이동평균선 전략 사용")
            return moving_average_crossover(df)
    except Exception as e:
        print(f"❌ RSI 전략 실패: {e}")
        return None


def quantum_predict(X_train, y_train, X_latest):
    """QSVC 양자 예측"""
    print("⚛️ QSVC 모델 학습 및 예측...")
    try:
        feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
        qsvc = QSVC()
        qsvc.feature_map = feature_map
        qsvc.fit(X_train, y_train)
        prediction = qsvc.predict(X_latest)
        return prediction[0]
    except Exception as e:
        print(f"❌ 양자 예측 실패: {e}")
        return None


# === 예측 메서드 선택 ===
def get_prediction(method, X_train, y_train, X_latest, df):
    """선택된 방법으로 예측 수행"""
    if method == "quantum":
        return quantum_predict(X_train, y_train, X_latest)
    elif method == "random_forest":
        return random_forest_predict(X_train, y_train, X_latest)
    elif method == "xgboost":
        return xgboost_predict(X_train, y_train, X_latest)
    elif method == "ma_crossover":
        return moving_average_crossover(df)
    elif method == "rsi":
        return rsi_strategy(df)
    else:
        print("⚠️ 알 수 없는 예측 방법, Random Forest 사용")
        return random_forest_predict(X_train, y_train, X_latest)


# === 포지션 정보 ===
def get_position_status():
    print("📊 계정 잔고 및 포지션 정보 조회...")
    try:
        balance_info = exchange.fetch_balance()
        usdt = balance_info['USDT']['total'] if 'USDT' in balance_info else 0.0

        positions = exchange.fetch_positions(symbols=[SYMBOL])
        pos = next((p for p in positions if float(p.get('contracts', 0)) != 0), None)

        if not pos:
            return usdt, None

        side = pos.get('side', 'N/A').upper()
        size = float(pos.get('contracts', 0))
        entry = float(pos.get('entryPrice', 0))
        pnl = float(pos.get('unrealizedPnl', 0))
        roi = float(pos.get('percentage', 0)) if pos.get('percentage') else (pnl / (
                entry * size)) * 100 if entry > 0 and size > 0 else 0

        position_info = {
            "symbol": SYMBOL, "side": side, "size": size,
            "entry_price": entry, "pnl": pnl, "roi": roi
        }

        POSITION_HISTORY[SYMBOL] = position_info
        return usdt, position_info
    except Exception as e:
        print(f"❌ 포지션 정보 조회 실패: {e}")
        return 0.0, None


# === 주문 실행 ===
def place_order(signal, amount=CONTRACT_AMOUNT, mode="isolated"):
    try:
        side = 'buy' if signal == 1 else 'sell'
        posSide = 'long' if signal == 1 else 'short'

        print(f"🚀 신규 주문: {posSide.upper()} / {amount} 계약 / 모드: {mode}")

        order_params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": amount,
            "params": {"tdMode": mode, "posSide": posSide}
        }

        try:
            result = exchange.create_order(**order_params)
            print(f"✅ 주문 성공: {result.get('id', 'N/A')}")
            return result
        except Exception as e:
            if "posSide" in str(e):
                print("⚠️ posSide 오류, 재시도...")
                del order_params["params"]["posSide"]
                result = exchange.create_order(**order_params)
                print(f"✅ 주문 성공: {result.get('id', 'N/A')}")
                return result
            else:
                raise e

    except Exception as e:
        print(f"❌ 주문 실행 실패: {e}")
        return None


# === 포지션 부분 청산 ===
def close_partial_position(position, close_percent=0.5, mode="isolated"):
    if not position:
        return False

    try:
        close_amount = max(position["size"] * close_percent, 0.01)
        close_amount = round(close_amount, 2)

        side = "sell" if position["side"] == "LONG" else "buy"
        posSide = "long" if position["side"] == "LONG" else "short"

        print(f"🔓 부분 청산: {position['side']} / {close_amount} 계약 ({close_percent * 100}%) / 모드: {mode}")

        order_params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": close_amount,
            "params": {"tdMode": mode, "posSide": posSide}
        }

        try:
            result = exchange.create_order(**order_params)
            print(f"✅ 부분 청산 성공: {result.get('id', 'N/A')}")

            if SYMBOL not in PARTIAL_CLOSE_RECORD:
                PARTIAL_CLOSE_RECORD[SYMBOL] = []
            PARTIAL_CLOSE_RECORD[SYMBOL].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": close_amount,
                "roi": position["roi"],
                "mode": mode
            })

            return True
        except Exception as e:
            if "posSide" in str(e):
                del order_params["params"]["posSide"]
                result = exchange.create_order(**order_params)
                print(f"✅ 부분 청산 성공: {result.get('id', 'N/A')}")

                if SYMBOL not in PARTIAL_CLOSE_RECORD:
                    PARTIAL_CLOSE_RECORD[SYMBOL] = []
                PARTIAL_CLOSE_RECORD[SYMBOL].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": close_amount,
                    "roi": position["roi"],
                    "mode": mode
                })
                return True
            else:
                raise e

    except Exception as e:
        print(f"❌ 부분 청산 실패: {e}")
        return False


# === 포지션 전체 청산 ===
def close_full_position(position, mode="cross"):
    if not position:
        return False

    try:
        side = "sell" if position["side"] == "LONG" else "buy"
        posSide = "long" if position["side"] == "LONG" else "short"

        print(f"🔒 전체 청산: {position['side']} / {position['size']} 계약 / 모드: {mode}")

        order_params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": position["size"],
            "params": {"tdMode": mode, "posSide": posSide}
        }

        try:
            result = exchange.create_order(**order_params)
            print(f"✅ 전체 청산 성공: {result.get('id', 'N/A')}")

            if SYMBOL in PARTIAL_CLOSE_RECORD:
                del PARTIAL_CLOSE_RECORD[SYMBOL]
            if SYMBOL in POSITION_HISTORY:
                del POSITION_HISTORY[SYMBOL]

            return True
        except Exception as e:
            if "posSide" in str(e):
                del order_params["params"]["posSide"]
                result = exchange.create_order(**order_params)
                print(f"✅ 전체 청산 성공: {result.get('id', 'N/A')}")

                if SYMBOL in PARTIAL_CLOSE_RECORD:
                    del PARTIAL_CLOSE_RECORD[SYMBOL]
                if SYMBOL in POSITION_HISTORY:
                    del POSITION_HISTORY[SYMBOL]
                return True
            else:
                raise e

    except Exception as e:
        print(f"❌ 전체 청산 실패: {e}")
        return False


# === 포지션 방향과 예측 방향 비교 ===
def is_position_direction_matched(position, signal):
    if not position or signal is None:
        return False
    position_direction = 1 if position['side'] == 'LONG' else 0
    return position_direction == signal


# === 손절 관리 ===
def manage_stop_loss(position, signal):
    if not position or position["roi"] >= 0:
        return False

    if is_position_direction_matched(position, signal):
        print(f"📈 손실 상태지만 예측 방향 일치하여 유지: {position['roi']:.2f}%")
        return False

    current_roi = position["roi"]
    print(f"📉 손실 관리: {current_roi:.2f}% (예측 방향 반대)")

    if current_roi <= SL_FULL:
        print(f"🚨 전체 손절: {current_roi:.2f}% ≤ {SL_FULL}%")
        return close_full_position(position, "cross")
    elif current_roi <= SL_PARTIAL and SYMBOL not in PARTIAL_CLOSE_RECORD:
        print(f"⚠️ 부분 손절: {current_roi:.2f}% ≤ {SL_PARTIAL}%")
        return close_partial_position(position, 0.5, "isolated")

    return False


# === 익절 관리 ===
def manage_take_profit(position, signal):
    if not position or position["roi"] <= 0:
        return False

    if is_position_direction_matched(position, signal):
        print(f"📈 이익 상태이고 예측 방향 일치하여 유지: {position['roi']:.2f}%")
        return False

    current_roi = position["roi"]
    print(f"💰 이익 실현: {current_roi:.2f}% (예측 방향 불일치)")

    # +30% 이상부터 +10% 증가마다 50%씩 청산
    if current_roi >= TP_START:
        # 현재 수익률이 도달한 임계값 계산
        threshold_level = int((current_roi - TP_START) // TP_INCREMENT)
        current_threshold = TP_START + (threshold_level * TP_INCREMENT)

        # 이 임계값에서 이미 청산했는지 확인
        partial_records = PARTIAL_CLOSE_RECORD.get(SYMBOL, [])
        threshold_executed = any(
            record["roi"] >= current_threshold and record["roi"] < current_threshold + TP_INCREMENT
            for record in partial_records
        )

        if not threshold_executed:
            print(f"✅ 부분 익절: {current_roi:.2f}% ≥ {current_threshold}% (레벨 {threshold_level + 1})")
            return close_partial_position(position, TP_CLOSE_PERCENT, "isolated")

    return False


# === 모니터링 간격 결정 ===
def determine_monitoring_interval(position):
    if not position:
        return INTERVAL_NORMAL
    current_roi = position["roi"]
    return INTERVAL_ACTIVE if current_roi <= -10 or current_roi >= 10 else INTERVAL_NORMAL


# === 상태 출력 ===
def print_status(usdt, position, next_run_in, signal=None, method_name="random_forest"):
    now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    pos_symbol = position['symbol'] if position else SYMBOL
    pos_side = position['side'] if position else '없음'
    pos_size = position['size'] if position else '0.0'
    pos_entry = f"{position['entry_price']:.2f}" if position else '-'
    pos_pnl = f"{position['pnl']:.2f} USDT" if position else '-'
    pos_roi = f"{position['roi']:.2f} %" if position else '-'

    partial_info = "없음"
    if SYMBOL in PARTIAL_CLOSE_RECORD:
        partial_records = PARTIAL_CLOSE_RECORD[SYMBOL]
        partial_info = f"{len(partial_records)}회"
        if partial_records:
            last_record = partial_records[-1]
            partial_info += f" (마지막: {last_record['roi']:.2f}% @ {last_record['mode']})"

    signal_info = f"{'상승(BUY)' if signal == 1 else '하락(SELL)'}" if signal is not None else '없음'
    direction_match = "일치" if position and is_position_direction_matched(position,
                                                                         signal) else "불일치" if position else "N/A"

    print(f"""
*** 양자 트레이딩 봇 상태 ({method_name}): {now} ***
================================================================================
계정 상태              | 포지션 정보                        | 예측 신호
----------------------|------------------------------------|-----------------
선물 잔고: {usdt:>6.2f} USDT | Symbol:    {pos_symbol:<15} | 신호:     {signal_info}
                      | Direction: {pos_side:<15} | 방향 일치: {direction_match}
                      | Size:      {pos_size:<15} | 방법:     {method_name}
                      | Entry:     {pos_entry:<15} |

수익 현황
-------------------
미실현 PNL:    {pos_pnl}
미실현 ROI:    {pos_roi}
부분 청산:     {partial_info}

익절 조건: {TP_START}% 이상부터 +{TP_INCREMENT}%마다 {TP_CLOSE_PERCENT * 100}% 청산

다음 실행: {next_run_in}초
================================================================================
""")


# === 메인 루프 ===
def main():
    error_count = 0
    max_errors = 5

    while True:
        try:
            start_time = time.time()

            # 1. 데이터 가져오기 및 전처리
            df = fetch_ohlcv()
            if df is None:
                error_count += 1
                if error_count >= max_errors:
                    print("❌ 연속 오류로 인해 프로그램 종료")
                    break
                time.sleep(INTERVAL_NORMAL)
                continue

            # 2. 시장 상황 분석 및 동적 방법 선택
            dynamic_method = analyze_market_condition(df)
            print(f"🎯 동적으로 선택된 방법: {dynamic_method}")

            (X_train, _, y_train, _), scaler = preprocess(df)

            # 3. 선택된 방법으로 예측
            latest_data = scaler.transform(
                [df[["return", "ma5", "ma20", "rsi", "macd", "macd_signal", "atr"]].values[-1]])
            signal = get_prediction(dynamic_method, X_train, y_train, latest_data, df)

            if signal is None:
                error_count += 1
                if error_count >= max_errors:
                    print("❌ 연속 오류로 인해 프로그램 종료")
                    break
                time.sleep(INTERVAL_NORMAL)
                continue

            print(f"🧠 {dynamic_method} 예측 결과: {'상승(BUY)' if signal == 1 else '하락(SELL)'}")

            # 4. 현재 포지션 확인
            usdt, position = get_position_status()

            # 5. 손익 관리 실행
            if position:
                sl_closed = manage_stop_loss(position, signal)
                if sl_closed:
                    time.sleep(2)
                    usdt, position = get_position_status()

                if position:
                    tp_closed = manage_take_profit(position, signal)
                    if tp_closed:
                        time.sleep(2)
                        usdt, position = get_position_status()

            # 6. 포지션이 없으면 신규 진입 (isolated 모드 사용)
            if not position:
                order_result = place_order(signal, CONTRACT_AMOUNT, "isolated")
                if order_result:
                    time.sleep(2)
                    usdt, position = get_position_status()

            # 7. 모니터링 간격 결정
            monitoring_interval = determine_monitoring_interval(position)

            # 8. 최종 상태 출력
            error_count = 0
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, monitoring_interval - elapsed_time)
            print_status(usdt, position, int(time_to_wait), signal, dynamic_method)

        except Exception as e:
            error_count += 1
            print(f"🔥 메인 루프 오류: {e}")
            print(traceback.format_exc())

            if error_count >= max_errors:
                print("❌ 연속 오류로 인해 프로그램 종료")
                break

            time_to_wait = INTERVAL_NORMAL

        time.sleep(time_to_wait)


# 스크립트 실행 지점
if __name__ == "__main__":
    main()