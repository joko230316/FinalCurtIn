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

warnings.filterwarnings("ignore")

# === 전역 설정 ===
SYMBOL = "BTC-USDT-SWAP"
CONTRACT_AMOUNT = 0.01  # 계약 수량 (⚠️ 실거래이므로 신중하게 설정)
INTERVAL_NORMAL = 120  # 일반 모니터링 간격 (2분)
INTERVAL_ACTIVE = 30  # 활성 모니터링 간격 (30초)

# === 손익 관리 설정 ===
SL_PARTIAL = -10.0  # 부분 손절 기준 (-10%)
SL_FULL = -20.0  # 전체 손절 기준 (-20%)
TP_PARTIAL_1 = 30.0  # 부분 익절 1차 (+30%)
TP_PARTIAL_2 = 50.0  # 부분 익절 2차 (+50%)
TP_FULL = 100.0  # 전체 익절 기준 (+100%)

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


# === OHLCV 데이터 가져오기 ===
def fetch_ohlcv(symbol=SYMBOL, timeframe="1h", limit=100):
    print(f"📈 {symbol}의 {timeframe} 캔들 데이터(최근 {limit}개)를 가져옵니다...")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["return"] = df["close"].pct_change()
        df["ma"] = df["close"].rolling(5).mean()
        df = df.dropna().reset_index(drop=True)
        print("...데이터 가져오기 및 가공 완료.")
        return df
    except Exception as e:
        print(f"❌ OHLCV 데이터 가져오기 실패: {e}")
        return None


# === 데이터 전처리 및 학습/예측 ===
def preprocess(df):
    print("🤖 데이터를 전처리하고 학습/테스트 세트로 분할합니다...")
    X = df[["return", "ma"]].values[:-1]
    y = (df["close"].diff().shift(-1).values[:-1] > 0).astype(int)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("...데이터 전처리 완료.")
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler


def quantum_predict(X_train, y_train, X_latest):
    print("⚛️ QSVC 모델을 생성하고 학습합니다...")
    try:
        feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
        qsvc = QSVC()
        qsvc.feature_map = feature_map
        qsvc.fit(X_train, y_train)
        print("...학습 완료. 최신 데이터로 예측합니다.")
        prediction = qsvc.predict(X_latest)
        return prediction[0]
    except Exception as e:
        print(f"❌ 양자 예측 실패: {e}")
        return None


# === 포지션 정보 ===
def get_position_status():
    print("📊 계정 잔고 및 포지션 정보를 조회합니다...")
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

        # 포지션 이력 업데이트
        POSITION_HISTORY[SYMBOL] = position_info

        return usdt, position_info
    except Exception as e:
        print(f"❌ 포지션 정보 조회 실패: {e}")
        return 0.0, None


# === 주문 실행 ===
def place_order(signal, amount=CONTRACT_AMOUNT):
    try:
        side = 'buy' if signal == 1 else 'sell'
        posSide = 'long' if signal == 1 else 'short'

        print(f"🚀 신규 주문 실행: {posSide.upper()} / {amount} 계약")

        # OKX swap에서의 올바른 주문 파라미터
        order_params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": amount,
            "params": {
                "tdMode": "isolated",
                "posSide": posSide
            }
        }

        # 일부 계정에서는 posSide를 생략해야 할 수 있음
        try:
            result = exchange.create_order(**order_params)
            print(f"✅ 주문 성공: {result.get('id', 'N/A')}")
            return result
        except Exception as e:
            if "posSide" in str(e):
                print("⚠️ posSide 오류 발생, posSide 없이 재시도...")
                # posSide 파라미터 제거 후 재시도
                del order_params["params"]["posSide"]
                result = exchange.create_order(**order_params)
                print(f"✅ 주문 성공 (posSide 없이): {result.get('id', 'N/A')}")
                return result
            else:
                raise e

    except Exception as e:
        print(f"❌ 주문 실행 실패: {e}")
        return None


# === 포지션 부분 청산 함수 ===
def close_partial_position(position, close_percent=0.5):
    """포지션의 일부를 시장가로 청산합니다."""
    if not position:
        return False

    try:
        close_amount = max(position["size"] * close_percent, 0.01)  # 최소 0.01 계약
        close_amount = round(close_amount, 2)  # 소수점 2자리로 반올림

        side = "sell" if position["side"] == "LONG" else "buy"
        posSide = "long" if position["side"] == "LONG" else "short"

        print(f"🔓 포지션 부분 청산: {position['side']} / {close_amount} 계약 ({close_percent * 100}%)")

        # 부분 청산 주문 실행
        order_params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": close_amount,
            "params": {
                "tdMode": "isolated",
                "posSide": posSide
            }
        }

        try:
            result = exchange.create_order(**order_params)
            print(f"✅ 부분 청산 주문 성공: {result.get('id', 'N/A')}")

            # 부분 청산 기록 업데이트
            if SYMBOL not in PARTIAL_CLOSE_RECORD:
                PARTIAL_CLOSE_RECORD[SYMBOL] = []
            PARTIAL_CLOSE_RECORD[SYMBOL].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": close_amount,
                "roi": position["roi"]
            })

            return True
        except Exception as e:
            if "posSide" in str(e):
                print("⚠️ posSide 오류 발생, posSide 없이 재시도...")
                del order_params["params"]["posSide"]
                result = exchange.create_order(**order_params)
                print(f"✅ 부분 청산 주문 성공 (posSide 없이): {result.get('id', 'N/A')}")

                # 부분 청산 기록 업데이트
                if SYMBOL not in PARTIAL_CLOSE_RECORD:
                    PARTIAL_CLOSE_RECORD[SYMBOL] = []
                PARTIAL_CLOSE_RECORD[SYMBOL].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": close_amount,
                    "roi": position["roi"]
                })

                return True
            else:
                raise e

    except Exception as e:
        print(f"❌ 부분 청산 주문 실패: {e}")
        return False


# === 포지션 전체 청산 함수 ===
def close_full_position(position):
    """포지션을 전체 시장가로 청산합니다."""
    if not position:
        return False

    try:
        side = "sell" if position["side"] == "LONG" else "buy"
        posSide = "long" if position["side"] == "LONG" else "short"

        print(f"🔒 포지션 전체 청산: {position['side']} / {position['size']} 계약")

        # 전체 청산 주문 실행
        order_params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": position["size"],
            "params": {
                "tdMode": "isolated",
                "posSide": posSide
            }
        }

        try:
            result = exchange.create_order(**order_params)
            print(f"✅ 전체 청산 주문 성공: {result.get('id', 'N/A')}")

            # 청산 후 기록 삭제
            if SYMBOL in PARTIAL_CLOSE_RECORD:
                del PARTIAL_CLOSE_RECORD[SYMBOL]
            if SYMBOL in POSITION_HISTORY:
                del POSITION_HISTORY[SYMBOL]

            return True
        except Exception as e:
            if "posSide" in str(e):
                print("⚠️ posSide 오류 발생, posSide 없이 재시도...")
                del order_params["params"]["posSide"]
                result = exchange.create_order(**order_params)
                print(f"✅ 전체 청산 주문 성공 (posSide 없이): {result.get('id', 'N/A')}")

                # 청산 후 기록 삭제
                if SYMBOL in PARTIAL_CLOSE_RECORD:
                    del PARTIAL_CLOSE_RECORD[SYMBOL]
                if SYMBOL in POSITION_HISTORY:
                    del POSITION_HISTORY[SYMBOL]

                return True
            else:
                raise e

    except Exception as e:
        print(f"❌ 전체 청산 주문 실패: {e}")
        return False


# === 포지션 방향과 예측 방향 비교 ===
def is_position_direction_matched(position, signal):
    """현재 포지션 방향과 예측 방향이 일치하는지 확인"""
    if not position or signal is None:
        return False

    position_direction = 1 if position['side'] == 'LONG' else 0
    return position_direction == signal


# === 손절 관리 함수 ===
def manage_stop_loss(position, signal):
    """손실 관리: -10%에서 50% 부분 청산, -20%에서 전체 청산 (예측 방향과 반대일 때만)"""
    if not position or position["roi"] >= 0:
        return False

    # 예측 방향과 포지션 방향이 일치하면 청산하지 않음
    if is_position_direction_matched(position, signal):
        print(f"📈 손실 상태이지만 예측 방향과 일치하여 유지: {position['roi']:.2f}%")
        return False

    current_roi = position["roi"]
    print(f"📉 손실 관리 모니터링: {current_roi:.2f}% (예측 방향과 반대)")

    # -20% 전체 청산
    if current_roi <= SL_FULL:
        print(f"🚨 전체 손절 실행: {current_roi:.2f}% ≤ {SL_FULL}%")
        return close_full_position(position)

    # -10% 부분 청산 (50%)
    elif current_roi <= SL_PARTIAL and SYMBOL not in PARTIAL_CLOSE_RECORD:
        print(f"⚠️ 부분 손절 실행: {current_roi:.2f}% ≤ {SL_PARTIAL}%")
        return close_partial_position(position, 0.5)

    return False


# === 익절 관리 함수 ===
def manage_take_profit(position, signal):
    """이익 관리: +30%, +50%에서 부분 청산, +100%에서 전체 청산 (예측 방향과 반대일 때만)"""
    if not position or position["roi"] <= 0:
        return False

    # 예측 방향과 포지션 방향이 일치하면 청산하지 않음
    if is_position_direction_matched(position, signal):
        print(f"📈 이익 상태이고 예측 방향과 일치하여 유지: {position['roi']:.2f}%")
        return False

    current_roi = position["roi"]
    print(f"💰 이익 실현: {current_roi:.2f}% (예측 방향과 불일치)")

    # +100% 전체 청산
    if current_roi >= TP_FULL:
        print(f"🎯 전체 익절 실행: {current_roi:.2f}% ≥ {TP_FULL}%")
        return close_full_position(position)

    # +50% 부분 청산 (50%)
    elif current_roi >= TP_PARTIAL_2:
        # 2차 부분 청산이 아직 실행되지 않았는지 확인
        partial_records = PARTIAL_CLOSE_RECORD.get(SYMBOL, [])
        tp2_executed = any(record["roi"] >= TP_PARTIAL_2 for record in partial_records)

        if not tp2_executed:
            print(f"✅ 2차 부분 익절 실행: {current_roi:.2f}% ≥ {TP_PARTIAL_2}%")
            return close_partial_position(position, 0.5)

    # +30% 부분 청산 (50%)
    elif current_roi >= TP_PARTIAL_1:
        # 1차 부분 청산이 아직 실행되지 않았는지 확인
        partial_records = PARTIAL_CLOSE_RECORD.get(SYMBOL, [])
        tp1_executed = any(record["roi"] >= TP_PARTIAL_1 for record in partial_records)

        if not tp1_executed:
            print(f"✅ 1차 부분 익절 실행: {current_roi:.2f}% ≥ {TP_PARTIAL_1}%")
            return close_partial_position(position, 0.5)

    return False


# === 모니터링 간격 결정 함수 ===
def determine_monitoring_interval(position):
    """포지션의 PnL%에 따라 모니터링 간격을 결정합니다."""
    if not position:
        return INTERVAL_NORMAL

    current_roi = position["roi"]

    # -10% < PnL% < +10% 범위 외부면 활성 모니터링
    if current_roi <= -10 or current_roi >= 10:
        return INTERVAL_ACTIVE

    # 범위 내부면 일반 모니터링
    return INTERVAL_NORMAL


# === 상태 출력 ===
def print_status(usdt, position, next_run_in, signal=None):
    now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    header = f"*** 양자 트레이딩 봇 상태 (실거래): {now} ***"
    pos_symbol = position['symbol'] if position else SYMBOL
    pos_side = position['side'] if position else '없음'
    pos_size = position['size'] if position else '0.0'
    pos_entry = f"{position['entry_price']:.2f}" if position else '-'
    pos_pnl = f"{position['pnl']:.2f} USDT" if position else '-'
    pos_roi = f"{position['roi']:.2f} %" if position else '-'

    # 부분 청산 기록
    partial_info = "없음"
    if SYMBOL in PARTIAL_CLOSE_RECORD:
        partial_records = PARTIAL_CLOSE_RECORD[SYMBOL]
        partial_info = f"{len(partial_records)}회 (마지막: {partial_records[-1]['roi']:.2f}%)"

    # 예측 신호 정보
    signal_info = f"{'상승(BUY)' if signal == 1 else '하락(SELL)'}" if signal is not None else '없음'

    # 포지션 방향과 예측 방향 일치 여부
    direction_match = "일치" if position and is_position_direction_matched(position,
                                                                         signal) else "불일치" if position else "N/A"

    print(f"\n{'=' * 80}")
    print(header)
    print(f"{'=' * 80}\n")
    print(f"  {'계정 상태':<22} | {'포지션 정보':<35} | {'예측 신호'}")
    print(f"  {'-' * 22} | {'-' * 35} | {'-' * 15}")
    print(f"  {'선물 계정 잔고:':<24} | {'심볼 (Symbol):':<20} {pos_symbol:<15} | {'신호:':<10} {signal_info}")
    print(f"  {usdt:<22.2f} USDT | {'방향 (Direction):':<20} {pos_side:<15} | {'방향 일치:':<10} {direction_match}")
    print(f"  {' ':<24} | {'수량 (Size):':<20} {pos_size:<15} |")
    print(f"  {' ':<24} | {'진입가 (Entry Price):':<20} {pos_entry:<15} |")
    print(f"\n  {'수익 현황':<20}")
    print(f"  {'-' * 20}")
    print(f"  {'미실현 손익 (PNL):':<24} {pos_pnl}")
    print(f"  {'미실현 손익률 (ROI%):':<24} {pos_roi}")
    print(f"  {'부분 청산 기록:':<24} {partial_info}")
    print(f"\n  {'손익 관리 설정':<20}")
    print(f"  {'-' * 20}")
    print(f"  {'부분 손절:':<24} {SL_PARTIAL}%")
    print(f"  {'전체 손절:':<24} {SL_FULL}%")
    print(f"  {'부분 익절 1:':<24} {TP_PARTIAL_1}%")
    print(f"  {'부분 익절 2:':<24} {TP_PARTIAL_2}%")
    print(f"  {'전체 익절:':<24} {TP_FULL}%")
    print(f"\n{'=' * 80}")
    print(f"  다음 실행까지: {next_run_in}초")
    print(f"{'=' * 80}\n")


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

            (X_train, _, y_train, _), scaler = preprocess(df)

            # 2. 최신 데이터로 예측
            latest_data = scaler.transform([df[["return", "ma"]].values[-1]])
            signal = quantum_predict(X_train, y_train, latest_data)

            if signal is None:
                error_count += 1
                if error_count >= max_errors:
                    print("❌ 연속 오류로 인해 프로그램 종료")
                    break
                time.sleep(INTERVAL_NORMAL)
                continue

            print(f"🧠 양자 모델 예측 결과: {'상승(BUY)' if signal == 1 else '하락(SELL)'}")

            # 3. 현재 포지션 확인
            usdt, position = get_position_status()

            # 4. 손익 관리 실행 (예측 방향과 반대일 때만)
            if position:
                # 손절 관리 (예측 방향과 반대일 때만)
                sl_closed = manage_stop_loss(position, signal)
                if sl_closed:
                    time.sleep(2)
                    usdt, position = get_position_status()

                # 익절 관리 (예측 방향과 반대일 때만, 손절로 청산되지 않은 경우에만)
                if position:
                    tp_closed = manage_take_profit(position, signal)
                    if tp_closed:
                        time.sleep(2)
                        usdt, position = get_position_status()

            # 5. 포지션이 없으면 신규 진입
            if not position:
                order_result = place_order(signal)
                if order_result:
                    time.sleep(2)
                    usdt, position = get_position_status()

            # 6. 모니터링 간격 결정
            monitoring_interval = determine_monitoring_interval(position)

            # 7. 최종 상태 출력
            error_count = 0  # 오류 카운트 초기화
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, monitoring_interval - elapsed_time)
            print_status(usdt, position, int(time_to_wait), signal)

        except Exception as e:
            error_count += 1
            print(f"🔥 메인 루프에서 심각한 오류 발생: {e}")
            print(traceback.format_exc())

            if error_count >= max_errors:
                print("❌ 연속 오류로 인해 프로그램 종료")
                break

            time_to_wait = INTERVAL_NORMAL

        time.sleep(time_to_wait)


# 스크립트 실행 지점
if __name__ == "__main__":
    main()