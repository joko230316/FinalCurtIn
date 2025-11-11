#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 필요한 라이브러리 임포트
import os
import time
from datetime import datetime
import pytz
import ccxt
import warnings
import traceback

warnings.filterwarnings("ignore")

# === 전역 변수 설정 ===
SYMBOL = "BTC-USDT-SWAP"
MONITORING_INTERVAL = 60  # 모니터링 간격 (초)
PRECISION = 3  # 소수점 자리수

# === 긴급 청산 설정 ===
EMERGENCY_CLOSE_PERCENT = 15.0  # 현재 잔고의 15% 손실 시 긴급 청산

# === 트레일링 스탑 설정 ===
TRAILING_STOP_PERCENT = 3.0  # 최고가 대비 -3% 하락 시 청산
TRAILING_STOP_INTERVAL = 60  # 1분 간격으로 최고값 갱신

# === 포지션 관리 ===
PNL_EXTREMES = {}  # Floating PnL% 최대값 저장

# === OKX 실거래 API 인증 ===
API_KEY = os.getenv("OKXYH_API_KEY")
API_SECRET = os.getenv("OKXYH_API_SECRET")
API_PASSPHRASE = os.getenv("OKXYH_API_PASSPHRASE")

if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    print("* 치명적 오류: OKX API 환경변수가 설정되지 않았습니다.")
    exit(1)

exchange = ccxt.okx({
    'apiKey': API_KEY, 'secret': API_SECRET, 'password': API_PASSPHRASE,
    'enableRateLimit': True, 'options': {'defaultType': 'swap'}
})
print("* OKX 실거래 모드가 활성화되었습니다.")


# === 잔고 및 포지션 상태 조회 ===
def get_account_and_position_status():
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {})
        balance_details = {
            'total': usdt_balance.get('total', 0.0),
            'free': usdt_balance.get('free', 0.0),
            'used': usdt_balance.get('used', 0.0)
        }

        all_positions = exchange.fetch_positions(symbols=[SYMBOL])
        active_positions = []
        for pos in all_positions:
            if float(pos.get('contracts', 0)) != 0:
                entry_price = float(pos.get('entryPrice', 0))
                size = float(pos.get('contracts', 0))
                pnl = float(pos.get('unrealizedPnl', 0))
                if entry_price > 0 and size > 0:
                    mark_price = float(pos.get('markPrice', entry_price))
                    leverage = float(pos.get('leverage', 1))
                    margin = (size * entry_price) / leverage if leverage > 0 else (size * entry_price)
                    floating_pnl_percent = (pnl / margin) * 100 if margin > 0 else 0.0
                else:
                    floating_pnl_percent = 0.0

                active_positions.append({
                    "symbol": pos.get('symbol'),
                    "side": pos.get('side', '').upper(),
                    "size": size,
                    "entry_price": entry_price,
                    "pnl": pnl,
                    "floating_pnl_percent": floating_pnl_percent,
                    "current_price": float(pos.get('markPrice', entry_price))
                })
        return balance_details, active_positions
    except Exception as e:
        print(f"* 잔고/포지션 정보 조회 실패: {e}")
        return None, []


# === 주문 실행 함수 ===
def execute_order(params):
    try:
        return exchange.create_order(**params)
    except ccxt.BaseError as e:
        if "posSide" in str(e):
            print("* posSide 오류 감지, 파라미터 없이 재시도...")
            if "params" in params and "posSide" in params["params"]:
                del params["params"]["posSide"]
            return exchange.create_order(**params)
        raise e


# === 포지션 청산 함수 ===
def close_position(position, amount, description, mode="cross"):
    try:
        side = "sell" if position["side"] == "LONG" else "buy"
        posSide = "long" if position["side"] == "LONG" else "short"
        print(f"* {description} 실행: {position['side']} / {amount:.{PRECISION}f} 계약 (모드: {mode})")
        params = {
            "symbol": SYMBOL,
            "type": "market",
            "side": side,
            "amount": amount,
            "params": {"tdMode": mode, "posSide": posSide}
        }
        result = execute_order(params)
        print(f"* 청산 성공: {result['id']}")
        return True
    except Exception as e:
        print(f"* 청산 실패: {e}")
        return False


# === 긴급 청산 기능 ===
def emergency_close_check(balance, position):
    """
    현재 잔고의 15% 손실 시 긴급 청산 실행
    """
    if not balance or not position:
        return False

    total_balance = balance['total']
    pnl_loss = position['pnl']

    # 손실이 잔고의 15%를 초과하는지 확인
    if pnl_loss < 0 and abs(pnl_loss) > (total_balance * EMERGENCY_CLOSE_PERCENT / 100):
        loss_percent = (abs(pnl_loss) / total_balance) * 100
        print(f"* 긴급 청산 조건 충족: 현재 손실 {loss_percent:.2f}% (설정값: {EMERGENCY_CLOSE_PERCENT}%)")
        print(f"* 손실 금액: {pnl_loss:.{PRECISION}f} USDT, 총 잔고: {total_balance:.{PRECISION}f} USDT")

        if close_position(position, position['size'], "긴급 청산"):
            print("* 긴급 청산 완료")
            return True

    return False


# === 트레일링 스탑 기능 ===
def trailing_stop_check(position):
    """
    PNL%의 최고값을 저장하고 최고가 대비 -3% 하락 시 전체 청산
    """
    if not position:
        return False

    symbol_key = SYMBOL
    current_pnl_percent = position['floating_pnl_percent']

    # 최고값 초기화 또는 갱신
    if symbol_key not in PNL_EXTREMES:
        PNL_EXTREMES[symbol_key] = {
            "max_pnl_percent": current_pnl_percent,
            "last_updated": time.time()
        }
        print(f"* 트레일링 스탑 최초 설정: {current_pnl_percent:.4f}%")
        return False

    # 1분 간격으로 최고값 갱신 확인
    current_time = time.time()
    time_since_update = current_time - PNL_EXTREMES[symbol_key]["last_updated"]

    # 현재 PNL%가 최고값보다 높으면 갱신
    if current_pnl_percent > PNL_EXTREMES[symbol_key]["max_pnl_percent"]:
        PNL_EXTREMES[symbol_key] = {
            "max_pnl_percent": current_pnl_percent,
            "last_updated": current_time
        }
        print(f"* 트레일링 스탑 최고값 갱신: {current_pnl_percent:.4f}%")
        return False

    # 1분이 지났을 때만 최고값 확인 (갱신은 아님)
    if time_since_update >= TRAILING_STOP_INTERVAL:
        max_pnl = PNL_EXTREMES[symbol_key]["max_pnl_percent"]
        drawdown_percent = ((current_pnl_percent - max_pnl) / max_pnl) * 100 if max_pnl > 0 else 0

        print(f"* 트레일링 스탑 모니터링: 현재 {current_pnl_percent:.4f}%, 최고 {max_pnl:.4f}%, 하락 {drawdown_percent:.2f}%")

        # 최고가 대비 -3% 이상 하락 시 청산
        if drawdown_percent <= -TRAILING_STOP_PERCENT:
            print(f"* 트레일링 스탑 발동: 최고가 대비 {drawdown_percent:.2f}% 하락")
            if close_position(position, position['size'], "트레일링 스탑 청산"):
                PNL_EXTREMES.pop(symbol_key, None)
                return True

        # 시간만 업데이트 (값은 유지)
        PNL_EXTREMES[symbol_key]["last_updated"] = current_time

    return False


# === 상태 출력 함수 ===
def print_status(balance, positions, next_run_in):
    now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    print("\n" + "=" * 80)
    print(f"** * 포지션 관리 봇 상태 ({now}) * **")
    print(f"| 다음 실행: {next_run_in}초 후 | 모니터링 간격: {MONITORING_INTERVAL}초")
    print("-" * 80)
    print("## * 계정 잔고 (USDT)")
    if balance:
        print(
            f"| 총액(Total): {balance['total']:.{PRECISION}f} | 사용 가능(Free): {balance['free']:.{PRECISION}f} | 사용 중(Used): {balance['used']:.{PRECISION}f}")
    else:
        print("| 잔고 정보를 가져올 수 없습니다.")
    print("-" * 80)
    print("## * 현재 포지션")
    if not positions:
        print("| 현재 진입한 포지션이 없습니다.")
    else:
        for i, pos in enumerate(positions):
            side_char = "▲" if pos['side'] == 'LONG' else "▼"
            print(f"| {i + 1}. {side_char} {pos['symbol']} ({pos['side']})")
            print(
                f"|    - 수량: {pos['size']:.{PRECISION}f} | 진입가: {pos['entry_price']:.4f} | 현재가: {pos['current_price']:.4f}")
            pnl_percent = pos['floating_pnl_percent']
            print(f"|    - 미실현 PNL: {pos['pnl']:.{PRECISION}f} USDT ({pnl_percent:.4f}%)")

            # 트레일링 스탑 정보 표시
            symbol_key = pos['symbol']
            if symbol_key in PNL_EXTREMES:
                max_pnl = PNL_EXTREMES[symbol_key]["max_pnl_percent"]
                drawdown = ((pnl_percent - max_pnl) / max_pnl) * 100 if max_pnl > 0 else 0
                print(f"|    - 트레일링 스탑: 최고 {max_pnl:.4f}%, 하락 {drawdown:.2f}%")

            # 긴급 청산 정보 표시 (USDT 환산값 추가)
            if balance:
                total_balance = balance['total']
                loss_percent = (abs(pos['pnl']) / total_balance) * 100 if pos['pnl'] < 0 else 0
                current_loss_usdt = abs(pos['pnl']) if pos['pnl'] < 0 else 0.0
                emergency_threshold_usdt = total_balance * EMERGENCY_CLOSE_PERCENT / 100
                print(
                    f"|    - 긴급 청산: {loss_percent:.2f}% / {EMERGENCY_CLOSE_PERCENT}% | {current_loss_usdt:.1f} USDT / {emergency_threshold_usdt:.1f} USDT")

            if i < len(positions) - 1:
                print("|" + "-" * 78)
    print("=" * 80 + "\n")


# === 메인 루프 ===
def main():
    error_count = 0
    max_errors = 5
    time_to_wait = 0

    print("* 포지션 관리 봇 시작...")
    print(f"* 주요 기능:")
    print(f"  1. 긴급 청산: 잔고의 {EMERGENCY_CLOSE_PERCENT}% 손실 시 자동 청산")
    print(f"  2. 트레일링 스탑: 최고가 대비 {TRAILING_STOP_PERCENT}% 하락 시 청산 (1분 간격)")

    while True:
        try:
            time.sleep(time_to_wait)
            start_time = time.time()

            # --- 잔고 및 포지션 상태 확인 ---
            balance, positions = get_account_and_position_status()
            active_position = positions[0] if positions else None

            # --- 포지션 관리 로직 ---
            if active_position:
                print(
                    f"* 포지션 감지: {active_position['side']}, PNL: {active_position['pnl']:.{PRECISION}f} USDT ({active_position['floating_pnl_percent']:.4f}%)")

                # 1. 긴급 청산 체크 (최우선)
                if emergency_close_check(balance, active_position):
                    print("* 긴급 청산 실행 후 상태 업데이트 중...")
                    time.sleep(3)
                    balance, positions = get_account_and_position_status()
                    PNL_EXTREMES.clear()

                # 2. 트레일링 스탑 체크 (긴급 청산이 실행되지 않았을 때만)
                elif trailing_stop_check(active_position):
                    print("* 트레일링 스탑 실행 후 상태 업데이트 중...")
                    time.sleep(3)
                    balance, positions = get_account_and_position_status()
                else:
                    print("* 현재 포지션 유지 - 관리 조건 미충족")

            # --- 다음 실행 준비 및 상태 출력 ---
            error_count = 0
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, MONITORING_INTERVAL - elapsed_time)
            print_status(balance, positions, int(time_to_wait))

        except KeyboardInterrupt:
            print("\n* 사용자에 의해 프로그램 종료")
            break
        except Exception as e:
            print(f"* 메인 루프 오류: {e}")
            traceback.print_exc()
            error_count += 1
            if error_count >= max_errors:
                print("* 최대 오류 횟수 도달, 프로그램 종료")
                break
            time_to_wait = MONITORING_INTERVAL


if __name__ == "__main__":
    main()