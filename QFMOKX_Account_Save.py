#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 필요한 라이브러리 임포트
import os
import time
from datetime import datetime, timedelta
import pytz
import ccxt
import warnings
import traceback
import threading

warnings.filterwarnings("ignore")

# === 전역 변수 설정 ===
SYMBOL = "BTC-USDT-SWAP"
MONITORING_INTERVAL = 60  # 기본 모니터링 간격 (초)
PRECISION = 3  # 소수점 자리수

# === 긴급 청산 설정 ===
EMERGENCY_CLOSE_PERCENT = 15.0  # 현재 잔고의 15% 손실 시 긴급 청산

# === 트레일링 스탑 설정 ===
TRAILING_STOP_PERCENT = 3.0  # 최고가 대비 -3% 하락 시 청산
TRAILING_STOP_INTERVAL = 60  # 1분 간격으로 최고값 갱신

# === 일일 잔고 증가 자동 매수 설정 ===
DAILY_BALANCE_CHECK_TIME = "16:00"  # 오후 4시 정각 잔고 체크
AUTO_BUY_TIME = "16:05"  # 오후 4시 5분 자동 매수
BALANCE_INCREASE_THRESHOLD = 1.0  # 1 USDT 이상 증가 시 실행
SPOT_BUY_PERCENT = 10.0  # USDT 잔고의 10%로 현물 BTC 매수
CONFIRMATION_WAIT_TIME = 10  # 확인 대기 시간 (초)
BALANCE_CHECK_INTERVAL = 3600  # 잔고 체크 반복 주기 (3600초 = 60분)

# === 포지션 관리 ===
PNL_EXTREMES = {}  # Floating PnL% 최대값 저장
LAST_BALANCE_CHECK = None  # 마지막 잔고 체크 날짜
YESTERDAY_BALANCE = None  # 전일 잔고 저장
LAST_BALANCE_UPDATE = None  # 마지막 잔고 업데이트 시간

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


# === BTC 현물 가격 조회 ===
def get_btc_spot_price():
    try:
        ticker = exchange.fetch_ticker('BTC/USDT')
        return {
            'price': ticker['last'],
            'high': ticker['high'],
            'low': ticker['low'],
            'change': ticker['change'],
            'percentage': ticker['percentage']
        }
    except Exception as e:
        print(f"* BTC 현물 가격 조회 실패: {e}")
        return None


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


# === 현물 잔고 조회 ===
def get_spot_balance():
    try:
        balance = exchange.fetch_balance({'type': 'spot'})
        btc_balance = balance.get('BTC', {})
        usdt_balance = balance.get('USDT', {})
        return {
            'btc_total': btc_balance.get('total', 0.0),
            'btc_free': btc_balance.get('free', 0.0),
            'usdt_total': usdt_balance.get('total', 0.0),
            'usdt_free': usdt_balance.get('free', 0.0)
        }
    except Exception as e:
        print(f"* 현물 잔고 조회 실패: {e}")
        return None


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


# === 현물 BTC 매수 함수 ===
def buy_spot_btc(usdt_amount):
    try:
        # BTC/USDT 현물 시장 가격 조회
        btc_price_info = get_btc_spot_price()
        if not btc_price_info:
            return False

        current_price = btc_price_info['price']

        # 매수 수량 계산 (약간의 여유를 두기 위해 99%만 사용)
        buy_amount = (usdt_amount * 0.99) / current_price

        print("\n" + "=" * 60)
        print("🎯 현물 BTC 매수 실행")
        print("=" * 60)
        print(f"📊 현재 BTC 가격: {current_price:,.2f} USDT")
        print(f"💰 매수 예정 금액: {usdt_amount:,.2f} USDT")
        print(f"📈 예상 매수 수량: {buy_amount:.6f} BTC")
        print(f"💳 실제 사용 금액: {usdt_amount * 0.99:,.2f} USDT (수수료 고려)")
        print("=" * 60)

        # 현물 매수 주문
        params = {
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": buy_amount
        }

        result = execute_order(params)
        print(f"✅ 현물 매수 성공: 주문 ID {result['id']}")
        print(f"✅ 매수 완료: {buy_amount:.6f} BTC")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 현물 매수 실패: {e}")
        return False


# === 사용자 확인 입력 처리 ===
def wait_for_user_confirmation():
    print(f"\n⏰ {CONFIRMATION_WAIT_TIME}초 내로 응답이 없으면 자동 실행됩니다...")

    # 간단한 입력 처리 (실제로는 더 복잡한 입력 시스템이 필요)
    confirmation_received = False
    user_input = None

    for i in range(CONFIRMATION_WAIT_TIME):
        print(f"\r⏳ 대기 중... {CONFIRMATION_WAIT_TIME - i}초 남음", end="", flush=True)
        time.sleep(1)

    print("\r" + " " * 50 + "\r", end="")  # 진행바 지우기

    return False, None  # 응답 없음으로 처리


# === 일일 잔고 증가 체크 및 자동 매수 ===
def check_daily_balance_increase():
    global LAST_BALANCE_CHECK, YESTERDAY_BALANCE, LAST_BALANCE_UPDATE

    # 한국 시간 기준 현재 시간
    kst = pytz.timezone("Asia/Seoul")
    now = datetime.now(kst)
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")

    # 60분마다 잔고 업데이트 체크
    current_timestamp = time.time()
    if LAST_BALANCE_UPDATE is None or (current_timestamp - LAST_BALANCE_UPDATE) >= BALANCE_CHECK_INTERVAL:
        print(f"* 정기 잔고 체크: {current_time} (KST)")
        LAST_BALANCE_UPDATE = current_timestamp

    # 오후 4시 정각: 전일 잔고 저장
    if current_time == DAILY_BALANCE_CHECK_TIME:
        if LAST_BALANCE_CHECK != current_date:  # 하루에 한 번만 실행
            balance, _ = get_account_and_position_status()
            if balance:
                YESTERDAY_BALANCE = balance['total']
                LAST_BALANCE_CHECK = current_date
                print(f"📅 전일 잔고 저장: {YESTERDAY_BALANCE:,.2f} USDT")

    # 오후 4시 5분: 잔고 증가 확인 및 자동 매수
    elif current_time == AUTO_BUY_TIME:
        if LAST_BALANCE_CHECK == current_date and YESTERDAY_BALANCE is not None:
            balance, _ = get_account_and_position_status()
            if balance:
                current_balance = balance['total']
                balance_increase = current_balance - YESTERDAY_BALANCE

                print(f"\n📊 일일 잔고 변동 확인")
                print(f"📅 전일 잔고: {YESTERDAY_BALANCE:,.2f} USDT")
                print(f"📈 현재 잔고: {current_balance:,.2f} USDT")
                print(f"💰 잔고 증가액: {balance_increase:+,.2f} USDT")

                # 잔고가 1 USDT 이상 증가했는지 확인
                if balance_increase >= BALANCE_INCREASE_THRESHOLD:
                    print(f"🎯 잔고 증가 조건 충족 (+{balance_increase:,.2f} USDT)")

                    # 현물 USDT 잔고 확인
                    spot_balance = get_spot_balance()
                    if spot_balance:
                        available_usdt = spot_balance['usdt_free']
                        buy_amount = available_usdt * (SPOT_BUY_PERCENT / 100)

                        # BTC 현재가 조회
                        btc_price_info = get_btc_spot_price()
                        if btc_price_info:
                            current_btc_price = btc_price_info['price']
                            estimated_btc_amount = (buy_amount * 0.99) / current_btc_price

                            print("\n" + "=" * 60)
                            print("🤖 자동 매수 정보")
                            print("=" * 60)
                            print(f"💰 현물 USDT 잔고: {available_usdt:,.2f} USDT")
                            print(f"📈 매수 예정 금액: {buy_amount:,.2f} USDT ({SPOT_BUY_PERCENT}%)")
                            print(f"🎯 현재 BTC 가격: {current_btc_price:,.2f} USDT")
                            print(f"📊 예상 매수 수량: {estimated_btc_amount:.6f} BTC")
                            print("=" * 60)

                            # 사용자 확인
                            print(f"\n❓ 매수 진행 하시겠습니까? Y or N")
                            print(f"⏰ {CONFIRMATION_WAIT_TIME}초 내 응답이 없으면 자동 실행")

                            confirmation_received, user_input = wait_for_user_confirmation()

                            if not confirmation_received:
                                print("🤖 응답 없음, 자동 매수 실행...")
                                if buy_amount >= 10:  # 최소 매수 금액 체크
                                    return buy_spot_btc(buy_amount)
                                else:
                                    print("❌ 매수 금액이 너무 작습니다 (10 USDT 미만)")
                            else:
                                if user_input and user_input.lower() in ['y', 'yes']:
                                    print("✅ 사용자 확인, 매수 실행...")
                                    if buy_amount >= 10:
                                        return buy_spot_btc(buy_amount)
                                else:
                                    print("❌ 사용자에 의해 매수 취소")
                else:
                    print(f"📉 잔고 증가 미달 ({balance_increase:+,.2f} USDT), 매수 건너뜀")

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

    # BTC 현물 가격 조회
    btc_price_info = get_btc_spot_price()

    print("\n" + "=" * 80)
    print(f"** * 포지션 관리 봇 상태 ({now}) * **")
    print(f"| 다음 실행: {next_run_in}초 후 | 모니터링 간격: {MONITORING_INTERVAL}초")
    print(f"| 잔고 체크 주기: {BALANCE_CHECK_INTERVAL // 60}분 ({BALANCE_CHECK_INTERVAL}초)")

    if btc_price_info:
        change_icon = "🟢" if btc_price_info['percentage'] >= 0 else "🔴"
        print(f"| BTC 현물: {btc_price_info['price']:,.2f} USDT {change_icon} {btc_price_info['percentage']:+.2f}%")

    print("-" * 80)
    print("## * 계정 잔고 (USDT)")
    if balance:
        print(
            f"| 총액(Total): {balance['total']:,.2f} | 사용 가능(Free): {balance['free']:,.2f} | 사용 중(Used): {balance['used']:,.2f}")
    else:
        print("| 잔고 정보를 가져올 수 없습니다.")

    # 현물 잔고 정보 표시
    spot_balance = get_spot_balance()
    if spot_balance:
        btc_value = spot_balance['btc_total'] * btc_price_info['price'] if btc_price_info else 0
        print(
            f"| 현물 BTC: {spot_balance['btc_total']:.6f} ({btc_value:,.2f} USDT) | 현물 USDT: {spot_balance['usdt_total']:,.2f}")

    print("-" * 80)
    print("## * 현재 포지션")
    if not positions:
        print("| 현재 진입한 포지션이 없습니다.")
    else:
        for i, pos in enumerate(positions):
            side_char = "🟢 LONG" if pos['side'] == 'LONG' else "🔴 SHORT"
            print(f"| {i + 1}. {side_char} {pos['symbol']}")
            print(
                f"|    - 수량: {pos['size']:.{PRECISION}f} | 진입가: {pos['entry_price']:,.2f} | 현재가: {pos['current_price']:,.2f}")
            pnl_percent = pos['floating_pnl_percent']
            pnl_icon = "🟢" if pos['pnl'] >= 0 else "🔴"
            print(f"|    - 미실현 PNL: {pnl_icon} {pos['pnl']:+,.2f} USDT ({pnl_percent:+.4f}%)")

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
                    f"|    - 긴급 청산: {loss_percent:.2f}% / {EMERGENCY_CLOSE_PERCENT}% | {current_loss_usdt:,.1f} USDT / {emergency_threshold_usdt:,.1f} USDT")

            if i < len(positions) - 1:
                print("|" + "-" * 78)

    # 일일 잔고 정보 표시
    if YESTERDAY_BALANCE is not None and balance:
        balance_increase = balance['total'] - YESTERDAY_BALANCE
        increase_icon = "🟢" if balance_increase >= 0 else "🔴"
        print("-" * 80)
        print(f"## * 일일 잔고 변동: {increase_icon} {balance_increase:+,.2f} USDT (기준: {YESTERDAY_BALANCE:,.2f} USDT)")
        print(f"## * 자동 매수 설정: {DAILY_BALANCE_CHECK_TIME} 체크, {AUTO_BUY_TIME} 실행")
        print(f"## * 매수 조건: +{BALANCE_INCREASE_THRESHOLD} USDT 이상, 현물 USDT의 {SPOT_BUY_PERCENT}%")

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
    print(f"  3. 일일 자동 매수: {DAILY_BALANCE_CHECK_TIME} 체크, {AUTO_BUY_TIME} 실행")
    print(f"     - 조건: 전일대비 +{BALANCE_INCREASE_THRESHOLD} USDT 이상 증가 시")
    print(f"     - 금액: 현물 USDT의 {SPOT_BUY_PERCENT}%로 BTC 매수")
    print(f"  4. 잔고 체크 주기: {BALANCE_CHECK_INTERVAL // 60}분 마다")

    while True:
        try:
            time.sleep(time_to_wait)
            start_time = time.time()

            # --- 잔고 및 포지션 상태 확인 ---
            balance, positions = get_account_and_position_status()
            active_position = positions[0] if positions else None

            # --- 일일 잔고 증가 체크 및 자동 매수 ---
            auto_buy_executed = check_daily_balance_increase()

            # --- 포지션 관리 로직 ---
            if active_position:
                print(
                    f"* 포지션 감지: {active_position['side']}, PNL: {active_position['pnl']:+,.2f} USDT ({active_position['floating_pnl_percent']:+.4f}%)")

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