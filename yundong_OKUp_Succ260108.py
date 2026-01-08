import os
import time
import pyupbit
import ccxt
from dotenv import load_dotenv
from datetime import datetime, date

# =========================================================
# 전역 변수
# =========================================================
CHECK_INTERVAL = 300          # 5분
LEVERAGE = 10
SIZE_MULTIPLIER = 0.1
SYMBOL = "BTC/USDT:USDT"
MAX_DAILY_LOSS_RATE = 0.30   # 1일 최대 손실 30%

# =========================================================
# 환경변수 로드
# =========================================================
load_dotenv()

UPBIT_ACCESS = os.getenv("upbit_YHaccess_key")
UPBIT_SECRET = os.getenv("upbit_YHsecret_key")

OKX_ACCESS = os.getenv("OKXYH_API_KEY")
OKX_SECRET = os.getenv("OKXYH_API_SECRET")
OKX_PASSPHRASE = os.getenv("OKXYH_API_PASSPHRASE")

# =========================================================
# 거래소 연결
# =========================================================
upbit = pyupbit.Upbit(UPBIT_ACCESS, UPBIT_SECRET)

okx = ccxt.okx({
    'apiKey': OKX_ACCESS,
    'secret': OKX_SECRET,
    'password': OKX_PASSPHRASE,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

okx.set_leverage(LEVERAGE, SYMBOL)

# =========================================================
# 기준 잔고 (1일 손실 제한용)
# =========================================================
def get_okx_usdt_balance():
    bal = okx.fetch_balance()
    return float(bal['total'].get('USDT', 0))

START_DATE = date.today()
START_BALANCE = get_okx_usdt_balance()
TRADING_ALLOWED = True

# =========================================================
# 업비트 평균 수익률
# =========================================================
def get_upbit_average_return():
    balances = upbit.get_balances()
    total_return = 0.0
    count = 0

    for b in balances:
        currency = b['currency']
        balance = float(b['balance'])
        avg_price = float(b['avg_buy_price'])

        if currency == "KRW" or balance <= 0 or avg_price <= 0:
            continue

        price = pyupbit.get_current_price(f"KRW-{currency}")
        if not price:
            continue

        ret = (price - avg_price) / avg_price * 100
        total_return += ret
        count += 1

    return (total_return / count if count > 0 else 0.0), count

# =========================================================
# OKX 포지션 조회
# =========================================================
def get_position():
    positions = okx.fetch_positions([SYMBOL])
    for p in positions:
        if float(p['contracts']) != 0:
            return p
    return None

# =========================================================
# 포지션 청산
# =========================================================
def close_all_positions():
    pos = get_position()
    if not pos:
        return

    side = 'sell' if pos['side'] == 'long' else 'buy'
    amount = abs(float(pos['contracts']))
    okx.create_market_order(SYMBOL, side, amount)
    print("🚨 OKX 포지션 전부 시장가 청산 완료")

# =========================================================
# 포지션 유지
# =========================================================
def maintain_position(avg_return):
    target_size = round(abs(int(avg_return)) * SIZE_MULTIPLIER, 2)
    if target_size <= 0:
        return

    side = 'buy' if avg_return > 0 else 'sell'
    target_side = 'long' if avg_return > 0 else 'short'

    pos = get_position()
    current_size = abs(float(pos['contracts'])) if pos else 0
    current_side = pos['side'] if pos else None

    if pos and current_side != target_side:
        close_all_positions()
        current_size = 0

    diff = round(target_size - current_size, 2)
    if diff > 0:
        okx.create_market_order(SYMBOL, side, diff)
        print(f"📌 {target_side.upper()} 유지 주문 | 수량: {diff}")

# =========================================================
# OKX 상태 출력
# =========================================================
def print_okx_status():
    balance = get_okx_usdt_balance()
    print("\n" + "=" * 80)
    print("📌 OKX 선물 계정 현황")
    print("=" * 80)
    print(f"💰 총 USDT 잔고 : {balance:.2f} USDT\n")

    pos = get_position()
    print(f"{'심볼':<20}{'방향':<15}{'수량':<10}{'진입가':<12}{'PnL':<10}{'PnL%'}")
    print("-" * 75)

    if pos:
        print(
            f"{pos['symbol']:<20}"
            f"{pos['side'].upper():<15}"
            f"{float(pos['contracts']):<10.2f}"
            f"{float(pos['entryPrice']):<12,.2f}"
            f"{float(pos['unrealizedPnl']):<10.2f}"
            f"{float(pos['percentage']):.2f}%"
        )
    else:
        print("현재 진입된 포지션 없음")

# =========================================================
# 1일 손실 제한 체크
# =========================================================
def check_daily_loss():
    global TRADING_ALLOWED
    current_balance = get_okx_usdt_balance()
    loss_rate = (START_BALANCE - current_balance) / START_BALANCE

    if loss_rate >= MAX_DAILY_LOSS_RATE:
        print("🛑 1일 최대 손실 30% 초과 → 거래 중단")
        close_all_positions()
        TRADING_ALLOWED = False

# =========================================================
# 메인 루프
# =========================================================
if __name__ == "__main__":
    print("🚀 업비트 평균 수익률 기반 OKX 선물 자동 유지 전략 시작")
    print(f"📅 기준일: {START_DATE} | 기준 잔고: {START_BALANCE:.2f} USDT")

    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg_return, coin_count = get_upbit_average_return()

            print(f"\n⏰ {now}")
            print(f"📊 KRW 제외 코인 평균 수익률 : {avg_return:.2f}%")

            check_daily_loss()

            if not TRADING_ALLOWED:
                print("⛔ 금일 거래 중단 상태")
            else:
                if avg_return == 0 and coin_count == 0:
                    close_all_positions()
                else:
                    maintain_position(avg_return)

            print_okx_status()

        except Exception as e:
            print("❌ 오류 발생:", e)

        time.sleep(CHECK_INTERVAL)