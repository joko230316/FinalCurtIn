import os
import pandas as pd
import numpy as np
import time
import schedule
import json
from datetime import datetime, timedelta
import pytz
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import math

# --- API 키 (환경변수에서 가져오기) ---
access_key = os.getenv("BINANCE_YH_API_KEY")
secret_key = os.getenv("BINANCE_YH_API_SECRET")

if not access_key or not secret_key:
    raise ValueError("BINANCE API 키가 환경변수에 설정되지 않았습니다.")

# --- 바이낸스 클라이언트 초기화 ---
client = Client(access_key, secret_key)

# --- 시간대 설정 ---
KST = pytz.timezone('Asia/Seoul')

# --- 전역 변수 설정 ---
NORMAL_INTERVAL = 30  # 분 단위 (30분)
CHECK_INTERVAL = 10  # 분 단위 (10분)
TRAILING_INTERVAL = 1  # 분 단위 (1분)

PROFIT_THRESHOLD = 3.0  # 수익률 임계값 (%)
TRAILING_STOP_LOSS = 1.0  # 트레일링 스탑로스 (%)
BTC_VOLATILITY_THRESHOLD = 2.0  # BTC 변동성 임계값 (ATR %)
MIN_HOLDING_VALUE = 5.0  # 최소 평가금액 (USDT)
MIN_NOTIONAL = 10.0  # 바이낸스 최소 주문 금액

# 모드 상태
current_mode = "normal"  # normal, check, trailing
mode_changed_time = datetime.now(KST)

# 보유 종목 및 최고 수익률 기록
holdings = {}
peak_profits = {}  # 각 종목의 최고 수익률 기록

# 심볼 정보 캐시
symbol_info_cache = {}


# --- 유틸리티 함수 ---
def get_symbol_info(symbol):
    """
    심볼 정보 가져오기 (캐싱 및 예외처리 적용)
    """
    if symbol in symbol_info_cache:
        return symbol_info_cache[symbol]
    try:
        info = client.get_symbol_info(symbol)
        if info:
            symbol_info_cache[symbol] = info
            return info
    except Exception as e:
        print(f"* {symbol} 심볼 정보 조회 오류: {e}")
    return None


def adjust_quantity_to_lot_size(symbol, quantity):
    """
    LOT_SIZE 필터에 맞게 수량 조정
    """
    info = get_symbol_info(symbol)
    if not info:
        return quantity

    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            min_qty = float(f['minQty'])

            if step_size > 0:
                adjusted_qty = math.floor(quantity / step_size) * step_size
            else:
                adjusted_qty = quantity

            if adjusted_qty < min_qty:
                adjusted_qty = min_qty

            decimal_places = len(str(step_size).split('.')[1]) if '.' in str(step_size) else 0
            return round(adjusted_qty, decimal_places)

    return quantity


def safe_get_current_price(ticker):
    """
    안전한 현재가 조회 함수
    """
    try:
        ticker_info = client.get_symbol_ticker(symbol=ticker)
        return float(ticker_info['price'])
    except BinanceAPIException as e:
        if e.code == -1121:
            # 유효하지 않은 심볼 오류는 무시하고 None 반환
            return None
        print(f"* {ticker} 현재가 조회 오류: {e}")
        return None
    except Exception as e:
        print(f"* {ticker} 현재가 조회 오류: {e}")
        return None


def safe_get_ohlcv(ticker, interval, count):
    """
    안전한 OHLCV 데이터 조회 함수
    """
    try:
        klines = client.get_klines(symbol=ticker, interval=interval, limit=count)
        df = pd.DataFrame(klines,
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume',
                                   'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"* {ticker} OHLCV 데이터 조회 오류: {e}")
        return None


def calculate_btc_atr():
    """
    BTC ATR 계산 함수
    """
    try:
        df = safe_get_ohlcv("BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, count=24)
        if df is None or len(df) < 14:
            print("* BTC ATR 계산: 데이터 부족")
            return 0.0

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = tr.max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]

        current_price = safe_get_current_price("BTCUSDT")
        if current_price and current_price > 0:
            atr_percent = (atr / current_price) * 100
            return round(atr_percent, 2)
        return 0.0
    except Exception as e:
        print(f"* BTC ATR 계산 오류: {e}")
        return 0.0


def determine_mode():
    """
    보유 종목 수익률과 BTC 변동성을 기반으로 모드 판단
    """
    global current_mode, mode_changed_time

    if not holdings:
        if current_mode != "normal":
            current_mode = "normal"
            mode_changed_time = datetime.now(KST)
            print("모드 변경: normal (보유 종목 없음)")
        return "normal"

    btc_atr = calculate_btc_atr()
    print(f"* BTC ATR: {btc_atr:.2f}%")

    high_profit_exists = any(
        (info['profit_percent'] >= PROFIT_THRESHOLD)
        for ticker, info in holdings.items()
    )

    new_mode = "normal"
    if high_profit_exists:
        new_mode = "trailing"
        print("trailing 모드 활성화 (고수익 종목 존재)")
    elif btc_atr >= BTC_VOLATILITY_THRESHOLD:
        new_mode = "check"
        print("check 모드 활성화 (BTC 변동성 높음)")

    if new_mode != current_mode:
        print(f"* 모드 변경: {current_mode} -> {new_mode}")
        current_mode = new_mode
        mode_changed_time = datetime.now(KST)
    return current_mode


def get_account_balances():
    """
    계좌 잔고를 안전하게 조회하는 함수
    """
    try:
        return client.get_account()['balances']
    except BinanceAPIException as e:
        print(f"* 잔고 조회 오류: {e}")
        return []


def get_weighted_average_buy_price(symbol):
    """
    가중 평균 매수가를 계산하는 함수
    """
    try:
        trades = client.get_my_trades(symbol=symbol)
        total_quantity = 0
        total_cost = 0

        for trade in trades:
            if trade['isBuyer']:
                quantity = float(trade['qty'])
                price = float(trade['price'])
                total_quantity += quantity
                total_cost += quantity * price

        return total_cost / total_quantity if total_quantity > 0 else 0
    except BinanceAPIException as e:
        if e.code == -1121:
            print(f"* {symbol} 거래 기록 조회 오류: Invalid symbol. 스킵합니다.")
        else:
            print(f"* {symbol} 거래 기록 조회 오류: {e}")
        return 0
    except Exception as e:
        print(f"* {symbol} 거래 기록 조회 중 예기치 않은 오류: {e}")
        return 0


def update_holdings_info():
    """
    보유 종목 정보 업데이트 함수
    """
    global holdings, peak_profits

    print(f"\n{'=' * 60}")
    print("* 보유 종목 정보 업데이트")
    print(f"{'=' * 60}")

    balances = get_account_balances()
    updated_holdings = {}
    skipped_count = 0

    for balance in balances:
        asset = balance['asset']
        amount = float(balance['free'])

        if asset == 'USDT' or amount < 0.000001:
            continue

        ticker = f"{asset}USDT"

        # 1. 심볼 유효성 검사
        if get_symbol_info(ticker) is None:
            print(f"* {ticker}는 유효하지 않은 심볼입니다. 스킵합니다.")
            skipped_count += 1
            continue

        try:
            current_price = safe_get_current_price(ticker)
            if not current_price:
                skipped_count += 1
                continue

            current_value = amount * current_price

            # 2. 최소 평가금액 검사
            if current_value <= MIN_HOLDING_VALUE:
                print(f"* {ticker} 제외 (평가금액: {current_value:.2f} USDT <= {MIN_HOLDING_VALUE} USDT)")
                skipped_count += 1
                continue

            # 3. 매수 평균가 계산
            entry_price = get_weighted_average_buy_price(ticker)

            # 매수 평균가가 없으면 (수동 매수 등) 현재가 기준으로 기록
            if entry_price <= 0:
                print(f"* {ticker}의 매수 평균가를 찾을 수 없습니다. 현재가를 매수가로 임시 설정합니다.")
                entry_price = current_price

            profit_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

            updated_holdings[ticker] = {
                "entry_price": round(entry_price, 8),
                "amount": round(amount, 8),
                "current_price": round(current_price, 8),
                "profit_percent": round(profit_percent, 2),
                "current_value": round(current_value, 2),
                "update_time": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
            }

            # 4. 최고 수익률 업데이트
            if ticker not in peak_profits or profit_percent > peak_profits[ticker]:
                peak_profits[ticker] = round(profit_percent, 2)
                print(f"* {ticker} 최고 수익률 갱신: {peak_profits[ticker]:.2f}%")

        except Exception as e:
            print(f"* {asset} 정보 업데이트 오류: {e}")
            skipped_count += 1

    holdings = updated_holdings
    print(f"* 보유 종목 {len(holdings)}개 정보 업데이트 완료")
    if skipped_count > 0:
        print(f"* 처리 제외: {skipped_count}개")


def execute_trailing_stop():
    """
    트레일링 스탑 실행 함수
    """
    global holdings, peak_profits
    print(f"\n{'=' * 60}")
    print("* 트레일링 스탑 실행")
    print(f"{'=' * 60}")

    sold_tickers = []

    for ticker, info in list(holdings.items()):
        current_price = safe_get_current_price(ticker)
        if not current_price or info['entry_price'] <= 0:
            continue

        profit_percent = ((current_price - info['entry_price']) / info['entry_price']) * 100

        # 트레일링 스탑 조건: 최고 수익률이 임계값을 넘고, 현재 수익률이 최고 수익률에서 일정 비율 하락했을 때
        if (ticker in peak_profits and
                peak_profits[ticker] >= PROFIT_THRESHOLD and
                profit_percent <= (peak_profits[ticker] - TRAILING_STOP_LOSS)):

            print(f"* {ticker} 트레일링 스탑 조건 충족")
            print(f"   최고 수익률: {peak_profits[ticker]:.2f}%")
            print(f"   현재 수익률: {profit_percent:.2f}%")

            try:
                amount_to_sell = adjust_quantity_to_lot_size(ticker, info['amount'])
                if amount_to_sell > 0:
                    print(f"   매도 수량: {amount_to_sell:.6f}")
                    order = client.order_market_sell(symbol=ticker, quantity=amount_to_sell)
                    print(f"   매도 결과: {order}")
                    sold_tickers.append(ticker)
                    time.sleep(1)
                else:
                    print(f"   {ticker} 매도 수량이 0이라 매도하지 않습니다.")
            except Exception as e:
                print(f"* {ticker} 매도 오류: {e}")

    for ticker in sold_tickers:
        holdings.pop(ticker, None)
        peak_profits.pop(ticker, None)

    if sold_tickers:
        print(f"* 매도 완료 종목: {sold_tickers}")
    else:
        print("* 트레일링 스탑 조건에 해당하는 종목 없음")


def main_execution():
    """
    주기적으로 실행되는 메인 로직
    """
    global current_mode
    current_time = datetime.now(KST)
    print(f"\n{'=' * 60}")
    print(f"* 실행 시간 (Asia/Seoul): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"{'=' * 60}")

    try:
        update_holdings_info()
        mode = determine_mode()
        print(f"* 현재 모드: {mode.upper()}")

        if mode == "trailing":
            execute_trailing_stop()

        print_holdings()

    except Exception as e:
        print(f"* 메인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        set_next_schedule(current_mode)


def print_holdings():
    """
    보유 내역을 콘솔에 출력하고 JSON 파일로 저장
    """
    print(f"\n{'=' * 60}")
    print("* 현재 보유 내역")
    print(f"{'=' * 60}")

    current_time = datetime.now(KST)
    balance_usdt = 0
    try:
        balance_usdt = float(client.get_asset_balance(asset='USDT')['free'])
    except Exception as e:
        print(f"* USDT 잔고 조회 오류: {e}")

    print(f"조회 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"현금 잔고: {balance_usdt:,.2f} USDT")
    print(f"현재 모드: {current_mode}")
    print(f"관리 대상 종목 수: {len(holdings)}개")
    print(f"{'-' * 60}")

    if not holdings:
        print("* 보유 중인 종목이 없습니다.")
        return

    total_investment = 0
    total_current_value = 0

    for ticker, info in holdings.items():
        # 최신 정보로 업데이트된 holdings 딕셔너리 사용
        total_investment += info.get("entry_price", 0) * info.get("amount", 0)
        total_current_value += info.get("current_value", 0)

        profit_sign = "[+]" if info['profit_percent'] >= 0 else "[-]"
        peak_info = f" (최고: {peak_profits.get(ticker, 0):.2f}%)" if ticker in peak_profits else ""

        print(f"* {ticker}")
        print(f"   매수가: {info['entry_price']:,.4f} USDT")
        print(f"   현재가: {info['current_price']:,.4f} USDT")
        print(f"   보유수량: {info['amount']:.6f}")
        print(f"   평가금액: {info['current_value']:,.2f} USDT")
        print(f"   수익률: {profit_sign} {info['profit_percent']:+.2f}%{peak_info}")
        print(f"{'-' * 40}")

    if total_investment > 0:
        total_profit = total_current_value - total_investment
        total_profit_percent = (total_profit / total_investment) * 100
        profit_sign = "[+]" if total_profit >= 0 else "[-]"

        print(f"* 총 투자금액: {total_investment:,.2f} USDT")
        print(f"* 총 평가금액: {total_current_value:,.2f} USDT")
        print(f"* 총 수익금: {profit_sign} {total_profit:+,.2f} USDT")
        print(f"* 총 수익률: {profit_sign} {total_profit_percent:+.2f}%")

    print(f"{'=' * 60}")

    filtered_holdings = {t: i for t, i in holdings.items() if i["current_value"] > MIN_HOLDING_VALUE}
    output = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Seoul",
        "balance_usdt": round(balance_usdt, 2),
        "current_mode": current_mode,
        "total_investment": round(total_investment, 2) if total_investment > 0 else 0,
        "total_current_value": round(total_current_value, 2) if total_current_value > 0 else 0,
        "total_profit": round(total_profit, 2) if total_investment > 0 else 0,
        "total_profit_percent": round(total_profit_percent, 2) if total_investment > 0 else 0,
        "holdings": filtered_holdings,
        "peak_profits": peak_profits
    }

    try:
        with open("holdings_trailing.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"* JSON 저장 오류: {e}")


def set_next_schedule(mode):
    """
    다음 실행 스케줄 설정 함수
    """
    schedule.clear()

    interval = TRAILING_INTERVAL if mode == "trailing" else CHECK_INTERVAL if mode == "check" else NORMAL_INTERVAL
    schedule.every(interval).minutes.do(main_execution)

    next_run = datetime.now(KST) + timedelta(minutes=interval)
    print(f"* 다음 실행: {next_run.strftime('%H:%M:%S')} ({interval}분 후)")
    print(f"* 현재 모드: {mode}, 실행 간격: {interval}분")


# --- 프로그램 시작 ---
if __name__ == "__main__":
    print("* 프로그램 시작 - 바이낸스 트레일링 스탑 손익 관리")
    print(f"* 현재 시간 (Asia/Seoul): {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"* 최소 평가금액: {MIN_HOLDING_VALUE} USDT 이상 종목만 관리")
    print(f"{'=' * 60}")

    try:
        main_execution()
    except Exception as e:
        print(f"* 초기 실행 중 오류: {e}")
        import traceback

        traceback.print_exc()
        set_next_schedule("normal")

    print("\n* 스케줄러 실행 중...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n* 프로그램 종료")
            break
        except Exception as e:
            print(f"* 스케줄러 오류: {e}")
            time.sleep(60)