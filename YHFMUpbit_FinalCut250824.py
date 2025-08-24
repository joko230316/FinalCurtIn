import os
import pyupbit
import pandas as pd
import numpy as np
import time
import schedule
import json
from datetime import datetime, timedelta
import pytz

# API 키 (환경변수에서 가져오기)
access_key = os.getenv("upbit_YHaccess_key")
secret_key = os.getenv("upbit_YHsecret_key")
upbit = pyupbit.Upbit(access_key, secret_key)

# 한국 시간대 설정
KST = pytz.timezone('Asia/Seoul')

# 전역 변수 설정
NORMAL_INTERVAL = 30  # 분 단위 (30분)
CHECK_INTERVAL = 10  # 분 단위 (10분)
TRAILING_INTERVAL = 1  # 분 단위 (1분)

PROFIT_THRESHOLD = 3.0  # 수익률 임계값 (%)
TRAILING_STOP_LOSS = 1.0  # 트레일링 스탑로스 (%)
BTC_VOLATILITY_THRESHOLD = 2.0  # BTC 변동성 임계값 (ATR %)
MIN_HOLDING_VALUE = 1000  # 최소 평가금액 (KRW)

# 모드 상태
current_mode = "normal"  # normal, check, trailing
mode_changed_time = datetime.now(KST)

# 보유 종목 및 최고 수익률 기록
holdings = {}
peak_profits = {}  # 각 종목의 최고 수익률 기록


# 안전한 가격 조회 함수
def safe_get_current_price(ticker, retry_count=3):
    for attempt in range(retry_count):
        try:
            price_data = pyupbit.get_current_price(ticker)

            # 다양한 응답 형식 처리
            if isinstance(price_data, (int, float)):
                return float(price_data)
            elif isinstance(price_data, dict):
                if 'trade_price' in price_data:
                    return float(price_data['trade_price'])
                elif 'price' in price_data:
                    return float(price_data['price'])
            elif isinstance(price_data, list) and len(price_data) > 0:
                if isinstance(price_data[0], dict) and 'trade_price' in price_data[0]:
                    return float(price_data[0]['trade_price'])

            # 응답 형식을 인식하지 못한 경우
            print(f"* 알 수 없는 응답 형식: {price_data}")
            return None

        except KeyError as e:
            print(f"* KeyError 발생 (시도 {attempt + 1}/{retry_count}): {e}")
            time.sleep(1)
        except Exception as e:
            print(f"* 가격 조회 오류 (시도 {attempt + 1}/{retry_count}): {e}")
            time.sleep(1)

    print(f"* {ticker} 가격 조회 실패 after {retry_count} attempts")
    return None


# 안전한 OHLCV 데이터 조회 함수
def safe_get_ohlcv(ticker, interval="minute60", count=24, retry_count=2):
    for attempt in range(retry_count):
        try:
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
            if df is not None and not df.empty:
                return df
            time.sleep(0.5)
        except Exception as e:
            print(f"* OHLCV 조회 오류 (시도 {attempt + 1}/{retry_count}): {e}")
            time.sleep(1)
    return None


# BTC ATR 계산 함수
def calculate_btc_atr():
    try:
        # BTC 1시간봉 데이터로 ATR 계산
        df = safe_get_ohlcv("KRW-BTC", interval="minute60", count=24)
        if df is None or len(df) < 14:
            print("* BTC ATR 계산: 데이터 부족")
            return 0.0

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = tr.max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]

        # ATR을 퍼센트로 변환 (현재가 대비)
        current_price = safe_get_current_price("KRW-BTC")
        if current_price and current_price > 0:
            atr_percent = (atr / current_price) * 100
            return round(atr_percent, 2)

        return 0.0
    except Exception as e:
        print(f"* BTC ATR 계산 오류: {e}")
        return 0.0


# 모드 판단 함수
def determine_mode():
    global current_mode, mode_changed_time

    # 보유 종목이 없는 경우 normal 모드
    if not holdings:
        if current_mode != "normal":
            current_mode = "normal"
            mode_changed_time = datetime.now(KST)
            print(f"모드 변경: normal (보유 종목 없음)")
        return "normal"

    # BTC 변동성 확인
    btc_atr = calculate_btc_atr()
    print(f"* BTC ATR: {btc_atr:.2f}%")

    # 보유 종목 중 수익률 10% 이상인 것이 있는지 확인
    high_profit_exists = False
    for ticker, info in holdings.items():
        current_price = safe_get_current_price(ticker)
        if current_price and 'entry_price' in info and info['entry_price'] > 0:
            profit_percent = ((current_price - info['entry_price']) / info['entry_price']) * 100
            if profit_percent >= PROFIT_THRESHOLD:
                high_profit_exists = True
                print(f"* {ticker} 수익률 {profit_percent:.2f}% (임계값 초과)")
                break

    # 모드 결정
    new_mode = "normal"
    if high_profit_exists:
        new_mode = "trailing"
        print("* trailing 모드 활성화 (고수익 종목 존재)")
    elif btc_atr >= BTC_VOLATILITY_THRESHOLD:
        new_mode = "check"
        print("* check 모드 활성화 (BTC 변동성 높음)")

    # 모드 변경 시 로그 출력
    if new_mode != current_mode:
        print(f"* 모드 변경: {current_mode} -> {new_mode}")
        current_mode = new_mode
        mode_changed_time = datetime.now(KST)

    return current_mode


# 보유 종목 정보 업데이트 함수
def update_holdings_info():
    global holdings, peak_profits

    print(f"\n{'=' * 60}")
    print("* 보유 종목 정보 업데이트")
    print(f"{'=' * 60}")

    # 업비트에서 현재 보유중인 종목 조회
    try:
        balances = upbit.get_balances()
    except Exception as e:
        print(f"* 잔고 조회 오류: {e}")
        return

    updated_count = 0
    skipped_count = 0

    for balance in balances:
        if balance['currency'] != 'KRW':
            ticker = f"KRW-{balance['currency']}"
            current_price = safe_get_current_price(ticker)

            if current_price and float(balance['balance']) > 0:
                avg_buy_price = float(balance['avg_buy_price'])
                amount = float(balance['balance'])
                current_value = amount * current_price

                # 평가금액이 1000원 이하인 경우 제외
                if current_value <= MIN_HOLDING_VALUE:
                    print(f"* {ticker} 제외 (평가금액: {current_value:.2f}원 <= {MIN_HOLDING_VALUE}원)")
                    skipped_count += 1
                    continue

                profit_percent = ((current_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price > 0 else 0

                # 보유 정보 업데이트
                holdings[ticker] = {
                    "entry_price": round(avg_buy_price, 2),
                    "amount": round(amount, 6),
                    "current_price": round(current_price, 2),
                    "profit_percent": round(profit_percent, 2),
                    "current_value": round(current_value, 2),
                    "update_time": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
                }

                # 최고 수익률 기록 초기화 (없는 경우)
                if ticker not in peak_profits:
                    peak_profits[ticker] = round(profit_percent, 2)
                else:
                    # 현재 수익률이 최고 수익률보다 높으면 업데이트
                    if profit_percent > peak_profits[ticker]:
                        peak_profits[ticker] = round(profit_percent, 2)
                        print(f"* {ticker} 최고 수익률 갱신: {peak_profits[ticker]:.2f}%")

                updated_count += 1

    print(f"* 보유 종목 {updated_count}개 정보 업데이트 완료")
    if skipped_count > 0:
        print(f"* 평가금액 부족 제외: {skipped_count}개")


# 트레일링 스탑 실행 함수
def execute_trailing_stop():
    global holdings, peak_profits

    print(f"\n{'=' * 60}")
    print("* 트레일링 스탑 실행")
    print(f"{'=' * 60}")

    sold_tickers = []

    for ticker, info in holdings.items():
        current_price = safe_get_current_price(ticker)
        if not current_price:
            continue

        # 현재 수익률 계산
        current_price = round(current_price, 2)
        profit_percent = round(((current_price - info['entry_price']) / info['entry_price']) * 100, 2)

        # 최고 수익률에서 -3% 이상 하락했는지 확인
        if (ticker in peak_profits and
                profit_percent <= (peak_profits[ticker] - TRAILING_STOP_LOSS) and
                peak_profits[ticker] >= PROFIT_THRESHOLD):

            print(f"* {ticker} 트레일링 스탑 조건 충족")
            print(f"   최고 수익률: {peak_profits[ticker]:.2f}%")
            print(f"   현재 수익률: {profit_percent:.2f}%")
            print(f"   하락폭: {peak_profits[ticker] - profit_percent:.2f}%")

            # 매도 실행
            try:
                amount = info['amount']
                print(f"   매도 수량: {amount:.6f}")

                result = upbit.sell_market_order(ticker, amount)
                print(f"   매도 결과: {result}")

                sold_tickers.append(ticker)
                time.sleep(1)  # 1초 대기

            except Exception as e:
                print(f"* {ticker} 매도 오류: {e}")

    # 매도된 종목 제거
    for ticker in sold_tickers:
        if ticker in holdings:
            del holdings[ticker]
        if ticker in peak_profits:
            del peak_profits[ticker]

    if sold_tickers:
        print(f"* 매도 완료 종목: {sold_tickers}")
    else:
        print("* 트레일링 스탑 조건에 해당하는 종목 없음")


# 메인 실행 함수
def main_execution():
    global current_mode

    current_time = datetime.now(KST)
    print(f"\n{'=' * 60}")
    print(f"* 실행 시간 (Asia/Seoul): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"{'=' * 60}")

    try:
        # 1. 보유 종목 정보 업데이트
        update_holdings_info()

        # 2. 현재 모드 판단
        mode = determine_mode()
        print(f"* 현재 모드: {mode.upper()}")

        # 3. 모드별 작업 실행
        if mode == "trailing":
            # trailing 모드: 트레일링 스탑 실행
            execute_trailing_stop()

            # 1분봉 데이터 요청 및 0.1초 대기
            for ticker in list(holdings.keys()):  # 리스트로 변환하여 안전하게 순회
                try:
                    df = safe_get_ohlcv(ticker, interval="minute1", count=1)
                    time.sleep(0.1)  # 0.1초 대기
                except Exception as e:
                    print(f"* {ticker} 1분봉 데이터 조회 오류: {e}")

        # 4. 현재 상태 출력
        print_holdings()

    except Exception as e:
        print(f"* 메인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 5. 다음 실행 간격 설정 (오류 발생해도 실행)
        set_next_schedule(current_mode)


# 보유 내역 출력 함수
def print_holdings():
    print(f"\n{'=' * 60}")
    print("* 현재 보유 내역")
    print(f"{'=' * 60}")

    current_time = datetime.now(KST)
    try:
        balance_krw = upbit.get_balance("KRW")
        balance_krw = round(balance_krw, 2)
    except Exception as e:
        print(f"* KRW 잔고 조회 오류: {e}")
        balance_krw = 0

    print(f"조회 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"현금 잔고: {balance_krw:,.2f} 원")
    print(f"현재 모드: {current_mode}")
    print(f"관리 대상 종목 수: {len(holdings)}개")
    print(f"{'-' * 60}")

    if not holdings:
        print("* 보유 중인 종목이 없습니다.")
        return

    total_investment = 0
    total_current_value = 0

    for ticker, info in holdings.items():
        # 최신 가격으로 업데이트
        current_price = safe_get_current_price(ticker)
        if current_price:
            current_price = round(current_price, 2)
            profit_percent = round(((current_price - info["entry_price"]) / info["entry_price"]) * 100, 2)
            current_value = round(info["amount"] * current_price, 2)

            info["current_price"] = current_price
            info["profit_percent"] = profit_percent
            info["current_value"] = current_value

            # 최고 수익률 업데이트
            if ticker in peak_profits:
                if profit_percent > peak_profits[ticker]:
                    peak_profits[ticker] = round(profit_percent, 2)

            total_investment += info["amount"] * info["entry_price"]
            total_current_value += current_value

            # 수익률에 따라 표시
            profit_sign = "[+]" if profit_percent >= 0 else "[-]"
            peak_info = f" (최고: {peak_profits.get(ticker, 0):.2f}%)" if ticker in peak_profits else ""

            print(f"* {ticker}")
            print(f"   매수가: {info['entry_price']:,.2f} 원")
            print(f"   현재가: {info['current_price']:,.2f} 원")
            print(f"   보유수량: {info['amount']:.6f}")
            print(f"   평가금액: {info['current_value']:,.2f} 원")
            print(f"   수익률: {profit_sign} {info['profit_percent']:+.2f}%{peak_info}")
            print(f"{'-' * 40}")

    # 총계 출력
    if total_investment > 0:
        total_investment = round(total_investment, 2)
        total_current_value = round(total_current_value, 2)
        total_profit = round(total_current_value - total_investment, 2)
        total_profit_percent = round((total_profit / total_investment) * 100, 2) if total_investment > 0 else 0
        profit_sign = "[+]" if total_profit >= 0 else "[-]"

        print(f"* 총 투자금액: {total_investment:,.2f} 원")
        print(f"* 총 평가금액: {total_current_value:,.2f} 원")
        print(f"* 총 수익금: {profit_sign} {total_profit:+,.2f} 원")
        print(f"* 총 수익률: {profit_sign} {total_profit_percent:+.2f}%")
        print(f"{'=' * 60}")

    # JSON 형태로 저장 (평가금액 1000원 이상 종목만)
    filtered_holdings = {}
    for ticker, info in holdings.items():
        if info["current_value"] > MIN_HOLDING_VALUE:
            filtered_holdings[ticker] = info

    output = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Seoul",
        "balance_krw": balance_krw,
        "current_mode": current_mode,
        "total_investment": round(total_investment, 2),
        "total_current_value": round(total_current_value, 2),
        "total_profit": round(total_profit, 2),
        "total_profit_percent": round(total_profit_percent, 2),
        "holdings": filtered_holdings,
        "peak_profits": peak_profits
    }

    try:
        with open("holdings_trailing.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"* JSON 저장 오류: {e}")


# 다음 실행 스케줄 설정 함수
def set_next_schedule(mode):
    # 기존 스케줄 클리어
    schedule.clear()

    # 모드에 따른 간격 설정
    if mode == "normal":
        interval = NORMAL_INTERVAL
    elif mode == "check":
        interval = CHECK_INTERVAL
    else:  # trailing
        interval = TRAILING_INTERVAL

    # 분 단위로 스케줄 설정
    schedule.every(interval).minutes.do(main_execution)

    next_run = datetime.now(KST) + timedelta(minutes=interval)
    print(f"* 다음 실행: {next_run.strftime('%H:%M:%S')} ({interval}분 후)")
    print(f"* 현재 모드: {mode}, 실행 간격: {interval}분")


# 프로그램 시작
print("* 프로그램 시작 - 트레일링 스탑 손익 관리")
print(f"* 현재 시간 (Asia/Seoul): {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
print(f"* 최소 평가금액: {MIN_HOLDING_VALUE}원 이상 종목만 관리")
print(f"{'=' * 60}")

# 초기 보유 종목 정보 업데이트
update_holdings_info()

# 초기 모드 판단 및 첫 실행
try:
    main_execution()
except Exception as e:
    print(f"* 초기 실행 중 오류: {e}")
    import traceback

    traceback.print_exc()
    # 오류 발생 시 기본적으로 normal 모드로 설정
    set_next_schedule("normal")

# 스케줄러 실행
print("\n* 스케줄러 실행 중...")
while True:
    try:
        schedule.run_pending()
        time.sleep(30)  # 30초마다 체크 (스케줄 정확도 향상)
    except KeyboardInterrupt:
        print("\n* 프로그램 종료")
        break
    except Exception as e:
        print(f"* 스케줄러 오류: {e}")
        time.sleep(60)