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

# 매수 기록을 저장할 딕셔너리
buy_records = {}
holdings = {}


# 역헤드앤숄더 판단 함수
def detect_reverse_head_and_shoulders(df):
    prices = df['close'].values[-15:]
    if len(prices) < 15:
        return False

    left_shoulder = np.min(prices[0:5])
    head = np.min(prices[5:10])
    right_shoulder = np.min(prices[10:15])

    is_rhs = (
            head < left_shoulder and
            head < right_shoulder and
            abs(left_shoulder - right_shoulder) / left_shoulder < 0.05
    )
    return is_rhs


# 동일 종목 재매수 가능 여부 확인 함수
def can_buy_again(ticker):
    if ticker not in buy_records:
        return True

    last_buy_time = buy_records[ticker]
    current_time = datetime.now(KST)

    # 24시간(1일)이 지났는지 확인
    if current_time - last_buy_time >= timedelta(hours=24):
        return True
    return False


# 매수 실행 함수
def execute_buy():
    current_time = datetime.now(KST)
    print(f"\n{'=' * 60}")
    print(f"매수 실행 시간 (Asia/Seoul): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"{'=' * 60}")

    # 내 총 자산 조회
    balance_krw = upbit.get_balance("KRW")
    print(f"보유 KRW: {balance_krw:,.0f} 원")

    # 1종목당 투자금액: 자본의 30%
    investment_per_asset = balance_krw * 0.30
    print(f"종목당 투자 금액: {investment_per_asset:,.0f} 원")
    print(f"{'-' * 60}")

    # 종목 리스트 가져오기
    tickers = pyupbit.get_tickers(fiat="KRW")
    selected = []

    for ticker in tickers:
        try:
            # 동일 종목 재매수 제한 확인
            if not can_buy_again(ticker):
                print(f"{ticker}: 재매수 제한 중 (24시간 내 매수 기록 있음)")
                continue

            df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            if df is not None and len(df) >= 15:
                if detect_reverse_head_and_shoulders(df):
                    selected.append(ticker)
                    print(f"✅ RHS 감지됨: {ticker}")

            if len(selected) >= 3:  # 최대 3종목 선택
                break

            time.sleep(0.3)  # API rate 제한 회피

        except Exception as e:
            print(f"{ticker} 오류: {e}")
            continue

    # 매수 실행
    if selected:
        print(f"\n📈 매수 대상 종목 ({len(selected)}개): {selected}")
        print(f"{'-' * 60}")
    else:
        print("❌ 매수 대상 종목이 없습니다.")
        print_holdings()
        return

    for ticker in selected:
        try:
            price = pyupbit.get_current_price(ticker)
            krw_amount = investment_per_asset

            print(f"\n🔹 {ticker} 매수 시도")
            print(f"   현재가: {price:,.0f} 원")
            print(f"   투자금액: {krw_amount:,.0f} 원")
            print(f"   예상 수량: {krw_amount / price:.6f} {ticker.replace('KRW-', '')}")

            result = upbit.buy_market_order(ticker, krw_amount)
            print(f"   매수 결과: {result}")

            # 매수 기록 저장
            buy_records[ticker] = current_time
            holdings[ticker] = {
                "entry_price": price,
                "amount": krw_amount / price,
                "total_value": krw_amount,
                "current_price": price,
                "profit_percent": 0.0,
                "buy_time": current_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"   매수 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            time.sleep(1)

        except Exception as e:
            print(f"❌ {ticker} 매수 오류: {e}")

    # 보유 내역 출력
    print_holdings()


# 보유 내역 출력 함수
def print_holdings():
    print(f"\n{'=' * 60}")
    print("📊 보유 내역")
    print(f"{'=' * 60}")

    current_time = datetime.now(KST)
    balance_krw = upbit.get_balance("KRW")

    print(f"조회 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    print(f"현금 잔고: {balance_krw:,.0f} 원")
    print(f"{'-' * 60}")

    if not holdings:
        print("보유 중인 종목이 없습니다.")
        return

    total_investment = 0
    total_current_value = 0

    for ticker, info in holdings.items():
        current_price = pyupbit.get_current_price(ticker)
        if current_price is not None:
            profit_percent = ((current_price - info["entry_price"]) / info["entry_price"]) * 100
            info["current_price"] = current_price
            info["profit_percent"] = profit_percent
            info["total_value"] = info["amount"] * current_price

            total_investment += info["amount"] * info["entry_price"]
            total_current_value += info["total_value"]

            # 수익률에 따라 색상 설정
            profit_color = "🟢" if profit_percent >= 0 else "🔴"

            print(f"🔸 {ticker}")
            print(f"   매수시간: {info.get('buy_time', 'N/A')}")
            print(f"   매수가: {info['entry_price']:,.0f} 원")
            print(f"   현재가: {info['current_price']:,.0f} 원")
            print(f"   보유수량: {info['amount']:.6f}")
            print(f"   평가금액: {info['total_value']:,.0f} 원")
            print(f"   수익률: {profit_color} {info['profit_percent']:+.2f}%")
            print(f"{'-' * 40}")

    # 총계 출력
    if total_investment > 0:
        total_profit = total_current_value - total_investment
        total_profit_percent = (total_profit / total_investment) * 100
        profit_color = "🟢" if total_profit >= 0 else "🔴"

        print(f"💰 총 투자금액: {total_investment:,.0f} 원")
        print(f"💰 총 평가금액: {total_current_value:,.0f} 원")
        print(f"💰 총 수익금: {profit_color} {total_profit:+,.0f} 원")
        print(f"💰 총 수익률: {profit_color} {total_profit_percent:+.2f}%")
        print(f"{'=' * 60}")

    # JSON 형태로 출력
    output = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Seoul",
        "balance_krw": balance_krw,
        "total_investment": total_investment,
        "total_current_value": total_current_value,
        "total_profit": total_profit,
        "total_profit_percent": total_profit_percent,
        "holdings": holdings
    }

    with open("holdings.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


# 즉시 한 번 실행
print("프로그램 시작 - 매일 4시간 간격으로 매수 실행됩니다.")
print(f"현재 시간 (Asia/Seoul): {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
print(f"{'=' * 60}")

# 즉시 첫 실행
execute_buy()

# 매일 4시간 간격으로 실행되도록 스케줄 설정
schedule.every(4).hours.do(execute_buy)

# 스케줄러 실행
while True:
    schedule.run_pending()
    time.sleep(60)  # 1분마다 체크