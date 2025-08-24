# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import schedule
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import sys
import math

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- 환경변수에서 API 키 가져오기 ---
access_key = os.getenv("BINANCE_YH_API_KEY")
secret_key = os.getenv("BINANCE_YH_API_SECRET")

if not access_key or not secret_key:
    logger.error("환경변수에서 API 키를 찾을 수 없습니다.")
    exit(1)

# --- 바이낸스 클라이언트 초기화 ---
client = Client(access_key, secret_key, testnet=False)

# --- 거래 파라미터 ---
MAX_POSITIONS = 5
POSITION_PERCENT = 0.33
COOLDOWN_HOURS = 24
FEE_RATIO = 0.001
INTERVAL_HOURS = 4
MIN_NOTIONAL = 10.0  # 바이낸스 최소 주문 금액 (10 USDT)

# 거래 기록 저장
trade_history = []
last_buy_times = {}
symbol_info_cache = {}


def get_symbol_info(symbol):
    """심볼 정보 가져오기 (캐싱 적용)"""
    if symbol in symbol_info_cache:
        return symbol_info_cache[symbol]

    try:
        info = client.get_symbol_info(symbol)
        symbol_info_cache[symbol] = info
        return info
    except Exception as e:
        logger.error("%s 정보 가져오기 오류: %s", symbol, e)
        return None


def get_lot_size_filters(symbol):
    """LOT_SIZE 필터 정보 가져오기"""
    info = get_symbol_info(symbol)
    if not info:
        return None

    for filter in info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            return {
                'min_qty': float(filter['minQty']),
                'max_qty': float(filter['maxQty']),
                'step_size': float(filter['stepSize'])
            }
    return None


def get_min_notional(symbol):
    """최소 주문 금액 정보 가져오기"""
    info = get_symbol_info(symbol)
    if not info:
        return MIN_NOTIONAL

    for filter in info['filters']:
        if filter['filterType'] == 'NOTIONAL':
            return float(filter.get('minNotional', MIN_NOTIONAL))

    return MIN_NOTIONAL


def adjust_quantity_to_lot_size(symbol, quantity):
    """LOT_SIZE 필터에 맞게 수량 조정"""
    lot_filters = get_lot_size_filters(symbol)
    if not lot_filters:
        return quantity

    min_qty = lot_filters['min_qty']
    max_qty = lot_filters['max_qty']
    step_size = lot_filters['step_size']

    # stepSize에 맞게 수량 조정 (내림 처리)
    if step_size > 0:
        adjusted_qty = math.floor(quantity / step_size) * step_size
    else:
        adjusted_qty = quantity

    # 최소 수량 확인
    if adjusted_qty < min_qty:
        logger.info("%s: 최소 주문 수량 %.8f 미달, 최소 수량으로 조정", symbol, min_qty)
        adjusted_qty = min_qty

    # 최대 수량 확인
    if adjusted_qty > max_qty:
        logger.info("%s: 최대 주문 수량 %.8f 초과, 최대 수량으로 조정", symbol, max_qty)
        adjusted_qty = max_qty

    # 소수점 자리수 조정 (step_size의 소수점 자리수에 맞춤)
    if step_size > 0:
        decimal_places = len(str(step_size).split('.')[1]) if '.' in str(step_size) else 0
        adjusted_qty = round(adjusted_qty, decimal_places)

    return adjusted_qty


def check_and_adjust_order_parameters(symbol, usdt_amount):
    """주문 파라미터 확인 및 조정"""
    try:
        # 현재 가격 확인
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])

        # 초기 수량 계산
        quantity = usdt_amount / current_price

        # LOT_SIZE 필터 적용
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, quantity)

        if adjusted_quantity <= 0:
            logger.warning("%s: 조정된 수량이 0입니다", symbol)
            return None, None, None

        # 실제 주문 금액 계산
        actual_amount = adjusted_quantity * current_price

        # 최소 주문 금액 확인
        min_notional = get_min_notional(symbol)
        if actual_amount < min_notional:
            logger.info("%s: 주문 금액 %.2f USDT가 최소 금액 %.2f USDT 미달",
                        symbol, actual_amount, min_notional)

            # 최소 주문 금액으로 재계산
            required_quantity = min_notional / current_price
            adjusted_quantity = adjust_quantity_to_lot_size(symbol, required_quantity)
            actual_amount = adjusted_quantity * current_price

            logger.info("%s: 최소 금액으로 조정 - 수량: %.8f, 금액: %.2f USDT",
                        symbol, adjusted_quantity, actual_amount)

        return adjusted_quantity, current_price, actual_amount

    except Exception as e:
        logger.error("%s: 주문 파라미터 계산 오류: %s", symbol, e)
        return None, None, None


def get_usdt_balance():
    """USDT 잔고 조회"""
    try:
        balance = client.get_asset_balance(asset='USDT')
        return float(balance['free'])
    except BinanceAPIException as e:
        logger.error("잔고 조회 오류: %s", e)
        return 0


def get_current_positions():
    """현재 보유 포지션 조회"""
    try:
        account = client.get_account()
        positions = []
        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked

            if total > 0.000001 and asset != 'USDT' and asset != 'BNB':
                try:
                    symbol = f"{asset}USDT"
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    value = total * price

                    # 매수 시간 정보 추가 (있는 경우)
                    buy_time = last_buy_times.get(symbol, None)

                    positions.append({
                        'symbol': symbol,
                        'asset': asset,
                        'quantity': total,
                        'current_price': price,
                        'value': value,
                        'buy_time': buy_time
                    })
                except:
                    continue

        return positions
    except BinanceAPIException as e:
        logger.error("포지션 조회 오류: %s", e)
        return []


def check_sell_signal(symbol):
    """매도 신호 확인"""
    try:
        df = get_binance_ohlcv(symbol, 'hourly', hours=100)
        if df is None or len(df) < 50:
            return False, None

        patterns = detect_head_shoulders(df)
        if patterns:
            latest_pattern = patterns[-1]
            pattern_time = df.index[latest_pattern[1]]

            if (datetime.now() - pattern_time).total_seconds() <= 12 * 3600:
                return True, pattern_time

        return False, None

    except Exception as e:
        logger.error("%s 매도 신호 확인 오류: %s", symbol, e)
        return False, None


def format_number(value, decimals=2):
    """숫자를 가독성 있게 포맷팅"""
    if abs(value) < 0.01:
        return f"{value:.8f}"
    elif abs(value) < 1:
        return f"{value:.6f}"
    else:
        return f"{value:,.{decimals}f}"


def format_time_difference(time_diff):
    """시간 차이를 가독성 있게 포맷팅"""
    if time_diff is None:
        return "알 수 없음"

    if isinstance(time_diff, timedelta):
        total_seconds = time_diff.total_seconds()
    else:
        total_seconds = time_diff

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    if hours > 0:
        return f"{hours}시간 {minutes}분"
    else:
        return f"{minutes}분"


def display_portfolio_summary():
    """포트폴리오 요약 정보 출력 (모든 보유 내역 포함)"""
    print("\n" + "=" * 90)
    print(f"포트폴리오 현황 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    usdt_balance = get_usdt_balance()
    print(f"USDT 잔고: {format_number(usdt_balance)} USDT")

    positions = get_current_positions()
    total_position_value = sum(pos['value'] for pos in positions)
    total_assets = usdt_balance + total_position_value

    print(f"총 자산: {format_number(total_assets)} USDT")
    print(f"보유 포지션: {len(positions)}개")
    print(f"포지션 총 가치: {format_number(total_position_value)} USDT")
    print("-" * 90)

    if positions:
        print("모든 보유 내역 (거래 실행 여부 포함):")
        print("-" * 90)
        print(f"{'종목':<8} {'수량':<15} {'현재가':<12} {'평가금액':<12} {'보유시간':<12} {'매도신호':<8}")
        print("-" * 90)

        for pos in positions:
            symbol = pos['symbol'].replace('USDT', '')
            quantity = pos['quantity']
            current_price = pos['current_price']
            value = pos['value']

            # 보유 시간 계산
            hold_time = "알 수 없음"
            if pos['buy_time']:
                time_diff = datetime.now() - pos['buy_time']
                hold_time = format_time_difference(time_diff)

            # 매도 신호 확인
            has_sell_signal, signal_time = check_sell_signal(pos['symbol'])
            sell_signal_text = "있음" if has_sell_signal else "없음"

            print(f"{symbol:<8} {format_number(quantity, 8):<15} {format_number(current_price):<12} "
                  f"{format_number(value):<12} {hold_time:<12} {sell_signal_text:<8}")

            # 매도 신호가 있는 경우 상세 정보 출력
            if has_sell_signal:
                print(f"  -> 매도 신호 시간: {signal_time.strftime('%Y-%m-%d %H:%M')}")

        print("-" * 90)

        # 매도 신호가 있는 포지션 요약
        sell_signal_positions = []
        for pos in positions:
            has_sell_signal, _ = check_sell_signal(pos['symbol'])
            if has_sell_signal:
                sell_signal_positions.append(pos['symbol'].replace('USDT', ''))

        if sell_signal_positions:
            print(f"매도 신호 있는 종목: {', '.join(sell_signal_positions)}")
        else:
            print("매도 신호 있는 종목: 없음")

    else:
        print("보유 중인 포지션이 없습니다.")

    print("=" * 90)

    return {
        'total_assets': total_assets,
        'usdt_balance': usdt_balance,
        'positions_value': total_position_value,
        'positions_count': len(positions)
    }


def get_binance_ohlcv(symbol, interval, hours=100):
    """바이낸스에서 OHLCV 데이터 가져오기"""
    try:
        interval_map = {
            'hourly': Client.KLINE_INTERVAL_1HOUR
        }

        klines = client.get_klines(
            symbol=symbol,
            interval=interval_map[interval],
            limit=hours
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df = df[['open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    except Exception as e:
        logger.error("%s 데이터 가져오기 오류: %s", symbol, e)
        return None


def detect_inverse_head_shoulders(df):
    """역헤드앤숄더 패턴 감지"""
    patterns = []
    for i in range(4, len(df) - 4):
        window = df.iloc[i - 4:i + 5]

        ls = window.iloc[1]
        head = window.iloc[3]
        rs = window.iloc[5]
        neckline_start = window.iloc[2]
        neckline_end = window.iloc[4]

        cond1 = ls['high'] > head['high'] and rs['high'] > head['high']
        cond2 = head['low'] < ls['low'] and head['low'] < rs['low']
        cond3 = neckline_start['high'] < neckline_end['high']

        if cond1 and cond2 and cond3:
            for j in range(i + 1, min(i + 4, len(df))):
                if df.iloc[j]['close'] > neckline_end['high']:
                    patterns.append((i, j))
                    break

    return patterns


def detect_head_shoulders(df):
    """헤드앤숄더 패턴 감지"""
    patterns = []
    for i in range(4, len(df) - 4):
        window = df.iloc[i - 4:i + 5]

        ls = window.iloc[1]
        head = window.iloc[3]
        rs = window.iloc[5]
        neckline_start = window.iloc[2]
        neckline_end = window.iloc[4]

        cond1 = ls['low'] < head['low'] and rs['low'] < head['low']
        cond2 = head['high'] > ls['high'] and head['high'] > rs['high']
        cond3 = neckline_start['low'] > neckline_end['low']

        if cond1 and cond2 and cond3:
            for j in range(i + 1, min(i + 4, len(df))):
                if df.iloc[j]['close'] < neckline_end['low']:
                    patterns.append((i, j))
                    break

    return patterns


def can_buy_symbol(symbol):
    """해당 종목을 매수할 수 있는지 확인"""
    if symbol in last_buy_times:
        last_buy = last_buy_times[symbol]
        time_diff = datetime.now() - last_buy
        if time_diff.total_seconds() < COOLDOWN_HOURS * 3600:
            return False
    return True


def place_buy_order(symbol, usdt_amount):
    """매수 주문 실행"""
    try:
        # 주문 파라미터 확인 및 조정
        quantity, price, actual_amount = check_and_adjust_order_parameters(symbol, usdt_amount)

        if quantity is None or quantity <= 0:
            logger.info("%s: 유효한 주문 수량이 없습니다", symbol)
            return False

        # 주문 실행
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity
        )

        logger.info("매수 완료: %s, 수량: %.8f, 단가: %.2f, 금액: %.2f USDT",
                    symbol, quantity, price, actual_amount)

        last_buy_times[symbol] = datetime.now()

        trade_history.append({
            'type': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'amount': actual_amount,
            'time': datetime.now()
        })

        return True

    except BinanceOrderException as e:
        logger.error("매수 주문 오류 (%s): %s", symbol, e)
        return False
    except Exception as e:
        logger.error("매수 중 오류 (%s): %s", symbol, e)
        return False


def place_sell_order(symbol, quantity):
    """매도 주문 실행"""
    try:
        # LOT_SIZE 필터 적용
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, quantity)

        if adjusted_quantity <= 0:
            logger.info("%s: 조정된 수량이 0이어서 매도 불가", symbol)
            return False

        order = client.order_market_sell(
            symbol=symbol,
            quantity=adjusted_quantity
        )

        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        realized_amount = adjusted_quantity * current_price * (1 - FEE_RATIO)

        logger.info("매도 완료: %s, 수량: %.8f, 실현 금액: %.2f USDT",
                    symbol, adjusted_quantity, realized_amount)

        trade_history.append({
            'type': 'SELL',
            'symbol': symbol,
            'quantity': adjusted_quantity,
            'price': current_price,
            'amount': realized_amount,
            'time': datetime.now()
        })

        return True

    except BinanceOrderException as e:
        logger.error("매도 주문 오류 (%s): %s", symbol, e)
        return False
    except Exception as e:
        logger.error("매도 중 오류 (%s): %s", symbol, e)
        return False


def scan_trading_opportunities():
    """거래 기회 스캔"""
    try:
        exchange_info = client.get_exchange_info()
        usdt_symbols = []

        for symbol_info in exchange_info['symbols']:
            if symbol_info['quoteAsset'] == 'USDT' and symbol_info['status'] == 'TRADING':
                usdt_symbols.append(symbol_info['symbol'])

        logger.info("스캔 대상 종목 수: %d", len(usdt_symbols))

        buy_candidates = []

        for i, symbol in enumerate(usdt_symbols[:30]):
            try:
                if not can_buy_symbol(symbol):
                    continue

                df = get_binance_ohlcv(symbol, 'hourly', hours=100)
                if df is None or len(df) < 50:
                    continue

                inverse_patterns = detect_inverse_head_shoulders(df)

                if inverse_patterns:
                    latest_pattern = inverse_patterns[-1]
                    pattern_time = df.index[latest_pattern[1]]

                    if (datetime.now() - pattern_time).total_seconds() <= 24 * 3600:
                        current_price = df['close'].iloc[-1]
                        buy_candidates.append({
                            'symbol': symbol,
                            'pattern_time': pattern_time,
                            'current_price': current_price,
                            'pattern_strength': len(inverse_patterns)
                        })

                time.sleep(0.1)

            except Exception as e:
                logger.error("%s 분석 중 오류: %s", symbol, e)
                continue

        buy_candidates.sort(key=lambda x: x['pattern_strength'], reverse=True)
        return buy_candidates[:10]

    except Exception as e:
        logger.error("거래 기회 스캔 중 오류: %s", e)
        return []


def execute_trading():
    """실제 거래 실행"""
    logger.info("=" * 50)
    logger.info("거래 실행 시작")
    logger.info("=" * 50)

    usdt_balance = get_usdt_balance()
    current_positions = get_current_positions()

    logger.info("현재 USDT 잔고: %.2f", usdt_balance)
    logger.info("현재 보유 포지션: %d개", len(current_positions))

    # 매도 신호 확인 및 실행
    sold_positions = []
    for position in current_positions:
        symbol = position['symbol']
        has_sell_signal, signal_time = check_sell_signal(symbol)

        if has_sell_signal:
            logger.info("매도 신호 발견: %s (신호 시간: %s)", symbol, signal_time.strftime('%Y-%m-%d %H:%M'))
            if place_sell_order(symbol, position['quantity']):
                sold_positions.append(symbol)

    if sold_positions:
        logger.info("매도 완료된 종목: %s", ", ".join(sold_positions))

    # 매수 기회 스캔 및 실행
    bought_positions = []
    if len(current_positions) - len(sold_positions) < MAX_POSITIONS and usdt_balance > MIN_NOTIONAL:
        available_symbols = MAX_POSITIONS - (len(current_positions) - len(sold_positions))
        investment_per_symbol = max(usdt_balance * POSITION_PERCENT, MIN_NOTIONAL)

        logger.info("매수 가능 종목 수: %d", available_symbols)
        logger.info("종목당 투자 금액: %.2f USDT", investment_per_symbol)

        buy_candidates = scan_trading_opportunities()

        for candidate in buy_candidates[:available_symbols]:
            symbol = candidate['symbol']
            if place_buy_order(symbol, investment_per_symbol):
                bought_positions.append(symbol)
                time.sleep(1)

    if bought_positions:
        logger.info("매수 완료된 종목: %s", ", ".join(bought_positions))

    # 포트폴리오 현황 출력 (모든 보유 내역 포함)
    portfolio_summary = display_portfolio_summary()

    logger.info("=" * 50)
    logger.info("거래 실행 완료")
    logger.info("=" * 50)

    return portfolio_summary


def trading_job():
    """스케줄링된 거래 작업"""
    try:
        logger.info("스케줄된 거래 실행 시작")
        execute_trading()
        next_run = datetime.now() + timedelta(hours=INTERVAL_HOURS)
        logger.info("다음 실행 예정: %s", next_run.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        logger.error("스케줄된 거래 실행 중 오류: %s", e)


def main():
    """메인 실행 함수"""
    logger.info("바이낸스 자동매매 프로그램 시작")

    # 초기 포트폴리오 현황 출력
    print("\n프로그램 시작 - 초기 포트폴리오 현황")
    initial_summary = display_portfolio_summary()

    # 즉시 첫 거래 실행
    trading_job()

    # 4시간 간격으로 스케줄링
    schedule.every(INTERVAL_HOURS).hours.do(trading_job)

    logger.info("4시간 간격으로 자동 실행됩니다. Ctrl+C로 종료할 수 있습니다.")

    # 무한 루프로 스케줄링 실행
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("프로그램을 종료합니다.")
    except Exception as e:
        logger.error("메인 루프 오류: %s", e)


if __name__ == "__main__":
    main()