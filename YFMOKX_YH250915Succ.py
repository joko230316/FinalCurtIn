# -*- coding: utf-8 -*-

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import ccxt
import pytz
import warnings
import traceback
import talib
import logging

warnings.filterwarnings("ignore")

# === 로깅 설정 ===
# __name__을 사용하여 현재 모듈의 로거를 가져옵니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingBot:
    # 클래스 생성자 이름을 'init'에서 '__init__'으로 수정했습니다.
    def __init__(self, symbol="BTC-USDT-SWAP"):
        # --- 기본 설정 ---
        self.symbol = symbol

        # --- API 인증 ---
        self.api_key = os.getenv("OKXYH_API_KEY")  # 환경 변수 이름을 명확하게 변경 (권장)
        self.api_secret = os.getenv("OKXYH_API_SECRET")
        self.api_passphrase = os.getenv("OKXYH_API_PASSPHRASE")
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            logger.critical("CRITICAL ERROR: OKX API environment variables are not set.")
            raise ValueError("OKX API 환경변수가 설정되지 않았습니다. (예: OKX_API_KEY)")

        self.exchange = ccxt.okx({
            'apiKey': self.api_key, 'secret': self.api_secret, 'password': self.api_passphrase,
            'enableRateLimit': True, 'options': {'defaultType': 'swap'}
        })
        logger.info("✅ OKX 실거래 모드가 활성화되었습니다.")

        # --- 상태 변수 ---
        self.position = None
        self.usdt_balance = 0.0
        self.best_price_since_entry = None  # 진입 후 최고/최저 가격 추적 (트레일링 스탑용)

        # --- 매매 전략 설정 ---
        self.TRAILING_STOP_PERCENT = 1.5  # 트레일링 스탑 콜백 (%)
        self.PROFIT_TAKE_PERCENT = 15.0  # 익절 기준 수익률 (%)

        # --- 실행 주기 설정 ---
        self.DEFAULT_INTERVAL = 120  # 기본 실행 주기 (60초)
        self.FAST_INTERVAL = 20  # 빠른 실행 주기 (10초, 익절 조건 근접 시)

    def fetch_ohlcv(self, timeframe="1h", limit=100):
        logger.info(f"📈 {self.symbol}의 {timeframe} 캔들 데이터(최근 {limit}개)를 가져옵니다...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["return"] = df["close"].pct_change()
            df["ma5"] = df["close"].rolling(5).mean()
            df["ma20"] = df["close"].rolling(20).mean()
            df["rsi"] = talib.RSI(df["close"], timeperiod=14)
            df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
            df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(df["close"], timeperiod=20)
            df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df = df.dropna().reset_index(drop=True)
            logger.info("...데이터 가져오기 및 가공 완료.")
            return df
        except Exception as e:
            logger.error(f"❌ OHLCV 데이터 가져오기 실패: {e}")
            return None

    def get_position_status(self):
        try:
            balance_info = self.exchange.fetch_balance()
            self.usdt_balance = balance_info.get('USDT', {}).get('total', 0.0)
            positions = self.exchange.fetch_positions(symbols=[self.symbol])
            # 계약(contracts) 수가 0이 아닌 포지션을 찾습니다.
            pos = next((p for p in positions if float(p.get('contracts', 0)) != 0), None)

            if not pos:
                self.position = None
                self.best_price_since_entry = None  # 포지션이 없으면 트레일링 스탑 리셋
                return

            # 'info' 딕셔너리에서 'uplRatio' (미실현 손익률) 값을 가져옵니다.
            # ccxt 라이브러리 버전에 따라 필드명이 다를 수 있으므로 'percentage'도 확인합니다.
            roi_ratio = float(pos.get('info', {}).get('uplRatio', 0))
            roi_percent = roi_ratio * 100

            self.position = {
                "symbol": self.symbol,
                "side": pos.get('side', 'N/A').upper(),
                "size": float(pos.get('contracts', 0)),
                "entry_price": float(pos.get('entryPrice', 0)),
                "pnl": float(pos.get('unrealizedPnl', 0)),
                "roi": roi_percent  # % 단위로 저장
            }

            # 현재 가격 가져오기
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            # 트레일링 스탑: 진입 후 최고/최저 가격 업데이트
            if self.best_price_since_entry is None:
                self.best_price_since_entry = current_price
            else:
                if (self.position['side'] == 'LONG' and current_price > self.best_price_since_entry) or \
                        (self.position['side'] == 'SHORT' and current_price < self.best_price_since_entry):
                    self.best_price_since_entry = current_price

        except Exception as e:
            logger.error(f"❌ 포지션 정보 조회 실패: {e}")
            self.position = None
            self.best_price_since_entry = None

    def place_order(self, side, amount, mode="cross"):
        try:
            logger.info(f"🚀 신규 주문: {side.upper()} / {amount} 계약")
            order = self.exchange.create_order(
                symbol=self.symbol, type='market', side=side, amount=amount,
                params={'tdMode': mode}
            )
            logger.info(f"✅ 주문 성공: {order.get('id', 'N/A')}")
            time.sleep(2)  # 주문 처리 시간 대기
            self.get_position_status()
            return order
        except Exception as e:
            logger.error(f"❌ 주문 실행 실패: {e}")
            return None

    def close_position(self, close_amount, reason, mode="cross"):
        if not self.position: return False
        try:
            side = "sell" if self.position["side"] == "LONG" else "buy"
            logger.info(f"🔐 포지션 청산 ({reason}): {self.position['side']} / {close_amount} 계약")
            order = self.exchange.create_order(
                symbol=self.symbol, type='market', side=side, amount=close_amount,
                params={'tdMode': mode}
            )
            logger.info(f"✅ 청산 주문 성공: {order.get('id', 'N/A')}")
            # 상태 변수 즉시 초기화
            self.position = None
            self.best_price_since_entry = None
            return True
        except Exception as e:
            logger.error(f"❌ 청산 실패: {e}")
            return False

    def check_trailing_stop(self):
        if not self.position or self.best_price_since_entry is None:
            return False

        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker['last']

        if self.position['side'] == 'LONG':
            stop_price = self.best_price_since_entry * (1 - self.TRAILING_STOP_PERCENT / 100)
            if current_price <= stop_price:
                logger.info(
                    f"🔻 트레일링 스탑 발동 (LONG): 현재가 {current_price:.2f} ≤ 스탑가 {stop_price:.2f} (최고가: {self.best_price_since_entry:.2f})")
                return self.close_position(self.position['size'], "TRAILING_STOP")
        elif self.position['side'] == 'SHORT':
            stop_price = self.best_price_since_entry * (1 + self.TRAILING_STOP_PERCENT / 100)
            if current_price >= stop_price:
                logger.info(
                    f"🔺 트레일링 스탑 발동 (SHORT): 현재가 {current_price:.2f} ≥ 스탑가 {stop_price:.2f} (최저가: {self.best_price_since_entry:.2f})")
                return self.close_position(self.position['size'], "TRAILING_STOP")
        return False

    def check_profit_take(self):
        """수익률(ROI)이 목표치에 도달했는지 확인하고 포지션을 종료하는 함수"""
        if not self.position:
            return False

        # self.position['roi']는 % 단위입니다.
        if self.position['roi'] >= self.PROFIT_TAKE_PERCENT:
            logger.info(
                f"💰 익절 조건 충족: 현재 수익률 {self.position['roi']:.2f}% ≥ 목표 수익률 {self.PROFIT_TAKE_PERCENT}%"
            )
            return self.close_position(self.position['size'], "PROFIT_TAKE")
        return False

    def print_status(self, next_run_in):
        now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
        pos_side = self.position['side'] if self.position else '없음'
        pos_size = f"{self.position['size']}" if self.position else '0.0'
        pos_entry = f"{self.position['entry_price']:.2f}" if self.position else '-'
        pos_pnl = f"{self.position['pnl']:.2f} USDT" if self.position else '-'
        pos_roi = f"{self.position['roi']:.2f} %" if self.position else '-'

        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker['last']

        trailing_info = "-"
        if self.position and self.best_price_since_entry:
            if self.position['side'] == 'LONG':
                stop_price = self.best_price_since_entry * (1 - self.TRAILING_STOP_PERCENT / 100)
                trailing_info = f"최고가: {self.best_price_since_entry:.2f} | 스탑가: {stop_price:.2f}"
            else:
                stop_price = self.best_price_since_entry * (1 + self.TRAILING_STOP_PERCENT / 100)
                trailing_info = f"최저가: {self.best_price_since_entry:.2f} | 스탑가: {stop_price:.2f}"

        status = f"""
        ================================================================================
        ** 🤖 트레이딩 봇 상태: {now} **
        --------------------------------------------------------------------------------
        - 계정 잔고: {self.usdt_balance:.2f} USDT
        - 현재 가격: {current_price:.2f} USDT
        - 포지션: {pos_side:<5} | 크기: {pos_size:<8} | 진입가: {pos_entry}
        - 미실현 PNL: {pos_pnl:<15} | 수익률(ROI): {pos_roi}
        - 트레일링 스탑 ({self.TRAILING_STOP_PERCENT}%): {trailing_info}
        - 익절 목표 ({self.PROFIT_TAKE_PERCENT}%): {'빠른 모니터링 중' if next_run_in < self.DEFAULT_INTERVAL else '대기'}
        --------------------------------------------------------------------------------
        다음 확인까지: {next_run_in}초
        ================================================================================
        """
        print(status)

    def run(self):
        error_count = 0
        max_errors = 5

        while True:
            try:
                start_time = time.time()

                # 1. 포지션 상태 최신화
                self.get_position_status()

                # 2. 포지션이 있는 경우 청산 조건 확인
                if self.position:
                    # 2-1. 익절 조건 확인
                    if self.check_profit_take():
                        time.sleep(2)  # 청산 후 상태 반영 대기
                        self.get_position_status()  # 포지션 상태 즉시 갱신

                    # 2-2. 트레일링 스탑 확인 (익절이 안된 경우에만)
                    elif self.check_trailing_stop():
                        time.sleep(2)
                        self.get_position_status()  # 포지션 상태 즉시 갱신

                # 3. 동적 실행 주기 설정
                current_interval = self.DEFAULT_INTERVAL
                if self.position and self.position['roi'] >= self.PROFIT_TAKE_PERCENT:
                    current_interval = self.FAST_INTERVAL
                    logger.info(f"🎯 익절 목표 근접! 모니터링 주기를 {self.FAST_INTERVAL}초로 변경합니다.")

                # 4. 상태 출력 및 대기
                elapsed_time = time.time() - start_time
                time_to_wait = max(0, current_interval - elapsed_time)
                self.print_status(int(time_to_wait))

                error_count = 0  # 성공 시 에러 카운트 초기화
                time.sleep(time_to_wait)

            except Exception as e:
                error_count += 1
                logger.error(f"🔥 메인 루프 오류 발생 (카운트: {error_count}/{max_errors}): {e}")
                traceback.print_exc()

                if error_count >= max_errors:
                    logger.critical("❌ 연속된 오류로 인해 프로그램을 종료합니다.")
                    break

                time.sleep(self.DEFAULT_INTERVAL)


# 이 블록은 클래스 외부에 최상위 레벨로 위치해야 합니다.
if __name__ == "__main__":
    bot = TradingBot(symbol="BTC-USDT-SWAP")
    bot.run()