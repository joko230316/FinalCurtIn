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

# === 로깅 설정 (FIXED) ===
# 정의되지 않은 'name' 변수 대신 '__name__'을 사용하여 로거를 올바르게 설정합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingBot:
    # --- 클래스 생성자 (FIXED) ---
    # 파이썬 클래스 생성자는 'init'이 아닌 '__init__'이어야 합니다.
    def __init__(self, symbol="BTC-USDT-SWAP"):
        """
        봇 초기화
        """
        # --- 기본 설정 ---
        self.symbol = symbol

        # --- API 인증 ---
        self.api_key = os.getenv("OKXYH_API_KEY")
        self.api_secret = os.getenv("OKXYH_API_SECRET")
        self.api_passphrase = os.getenv("OKXYH_API_PASSPHRASE")
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            logger.critical("CRITICAL ERROR: OKX API environment variables are not set.")
            raise ValueError("OKX API 환경변수가 설정되지 않았습니다.")

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
        self.TRAILING_STOP_PERCENT = 3.0  # 트레일링 스탑 콜백 (%)
        self.STOP_LOSS_PERCENT = -15.0  # 손절 기준 수익률 (%)
        self.PROFIT_TAKE_PERCENT = 30.0  # 익절 기준 수익률 (%)

        # --- 실행 주기 설정 ---
        self.DEFAULT_INTERVAL = 60  # 기본 실행 주기 (60초)
        self.FAST_INTERVAL = 10  # 빠른 실행 주기 (10초, 조건 근접 시)

    def fetch_ohlcv(self, timeframe="1h", limit=100):
        """
        OHLCV 데이터를 가져오고 기술적 지표를 계산합니다.
        """
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
        """
        현재 계정의 포지션 상태와 잔고를 조회하고 업데이트합니다.
        """
        try:
            balance_info = self.exchange.fetch_balance()
            self.usdt_balance = balance_info.get('USDT', {}).get('total', 0.0)
            positions = self.exchange.fetch_positions(symbols=[self.symbol])
            pos = next((p for p in positions if float(p.get('contracts', 0)) != 0), None)

            if not pos:
                self.position = None
                self.best_price_since_entry = None
                return

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
        """
        신규 주문을 실행합니다.
        """
        try:
            logger.info(f"🚀 신규 주문: {side.upper()} / {amount} 계약")
            order = self.exchange.create_order(
                symbol=self.symbol, type='market', side=side, amount=amount,
                params={'tdMode': mode}
            )
            logger.info(f"✅ 주문 성공: {order.get('id', 'N/A')}")
            time.sleep(2)  # 주문 체결 대기
            self.get_position_status()
            return order
        except Exception as e:
            logger.error(f"❌ 주문 실행 실패: {e}")
            return None

    def close_position(self, close_amount, reason, mode="cross"):
        """
        현재 포지션을 청산합니다.
        """
        if not self.position: return False
        try:
            side = "sell" if self.position["side"] == "LONG" else "buy"
            logger.info(f"🔐 포지션 청산 ({reason}): {self.position['side']} / {close_amount} 계약")
            order = self.exchange.create_order(
                symbol=self.symbol, type='market', side=side, amount=close_amount,
                params={'tdMode': mode}
            )
            logger.info(f"✅ 청산 주문 성공: {order.get('id', 'N/A')}")
            self.position = None
            self.best_price_since_entry = None
            return True
        except Exception as e:
            logger.error(f"❌ 청산 실패: {e}")
            return False

    def check_trailing_stop(self):
        """
        트레일링 스탑 조건을 확인하고 충족 시 포지션을 청산합니다.
        """
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

    def check_stop_loss(self):
        """
        손절 조건을 확인하고 충족 시 포지션을 청산합니다.
        """
        if not self.position:
            return False

        if self.position.get('roi', 0.0) <= self.STOP_LOSS_PERCENT:
            logger.info(
                f"💔 손절 조건 충족: 현재 수익률 {self.position.get('roi', 0.0):.2f}% ≤ 손절 기준 {self.STOP_LOSS_PERCENT}%"
            )
            return self.close_position(self.position['size'], "STOP_LOSS")
        return False

    def check_profit_take(self):
        """
        익절 조건을 확인하고 충족 시 포지션을 청산합니다.
        """
        if not self.position:
            return False

        if self.position.get('roi', 0.0) >= self.PROFIT_TAKE_PERCENT:
            logger.info(
                f"💰 익절 조건 충족: 현재 수익률 {self.position.get('roi', 0.0):.2f}% ≥ 목표 수익률 {self.PROFIT_TAKE_PERCENT}%"
            )
            return self.close_position(self.position['size'], "PROFIT_TAKE")
        return False

    def print_status(self, next_run_in):
        """
        현재 봇의 상태를 콘솔에 출력합니다.
        """
        now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")

        # --- 안정성 향상 (IMPROVED) ---
        # self.position이 None일 경우를 대비해 .get()을 사용하여 안전하게 값에 접근합니다.
        pos_side = self.position.get('side', '없음') if self.position else '없음'
        pos_size = f"{self.position.get('size', 0.0)}" if self.position else '0.0'
        pos_entry = f"{self.position.get('entry_price', 0.0):.2f}" if self.position else '-'
        pos_pnl = f"{self.position.get('pnl', 0.0):.2f} USDT" if self.position else '-'
        pos_roi = f"{self.position.get('roi', 0.0):.2f} %" if self.position else '-'

        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker.get('last', 0.0)

        trailing_info = "-"
        if self.position and self.best_price_since_entry:
            if self.position.get('side') == 'LONG':
                stop_price = self.best_price_since_entry * (1 - self.TRAILING_STOP_PERCENT / 100)
                trailing_info = f"최고가: {self.best_price_since_entry:.2f} | 스탑가: {stop_price:.2f}"
            else:
                stop_price = self.best_price_since_entry * (1 + self.TRAILING_STOP_PERCENT / 100)
                trailing_info = f"최저가: {self.best_price_since_entry:.2f} | 스탑가: {stop_price:.2f}"

        # 빠른 모니터링 조건 확인
        fast_monitoring = False
        if self.position:
            current_roi = self.position.get('roi', 0.0)
            if (current_roi >= self.PROFIT_TAKE_PERCENT * 0.8 or
                    current_roi <= self.STOP_LOSS_PERCENT * 0.8):
                fast_monitoring = True

        status = f"""
================================================================================
** 🤖 트레이딩 봇 상태: {now} **
--------------------------------------------------------------------------------
- 계정 잔고: {self.usdt_balance:.2f} USDT
- 현재 가격: {current_price:.2f} USDT
- 포지션: {pos_side:<5} | 크기: {pos_size:<8} | 진입가: {pos_entry}
- 미실현 PNL: {pos_pnl:<15} | 수익률(ROI): {pos_roi}
- 트레일링 스탑 ({self.TRAILING_STOP_PERCENT}%): {trailing_info}
- 손절 목표 ({self.STOP_LOSS_PERCENT}%): {'빠른 모니터링 중' if fast_monitoring else '대기'}
- 익절 목표 ({self.PROFIT_TAKE_PERCENT}%): {'빠른 모니터링 중' if fast_monitoring else '대기'}
--------------------------------------------------------------------------------
다음 확인까지: {next_run_in}초
================================================================================
        """
        print(status)

    def run(self):
        """
        봇의 메인 실행 루프
        """
        error_count = 0
        max_errors = 5

        while True:
            try:
                start_time = time.time()

                # 1. 포지션 상태 최신화
                self.get_position_status()

                # 2. 포지션이 있는 경우 청산 조건 확인
                if self.position:
                    # 우선순위: 손절 > 익절 > 트레일링 스탑
                    if self.check_stop_loss() or self.check_profit_take() or self.check_trailing_stop():
                        time.sleep(2)  # 청산 후 상태 업데이트를 위해 잠시 대기
                        self.get_position_status()  # 상태 즉시 갱신

                # 3. 포지션이 없는 경우: 진입 로직 (현재 코드에는 없음, 필요시 추가)
                # else:
                #     # df = self.fetch_ohlcv("1h", 100)
                #     # if self.check_entry_conditions(df):
                #     #     self.place_order(...)

                # 4. 동적 실행 주기 설정
                current_interval = self.DEFAULT_INTERVAL
                if self.position:
                    current_roi = self.position.get('roi', 0.0)
                    # 익절 또는 손절 목표에 근접한 경우 빠른 모니터링
                    if (current_roi >= self.PROFIT_TAKE_PERCENT * 0.8 or
                            current_roi <= self.STOP_LOSS_PERCENT * 0.8):
                        current_interval = self.FAST_INTERVAL
                        logger.info(f"🎯 조건 근접! 모니터링 주기를 {self.FAST_INTERVAL}초로 변경합니다.")

                # 5. 상태 출력 및 대기
                elapsed_time = time.time() - start_time
                time_to_wait = max(0, current_interval - elapsed_time)
                self.print_status(int(time_to_wait))

                error_count = 0  # 성공적으로 루프를 마치면 에러 카운트 초기화
                time.sleep(time_to_wait)

            except Exception as e:
                error_count += 1
                logger.error(f"🔥 메인 루프 오류 발생 (카운트: {error_count}/{max_errors}): {e}")
                traceback.print_exc()

                if error_count >= max_errors:
                    logger.critical("❌ 연속된 오류로 인해 프로그램을 종료합니다.")
                    break

                time.sleep(self.DEFAULT_INTERVAL)


# --- 프로그램 실행 블록 (FIXED) ---
# 파이썬 파일이 직접 실행될 때만 아래 코드가 동작하도록 하는 표준 방식입니다.
if __name__ == "__main__":
    bot = TradingBot(symbol="BTC-USDT-SWAP")
    bot.run()