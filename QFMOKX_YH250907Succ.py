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
import argparse
import logging
import json
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC

warnings.filterwarnings("ignore")

# === 로깅 설정 ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# === 매매 기록 로거 클래스 ===
class TradeLogger:
    def __init__(self, filename="trade_history.csv"):
        self.filename = filename
        self.file_exists = os.path.isfile(self.filename)
        if not self.file_exists:
            self._create_header()

    def _create_header(self):
        with open(self.filename, "w", encoding='utf-8') as f:
            f.write("timestamp,symbol,type,side,amount,price,pnl,roi_percent\n")

    def log_trade(self, symbol, trade_type, side, amount, price, pnl=0.0, roi_percent=0.0):
        try:
            timestamp = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp},{symbol},{trade_type},{side},{amount},{price},{pnl},{roi_percent}\n"
            with open(self.filename, "a", encoding='utf-8') as f:
                f.write(log_entry)
            logger.info(f"💾 매매 기록 저장: {trade_type} {side} {amount} at {price}")
        except Exception as e:
            logger.error(f"❌ 매매 기록 저장 실패: {e}")


# === Gemini AI 자문 클래스 ===
class GeminiAdvisor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.critical("CRITICAL ERROR: GEMINI_API_KEY is not set.")
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        self.timeout = 50

    def get_advice(self, ohlcv_df: pd.DataFrame):
        logger.info("🤖 Gemini AI에게 시장 분석 및 전략 자문을 요청합니다...")
        try:
            df_for_ai = ohlcv_df[['ts', 'open', 'high', 'low', 'close', 'volume']].copy()
            df_for_ai['ts'] = pd.to_datetime(df_for_ai['ts'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
            data_str = df_for_ai.to_string(index=False)

            prompt = f"""
            You are an expert cryptocurrency futures trader. Based on the following 200 recent 15-minute candles for BTC/USDT futures, analyze the current market volatility and trend. Recommend the most suitable risk management strategy for short-term trading.

            After your analysis, please respond ONLY with a JSON object containing the following keys and appropriate numerical values. Do not add any other text or explanations.
            - SL_PARTIAL: Partial stop-loss percentage (negative float, e.g., -10.0)
            - SL_FULL: Full stop-loss percentage (negative float smaller than SL_PARTIAL, e.g., -20.0)
            - TP_START: Take-profit start percentage (positive float, e.g., 30.0)
            - TP_INCREMENT: Incremental take-profit percentage (positive float, e.g., 10.0)
            - TP_CLOSE_PERCENT: Partial close ratio (float between 0.0 and 1.0, e.g., 0.5)

            {{
              "SL_PARTIAL": -10.0,
              "SL_FULL": -20.0,
              "TP_START": 30.0,
              "TP_INCREMENT": 10.0,
              "TP_CLOSE_PERCENT": 0.5
            }}
            """

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "response_mime_type": "application/json"
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            response_json = response.json()
            advice_str = response_json['candidates'][0]['content']['parts'][0]['text']
            advice = json.loads(advice_str)

            logger.info(f"✅ Gemini AI 자문 수신 완료: {advice}")
            return advice

        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Gemini AI API 요청 실패: {e}. 기본 설정으로 계속합니다.")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"⚠️ Gemini AI 응답 파싱 실패: {e}. 기본 설정으로 계속합니다.")
            return None


# === 메인 트레이딩 봇 클래스 ===
class TradingBot:
    def __init__(self, method_name, symbol="BTC-USDT-SWAP"):
        # --- 기본 설정 ---
        self.symbol = symbol
        self.method_name = method_name
        self.trade_logger = TradeLogger()
        self.gemini_advisor = GeminiAdvisor()

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
        self.partial_close_record = {}

        ### 추가됨: AI 자문 요청 시간 제어를 위한 변수 ###
        self.last_ai_update_time = 0
        self.ai_update_interval = 3600  # AI 자문 간격: 3600초 (1시간)

        # --- 동적 설정 변수 (기본값) ---
        self.config = {
            'CONTRACT_AMOUNT': 0.01,
            'INTERVAL_NORMAL': 60,
            'INTERVAL_ACTIVE': 30,
            'SL_PARTIAL': -70.0,
            'SL_FULL': -90.0,
            'TP_START': 100.0,
            'TP_INCREMENT': 20.0,
            'TP_CLOSE_PERCENT': 0.5,
            'ATR_THRESHOLD_LOW': 20.0
        }
        self.default_config = self.config.copy()

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

    def update_config_from_ai(self):
        df_15m = self.fetch_ohlcv(timeframe="15m", limit=200)
        if df_15m is None:
            return

        advice = self.gemini_advisor.get_advice(df_15m)
        if advice:
            try:
                self.config['SL_PARTIAL'] = float(advice.get('SL_PARTIAL', self.default_config['SL_PARTIAL']))
                self.config['SL_FULL'] = float(advice.get('SL_FULL', self.default_config['SL_FULL']))
                self.config['TP_START'] = float(advice.get('TP_START', self.default_config['TP_START']))
                self.config['TP_INCREMENT'] = float(advice.get('TP_INCREMENT', self.default_config['TP_INCREMENT']))
                self.config['TP_CLOSE_PERCENT'] = float(
                    advice.get('TP_CLOSE_PERCENT', self.default_config['TP_CLOSE_PERCENT']))
                logger.info(f"⚙️ AI 자문을 통해 매매 설정을 업데이트했습니다.")
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ AI 응답 값 변환 실패: {e}. 기본 설정으로 복원합니다.")
                self.config = self.default_config.copy()
        else:
            self.config = self.default_config.copy()
            logger.info("⚙️ AI 자문 실패. 기본 매매 설정을 사용합니다.")

    def random_forest_predict(self, X_train, y_train, X_latest):
        logger.info("🌲 Random Forest 모델 학습 및 예측...")
        try:
            model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42,
                                           n_jobs=-1)
            model.fit(X_train, y_train)
            prediction = model.predict(X_latest)
            proba = model.predict_proba(X_latest)[0]
            logger.info(f"📊 예측 확률: {max(proba) * 100:.1f}%")
            return prediction[0]
        except Exception as e:
            logger.error(f"❌ Random Forest 예측 실패: {e}")
            return None

    def get_prediction(self, method, X_train, y_train, X_latest, df):
        if method == "random_forest":
            return self.random_forest_predict(X_train, y_train, X_latest)
        else:
            logger.warning("⚠️ 알 수 없는 예측 방법, Random Forest 사용")
            return self.random_forest_predict(X_train, y_train, X_latest)

    def get_position_status(self):
        try:
            balance_info = self.exchange.fetch_balance()
            self.usdt_balance = balance_info.get('USDT', {}).get('total', 0.0)
            positions = self.exchange.fetch_positions(symbols=[self.symbol])
            pos = next((p for p in positions if float(p.get('contracts', 0)) != 0), None)

            if not pos:
                self.position = None
                return

            self.position = {
                "symbol": self.symbol,
                "side": pos.get('side', 'N/A').upper(),
                "size": float(pos.get('contracts', 0)),
                "entry_price": float(pos.get('entryPrice', 0)),
                "pnl": float(pos.get('unrealizedPnl', 0)),
                "roi": float(pos.get('percentage', 0))
            }
        except Exception as e:
            logger.error(f"❌ 포지션 정보 조회 실패: {e}")
            self.position = None

    def place_order(self, signal, amount, mode="isolated"):
        try:
            side = 'buy' if signal == 1 else 'sell'
            logger.info(f"🚀 신규 주문: {side.upper()} / {amount} 계약")

            ### 수정됨: 'posSide' 파라미터 제거 ###
            order = self.exchange.create_order(
                symbol=self.symbol, type='market', side=side, amount=amount,
                params={'tdMode': mode}
            )
            logger.info(f"✅ 주문 성공: {order.get('id', 'N/A')}")
            time.sleep(2)
            self.get_position_status()
            if self.position:
                self.trade_logger.log_trade(self.symbol, "ENTRY", self.position['side'], self.position['size'],
                                            self.position['entry_price'])
            return order
        except Exception as e:
            logger.error(f"❌ 주문 실행 실패: {e}")
            return None

    def close_position(self, close_amount, reason, mode="isolated"):
        if not self.position: return False
        try:
            side = "sell" if self.position["side"] == "LONG" else "buy"
            logger.info(f"🔐 포지션 청산 ({reason}): {self.position['side']} / {close_amount} 계약")

            original_position = self.position.copy()

            ### 수정됨: 'posSide' 파라미터 제거 ###
            order = self.exchange.create_order(
                symbol=self.symbol, type='market', side=side, amount=close_amount,
                params={'tdMode': mode}
            )
            logger.info(f"✅ 청산 주문 성공: {order.get('id', 'N/A')}")

            close_price = float(order.get('average', original_position['entry_price']))
            pnl_estimate = (close_price - original_position['entry_price']) * close_amount if original_position[
                                                                                                  'side'] == 'LONG' else (
                                                                                                                                     original_position[
                                                                                                                                         'entry_price'] - close_price) * close_amount

            self.trade_logger.log_trade(
                self.symbol, f"EXIT_{reason}", original_position['side'],
                close_amount, close_price, pnl=pnl_estimate,
                roi_percent=original_position['roi']
            )

            if reason.startswith("FULL"):
                if self.symbol in self.partial_close_record:
                    del self.partial_close_record[self.symbol]
            else:
                if self.symbol not in self.partial_close_record:
                    self.partial_close_record[self.symbol] = []
                self.partial_close_record[self.symbol].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "amount": close_amount,
                    "roi": original_position["roi"]
                })
            return True
        except Exception as e:
            logger.error(f"❌ 청산 실패: {e}")
            return False

    def manage_risk(self, signal):
        if not self.position: return False

        roi = self.position['roi']
        position_direction_matched = (self.position['side'] == 'LONG' and signal == 1) or \
                                     (self.position['side'] == 'SHORT' and signal == 0)

        # 1. 손절 관리
        if roi < 0:
            if roi <= self.config['SL_FULL']:
                logger.info(f"🚨 전체 손절: {roi:.2f}% ≤ {self.config['SL_FULL']}%")
                return self.close_position(self.position['size'], "FULL_SL", "cross")

            if not position_direction_matched and roi <= self.config[
                'SL_PARTIAL'] and self.symbol not in self.partial_close_record:
                logger.info(f"⚠️ 부분 손절: {roi:.2f}% ≤ {self.config['SL_PARTIAL']}%")
                return self.close_position(self.position['size'] * self.config['TP_CLOSE_PERCENT'], "PARTIAL_SL")

        # 2. 익절 관리
        if roi > self.config['TP_START']:
            threshold_level = int((roi - self.config['TP_START']) // self.config['TP_INCREMENT'])
            current_threshold = self.config['TP_START'] + (threshold_level * self.config['TP_INCREMENT'])

            partial_records = self.partial_close_record.get(self.symbol, [])
            threshold_executed = any(
                record["roi"] >= current_threshold and record["roi"] < current_threshold + self.config['TP_INCREMENT']
                for record in partial_records
            )

            if not threshold_executed:
                logger.info(f"💰 부분 익절: {roi:.2f}% ≥ {current_threshold}%")
                return self.close_position(self.position['size'] * self.config['TP_CLOSE_PERCENT'],
                                           f"PARTIAL_TP_L{threshold_level + 1}")
        return False

    def print_status(self, next_run_in, signal, method_name):
        now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
        pos_side = self.position['side'] if self.position else '없음'
        pos_size = f"{self.position['size']}" if self.position else '0.0'
        pos_entry = f"{self.position['entry_price']:.2f}" if self.position else '-'
        pos_pnl = f"{self.position['pnl']:.2f} USDT" if self.position else '-'
        pos_roi = f"{self.position['roi']:.2f} %" if self.position else '-'
        signal_info = f"{'상승(BUY)' if signal == 1 else '하락(SELL)'}" if signal is not None else '대기'

        status = f"""
*** AI 트레이딩 봇 상태 ({method_name}): {now} ***
================================================================================
  계정 잔고: {self.usdt_balance:.2f} USDT
--------------------------------------------------------------------------------
  포지션 정보:
    - 방향: {pos_side:<8} | 크기: {pos_size:<8} | 진입가: {pos_entry}
    - 미실현 PNL: {pos_pnl:<15} | ROI: {pos_roi}
--------------------------------------------------------------------------------
  AI 기반 설정 (다음 업데이트까지: {max(0, int(self.ai_update_interval - (time.time() - self.last_ai_update_time)))}초):
    - 부분손절: {self.config['SL_PARTIAL']}% | 전체손절: {self.config['SL_FULL']}%
    - 부분익절: {self.config['TP_START']}% 부터 {self.config['TP_INCREMENT']}% 마다 {self.config['TP_CLOSE_PERCENT'] * 100}%씩
--------------------------------------------------------------------------------
  예측 신호:
    - 신호: {signal_info:<10} | 방법: {method_name}
================================================================================
  다음 실행까지: {next_run_in}초
"""
        print(status)

    def run(self):
        error_count = 0
        max_errors = 5
        while True:
            try:
                start_time = time.time()

                if start_time - self.last_ai_update_time > self.ai_update_interval:
                    self.update_config_from_ai()
                    self.last_ai_update_time = start_time

                df = self.fetch_ohlcv()
                if df is None: raise Exception("데이터 가져오기 실패")

                features = ["return", "ma5", "ma20", "rsi", "macd", "macd_signal", "atr"]
                X = df[features].values[:-1]
                y = (df["close"].diff().shift(-1).values[:-1] > 0).astype(int)
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                latest_data = scaler.transform([df[features].values[-1]])
                signal = self.get_prediction(self.method_name, X_train, y_train, latest_data, df)
                if signal is None: raise Exception("예측 신호 생성 실패")

                self.get_position_status()

                if self.position:
                    closed = self.manage_risk(signal)
                    if closed:
                        time.sleep(2)
                        self.get_position_status()

                current_atr = df['atr'].iloc[-1]
                logger.info(f"📊 현재 ATR: {current_atr:.2f} (횡보 기준: < {self.config['ATR_THRESHOLD_LOW']})")

                if not self.position:
                    if current_atr >= self.config['ATR_THRESHOLD_LOW']:
                        self.place_order(signal, self.config['CONTRACT_AMOUNT'], "isolated")
                    else:
                        logger.info("횡보 구간으로 판단되어 신규 주문을 실행하지 않습니다.")

                monitoring_interval = self.config['INTERVAL_NORMAL'] if current_atr < self.config[
                    'ATR_THRESHOLD_LOW'] * 1.5 else self.config['INTERVAL_ACTIVE']

                elapsed_time = time.time() - start_time
                time_to_wait = max(0, monitoring_interval - elapsed_time)
                self.print_status(int(time_to_wait), signal, self.method_name)

                error_count = 0
                time.sleep(time_to_wait)

            except Exception as e:
                error_count += 1
                logger.error(f"🔥 메인 루프 오류 발생 (카운트: {error_count}/{max_errors}): {e}")
                traceback.print_exc()

                if error_count >= max_errors:
                    logger.critical("❌ 연속된 오류로 인해 프로그램을 종료합니다.")
                    break

                time.sleep(self.config['INTERVAL_NORMAL'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI 기반 양자 트레이딩 봇')
    parser.add_argument('--method', type=str, default="random_forest", choices=["random_forest"],
                        help='사용할 예측 방법 선택 (현재 random_forest 지원)')
    args = parser.parse_args()

    bot = TradingBot(method_name=args.method)
    bot.run()