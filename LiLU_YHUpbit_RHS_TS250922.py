import sys
import os
import requests
import json
import time
import hmac
import hashlib
import uuid
import jwt
import threading
import schedule
import csv
import pandas as pd
from datetime import datetime, timedelta

# ------------------- 설정 영역 ------------------- #
RUN_MODE = 'linux'  #windows
INVESTMENT_DIVISION = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
# ----------------------------------------------- #

if RUN_MODE == 'linux':
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QLabel, QPushButton, QVBoxLayout, QWidget, \
    QHBoxLayout, QSpinBox, QDoubleSpinBox, QTextEdit
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont

access_key = os.getenv("upbit_YHaccess_key", "your_access_key_here")
secret_key = os.getenv("upbit_YHsecret_key", "your_secret_key_here")

# 전역 변수 설정
MAX_POSITIONS = 5
REBUY_COOLDOWN_HOURS = 240
TRAILING_STOP_PERCENT = 5
SCAN_INTERVAL_HOURS = 24
DAILY_START_TIME = "05:13"
MONITORING_INTERVAL_MINUTES = 5

# 상태 관리 변수
purchase_history = {}
active_positions = {}
pending_buy_list = {}  # 감시 목록: { "KRW-BTC": "discovered_time", ... }

# 디렉토리 설정
ORDER_HISTORY_DIR = "order_history"
SCANNED_PATTERNS_DIR = "scanned_patterns"

for dir_path in [ORDER_HISTORY_DIR, SCANNED_PATTERNS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class PatternRecognizer:
    """RHS 패턴(역헤드앤숄더) 분석 클래스"""

    def __init__(self):
        self.ohlcv_data = {}

    def add_data(self, code, data_list):
        self.ohlcv_data[code] = data_list

    def detect_inverse_head_shoulders(self, code, window=60):
        if code not in self.ohlcv_data or len(self.ohlcv_data[code]) < window:
            return False
        recent_data = self.ohlcv_data[code][-window:]
        lows, highs = [], []
        for i in range(2, len(recent_data) - 2):
            if recent_data[i]['low'] < min(p['low'] for p in recent_data[i - 2:i] + recent_data[i + 1:i + 3]):
                lows.append((i, recent_data[i]['low']))
            if recent_data[i]['high'] > max(p['high'] for p in recent_data[i - 2:i] + recent_data[i + 1:i + 3]):
                highs.append((i, recent_data[i]['high']))

        if len(lows) < 3 or len(highs) < 2: return False
        for i in range(len(lows) - 2):
            ls_idx, ls_val = lows[i]
            h_idx, h_val = lows[i + 1]
            rs_idx, rs_val = lows[i + 2]
            if not (ls_idx < h_idx < rs_idx and h_val < ls_val and h_val < rs_val): continue
            p1 = [h for idx, h in highs if ls_idx < idx < h_idx]
            p2 = [h for idx, h in highs if h_idx < idx < rs_idx]
            if p1 and p2 and recent_data[-1]['close'] > max(max(p1), max(p2)):
                return True
        return False


class MACDAnalyzer:
    """MACD 분석 클래스"""

    def __init__(self, upbit_api):
        self.upbit_api = upbit_api

    def check_macd_increase(self, market):
        """1분봉 기준 MACD 값 증가 여부 확인"""
        try:
            ohlcv_data = self.upbit_api.get_ohlcv(market, count=100, timeframe="minutes/1")
            if not ohlcv_data or len(ohlcv_data) < MACD_SLOW: return False, 0, 0

            closes = pd.Series([data['close'] for data in ohlcv_data])
            macd = closes.ewm(span=MACD_FAST, adjust=False).mean() - closes.ewm(span=MACD_SLOW, adjust=False).mean()

            if len(macd) < 2: return False, 0, 0
            return macd.iloc[-1] > macd.iloc[-2], macd.iloc[-2], macd.iloc[-1]
        except Exception as e:
            print(f"MACD 분석 오류 ({market}): {e}")
            return False, 0, 0


class UpbitAPI:
    """Upbit API 래퍼 클래스"""

    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key
        self.server_url = "https://api.upbit.com/v1"

    def get_headers(self, query=None):
        payload = {"access_key": self.access_key, "nonce": str(uuid.uuid4())}
        if query:
            m = hashlib.sha512(query.encode())
            query_hash = m.hexdigest()
            payload.update({"query_hash": query_hash, "query_hash_alg": "SHA512"})
        return {"Authorization": f"Bearer {jwt.encode(payload, self.secret_key)}"}

    def _request(self, method, url, params=None, is_protected=False):
        try:
            headers = self.get_headers(
                params and "&".join([f"{k}={v}" for k, v in params.items()])) if is_protected else {}
            res = requests.request(method, url, params=params, headers=headers)
            res.raise_for_status()
            return res.json()
        except requests.RequestException as e:
            print(f"API 요청 실패: {e}")
            return None

    def get_market_all(self):
        data = self._request("GET", f"{self.server_url}/market/all")
        return [m for m in data if m['market'].startswith('KRW-')] if data else []

    def get_ohlcv(self, market, count=250, timeframe="days"):
        data = self._request("GET", f"{self.server_url}/candles/{timeframe}", params={"market": market, "count": count})
        return data and sorted([{'date': d['candle_date_time_kst'], 'open': d['opening_price'], 'high': d['high_price'],
                                 'low': d['low_price'], 'close': d['trade_price'],
                                 'volume': d['candle_acc_trade_volume']}
                                for d in data], key=lambda x: x['date'])

    def get_current_price(self, market):
        data = self._request("GET", f"{self.server_url}/ticker", params={"markets": market})
        if data:
            d = data[0]
            return {'code': market, 'name': market.replace('KRW-', ''), 'current_price': d['trade_price'],
                    'diff_rate': d['signed_change_rate'] * 100}
        return None

    def get_balance(self, currency="KRW"):
        accounts = self._request("GET", f"{self.server_url}/accounts", is_protected=True)
        if accounts:
            for acc in accounts:
                if acc['currency'] == currency:
                    return int(float(acc['balance']))  # KRW 잔고는 정수로 반환
        return 0

    def _execute_order(self, query):
        try:
            query_string = "&".join([f"{k}={v}" for k, v in query.items()])
            headers = self.get_headers(query_string)
            res = requests.post(f"{self.server_url}/orders", params=query, headers=headers)
            res.raise_for_status()
            order_info = res.json()
            self._save_order_history(order_info, "BUY" if query['side'] == 'bid' else "SELL")
            return {'success': True, 'data': order_info}
        except requests.RequestException as e:
            print(f"주문 실패: {e.response.text if e.response else str(e)}")
            return {'success': False, 'data': e.response.json() if e.response else {}}

    def buy_market_order(self, market, amount):
        print(f"시장가 매수 주문: {market}, 금액: {amount:,.0f} KRW")
        return self._execute_order({"market": market, "side": "bid", "price": str(amount), "ord_type": "price"})

    def sell_market_order(self, market, volume):
        return self._execute_order({"market": market, "side": "ask", "volume": str(volume), "ord_type": "market"})

    def _save_order_history(self, order_info, order_type):
        try:
            filename = f"{ORDER_HISTORY_DIR}/{datetime.now().strftime('%Y%m%d')}_orders.csv"
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), order_type, order_info.get('market')])
        except Exception as e:
            print(f"주문 내역 저장 오류: {e}")


class TrailingStopManager(QThread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True

    def run(self):
        while self.running:
            try:
                for market, position in list(active_positions.items()):
                    price_info = self.parent.upbit_api.get_current_price(market)
                    if not price_info: continue
                    current_price = price_info['current_price']
                    highest_price = max(position.get('highest_price', current_price), current_price)
                    active_positions[market]['highest_price'] = highest_price

                    stop_price = highest_price * (1 - TRAILING_STOP_PERCENT / 100)
                    if current_price <= stop_price:
                        self.parent.add_to_log(f"{market} 트레일링 스탑 발동: {current_price:.1f} <= {stop_price:.1f}")
                        # 실제 매도 로직은 MyWindow 클래스에서 처리하도록 시그널 전송 또는 직접 호출
                        # self.parent.sell_logic(market)
            except Exception as e:
                print(f"트레일링 스탑 오류: {e}")
            time.sleep(60)

    def stop(self):
        self.running = False


class RHS_DiscoveryThread(QThread):
    """RHS 패턴 탐색 스레드"""
    progress_updated = pyqtSignal(str, int, int)

    def __init__(self, market_list, parent):
        super().__init__()
        self.market_list = market_list
        self.parent = parent
        self.running = True

    def run(self):
        for i, market in enumerate(self.market_list):
            if not self.running: break
            try:
                self.progress_updated.emit(market['market'], i + 1, len(self.market_list))
                if market['market'] in active_positions or market['market'] in pending_buy_list:
                    continue
                time.sleep(0.1)
                if self.parent.analyzeStock(market['market']):
                    pending_buy_list[market['market']] = datetime.now()
                    self.parent.add_to_log(f"RHS 패턴 발견, 매수 감시 목록 추가: {market['market']}")
            except Exception as e:
                print(f"{market['market']} 탐색 중 오류: {e}")
        self.parent.add_to_log(f"RHS 패턴 탐색 완료. 현재 감시 목록: {len(pending_buy_list)}개")

    def stop(self):
        self.running = False


class ExecutionManagerThread(QThread):
    """감시 목록 모니터링 및 매수 실행 스레드"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True
        self.macd_analyzer = MACDAnalyzer(self.parent.upbit_api)

    def run(self):
        self.parent.add_to_log("매수 감시 스레드 시작 (1분 간격)")
        while self.running:
            try:
                if len(active_positions) >= MAX_POSITIONS:
                    time.sleep(60)
                    continue

                for market in list(pending_buy_list.keys()):
                    if not self.running: break
                    is_increasing, prev, curr = self.macd_analyzer.check_macd_increase(market)
                    if is_increasing:
                        self.parent.add_to_log(f"{market} 1분봉 MACD 증가 포착! (이전: {prev:.4f}, 현재: {curr:.4f}) 매수 시도.")
                        self.parent.execute_buy_order(market)
                    # 감시 시간 초과 로직 (예: 1시간)
                    if datetime.now() - pending_buy_list[market] > timedelta(hours=1):
                        self.parent.add_to_log(f"{market} 감시 시간 초과로 목록에서 제거.")
                        pending_buy_list.pop(market, None)

            except Exception as e:
                print(f"매수 감시 스레드 오류: {e}")
            time.sleep(60)

    def stop(self):
        self.running = False


class SchedulerThread(QThread):
    """탐색 스케줄러 스레드"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True

    def run(self):
        schedule.every().day.at(DAILY_START_TIME).do(self.parent.start_discovery_scan)
        schedule.every(SCAN_INTERVAL_HOURS).hours.do(self.parent.start_discovery_scan)
        self.parent.add_to_log(f"스케줄러 시작: {DAILY_START_TIME} 및 {SCAN_INTERVAL_HOURS}시간마다 탐색 시작")
        while self.running:
            schedule.run_pending()
            time.sleep(60)

    def stop(self):
        self.running = False
        schedule.clear()


class MyWindow(QMainWindow):
    """메인 애플리케이션 클래스"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"자동매매 프로그램 (최대 {MAX_POSITIONS}종목, {INVESTMENT_DIVISION}분할 투자)")
        self.upbit_api = UpbitAPI(access_key, secret_key)
        self.pattern_recognizer = PatternRecognizer()
        self.init_threads()
        self.init_market_list()

        if RUN_MODE == 'windows':
            self.initUI()

        self.start_threads()

    def init_threads(self):
        self.discovery_thread = None
        self.execution_thread = ExecutionManagerThread(self)
        self.scheduler_thread = SchedulerThread(self)
        self.trailing_stop_thread = TrailingStopManager(self)

    def init_market_list(self):
        self.market_list = self.upbit_api.get_market_all()
        self.add_to_log(f"업비트 KRW 마켓 로드: {len(self.market_list)}개")

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.setCentralWidget(self.log_text)

    def start_threads(self):
        self.execution_thread.start()
        self.scheduler_thread.start()
        self.trailing_stop_thread.start()

    def add_to_log(self, message):
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(log_message)
        if RUN_MODE == 'windows' and hasattr(self, 'log_text'):
            self.log_text.append(log_message)

    def analyzeStock(self, market, days=250):
        ohlcv = self.upbit_api.get_ohlcv(market, days)
        if not ohlcv: return False
        self.pattern_recognizer.add_data(market, ohlcv)
        return self.pattern_recognizer.detect_inverse_head_shoulders(market)

    def execute_buy_order(self, market):
        if market not in pending_buy_list: return

        try:
            krw_balance = self.upbit_api.get_balance("KRW")
            remaining_slots = max(1, INVESTMENT_DIVISION - len(purchase_history))
            amount = krw_balance / remaining_slots

            if amount < 5000:
                self.add_to_log(f"매수금액 부족({amount:,.0f}원)으로 {market} 매수 취소")
                pending_buy_list.pop(market, None)
                return

            market_info = self.upbit_api.get_current_price(market)
            if not market_info: return

            res = self.upbit_api.buy_market_order(market, amount)
            if res['success']:
                purchase_history[market] = datetime.now()
                active_positions[market] = {
                    'purchase_price': market_info['current_price'],
                    'purchase_time': datetime.now(), 'investment_amount': amount,
                    'highest_price': market_info['current_price']
                }
                pending_buy_list.pop(market, None)
                self.add_to_log(f"매수 성공: {market} / {amount:,.0f}원 / 가격: {market_info['current_price']:.1f}")
            else:
                err_name = res.get('data', {}).get('error', {}).get('name', '')
                if 'insufficient_funds' in err_name:
                    self.add_to_log(f"잔고 부족으로 {market} 매수 실패. 감시 목록에서 제거.")
                    pending_buy_list.pop(market, None)
        except Exception as e:
            self.add_to_log(f"매수 실행 오류: {e}")

    def start_discovery_scan(self):
        if self.discovery_thread and self.discovery_thread.isRunning():
            self.add_to_log("탐색이 이미 진행 중입니다.")
            return
        self.add_to_log("RHS 패턴 탐색을 시작합니다.")
        self.discovery_thread = RHS_DiscoveryThread(self.market_list, self)
        self.discovery_thread.start()

    def closeEvent(self, event):
        self.add_to_log("프로그램 종료 절차 시작...")
        if self.discovery_thread: self.discovery_thread.stop()
        self.execution_thread.stop()
        self.scheduler_thread.stop()
        self.trailing_stop_thread.stop()
        time.sleep(1)  # 스레드 종료 대기
        if event: event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    if RUN_MODE == 'windows':
        window.show()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        window.closeEvent(None)