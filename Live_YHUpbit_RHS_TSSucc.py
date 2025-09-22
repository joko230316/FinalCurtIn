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
from datetime import datetime, timedelta

# ------------------- 설정 영역 ------------------- #

# 실행 모드 설정 ('linux' 또는 'windows')
# - linux: GUI 없이 서버 환경에서 실행 시 사용 (기본값)
# - windows: Windows 등 GUI 환경에서 실행 시 'windows'로 변경  linux linux
RUN_MODE = 'windows'

# 투자금 분할 설정
# 보유 현금을 몇 등분하여 투자할지 결정합니다. (예: 3으로 설정 시 3회에 걸쳐 균등 분할 매수)
INVESTMENT_DIVISION = 3

# ----------------------------------------------- #

# 실행 모드에 따라 Qt 플러그인 설정
if RUN_MODE == 'linux':
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # 헤드리스 모드 사용

from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QLabel, QPushButton, QVBoxLayout, QWidget, \
    QHBoxLayout, QSpinBox, QDoubleSpinBox, QTextEdit
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

# API 키 설정 (환경변수에서 로드)
access_key = os.getenv("upbit_YHaccess_key")
secret_key = os.getenv("upbit_YHsecret_key")

# 환경변수가 없을 경우 기본값 설정 (실제 사용시 반드시 환경변수로 설정하세요)
if not access_key:
    access_key = "your_access_key_here"
if not secret_key:
    secret_key = "your_secret_key_here"

# 전역 변수 설정
MAX_POSITIONS = 5  # 최대 보유 종목 수
REBUY_COOLDOWN_HOURS = 240  # 240시간 재매수 쿨다운
TRAILING_STOP_PERCENT = 2  # 2% 트레일링 스탑
SCAN_INTERVAL_HOURS = 24  # 24시간 간격으로 스캔
DAILY_START_TIME = "04:43"  # 매일 오전 1시 43분에 시작
MONITORING_INTERVAL_MINUTES = 5  # 5분 간격 모니터링

# 구매 내역 추적
purchase_history = {}
active_positions = {}

# 주문 내역 저장 디렉토리
ORDER_HISTORY_DIR = "order_history"

# 주문 내역 저장 디렉토리 생성
if not os.path.exists(ORDER_HISTORY_DIR):
    os.makedirs(ORDER_HISTORY_DIR)


class PatternRecognizer:
    """Analyzes OHLCV data to find specific chart patterns."""

    def __init__(self):
        # Data structure: { "KRW-BTC": [ {'date':..., 'open':...}, ... ] }
        self.ohlcv_data = {}

    def add_data(self, code, data_list):
        self.ohlcv_data[code] = data_list

    def is_near_yearly_low(self, code, threshold=0.2):
        """Checks if the current price is near the yearly low."""
        if code not in self.ohlcv_data:
            return False

        ohlcv = self.ohlcv_data[code]
        if len(ohlcv) < 200:
            return False

        # Find the minimum low price from the list of dictionaries
        yearly_low = min(day['low'] for day in ohlcv)
        # Get the latest closing price
        current_price = ohlcv[-1]['close']

        return current_price <= yearly_low * (1 + threshold)

    def detect_inverse_head_shoulders(self, code, window=60):
        """Detects an inverse head and shoulders pattern."""
        if code not in self.ohlcv_data:
            return False

        ohlcv = self.ohlcv_data[code]
        if len(ohlcv) < window:
            return False

        recent_data = ohlcv[-window:]
        lows = []  # List of (index, value) for pivot lows
        highs = []  # List of (index, value) for pivot highs

        # Find pivot lows
        for i in range(2, len(recent_data) - 2):
            if (recent_data[i]['low'] < recent_data[i - 1]['low'] and
                    recent_data[i]['low'] < recent_data[i - 2]['low'] and
                    recent_data[i]['low'] < recent_data[i + 1]['low'] and
                    recent_data[i]['low'] < recent_data[i + 2]['low']):
                lows.append((i, recent_data[i]['low']))

        # Find pivot highs
        for i in range(2, len(recent_data) - 2):
            if (recent_data[i]['high'] > recent_data[i - 1]['high'] and
                    recent_data[i]['high'] > recent_data[i - 2]['high'] and
                    recent_data[i]['high'] > recent_data[i + 1]['high'] and
                    recent_data[i]['high'] > recent_data[i + 2]['high']):
                highs.append((i, recent_data[i]['high']))

        if len(lows) < 3 or len(highs) < 2:
            return False

        # Check for the pattern
        for i in range(len(lows) - 2):
            left_shoulder_idx, left_shoulder_val = lows[i]
            head_idx, head_val = lows[i + 1]
            right_shoulder_idx, right_shoulder_val = lows[i + 2]

            # Conditions for a valid pattern
            is_sequential = left_shoulder_idx < head_idx < right_shoulder_idx
            has_min_spacing = (head_idx - left_shoulder_idx >= 5) and (right_shoulder_idx - head_idx >= 5)
            head_is_lowest = head_val < left_shoulder_val and head_val < right_shoulder_val
            shoulders_are_similar = abs(left_shoulder_val - right_shoulder_val) / max(left_shoulder_val, 0.0001) < 0.2
            is_recent = right_shoulder_idx >= len(recent_data) - 10

            if not (is_sequential and has_min_spacing and head_is_lowest and shoulders_are_similar and is_recent):
                continue

            # Find neckline peaks
            p1_candidates = [h_val for h_idx, h_val in highs if left_shoulder_idx < h_idx < head_idx]
            p2_candidates = [h_val for h_idx, h_val in highs if head_idx < h_idx < right_shoulder_idx]

            if not p1_candidates or not p2_candidates:
                continue

            neckline = max(max(p1_candidates), max(p2_candidates))

            # Check for neckline breakout
            if recent_data[-1]['close'] > neckline:
                return True

        return False


class UpbitAPI:
    """Upbit API를 사용하여 데이터를 가져오는 클래스"""

    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key
        self.server_url = "https://api.upbit.com/v1"

    def get_headers(self, query=None):
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
        }

        if query:
            m = hashlib.sha512()
            m.update(query.encode())
            query_hash = m.hexdigest()
            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"

        jwt_token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return {"Authorization": f"Bearer {jwt_token}"}

    def get_market_all(self):
        """모든 마켓 코드를 가져옵니다."""
        url = f"{self.server_url}/market/all"
        response = requests.get(url)

        if response.status_code == 200:
            markets = response.json()
            # KRW 마켓만 필터링
            krw_markets = [market for market in markets if market['market'].startswith('KRW-')]
            return krw_markets
        else:
            print(f"마켓 정보 요청 실패: {response.status_code}")
            return []

    def get_ohlcv(self, market, count=250, timeframe="days"):
        """OHLCV 데이터를 가져옵니다."""
        url = f"{self.server_url}/candles/{timeframe}"
        params = {
            "market": market,
            "count": count
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            ohlcv_data = response.json()
            # 데이터를 올바른 순서로 정렬 (과거 -> 현재)
            ohlcv_data.reverse()

            processed_data = []
            for item in ohlcv_data:
                processed_data.append({
                    'date': item['candle_date_time_kst'],
                    'open': item['opening_price'],
                    'high': item['high_price'],
                    'low': item['low_price'],
                    'close': item['trade_price'],
                    'volume': item['candle_acc_trade_volume']
                })

            return processed_data
        else:
            print(f"OHLCV 데이터 요청 실패 ({market}): {response.status_code}")
            return None

    def get_current_price(self, market):
        """현재 가격 정보를 가져옵니다."""
        url = f"{self.server_url}/ticker"
        params = {"markets": market}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data:
                return {
                    'code': market,
                    'name': market.replace('KRW-', ''),
                    'current_price': data[0]['trade_price'],
                    'diff': data[0]['signed_change_price'],
                    'diff_rate': data[0]['signed_change_rate'] * 100,
                    'volume': data[0]['acc_trade_volume_24h']
                }
        return None

    def get_accounts(self):
        """계좌 정보를 가져옵니다."""
        url = f"{self.server_url}/accounts"
        headers = self.get_headers()
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"계좌 정보 요청 실패: {response.status_code}")
            return None

    def get_balance(self, currency="KRW"):
        """특정 화폐의 잔고를 가져옵니다."""
        accounts = self.get_accounts()
        if accounts:
            for account in accounts:
                if account['currency'] == currency:
                    return float(account['balance'])
        return 0.0

    def get_coin_balance(self, market):
        """특정 코인의 보유 수량을 가져옵니다."""
        accounts = self.get_accounts()
        if accounts:
            currency = market.replace('KRW-', '')
            for account in accounts:
                if account['currency'] == currency:
                    return float(account['balance'])
        return 0.0

    def buy_market_order(self, market, amount):
        """시장가 매수 주문을 실행합니다."""
        query = {
            "market": market,
            "side": "bid",
            "price": str(amount),
            "ord_type": "price"
        }

        query_string = "&".join([f"{key}={value}" for key, value in query.items()])
        headers = self.get_headers(query_string)

        response = requests.post(f"{self.server_url}/orders", params=query, headers=headers)

        if response.status_code == 201:
            order_info = response.json()
            print(f"매수 주문 성공: {market}, 금액: {amount} KRW")

            # 주문 내역 저장
            self._save_order_history(order_info, "BUY")
            return {'success': True, 'data': order_info}
        else:
            print(f"매수 주문 실패: {response.status_code}, {response.text}")
            try:
                error_data = response.json()
            except json.JSONDecodeError:
                error_data = {'error': {'message': response.text, 'name': 'unknown_error'}}
            return {'success': False, 'data': error_data}

    def sell_market_order(self, market, volume):
        """시장가 매도 주문을 실행합니다."""
        query = {
            "market": market,
            "side": "ask",
            "volume": str(volume),
            "ord_type": "market"
        }

        query_string = "&".join([f"{key}={value}" for key, value in query.items()])
        headers = self.get_headers(query_string)

        response = requests.post(f"{self.server_url}/orders", params=query, headers=headers)

        if response.status_code == 201:
            order_info = response.json()
            print(f"매도 주문 성공: {market}, 수량: {volume}")

            # 주문 내역 저장
            self._save_order_history(order_info, "SELL")
            return {'success': True, 'data': order_info}
        else:
            print(f"매도 주문 실패: {response.status_code}, {response.text}")
            try:
                error_data = response.json()
            except json.JSONDecodeError:
                error_data = {'error': {'message': response.text, 'name': 'unknown_error'}}
            return {'success': False, 'data': error_data}

    def _save_order_history(self, order_info, order_type):
        """주문 내역을 CSV 파일로 저장합니다."""
        try:
            today = datetime.now().strftime("%Y%m%d")
            filename = f"{ORDER_HISTORY_DIR}/{today}_orders.csv"

            file_exists = os.path.isfile(filename)

            with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'type', 'market', 'price', 'volume', 'funds', 'uuid']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'type': order_type,
                    'market': order_info.get('market', ''),
                    'price': order_info.get('price', ''),
                    'volume': order_info.get('volume', ''),
                    'funds': order_info.get('executed_funds', ''),
                    'uuid': order_info.get('uuid', '')
                })

            print(f"주문 내역 저장 완료: {filename}")

        except Exception as e:
            print(f"주문 내역 저장 오류: {e}")

    def get_order(self, uuid):
        """주문 정보를 조회합니다."""
        params = {"uuid": uuid}
        query_string = "&".join([f"{key}={value}" for key, value in params.items()])
        headers = self.get_headers(query_string)

        response = requests.get(f"{self.server_url}/order", params=params, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"주문 조회 실패: {response.status_code}")
            return None


class TrailingStopManager:
    """트레일링 스탑 매니저"""

    def __init__(self, upbit_api, stop_percent):
        self.upbit_api = upbit_api
        self.stop_percent = stop_percent / 100
        self.running = False

    def start(self):
        """트레일링 스탑 모니터링 시작"""
        self.running = True
        thread = threading.Thread(target=self._monitor_positions)
        thread.daemon = True
        thread.start()

    def stop(self):
        """트레일링 스탑 모니터링 중지"""
        self.running = False

    def _monitor_positions(self):
        """포지션 모니터링"""
        while self.running:
            try:
                for market, position in list(active_positions.items()):
                    current_price_info = self.upbit_api.get_current_price(market)
                    if not current_price_info:
                        continue

                    current_price = current_price_info['current_price']
                    purchase_price = position['purchase_price']
                    highest_price = position.get('highest_price', purchase_price)

                    # 최고가 업데이트
                    if current_price > highest_price:
                        highest_price = current_price
                        active_positions[market]['highest_price'] = highest_price
                        active_positions[market]['max_profit'] = ((
                                                                          highest_price - purchase_price) / purchase_price) * 100
                        print(
                            f"{market} 최고가 업데이트: {highest_price:,.0f} KRW (최대수익: {active_positions[market]['max_profit']:.2f}%)")

                    # 트레일링 스탑 가격 계산 (최고가 대비 2% 하락)
                    stop_price = highest_price * (1 - self.stop_percent)

                    # 현재 가격이 스탑 가격 아래로 떨어지면 매도
                    if current_price <= stop_price:
                        print(f"{market} 트레일링 스탑 발동: {current_price:,.0f} KRW <= {stop_price:,.0f} KRW")

                        # 보유 수량 확인
                        sell_volume = self.upbit_api.get_coin_balance(market)

                        if sell_volume > 0:
                            # 매도 주문 실행
                            result = self.upbit_api.sell_market_order(market, sell_volume)
                            if result['success']:
                                print(f"{market} 매도 완료: {sell_volume} 수량")

                                # 수익률 계산
                                profit_loss = ((current_price - purchase_price) / purchase_price) * 100
                                print(f"수익률: {profit_loss:+.2f}%")

                                # 포지션 제거
                                active_positions.pop(market, None)

                time.sleep(MONITORING_INTERVAL_MINUTES * 60)
            except Exception as e:
                print(f"트레일링 스탑 모니터링 오류: {e}")
                time.sleep(60)  # 1분 후 재시도


class StockScannerThread(QThread):
    """Worker thread to scan stocks without freezing the UI."""
    scan_completed = pyqtSignal(list)
    progress_updated = pyqtSignal(str, int, int)  # 진행 상황 업데이트

    def __init__(self, market_list, parent):
        super().__init__()
        self.market_list = market_list
        self.parent = parent
        self.running = True
        self.upbit_api = UpbitAPI(access_key, secret_key)
        self.investment_amount = 0  # 이번 스캔 주기의 투자 금액

    def run(self):
        # --- 투자금액 계산 로직 ---
        krw_balance = self.upbit_api.get_balance("KRW")

        # 투자해야 할 남은 분할 횟수 계산
        remaining_investment_slots = INVESTMENT_DIVISION - len(active_positions)

        if remaining_investment_slots <= 0:
            print(f"모든 투자금({INVESTMENT_DIVISION}분할)이 소진되어 매수를 진행하지 않습니다.")
            self.scan_completed.emit([])
            return

        if len(active_positions) >= MAX_POSITIONS:
            print(f"최대 보유 종목({MAX_POSITIONS})에 도달하여 매수를 진행하지 않습니다.")
            self.scan_completed.emit([])
            return

        # 남은 현금을 남은 투자 횟수로 균등하게 분배
        self.investment_amount = krw_balance / remaining_investment_slots
        self.parent.add_to_log(
            f"이번 스캔의 종목당 투자금액 설정: {self.investment_amount:,.0f} KRW ({remaining_investment_slots}회 남음)")
        # -------------------------

        found_patterns = []
        total = len(self.market_list)

        for i, market in enumerate(self.market_list):
            if not self.running:
                break

            try:
                self.progress_updated.emit(market['market'], i + 1, total)
                time.sleep(0.1)

                if len(active_positions) >= MAX_POSITIONS:
                    print(f"최대 보유 종목 수({MAX_POSITIONS})에 도달하여 추가 매수 중단")
                    break

                if self._is_in_cooldown(market['market']):
                    continue

                if market['market'] in active_positions:
                    continue

                if self.parent.analyzeStock(market['market'], days=250):
                    market_info = self.upbit_api.get_current_price(market['market'])
                    if market_info:
                        found_patterns.append(market_info)
                        self.parent.print_market_info(market_info, "패턴 발견!")
                        self._execute_buy_order(market_info)
            except Exception as e:
                print(f"마켓 {market['market']} 분석 중 오류: {e}")

        if self.running:
            self.scan_completed.emit(found_patterns)

    def _is_in_cooldown(self, market):
        """재매수 쿨다운 확인"""
        if market in purchase_history:
            last_purchase = purchase_history[market]
            cooldown_end = last_purchase + timedelta(hours=REBUY_COOLDOWN_HOURS)
            if datetime.now() < cooldown_end:
                remaining_time = cooldown_end - datetime.now()
                hours_remaining = remaining_time.total_seconds() / 3600
                print(f"{market} 재매수 쿨다운 중: {hours_remaining:.1f}시간 남음")
                return True
        return False

    def _execute_buy_order(self, market_info):
        """매수 주문 실행"""
        try:
            investment_amount = self.investment_amount

            if investment_amount < 5000:  # 최소 주문 금액
                log_msg = f"계산된 투자금({investment_amount:,.0f} KRW)이 최소 주문 금액(5,000 KRW)보다 작아 매수하지 않습니다."
                print(log_msg)
                self.parent.add_to_log(log_msg)
                return

            print(f"{market_info['code']} 매수 시도: {investment_amount:,.0f} KRW")

            # 매수 주문 실행
            order_response = self.upbit_api.buy_market_order(market_info['code'], investment_amount)

            if order_response['success']:
                purchase_history[market_info['code']] = datetime.now()
                active_positions[market_info['code']] = {
                    'purchase_price': market_info['current_price'],
                    'purchase_time': datetime.now(),
                    'investment_amount': investment_amount,
                    'coin_name': market_info['name'],
                    'highest_price': market_info['current_price'],
                    'max_profit': 0.0
                }
                print(f"{market_info['code']} 매수 완료: {investment_amount:,.0f} KRW")
                print(f"현재 보유 종목 수: {len(active_positions)}/{MAX_POSITIONS}")

                self.parent.add_to_log(f"매수: {market_info['code']} - {investment_amount:,.0f} KRW")
                self.parent.updatePositionsDisplay()
            else:
                error_info = order_response.get('data', {})
                error_name = error_info.get('error', {}).get('name', '')
                if 'insufficient_funds' in error_name:
                    log_msg = f"{market_info['code']} 매수 실패: 잔고 부족"
                else:
                    error_message = error_info.get('error', {}).get('message', '알 수 없는 오류')
                    log_msg = f"{market_info['code']} 매수 실패: {error_message}"
                print(log_msg)
                self.parent.add_to_log(log_msg)
        except Exception as e:
            print(f"매수 주문 실행 오류: {e}")

    def stop(self):
        self.running = False


class SchedulerThread(QThread):
    """스케줄러 스레드"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True

    def run(self):
        schedule.every().day.at(DAILY_START_TIME).do(self._start_daily_scan)
        schedule.every(SCAN_INTERVAL_HOURS).hours.do(self._start_interval_scan)

        self.parent.add_to_log(f"스케줄러 시작: 매일 {DAILY_START_TIME}, {SCAN_INTERVAL_HOURS}시간 간격")
        self.parent.add_to_log(f"최대 보유 종목: {MAX_POSITIONS}, 잔고 {INVESTMENT_DIVISION}분할 투자")
        self.parent.add_to_log(f"트레일링 스탑: {TRAILING_STOP_PERCENT}%, 모니터링 간격: {MONITORING_INTERVAL_MINUTES}분")

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                print(f"스케줄러 오류: {e}")
                time.sleep(300)

    def _start_daily_scan(self):
        self.parent.add_to_log(f"일일 스캔 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.parent.startScan()

    def _start_interval_scan(self):
        self.parent.add_to_log(f"주기적 스캔 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.parent.startScan()

    def stop(self):
        self.running = False
        schedule.clear()


class MyWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"역헤드앤숄더 패턴 탐지기 - 업비트 (최대 {MAX_POSITIONS}종목, {INVESTMENT_DIVISION}분할 투자)")
        self.setGeometry(100, 100, 1600, 900)

        self.pattern_recognizer = PatternRecognizer()
        self.scanner_thread = None
        self.scheduler_thread = None
        self.upbit_api = UpbitAPI(access_key, secret_key)
        self.trailing_stop_manager = TrailingStopManager(self.upbit_api, TRAILING_STOP_PERCENT)

        self.market_list = []
        self.max_positions = MAX_POSITIONS
        self.rebuy_cooldown = REBUY_COOLDOWN_HOURS
        self.trailing_stop_percent = TRAILING_STOP_PERCENT
        self.scan_interval = SCAN_INTERVAL_HOURS
        self.daily_start_time = DAILY_START_TIME
        self.monitoring_interval = MONITORING_INTERVAL_MINUTES

        self.initUI()
        self.init_market_list()
        self.startTrailingStop()
        self.startScheduler()

    def init_market_list(self):
        """업비트에서 KRW 마켓 목록을 가져옵니다."""
        print("마켓 목록 로딩 중...")
        self.market_list = self.upbit_api.get_market_all()
        print(f"로드된 마켓 수: {len(self.market_list)}")

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 설정 패널
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("최대 종목 수:"))
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 10)
        self.max_positions_spin.setValue(self.max_positions)
        self.max_positions_spin.valueChanged.connect(self.updateMaxPositions)
        settings_layout.addWidget(self.max_positions_spin)

        settings_layout.addWidget(QLabel("재매수 쿨다운 (시간):"))
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(1, 240)
        self.cooldown_spin.setValue(self.rebuy_cooldown)
        self.cooldown_spin.valueChanged.connect(self.updateRebuyCooldown)
        settings_layout.addWidget(self.cooldown_spin)

        settings_layout.addWidget(QLabel("트레일링 스탑 (%):"))
        self.stop_spin = QDoubleSpinBox()
        self.stop_spin.setRange(0.1, 10)
        self.stop_spin.setValue(self.trailing_stop_percent)
        self.stop_spin.valueChanged.connect(self.updateTrailingStop)
        settings_layout.addWidget(self.stop_spin)

        settings_layout.addWidget(QLabel("스캔 간격 (시간):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 48)
        self.interval_spin.setValue(self.scan_interval)
        self.interval_spin.valueChanged.connect(self.updateScanInterval)
        settings_layout.addWidget(self.interval_spin)
        layout.addLayout(settings_layout)

        # 시간 설정
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("일일 시작 시간:"))
        self.time_edit = QLabel(DAILY_START_TIME)
        time_layout.addWidget(self.time_edit)
        time_layout.addStretch()
        layout.addLayout(time_layout)

        self.next_scan_label = QLabel("다음 스캔 시간: 계산 중...")
        layout.addWidget(self.next_scan_label)
        self.positions_label = QLabel("보유 포지션: 0/0")
        layout.addWidget(self.positions_label)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        layout.addWidget(self.list_widget)
        header_text = f"{'마켓':<12s} | {'종목명':<15s} | {'현재가':>12s} | {'등락율':>10s} | {'거래량':>16s} | {'패턴':^6s} | {'1년저점':^8s}"
        self.list_widget.addItem(header_text)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("상태: 대기중")
        status_layout.addWidget(self.status_label)
        self.progress_label = QLabel("")
        status_layout.addWidget(self.progress_label)
        self.balance_label = QLabel("")
        status_layout.addWidget(self.balance_label)
        status_layout.addStretch()

        self.scan_button = QPushButton("즉시 스캔")
        self.scan_button.clicked.connect(self.startScan)
        status_layout.addWidget(self.scan_button)
        self.stop_button = QPushButton("스캔 중지")
        self.stop_button.clicked.connect(self.stopScan)
        status_layout.addWidget(self.stop_button)
        self.refresh_button = QPushButton("마켓 새로고침")
        self.refresh_button.clicked.connect(self.init_market_list)
        status_layout.addWidget(self.refresh_button)
        layout.addLayout(status_layout)

        self.balance_timer = QTimer(self)
        self.balance_timer.timeout.connect(self.updateBalance)
        self.balance_timer.start(60000)
        self.schedule_timer = QTimer(self)
        self.schedule_timer.timeout.connect(self.updateNextScanTime)
        self.schedule_timer.start(60000)
        self.positions_timer = QTimer(self)
        self.positions_timer.timeout.connect(self.updatePositionsDisplay)
        self.positions_timer.start(30000)

        self.updateBalance()
        self.updateNextScanTime()
        self.updatePositionsDisplay()

    def updateMaxPositions(self, value):
        self.max_positions = value

    def updateRebuyCooldown(self, value):
        self.rebuy_cooldown = value

    def updateTrailingStop(self, value):
        self.trailing_stop_percent = value
        self.trailing_stop_manager.stop_percent = value / 100

    def updateScanInterval(self, value):
        self.scan_interval = value
        self.restartScheduler()

    def updateBalance(self):
        try:
            balance = self.upbit_api.get_balance("KRW")
            self.balance_label.setText(f"잔고: {balance:,.0f} KRW")
        except:
            self.balance_label.setText("잔고: 확인 불가")

    def updateNextScanTime(self):
        try:
            idle = schedule.idle_seconds()
            if idle is None or idle < 0:
                self.next_scan_label.setText("다음 스캔 시간: 대기 중...")
                return
            next_run_time = datetime.now() + timedelta(seconds=idle)
            self.next_scan_label.setText(f"다음 스캔 시간: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            error_message = f"다음 스캔 시간 업데이트 오류: {e}"
            print(error_message)
            self.next_scan_label.setText("다음 스캔 시간: 계산 오류")

    def updatePositionsDisplay(self):
        try:
            current_count = len(active_positions)
            self.positions_label.setText(f"보유 포지션: {current_count}/{self.max_positions}")
            if current_count > 0:
                print("\n=== 현재 보유 포지션 ===")
                total_investment = 0
                for market, position in active_positions.items():
                    current_price_info = self.upbit_api.get_current_price(market)
                    if current_price_info:
                        current_price = current_price_info['current_price']
                        purchase_price = position['purchase_price']
                        profit_loss = ((current_price - purchase_price) / purchase_price) * 100
                        max_profit = position.get('max_profit', 0)
                        print(
                            f"{market}: 매수가 {purchase_price:,.0f}, 현재가 {current_price:,.0f}, 수익률 {profit_loss:+.2f}%, 최대수익 {max_profit:.2f}%")
                    total_investment += position['investment_amount']
                print(f"총 투자금: {total_investment:,.0f} KRW")
                print("=======================\n")
        except Exception as e:
            print(f"포지션 정보 업데이트 오류: {e}")

    def startTrailingStop(self):
        self.trailing_stop_manager.start()
        self.add_to_log("트레일링 스탑 모니터링 시작")

    def startScheduler(self):
        if self.scheduler_thread and self.scheduler_thread.isRunning():
            return
        self.scheduler_thread = SchedulerThread(self)
        self.scheduler_thread.start()
        self.add_to_log("스케줄러 시작됨")

    def restartScheduler(self):
        if self.scheduler_thread:
            self.scheduler_thread.stop()
            self.scheduler_thread.wait()
        self.startScheduler()

    def startScan(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            print("스캔이 이미 진행 중입니다.")
            return
        if not self.market_list:
            self.init_market_list()
            if not self.market_list:
                self.status_label.setText("상태: 마켓 목록을 가져올 수 없음")
                return
        self.status_label.setText("상태: 종목 스캔 중...")
        self.progress_label.setText("")
        self.clearListExceptHeader()

        print("\n" + "=" * 120)
        print(f"1일봉 역헤드앤숄더 패턴 스캔 시작 (최대 {self.max_positions}종목, {INVESTMENT_DIVISION}분할 투자)")
        print("-" * 120)
        print(f"{'상태':<12} | {'마켓':<10} | {'종목명':<12} | {'현재가':>12} | {'등락율':>10} | {'거래량':>16} | {'패턴'} | {'1년저점'}")
        print("-" * 120)

        self.scanner_thread = StockScannerThread(self.market_list, self)
        self.scanner_thread.scan_completed.connect(self.onScanCompleted)
        self.scanner_thread.progress_updated.connect(self.updateProgress)
        self.scanner_thread.start()

    def stopScan(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.stop()
            self.status_label.setText("상태: 스캔 중지 중...")
            self.scanner_thread.wait()
            self.status_label.setText("상태: 스캔 중지됨")

    def updateProgress(self, market, current, total):
        self.progress_label.setText(f"진행: {current}/{total} ({market})")

    def onScanCompleted(self, found_patterns):
        if not found_patterns:
            print("패턴 발견 종목 없음")
            self.list_widget.addItem("패턴이 발견된 종목이 없습니다.")
            self.add_to_log("패턴 발견 종목 없음")
        else:
            for info in found_patterns:
                self.add_item_to_list(info)
        self.status_label.setText(f"상태: 스캔 완료 (발견: {len(found_patterns)} 종목)")
        self.progress_label.setText("")
        print("=" * 120 + "\n")

    def add_item_to_list(self, info):
        pattern_status = 'O' if self.pattern_recognizer.detect_inverse_head_shoulders(info['code']) else 'X'
        yearly_low_status = 'O' if self.pattern_recognizer.is_near_yearly_low(info['code']) else 'X'
        item_text = (f"{info['code']:<12s} | {info['name']:<15s} | {info['current_price']:>12,.0f} | "
                     f"{info['diff_rate']:>+9.2f}% | {info['volume']:>16,.0f} | "
                     f"{pattern_status:^6s} | {yearly_low_status:^8s}")
        self.list_widget.addItem(item_text)

    def add_to_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_text.append(log_message)
        print(log_message)

    def clearListExceptHeader(self):
        while self.list_widget.count() > 1:
            self.list_widget.takeItem(1)

    def analyzeStock(self, market, days=250):
        ohlcv_data = self.upbit_api.get_ohlcv(market, days, "days")
        if not ohlcv_data:
            return False
        self.pattern_recognizer.add_data(market, ohlcv_data)
        return self.pattern_recognizer.detect_inverse_head_shoulders(market)

    def print_market_info(self, market_info, status):
        if not market_info:
            return
        pattern = 'O' if self.pattern_recognizer.detect_inverse_head_shoulders(market_info['code']) else 'X'
        low_near = 'O' if self.pattern_recognizer.is_near_yearly_low(market_info['code']) else 'X'
        print(f"{status:<12} | {market_info['code']:<10} | {market_info['name']:<12} | "
              f"{market_info['current_price']:>12,.0f} | {market_info['diff_rate']:>+9.2f}% | "
              f"{market_info['volume']:>16,.0f} | {pattern:^4s} | {low_near:^6s}")

    def closeEvent(self, event):
        self.stopScan()
        if self.scheduler_thread:
            self.scheduler_thread.stop()
            self.scheduler_thread.wait()
        self.trailing_stop_manager.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if RUN_MODE == 'linux':
        print("헤드리스 모드(리눅스 서버)로 실행 중...")
        print(f"설정: 최대 {MAX_POSITIONS}종목, 잔고 {INVESTMENT_DIVISION}분할 투자")
        print(f"트레일링 스탑: {TRAILING_STOP_PERCENT}%, 모니터링 간격: {MONITORING_INTERVAL_MINUTES}분")
        myWindow = MyWindow()
        sys.exit(app.exec_())
    else:
        myWindow = MyWindow()
        myWindow.show()
        sys.exit(app.exec_())