# -*- coding: utf-8 -*-
import sys
import os
import win32com.client
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit, \
    QHBoxLayout, QSpinBox
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer, QDateTime
import time
import pythoncom
from datetime import datetime, time as dt_time, timedelta
import google.generativeai as genai

# --- 전역 설정 ---
# 1주씩 주문하도록 설정
ORDER_QUANTITY = 1

# 스캔 시간 설정 (기본값: 오전 8시)
SCAN_HOUR = 8
SCAN_MINUTE = 0

# API 키 및 계좌번호 환경 변수에서 로드
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    CREON_ACCOUNT_NUM = os.getenv("CREON_ACCOUNT")
    if not GEMINI_API_KEY:
        print("오류: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    if not CREON_ACCOUNT_NUM:
        print("오류: CREON_ACCOUNT 환경변수가 설정되지 않았습니다.")
except Exception as e:
    print(f"환경 변수 로딩 중 오류: {e}")


# -----------------

class CpStockChart:
    """주식 차트 데이터 요청 클래스"""

    def __init__(self):
        pass

    def Request(self, code, days=61):
        pythoncom.CoInitialize()
        try:
            objCpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
            if objCpCybos.IsConnect == 0:
                return None

            objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
            objStockChart.SetInputValue(0, code)
            objStockChart.SetInputValue(1, ord('2'))
            objStockChart.SetInputValue(4, days)
            objStockChart.SetInputValue(5, [0, 2, 3, 4, 5, 8])
            objStockChart.SetInputValue(6, ord('D'))
            objStockChart.SetInputValue(9, ord('1'))
            objStockChart.BlockRequest()

            if objStockChart.GetDibStatus() != 0:
                return None

            num_data = objStockChart.GetHeaderValue(3)
            data = []
            for i in range(num_data):
                data.append({
                    'date': objStockChart.GetDataValue(0, i),
                    'open': objStockChart.GetDataValue(1, i),
                    'high': objStockChart.GetDataValue(2, i),
                    'low': objStockChart.GetDataValue(3, i),
                    'close': objStockChart.GetDataValue(4, i),
                    'volume': objStockChart.GetDataValue(5, i)
                })
            data.reverse()
            return data
        finally:
            pythoncom.CoUninitialize()


class CpStockMst:
    """현재 주식 정보 요청 클래스"""

    def request_stock_info(self, code):
        pythoncom.CoInitialize()
        try:
            objStockMst = win32com.client.Dispatch("DsCbo1.StockMst")
            objStockMst.SetInputValue(0, code)
            objStockMst.BlockRequest()

            if objStockMst.GetDibStatus() != 0:
                return None

            change = objStockMst.GetHeaderValue(12)
            return {
                'code': code,
                'name': objStockMst.GetHeaderValue(1),
                'current_price': objStockMst.GetHeaderValue(11),
                'prev_close': objStockMst.GetHeaderValue(10),
                'volume': objStockMst.GetHeaderValue(18),
                'volume_amount': objStockMst.GetHeaderValue(19),
                'change_rate': (change / objStockMst.GetHeaderValue(10)) * 100 if objStockMst.GetHeaderValue(
                    10) != 0 else 0
            }
        finally:
            pythoncom.CoUninitialize()


class StockAnalyzer:
    """주식 분석 클래스"""

    def calculate_moving_average(self, prices, period):
        if len(prices) < period: return None
        return [sum(prices[i - period + 1:i + 1]) / period if i >= period - 1 else None for i in range(len(prices))]

    def analyze_stock(self, chart_data):
        if not chart_data or len(chart_data) < 61: return None
        closes = [d['close'] for d in chart_data]
        prev_closes = closes[:-1]

        data = {
            'current_open': chart_data[-1]['open'], 'current_close': chart_data[-1]['close'],
            'prev_close': chart_data[-2]['close'], 'ma5': self.calculate_moving_average(closes, 5)[-1],
            'ma10': self.calculate_moving_average(closes, 10)[-1],
            'ma20': self.calculate_moving_average(closes, 20)[-1],
            'ma60': self.calculate_moving_average(closes, 60)[-1],
            'prev_ma5': self.calculate_moving_average(prev_closes, 5)[-1],
            'prev_ma20': self.calculate_moving_average(prev_closes, 20)[-1],
            'volume': chart_data[-1]['volume']
        }
        for key, value in data.items():
            if value is None: return None  # 하나라도 None이면 분석 불가
        return data

    def check_all_conditions(self, data):
        results = {}
        results['A'] = data['current_open'] < data['current_close']
        results['C'] = all(data['current_close'] > data[ma] for ma in ['ma5', 'ma10', 'ma20'])
        results['D'] = all(data['current_open'] < data[ma] for ma in ['ma5', 'ma10', 'ma20'])
        results['I'] = all(data['current_close'] > data[ma] for ma in ['ma5', 'ma10', 'ma60'])
        results['J'] = all(data['current_open'] < data[ma] for ma in ['ma5', 'ma10', 'ma60'])
        results['H'] = ((data['current_close'] - data['prev_close']) / data['prev_close']) * 100 >= 5 if data[
                                                                                                             'prev_close'] != 0 else False
        results['F'] = abs(data['ma5'] - data['ma20']) / data['ma20'] * 100 <= 2 if data['ma20'] != 0 else False
        results['K'] = abs(data['prev_ma5'] - data['prev_ma20']) / data['prev_ma20'] * 100 <= 2 if data[
                                                                                                       'prev_ma20'] != 0 else False
        results['S'] = (data['volume'] * data['current_close']) >= 10_000_000
        results['T'] = data['ma20'] >= data['prev_ma20']
        return results


class GeminiAnalyzer:
    """Gemini AI를 이용한 주식 분석 클래스"""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API 키가 설정되지 않았습니다.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_stock_potential(self, stock_name, stock_code):
        prompt = f"""
        대한민국 주식 시장의 숙련된 AI 애널리스트로서, 다음 종목의 향후 1개월 수익률 잠재력을 분석해줘.
        분석은 KRX(한국거래소)에 공시된 최신 재무 데이터와 기술적 지표를 기반으로 해야 해.

        - 종목명: {stock_name}
        - 종목코드: {stock_code}

        분석 결과를 바탕으로, 이 종목의 향후 1개월 수익률 잠재력을 1점(매우 낮음)부터 10점(매우 높음)까지의 정수 점수로만 평가해줘.
        다른 설명은 모두 제외하고, 오직 "SCORE: [점수]" 형식으로 마지막 줄에 점수만 출력해줘. 예를 들어 "SCORE: 8" 과 같이 말이야.
        """
        try:
            response = self.model.generate_content(prompt)
            last_line = response.text.strip().split('\n')[-1]
            if "SCORE:" in last_line:
                score = int(last_line.replace("SCORE:", "").strip())
                return score
        except Exception as e:
            print(f"{stock_name}({stock_code}) Gemini 분석 오류: {e}")
        return 0


class CreonTrader:
    """대신증권 Creon API를 이용한 주식 주문 클래스"""

    def __init__(self):
        self.obj_trade_util = win32com.client.Dispatch("CpTrade.CpTdUtil")
        if self.obj_trade_util.TradeInit(0) != 0:
            raise ConnectionError("Creon 거래 서비스 초기화 실패")

    def place_market_buy_order(self, stock_code, quantity):
        if not CREON_ACCOUNT_NUM:
            raise ValueError("증권 계좌번호가 설정되지 않았습니다.")

        obj_order = win32com.client.Dispatch("CpTrade.CpTd0311")
        obj_order.SetInputValue(0, "2")  # 2: 매수
        obj_order.SetInputValue(1, CREON_ACCOUNT_NUM)  # 계좌번호
        obj_order.SetInputValue(3, stock_code)  # 종목코드
        obj_order.SetInputValue(4, quantity)  # 수량
        obj_order.SetInputValue(5, 0)  # 가격 (0: 시장가)
        obj_order.SetInputValue(8, "03")  # 주문조건 (03: IOC)

        ret = obj_order.BlockRequest()
        if ret == 0:
            return True, "매수 주문 성공"
        else:
            return False, f"매수 주문 실패 (에러코드: {ret})"


class StockScannerThread(QThread):
    """주식 스캔 스레드"""
    scan_completed = pyqtSignal(list)
    progress_updated = pyqtSignal(str)
    stock_found = pyqtSignal(object)

    def __init__(self, stock_list):
        super().__init__()
        self.stock_list = stock_list
        self.running = True

    def run(self):
        analyzer = StockAnalyzer()
        chart_requester = CpStockChart()
        info_requester = CpStockMst()
        found_stocks = []

        for i, code in enumerate(self.stock_list):
            if not self.running: break
            self.progress_updated.emit(f"종목 분석 중: {code} ({i + 1}/{len(self.stock_list)})")
            time.sleep(0.25)

            chart_data = chart_requester.Request(code)
            if not chart_data: continue

            analyzed_data = analyzer.analyze_stock(chart_data)
            if not analyzed_data: continue

            results = analyzer.check_all_conditions(analyzed_data)
            if all(results.values()):
                stock_info = info_requester.request_stock_info(code)
                if stock_info:
                    found_stocks.append(stock_info)
                    self.stock_found.emit(stock_info)

        if self.running:
            self.scan_completed.emit(found_stocks)

    def stop(self):
        self.running = False


class AnalysisAndTradeThread(QThread):
    """AI 분석 및 자동매매 스레드"""
    log_message = pyqtSignal(str)
    analysis_finished = pyqtSignal()
    trade_executed = pyqtSignal(object, bool, str)  # stock, success, message

    def __init__(self, found_stocks):
        super().__init__()
        self.found_stocks = found_stocks
        self.best_stock = None
        self.max_score = -1

    def run(self):
        try:
            if not self.found_stocks:
                self.log_message.emit("조건을 만족하는 종목이 없습니다.")
                self.analysis_finished.emit()
                return

            self.log_message.emit("\n===== Gemini AI 수익률 잠재력 분석 시작 =====")
            gemini = GeminiAnalyzer()

            for stock in self.found_stocks:
                self.log_message.emit(f"分析中... {stock['name']}({stock['code']})")
                score = gemini.analyze_stock_potential(stock['name'], stock['code'])
                self.log_message.emit(f"-> {stock['name']} 잠재력 점수: {score}점")
                if score > self.max_score:
                    self.max_score = score
                    self.best_stock = stock

            if not self.best_stock:
                self.log_message.emit("AI 분석 결과, 투자에 적합한 종목을 찾지 못했습니다.")
                self.analysis_finished.emit()
                return

            self.log_message.emit(f"\n===== 최종 투자 결정 =====")
            self.log_message.emit(f"종목: {self.best_stock['name']} (점수: {self.max_score}점)")

            # 장 시작 시간까지 대기 (9시 5분)
            current_time = datetime.now()
            target_time = current_time.replace(hour=9, minute=5, second=0, microsecond=0)

            if current_time < target_time:
                wait_seconds = (target_time - current_time).total_seconds()
                self.log_message.emit(f"장 시작 5분 후({target_time.strftime('%H:%M:%S')})까지 대기합니다...")
                time.sleep(wait_seconds)

            self.execute_trade()

        except Exception as e:
            self.log_message.emit(f"AI 분석 및 주문 처리 중 오류 발생: {e}")
            self.analysis_finished.emit()

    def execute_trade(self):
        try:
            self.log_message.emit("\n===== 자동 매수 주문 시도 =====")
            trader = CreonTrader()
            success, msg = trader.place_market_buy_order(self.best_stock['code'], ORDER_QUANTITY)
            self.log_message.emit(f"주문 결과: {msg}")
            self.trade_executed.emit(self.best_stock, success, msg)
        except Exception as e:
            error_msg = f"주문 실행 중 오류: {e}"
            self.log_message.emit(error_msg)
            self.trade_executed.emit(self.best_stock, False, error_msg)
        finally:
            self.analysis_finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        pythoncom.CoInitialize()
        self.setWindowTitle("주식 조건 검색 및 자동주문 시스템 - 자동 모드")
        self.setGeometry(100, 100, 900, 700)

        self.scanner_thread = None
        self.analysis_thread = None
        self.stock_list = self.get_filtered_stock_codes()
        self.found_stocks = []
        self.scheduled_timer = QTimer()
        self.scheduled_timer.timeout.connect(self.check_schedule)
        self.scheduled_timer.start(60000)  # 1분마다 체크

        self.init_ui()
        self.append_text("시스템이 시작되었습니다.")
        self.update_next_run_time()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 스캔 시간 설정 UI
        time_setting_layout = QHBoxLayout()
        time_setting_layout.addWidget(QLabel("스캔 시간 설정:"))

        self.hour_spinbox = QSpinBox()
        self.hour_spinbox.setRange(0, 23)
        self.hour_spinbox.setValue(SCAN_HOUR)
        self.hour_spinbox.valueChanged.connect(self.update_scan_time)
        time_setting_layout.addWidget(self.hour_spinbox)
        time_setting_layout.addWidget(QLabel("시"))

        self.minute_spinbox = QSpinBox()
        self.minute_spinbox.setRange(0, 59)
        self.minute_spinbox.setValue(SCAN_MINUTE)
        self.minute_spinbox.valueChanged.connect(self.update_scan_time)
        time_setting_layout.addWidget(self.minute_spinbox)
        time_setting_layout.addWidget(QLabel("분"))

        layout.addLayout(time_setting_layout)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.status_label = QLabel("상태: 대기중")
        self.scan_button = QPushButton("지금 바로 스캔")
        self.stop_button = QPushButton("스캔/분석 중지")

        layout.addWidget(self.text_edit)
        layout.addWidget(self.status_label)
        layout.addWidget(self.scan_button)
        layout.addWidget(self.stop_button)

        self.scan_button.clicked.connect(self.start_scan)
        self.stop_button.clicked.connect(self.stop_all_threads)
        self.stop_button.setEnabled(False)

        self.append_text(f"총 {len(self.stock_list)}개 종목 대상 검색 준비 완료.")
        self.append_text(f"현재 스캔 시간 설정: {SCAN_HOUR:02d}:{SCAN_MINUTE:02d}")

    def update_scan_time(self):
        global SCAN_HOUR, SCAN_MINUTE
        SCAN_HOUR = self.hour_spinbox.value()
        SCAN_MINUTE = self.minute_spinbox.value()
        self.append_text(f"스캔 시간이 {SCAN_HOUR:02d}:{SCAN_MINUTE:02d}으로 변경되었습니다.")
        self.update_next_run_time()

    def check_schedule(self):
        now = datetime.now()
        # 설정된 시간에 실행
        if now.weekday() < 5 and now.hour == SCAN_HOUR and now.minute == SCAN_MINUTE:
            self.append_text(f"⏰ 예약된 시간입니다! ({SCAN_HOUR:02d}:{SCAN_MINUTE:02d}) 자동 스캔을 시작합니다.")
            self.start_scan()

    def update_next_run_time(self):
        now = datetime.now()
        today_target = now.replace(hour=SCAN_HOUR, minute=SCAN_MINUTE, second=0, microsecond=0)

        if now.weekday() >= 5:  # 주말
            # 다음 평일까지의 일수 계산
            days_to_add = 7 - now.weekday() if now.weekday() == 5 else 1
            next_run = today_target + timedelta(days=days_to_add)
        else:
            if now.time() < today_target.time():
                next_run = today_target
            else:
                # 다음 평일까지의 일수 계산
                next_weekday = now.weekday() + 1
                if next_weekday >= 5:  # 금요일 이후면 월요일로
                    days_to_add = 7 - now.weekday()
                else:
                    days_to_add = 1
                next_run = today_target + timedelta(days=days_to_add)

        self.status_label.setText(f"상태: 대기중 (다음 실행: {next_run.strftime('%m월 %d일 %H:%M')})")

    def get_filtered_stock_codes(self):
        codemgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
        all_codes = codemgr.GetStockListByMarket(1) + codemgr.GetStockListByMarket(2)
        return [code for code in all_codes if codemgr.GetStockSupervisionKind(code) == 0 and codemgr.GetStockStatusKind(
            code) == 0 and not codemgr.IsSPAC(code)]

    def start_scan(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            return
        self.text_edit.clear()
        self.append_text(f"주식 검색을 시작합니다. (대상: {len(self.stock_list)} 종목)")
        self.append_text(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.scan_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.scanner_thread = StockScannerThread(self.stock_list)
        self.scanner_thread.progress_updated.connect(self.update_progress)
        self.scanner_thread.stock_found.connect(self.on_stock_found)
        self.scanner_thread.scan_completed.connect(self.on_scan_completed)
        self.scanner_thread.start()

    def stop_all_threads(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.stop()
            self.status_label.setText("상태: 스캔 중지 중...")
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.status_label.setText("상태: 분석/주문 중지 중...")
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.append_text("\n사용자에 의해 모든 작업이 중지되었습니다.")

    def on_stock_found(self, stock_info):
        self.append_text(f"✅ 조건 충족: {stock_info['name']}({stock_info['code']})")

    def on_scan_completed(self, found_stocks):
        self.found_stocks = found_stocks
        self.append_text(f"\n===== 검색 완료: 총 {len(found_stocks)}개 종목 발견 =====")
        self.save_results_to_file(found_stocks)

        # 장 시간 확인 (9시 ~ 15시 30분)
        now = datetime.now()
        market_open_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if now.weekday() < 5 and market_open_time.time() <= now.time() <= market_close_time.time():
            if GEMINI_API_KEY and CREON_ACCOUNT_NUM:
                self.analysis_thread = AnalysisAndTradeThread(found_stocks)
                self.analysis_thread.log_message.connect(self.append_text)
                self.analysis_thread.trade_executed.connect(self.on_trade_executed)
                self.analysis_thread.analysis_finished.connect(self.on_analysis_finished)
                self.analysis_thread.start()
            else:
                self.append_text("\n[오류] API 키 또는 계좌번호가 설정되지 않아 AI 분석 및 주문을 건너뜁니다.")
                self.reset_ui_state()
        else:
            self.append_text("\n장 시간이 아니므로 AI 분석 및 자동주문을 실행하지 않습니다.")
            self.print_final_summary(found_stocks)
            self.reset_ui_state()

    def on_trade_executed(self, stock, success, message):
        if success:
            self.append_text(f"🎉 매수 주문 완료: {stock['name']}({stock['code']})")
        else:
            self.append_text(f"❌ 매수 주문 실패: {message}")

    def save_results_to_file(self, found_stocks):
        filename = f"{datetime.now().strftime('%Y-%m-%d')}_found_stocks.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"검색된 종목 리스트 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("=" * 40 + "\n")
            if not found_stocks:
                f.write("조건을 만족하는 종목을 찾지 못했습니다.\n")
            else:
                for stock in found_stocks:
                    f.write(f"- 종목명: {stock['name']} ({stock['code']})\n")
                    f.write(f"  현재가: {stock['current_price']:,}원\n")
                    f.write(f"  등락률: {stock['change_rate']:.2f}%\n")
                    f.write(f"  거래대금: {stock['volume_amount']:,}원\n\n")
        self.append_text(f"\n검색 결과를 '{filename}' 파일로 저장했습니다.")

    def print_final_summary(self, stocks):
        self.append_text("\n--- 최종 요약 ---")
        for stock in stocks:
            self.append_text(
                f"{stock['name']}({stock['code']}) | 현재가: {stock['current_price']:,}원 | 등락률: {stock['change_rate']:.2f}% | 거래량: {stock['volume']:,}")

    def on_analysis_finished(self):
        self.append_text("\n===== 모든 절차 완료 =====")
        self.reset_ui_state()
        self.update_next_run_time()

    def update_progress(self, message):
        self.status_label.setText(message)

    def append_text(self, text):
        self.text_edit.append(text)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())

    def reset_ui_state(self):
        self.status_label.setText("상태: 대기중")
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_all_threads()
        pythoncom.CoUninitialize()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()