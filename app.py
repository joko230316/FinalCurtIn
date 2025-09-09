import os
import json
import logging
from flask import Flask, request, jsonify
import ccxt

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask 앱 초기화 ---
app = Flask(__name__)

# --- API 자격 증명 로드 ---
API_KEY = os.getenv("OKXYH_API_KEY")
API_SECRET = os.getenv("OKXYH_API_SECRET")
API_PASSPHRASE = os.getenv("OKXYH_API_PASSPHRASE")

# API 키 존재 여부 확인
if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logging.critical("치명적 오류: OKX API 환경변수가 설정되지 않았습니다.")
    # 이 경우 앱이 시작되지 않도록 처리할 수 있으나, 우선 경고만 로깅합니다.
    # 실서버에서는 exit(1) 등으로 종료하는 것이 좋습니다.

# --- CCXT 거래소 객체 생성 ---
try:
    exchange = ccxt.okx({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'password': API_PASSPHRASE,
        'options': {
            'defaultType': 'swap',  # 선물 거래
        },
    })
    logging.info("✅ OKX 거래소 객체가 성공적으로 생성되었습니다.")
except Exception as e:
    logging.error(f"❌ OKX 거래소 객체 생성 실패: {e}")
    exchange = None


# --- 웹훅 수신 라우트 ---
@app.route('/webhook', methods=['POST'])
def webhook():
    if not exchange:
        logging.error("거래소 객체가 초기화되지 않아 주문을 처리할 수 없습니다.")
        return jsonify({'status': 'error', 'message': 'Exchange not initialized'}), 500

    try:
        # TradingView에서 보낸 JSON 데이터 파싱
        data = request.get_json()
        if data is None:
            # JSON이 아닌 일반 텍스트로 왔을 경우를 대비
            plain_text_data = request.data.decode('utf-8')
            logging.warning(f"수신된 데이터가 JSON 형식이 아닙니다. 텍스트로 파싱 시도: {plain_text_data}")
            data = json.loads(plain_text_data)

        logging.info(f"📥 웹훅 수신: {data}")

        # 필수 파라미터 확인
        symbol = data.get('symbol')
        action = data.get('action')
        amount = data.get('amount')

        if not all([symbol, action, amount]):
            return jsonify({'status': 'error', 'message': 'Missing parameters: symbol, action, or amount'}), 400

        # action에 따른 주문 정보 설정
        # 단방향 모드 기준: 포지션 진입은 buy/sell, 포지션 종료는 반대매매
        if action.lower() in ['buy', 'long']:
            side = 'buy'
        elif action.lower() in ['sell', 'short']:
            side = 'sell'
        elif action.lower() == 'close_long':
            side = 'sell'
        elif action.lower() == 'close_short':
            side = 'buy'
        else:
            return jsonify({'status': 'error', 'message': f"Invalid action: {action}"}), 400

        # 주문 실행
        logging.info(f"🚀 주문 실행 준비: {symbol}, {side}, {amount}")
        order = exchange.create_order(
            symbol=symbol,
            type='market',  # 시장가 주문
            side=side,
            amount=float(amount),
            params={
                'tdMode': 'cross'  # 교차 모드
            }
        )
        logging.info(f"✅ 주문 성공: {order}")
        return jsonify({'status': 'success', 'order': order}), 200

    except json.JSONDecodeError:
        logging.error(f"JSON 파싱 오류. 수신된 원본 데이터: {request.data.decode('utf-8')}")
        return jsonify({'status': 'error', 'message': 'Invalid JSON format'}), 400
    except ccxt.BaseError as e:
        logging.error(f"CCXT 오류 (거래소 API 오류): {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생: {e}")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred'}), 500


@app.route('/', methods=['GET'])
def health_check():
    return "Webhook listener is running.", 200


# --- 앱 실행 ---
if __name__ == '__main__':
    # 80 포트는 관리자 권한이 필요할 수 있습니다.
    # Linux/macOS: sudo python app.py
    # Windows: 관리자 권한으로 터미널 실행
    app.run(host='0.0.0.0', port=80)