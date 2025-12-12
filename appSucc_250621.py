import os
import hmac
import base64
import hashlib
import requests
import json
import time
from flask import Flask, request, jsonify, abort
import logging

# ----------- 전역 변수 설정 -----------
LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", 100))         # 기본 레버리지 (100, 변경 가능)
AMOUNT_MULTIPLIER = int(os.getenv("DEFAULT_AMOUNT_MULTIPLIER", 1))  # 승수 기본 1
TD_MODE = os.getenv("DEFAULT_TDMODE", "cross")             # cross, 변경 가능
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your_key_here")
PERPLEXITY_API_URL = "https://api.perplexity.ai/v1/your_endpoint"

OKX_API_KEY = os.getenv("OKX_API_KEY", "your_okx_key")
OKX_API_SECRET = os.getenv("OKX_API_SECRET", "your_okx_secret")
OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "your_passphrase")
OKX_BASE_URL = "https://www.okx.com"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ----------- Helper for OKX API Auth -----------
def get_iso_timestamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')

def generate_okx_headers(method, request_path, body=''):
    timestamp = str(time.time())
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    mac = hmac.new(OKX_API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    sign = base64.b64encode(mac.digest()).decode()
    return {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": OKX_API_PASSPHRASE,
        "Content-Type": "application/json"
    }

# ----------- 잔고 조회 함수 -----------
def get_balance_usdt():
    url_path = "/api/v5/account/balance"
    headers = generate_okx_headers("GET", url_path)
    try:
        response = requests.get(OKX_BASE_URL+url_path, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        usdt_balance = 0.0
        if data.get("code") == "0":
            for asset in data.get("data", [])[0].get("details", []):
                if asset.get("ccy") == "USDT":
                    usdt_balance = float(asset.get("availBal", 0))
        return usdt_balance
    except Exception as e:
        logging.error(f"잔고조회 오류: {e}")
        return 0.0

# ----------- 주문 함수 -----------
def place_order_okx(symbol, amount, side, leverage, td_mode):
    url_path = "/api/v5/trade/order"
    body = {
        "instId": symbol,
        "tdMode": td_mode,
        "side": side.lower(),      # buy/sell
        "ordType": "market",
        "sz": str(amount),
        "lever": str(leverage)
    }
    json_body = json.dumps(body)
    headers = generate_okx_headers("POST", url_path, json_body)
    try:
        response = requests.post(OKX_BASE_URL+url_path, headers=headers, data=json_body, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "0":
            return {"status": "success", "order_id": data["data"][0]["ordId"]}
        return {"status": "error", "detail": data}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ----------- Perplexity AI 자문 함수 -----------
def call_perplexity_advisor(signal):
    headers = {
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        "input": signal,
        "prompt": "해당 신호에 맞는 레버리지 선물 트레이딩 AI 자문을 해줘. 결과는 JSON만 반환."
    }
    try:
        resp = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"advisor": "error", "reason": str(e)}

# ----------- 승수 계산 함수 -----------
def calculate_amount_multiplier(balance, unit=20, multiplier_base=1):
    # 20USDT 단위마다 승수 1씩 증가, 최소 1
    return max(multiplier_base, int(balance // unit))

# ----------- 웹훅 주문 처리 -----------
@app.route('/webhook', methods=['POST'])
def webhook_receiver():
    if not request.is_json:
        abort(415)
    data = request.get_json()
    symbol = data.get("symbol")
    action = data.get("action")
    req_amount = int(data.get("amount", 0))
    if not (symbol and action and req_amount):
        return jsonify({"error": "Missing required fields"}), 400

    # OKX API에 최적화된 심볼 형식으로 변환 필요할 수 있음
    okx_symbol = symbol.replace("USDT", "-USDT-SWAP")

    # 잔고 확인해서 승수 결정
    balance = get_balance_usdt()
    amount_multiplier = calculate_amount_multiplier(balance, unit=20, multiplier_base=AMOUNT_MULTIPLIER)
    final_amount = req_amount * amount_multiplier
    leverage = LEVERAGE
    td_mode = TD_MODE

    # Perplexity API 자문 요청
    ai_advice = call_perplexity_advisor({
        "symbol": okx_symbol,
        "action": action,
        "req_amount": req_amount,
        "balance": balance,
        "amount_multiplier": amount_multiplier,
        "final_amount": final_amount,
        "leverage": leverage,
        "tdMode": td_mode
    })

    # 실제 주문 실행(OKX futures 시장가 주문)
    order_result = place_order_okx(okx_symbol, final_amount, action, leverage, td_mode)

    # 결과 출력 형식(요청시 참고 코드 예시)
    result = {
        "symbol": okx_symbol,
        "action": action,
        "requested_amount": req_amount,
        "balance_usdt": balance,
        "amount_multiplier": amount_multiplier,
        "final_order_amount": final_amount,
        "leverage": leverage,
        "td_mode": td_mode,
        "order_result": order_result,
        "ai_advice": ai_advice,
    }
    return jsonify(result), 200

# ----------- 서버 상태 체크 -----------
@app.route('/status', methods=['GET'])
def status_check():
    return jsonify({"status": "OKX AI trading server is running"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)