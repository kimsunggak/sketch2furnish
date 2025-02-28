from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 추가
import requests

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 접근 허용

# 외부 맞춤법 검사 API 정보
API_URL = "https://api-cloud-function.elice.io/c9d3f335-47c2-4401-b616-73cebcf53593/check"
API_KEY = "YOUR_API_KEY"  # 실제 API 키로 교체하세요

@app.route("/check", methods=["POST"])
def check_spelling():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "검사할 텍스트가 제공되지 않았습니다."}), 400

    payload = {"text": data["text"]}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({
            "error": "맞춤법 검사 API 요청 실패",
            "status": response.status_code,
            "details": response.text
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
