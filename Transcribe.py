from flask import Flask, request, jsonify, make_response
import requests
from flask_cors import CORS  # pip install flask-cors

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NDA1NDYyMjcsIm5iZiI6MTc0MDU0NjIyNywiZXhwIjoxNzUwNDYzOTk5LCJrZXlfaWQiOiI4NDE5M2JlMi0xOTllLTQ1NjUtOGJkNC1mYjcwMDRjYWViMzgifQ.Tt272G6boYh26O_WppZMy0PAlA77Ueay1Cq15EMW0GU"  # 여기에 실제 API 키 입력

if API_KEY == "YOUR_API_KEY":
    raise ValueError("API_KEY를 설정하세요.")

def transcribe_audio(audio_data, api_key):
    url = "https://api-cloud-function.elice.io/76bff628-acfc-45a2-9c39-e3dd316f33f2/transcribe"
    files = {
        "audio": ("recorded_audio.wav", audio_data, "audio/wav")
    }
    data = {
        "environment": "Production"
    }
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(url, headers=headers, data=data, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"API 요청 실패 (코드: {response.status_code})",
            "details": response.text,
        }

@app.route("/stt", methods=["POST"])
def stt():
    if "audio" not in request.files:
        return make_response(jsonify({"error": "No audio file provided"}), 400)
    audio_file = request.files["audio"]
    result = transcribe_audio(audio_file, API_KEY)
    return make_response(jsonify(result), 200)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
