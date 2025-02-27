import requests
import base64
from io import BytesIO
from PIL import Image

# 기본 프롬프트: 특정 가구(쇼파)가 아닌 일반 '가구'로 표현
default_prompt = "단색 흰 배경 위에 중앙에 오직 한 개의 가구만 등장하며, 어떠한 다른 가구나 소품도 포함되지 않은, 깔끔한 구성의 고해상도 실제 사진 스타일 이미지."

# 사용자로부터 추가 프롬프트 입력 받기 (없을 경우엔 기본 프롬프트만 사용)
user_prompt = input("추가 프롬프트를 입력하세요 (없으면 엔터키): ").strip()

# 최종 프롬프트 조합 (사용자 입력이 있을 경우 기본 프롬프트 뒤에 추가)
final_prompt = default_prompt
if user_prompt:
    final_prompt += " " + user_prompt

# 엘리스 ML API 엔드포인트 URL
url = "https://api-cloud-function.elice.io/0133c2f7-9f3f-44b6-a3d6-c24ba8ef4510/generate"

# API 요청에 사용할 데이터
payload = {
    "prompt": final_prompt,
    "style": "3d_animation",
    "num_inference_steps": 1,
    "width": 256,
    "height": 256,
    "seed": 42
}

# 헤더에 실제 API 키를 입력하세요.
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"  # 실제 API 키로 교체
}

print("API 요청을 보내는 중...")
response = requests.post(url, json=payload, headers=headers)
print("응답 코드:", response.status_code)

if response.status_code == 200:
    result = response.json()
    print("API 응답 JSON:", result)
    
    # API 응답의 이미지 데이터가 "predictions" 키에 있다고 가정합니다.
    image_base64 = result.get("predictions")
    if image_base64:
        try:
            # Base64 문자열 디코딩
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # 이미지 파일로 저장
            image.save("generated_image.png")
            print("이미지가 'generated_image.png'로 저장되었습니다.")
            
            # 이미지 표시
            image.show()
        except Exception as e:
            print("이미지 디코딩 중 오류 발생:", e)
    else:
        print("응답에 이미지 데이터가 없습니다.")
else:
    print("Error:", response.status_code, response.text)
