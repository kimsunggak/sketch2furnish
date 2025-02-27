import requests
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 1. CLIP 모델과 프로세서 로드 (가구 이미지 검증용)
print("CLIP 모델과 프로세서를 로드합니다...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. design_preview.png 파일 로드 및 CLIP 특징 추출
try:
    furniture_image = Image.open("design_preview.png")
except Exception as e:
    print("design_preview.png를 불러오는 중 오류 발생:", e)
    exit(1)

clip_inputs = clip_processor(images=furniture_image, return_tensors="pt")
with torch.no_grad():
    image_features = clip_model.get_image_features(**clip_inputs)

# 3. "가구"에 해당하는 텍스트 프롬프트 처리 및 특징 추출 (검증 용도)
furniture_prompt = "a piece of furniture"
text_inputs = clip_processor(text=[furniture_prompt], return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)

# 4. 이미지와 텍스트 특징 간 코사인 유사도 계산 및 출력 (단순 검증)
image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
similarity = (image_features_norm @ text_features_norm.T).item()
print("이미지와 'a piece of furniture' 텍스트 간의 유사도:", similarity)

threshold = 0.2  # 임계치 (필요에 따라 조정)
if similarity < threshold:
    print("경고: 이미지 유사도가 낮습니다. 해당 이미지가 가구 이미지가 아닐 수 있습니다.")

# 5. design_preview.png를 conditioning image로 사용하기 위해 Base64로 인코딩
with open("design_preview.png", "rb") as img_file:
    conditioning_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# 6. 사용자로부터 간단한 프롬프트 입력 받기 (입력 없으면 기본 프롬프트 사용)
user_prompt = input("추가 프롬프트를 입력하세요 (없으면 엔터키): ").strip()
default_text_prompt = "기존 디자인 스타일을 유지한 단색 배경의 가구 이미지."
# 최종 프롬프트는 기본 프롬프트와 사용자 입력만으로 구성됩니다.
final_prompt = user_prompt if user_prompt else default_text_prompt

# 7. 엘리스 ML API 엔드포인트 URL (실제 엔드포인트 URL 사용)
url = "https://api-cloud-function.elice.io/0133c2f7-9f3f-44b6-a3d6-c24ba8ef4510/generate"

# 8. API 요청 데이터 구성 (conditioning image 포함)
payload = {
    "prompt": final_prompt,
    "conditioning_image": conditioning_image_base64,  # 스타일 유지용 조건 이미지
    "style": "3d_animation",
    "num_inference_steps": 1,
    "width": 256,
    "height": 256,
    "seed": 42
}

# 9. API 요청 헤더 (실제 API 키로 교체)
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"  # 실제 API 키로 교체
}

print("API 요청을 보내는 중...")
response = requests.post(url, json=payload, headers=headers)
print("응답 코드:", response.status_code)

if response.status_code == 200:
    try:
        result = response.json()
        print("API 응답 JSON:", result)
    except Exception as e:
        print("JSON 파싱 오류:", e)
        result = None

    if result:
        image_base64 = result.get("predictions")
        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))
                image.save("generated_image.png")
                print("이미지가 'generated_image.png'로 저장되었습니다.")
                image.show()
            except Exception as e:
                print("이미지 디코딩 중 오류 발생:", e)
        else:
            print("응답에 이미지 데이터가 없습니다.")
else:
    print("Error:", response.status_code, response.text)
