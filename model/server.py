from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
import re
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from models import networks 
from data.base_dataset import get_params, get_transform
from dotenv import load_dotenv
from openai import OpenAI
import os
import requests
from pymongo import MongoClient
from gridfs import GridFS
from transformers import CLIPProcessor, CLIPModel
from pydantic import BaseModel
from io import BytesIO

# ---------------------------
# MongoDB 연결 설정
# ---------------------------
try:
    MONGO_URI = "mongodb+srv://sth0824:daniel0824@sthcluster.sisvx.mongodb.net/?retryWrites=true&w=majority&appName=STHCluster"
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # 연결 테스트
    client.server_info()
    print("MongoDB 연결 성공!")
except Exception as e:
    print(f"MongoDB 연결 실패: {str(e)}")
    print("주의: MongoDB가 없으면 추천 기능이 작동하지 않습니다!")

db = client["furniture_db"]
collection = db["furniture_embeddings"]
fs = GridFS(db)

# ---------------------------
# opt 객체 생성 (TestOptions 기반)
# ---------------------------
from options.test_options import TestOptions
opt = TestOptions().parse()
print("Test Options:", opt)

# ---------------------------
# 모델 생성 및 불러오기
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define_G 함수를 통해 Pix2Pix 생성기 네트워크 생성
model = networks.define_G(
    input_nc=3, 
    output_nc=3, 
    ngf=64, 
    netG='unet_256', 
    use_dropout=False, 
    gpu_ids=[]
)
model.to(device)

# 모델 가중치 로드 
model_path = os.path.join("checkpoints", "furniture_pix2pix", "latest_net_G.pth")
model.load_state_dict(torch.load(model_path, map_location=device))

# ---------------------------
# CLIP 모델 로드 (가구 추천용)
# ---------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------------------
# FastAPI 앱 및 CORS 설정
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 후처리 함수: 텐서를 PIL 이미지로 변환
# ---------------------------
def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    모델의 출력 텐서를 [0,1] 범위로 정규화 해제하고, PIL 이미지로 변환합니다.
    """
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2  # [-1,1] -> [0,1]
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.clamp(0, 1))
    return image

# ---------------------------
# CLIP 임베딩 추출 함수
# ---------------------------
def extract_embedding(image: Image) -> dict:
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        clip_embedding = clip_model.get_image_features(**inputs).cpu().numpy().flatten()
    return {"clip_embedding": clip_embedding, "cnn_embedding": clip_embedding, "vit_embedding": clip_embedding}

# ---------------------------
# 코사인 유사도 계산 함수
# ---------------------------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ---------------------------
# 가격 문자열에서 숫자만 추출하는 함수
# ---------------------------
def extract_numeric_price(price_str):
    if not isinstance(price_str, str):
        return 0  # 문자열이 아니면 변환할 수 없음

    price_str = price_str.replace("₩", "").replace(",", "").replace("(won)", "").replace("원", "").strip()
    numeric_part = re.findall(r'\d+', price_str)
    
    if numeric_part:
        numeric_value = int("".join(numeric_part))
    else:
        numeric_value = 0  # 변환 실패 시 0원 처리
    
    return numeric_value

# ---------------------------
# 유사 가구 검색 함수
# ---------------------------
def find_similar_furniture(query_embedding: dict, min_price: int = 0, max_price: int = 100000000, top_k: int = 4):
    all_furniture = list(collection.find({}, {
        "cnn_embedding": 1, "vit_embedding": 1, "clip_embedding": 1, 
        "filename": 1, "category": 1, "price": 1, "brand": 1, "coupang_link": 1, "_id": 0
    }))

    similarities = []
    for furniture in all_furniture:
        price = extract_numeric_price(furniture.get("price", "0"))
        
        if min_price <= price <= max_price:
            stored_cnn = np.array(furniture.get("cnn_embedding", []))
            stored_vit = np.array(furniture.get("vit_embedding", []))
            stored_clip = np.array(furniture.get("clip_embedding", []))

            cnn_similarity = cosine_similarity(query_embedding["cnn_embedding"], stored_cnn) if len(stored_cnn) > 0 else 0
            vit_similarity = cosine_similarity(query_embedding["vit_embedding"], stored_vit) if len(stored_vit) > 0 else 0
            clip_similarity = cosine_similarity(query_embedding["clip_embedding"], stored_clip) if len(stored_clip) > 0 else 0

            avg_similarity = (cnn_similarity + vit_similarity + clip_similarity) / 3
            similarities.append((avg_similarity, furniture))

    top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
    return [match[1] for match in top_matches]

# ---------------------------
# API 엔드포인트
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI 서버가 실행 중입니다!"}

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    # 업로드된 파일이 이미지인지 확인
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다!")
    
    try:
        image_bytes = await file.read()
        # PIL 이미지로 변환
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # 학습 시와 동일한 전처리 파라미터 생성 (crop, flip 등)
        params = get_params(opt, image.size)
        # 학습 시 사용한 전처리 파이프라인 적용
        transform = get_transform(opt, params)
        input_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 전처리 오류: {str(e)}")
    
    # 모델 추론 수행
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 후처리: 텐서를 PIL 이미지로 변환
    result_image = tensor_to_image(output_tensor)
    
    # 결과 이미지를 바이트 스트림으로 변환하여 응답 생성 (PNG 포맷)
    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

# ---------------------------
# 맞춤법 검사 API 엔드포인트
# ---------------------------
@app.post("/check")
async def check_spelling(text_data: dict):
    if not text_data or "text" not in text_data:
        return JSONResponse(status_code=400, content={"error": "검사할 텍스트가 제공되지 않았습니다."})

    # 환경 변수에서 API 키 가져오기
    API_URL = "https://api-cloud-function.elice.io/c9d3f335-47c2-4401-b616-73cebcf53593/check"
    API_KEY = os.getenv("SPELLING_API_KEY")  # .env 파일에서 API 키를 로드
    
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "맞춤법 검사 API 키가 설정되지 않았습니다."})
    
    payload = {"text": text_data["text"]}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return JSONResponse(
                status_code=500, 
                content={
                    "error": "맞춤법 검사 API 요청 실패",
                    "status": response.status_code,
                    "details": response.text
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"맞춤법 검사 중 오류 발생: {str(e)}"}
        )

# ---------------------------
# OpenAI API 클라이언트 설정 및 이미지 편집 엔드포인트
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@app.post("/edit-image")
async def edit_image(
    original_image: UploadFile = File(...),
    prompt: str = Form(...),
):
    """
    멀티모달 AI 모델을 사용하여 이미지를 분석하고 새 이미지를 생성합니다.
    """
    # 1. 원본 이미지 처리
    try:
        original_image_bytes = await original_image.read()
        img = Image.open(io.BytesIO(original_image_bytes))
        
        # 이미지 크기 및 형식 정보 출력
        print(f"원본 이미지 정보: 크기={img.size}, 모드={img.mode}")
        
        # 이미지 크기 조정 및 RGB 변환
        img = img.resize((512, 512))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 디버깅용 이미지 저장
        debug_path = os.path.join(os.path.dirname(__file__), "debug_original.png")
        img.save(debug_path)
        print(f"원본 이미지 저장됨: {debug_path}")
        
        # 이미지를 Base64로 인코딩
        image_buffer = io.BytesIO()
        img.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        base64_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        print(f"Base64 인코딩 완료: {len(base64_image)} 바이트")
        
    except Exception as e:
        print(f"이미지 처리 오류: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"이미지 처리 오류: {str(e)}"})

    # 2. 멀티모달 모델에 이미지 분석 요청
    try:
        print(f"GPT-4o 모델로 이미지 분석 시작...")
        print(f"분석 프롬프트: {prompt}")
        
        # 더 구체적인 가구 중심 프롬프트
        multimodal_prompt = f"""
        이미지에 있는 가구를 분석해주세요. 이 가구의 종류, 형태, 디자인 특징을 상세히 설명해주세요.
        그 후 '{prompt}' 스타일로 변경했을 때의 가구 모습을 설명해주세요.
        반드시 가구에 대한 설명만 제공하세요.
        """
        
        # GPT-4o 모델에 이미지와 프롬프트 전송
        try:
            vision_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": multimodal_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # 응답 전체 출력
            full_response = vision_response.choices[0].message.content
            print(f"GPT 응답 (전체):\n{full_response}")
            
            # 가구 특화 프롬프트로 변환 - 사용자 프롬프트 강조
            user_prompt_emphasized = prompt.upper()  # 대문자로 변환하여 강조
            dalle_prompt = f"【{prompt}】스타일의 가구 제품 사진. {full_response[:150]}. 사실적인 가구 3D 렌더링."
            
        except Exception as vision_err:
            print(f"GPT-4o API 오류: {str(vision_err)}")
            # 오류 발생 시 기본 프롬프트 사용
            dalle_prompt = f"【{prompt}】스타일의 고급 가구 디자인. 사실적인 3D 렌더링의 가구."
            full_response = f"비전 모델 오류: {str(vision_err)}"
        
    except Exception as e:
        print(f"비전 모델 처리 오류: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"비전 모델 오류: {str(e)}"})

    # 3. DALL-E로 새 이미지 생성
    try:
        # 가구에 초점을 명확히 하고 사용자 프롬프트를 더 강조
        final_prompt = f"""### 전문 가구 제품 이미지 생성 - 스타일: 【{prompt}】 ###

생성해야 할 것: 오직 {prompt} 스타일의 단일 가구 아이템만 포함된 깨끗한 제품 이미지
배경: 완전히 순수한 흰색 또는 매우 연한 회색 배경 (그라데이션 없음)
카메라 각도: 가구가 명확히 보이는 3/4 앵글의 프로페셔널한 제품 사진

제약사항:
- 배경에 다른 가구, 소품, 장식, 사람, 동물, 식물 등의 요소 포함하지 않음
- 텍스트, 워터마크, 로고 등의 그래픽 요소 없음
- 창문, 벽, 바닥, 그림자 등의 환경 요소 최소화
- 음식, 과일, 음료 등의 비가구 항목 완전히 제외

이미지 요구사항:
- 고품질 3D 렌더링, 포토리얼리스틱한 조명
- 제품 사진 스타일의 단순하고 깔끔한 구도
- 가구의 재질, 질감, 디테일이 선명하게 표현되어야 함

가구 세부 정보: {full_response[:100]}
"""
        
        print(f"DALL-E 생성 프롬프트: {final_prompt[:150]}...")
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        print("이미지 생성 완료!")
        result_url = response.data[0].url
        
        # 결과에 원본 GPT 응답도 포함
        return {
            "result_url": result_url,
            "gpt_response": full_response,
            "final_prompt": final_prompt
        }
        
    except Exception as e:
        print(f"DALL-E API 오류: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"DALL-E API 오류: {str(e)}"})

# ---------------------------
# 가구 추천 관련 클래스 및 엔드포인트
# ---------------------------
class RecommendRequest(BaseModel):
    image_data: str
    min_price: int = 0
    max_price: int = 100000000

@app.post("/recommend")
async def recommend_furniture_endpoint(request: RecommendRequest):
    """
    사용자가 업로드한 이미지와 유사한 가구를 추천합니다.
    가격 범위 필터링도 지원합니다.
    """
    try:
        min_price = request.min_price
        max_price = request.max_price
        print(f"적용된 가격 필터 값 - 최소 가격: {min_price}, 최대 가격: {max_price}")
        
        # Base64 데이터 처리 코드 개선
        try:
            print(f"수신된 이미지 데이터 길이: {len(request.image_data)}")
            # 패딩 문자(=)가 없는 경우 추가
            padded_data = request.image_data
            padding = len(padded_data) % 4
            if padding:
                padded_data += '=' * (4 - padding)
            
            try:
                # base64 디코딩
                image_bytes = base64.b64decode(padded_data)
                print(f"디코딩된 이미지 바이트 길이: {len(image_bytes)}")
                
                # 추가 디버깅 - 바이트 시작 부분 출력
                print(f"바이트 시작: {image_bytes[:20]}")
                
                # 이미지 파일인지 확인 (헤더 검사)
                if not (image_bytes.startswith(b'\xff\xd8') or  # JPEG
                        image_bytes.startswith(b'\x89PNG') or   # PNG
                        image_bytes.startswith(b'GIF') or       # GIF 
                        image_bytes.startswith(b'BM')):         # BMP
                    print("경고: 이미지 헤더가 인식되지 않습니다. 계속 진행합니다...")
                
                # PIL 이미지로 열기 시도
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                print(f"이미지 열기 성공: 크기={image.size}, 모드={image.mode}")
                
                # 임베딩 추출 및 추천 진행
                query_embedding = extract_embedding(image)
                recommended_furniture = find_similar_furniture(query_embedding, min_price, max_price)
                print(f"최종 추천된 가구 개수: {len(recommended_furniture)}개")
                return {"recommendations": recommended_furniture}
                
            except Exception as e:
                print(f"이미지 처리 실패: {str(e)}")
                # 추가 디버깅 - 오류가 발생한 경우 더 자세한 정보
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {str(e)}")
                
        except Exception as e:
            print(f"Base64 데이터 처리 실패: {str(e)}")
            raise HTTPException(status_code=400, detail=f"이미지 데이터 처리 오류: {str(e)}")
            
    except Exception as e:
        print(f"추천 과정에서 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"가구 추천 중 오류 발생: {str(e)}")

@app.get("/image/{filename}")
async def get_image(filename: str):
    """
    MongoDB에 저장된 가구 이미지를 다운로드하는 API
    """
    try:
        image_data = fs.find_one({"filename": filename})
        if not image_data:
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다!")
        return StreamingResponse(io.BytesIO(image_data.read()), media_type="image/jpeg")
    except Exception as e:
        print(f"이미지 제공 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 제공 중 오류 발생: {str(e)}")

# ---------------------------
# 서버 실행
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)