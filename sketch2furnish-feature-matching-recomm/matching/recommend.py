import os
import numpy as np
import traceback
import base64
from io import BytesIO
from pymongo import MongoClient
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gridfs import GridFS
import io
import re

# FastAPI 서버 생성
app = FastAPI()

# CORS 설정 추가 (로컬 프론트엔드에서 백엔드 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],  # 로컬 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📌 예외 핸들러 추가 (500 Internal Server Error 원인 파악)
@app.exception_handler(Exception)
async def exception_handler(request, exc):
    """ 모든 예외를 잡아서 상세한 로그를 출력하는 핸들러 """
    error_message = f"🔥 ERROR: {str(exc)}\n{traceback.format_exc()}"
    print(error_message)  # 터미널에서 오류 로그 확인
    raise HTTPException(status_code=500, detail="🚨 서버 내부 오류 발생! 로그를 확인하세요.")

# MongoDB 연결 (로컬 환경에서 실행)
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["furniture_db"]
collection = db["furniture_embeddings"]
fs = GridFS(db)

# 🔹 CLIP 모델 및 프로세서 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_embedding(image: Image) -> dict:
    """CLIP Vision Encoder를 사용하여 이미지 임베딩 추출"""
    try:
        # 🔹 이미지 전처리 (CLIP Processor 사용)
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            clip_embedding = clip_model.get_image_features(**inputs)

        # 🔹 numpy 변환 후 평탄화 (MongoDB 저장 구조 유지)
        clip_embedding = clip_embedding.cpu().numpy().flatten()

        return {
            "clip_embedding": clip_embedding,
            "cnn_embedding": clip_embedding,  # CNN 모델이 없으면 CLIP과 동일하게 사용
            "vit_embedding": clip_embedding  # ViT 모델이 없으면 CLIP과 동일하게 사용
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🚨 이미지 임베딩 추출 오류: {str(e)}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산 함수"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_furniture(query_embedding: dict, top_k: int = 4):
    """MongoDB에 저장된 가구 임베딩들과 비교하여 가장 유사한 가구 4개 찾기"""
    all_furniture = list(collection.find({}, {
        "cnn_embedding": 1, "vit_embedding": 1, "clip_embedding": 1, 
        "filename": 1, "category": 1, "price": 1, "brand": 1, "coupang_link": 1, "_id": 0
    }))

    similarities = []
    for furniture in all_furniture:
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

# 📌 JSON 데이터를 받기 위한 Pydantic 모델 정의
class ImageRequest(BaseModel):
    image_data: str  # Base64 인코딩된 이미지

import re  # 문자열에서 숫자만 추출하는 정규식 사용

def extract_numeric_price(price_str):
    """📌 '100600원(won)' 같은 문자열에서 숫자 부분만 추출하여 정수 변환"""
    numeric_part = re.findall(r'\d+', price_str)  # 숫자만 추출
    return int("".join(numeric_part)) if numeric_part else 0  # 숫자가 있으면 변환, 없으면 0

@app.post("/recommend")
async def recommend_furniture(request: ImageRequest, min_price: int = 0, max_price: int = 100000000):
    try:
        image_bytes = base64.b64decode(request.image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        query_embedding = extract_embedding(image)
        recommended_furniture = find_similar_furniture(query_embedding)

        # 🔹 가격 필터 적용 (가격 데이터 변환 후 비교)
        filtered_furniture = [
            item for item in recommended_furniture
            if min_price <= extract_numeric_price(item["price"]) <= max_price
        ]

        # 🔹 가격 범위 내 가구가 4개 미만이면 기존 추천 가구에서 추가
        while len(filtered_furniture) < 4 and recommended_furniture:
            candidate = recommended_furniture.pop(0)
            if candidate not in filtered_furniture:
                filtered_furniture.append(candidate)

        return {"recommendations": filtered_furniture[:4]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🚨 추천 과정에서 오류 발생: {str(e)}")



@app.get("/image/{filename}")
async def get_image(filename: str):
    """MongoDB에 저장된 가구 이미지를 다운로드하는 API"""
    try:
        image_data = fs.find_one({"filename": filename})
        if not image_data:
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다!")
        return StreamingResponse(io.BytesIO(image_data.read()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🚨 이미지 제공 중 오류 발생: {str(e)}")
