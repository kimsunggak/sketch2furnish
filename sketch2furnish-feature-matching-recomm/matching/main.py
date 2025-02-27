import os
import numpy as np
from pymongo import MongoClient
import torch
import torchvision.transforms as transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List

# MongoDB Atlas 연결
MONGO_URI = os.getenv("MONGO_URI")  # 환경 변수에서 MongoDB URI 불러오기
client = MongoClient(MONGO_URI)
db = client["furniture_db"]
collection = db["furniture_embeddings"]

# 이미지 임베딩 모델 (CLIP 사용)
model = SentenceTransformer("clip-ViT-B-32")

# 이미지 전처리 변환기
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(image: Image) -> dict:
    """사용자가 업로드한 이미지에서 CLIP, ViT, CNN 기반 임베딩을 추출"""
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        clip_embedding = model.encode(image)  # CLIP 모델을 사용한 임베딩 추출

    # CNN, ViT 모델 추가 가능 (예제 코드에선 CLIP만 사용)
    return {
        "clip_embedding": clip_embedding,
        "cnn_embedding": clip_embedding,  # CNN 모델이 없으면 CLIP과 동일하게 사용
        "vit_embedding": clip_embedding  # ViT 모델이 없으면 CLIP과 동일하게 사용
    }

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

        # 평균 유사도를 사용하여 가장 유사한 가구 찾기
        avg_similarity = (cnn_similarity + vit_similarity + clip_similarity) / 3
        similarities.append((avg_similarity, furniture))

    # 유사도 높은 순으로 정렬 후 top_k개 선택
    top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
    return [match[1] for match in top_matches]
