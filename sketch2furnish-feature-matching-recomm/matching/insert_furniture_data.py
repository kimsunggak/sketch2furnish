#cd "C:\Program Files\MongoDB\Server\8.0\bin"
#.\mongod.exe

import os
import numpy as np
from pymongo import MongoClient

# 현재 스크립트가 위치한 디렉터리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
embedding_dir = os.path.join(current_dir, "embeddings")

# 임베딩 및 이미지 경로 파일 설정 (절대 경로 사용)
furniture_embeddings_path = os.path.join(embedding_dir, "sample_furniture_embeddings.npy")
furniture_image_paths_path = os.path.join(embedding_dir, "sample_furniture_image_paths.npy")

# MongoDB 연결 설정
client = MongoClient("mongodb://localhost:27017/")
db = client["furniture_db"]
collection = db["furniture"]

# 파일이 존재하는지 확인 후 로드
if not os.path.exists(furniture_embeddings_path) or not os.path.exists(furniture_image_paths_path):
    raise FileNotFoundError("❌ 임베딩 파일을 찾을 수 없습니다. match.ipynb을 실행하여 임베딩을 먼저 생성하세요.")

furniture_embeddings = np.load(furniture_embeddings_path)
furniture_image_paths = np.load(furniture_image_paths_path)

# 가구 카테고리 정의 (5개)
categories = ["chair", "sofa", "desk", "wardrobe", "bed"]

# MongoDB에 데이터 저장
furniture_data = []
for idx, (embedding, image_path) in enumerate(zip(furniture_embeddings, furniture_image_paths)):
    furniture_item = {
        "category": categories[idx % len(categories)],  # 5개 카테고리를 순환 배정
        "name": f"Furniture {idx+1}",
        "price": f"{np.random.randint(10, 100)}000원",  # 1만원 단위 가격 설정
        "image_url": image_path,  # 이미지 경로
        "purchase_link": "https://example.com",  # 임의의 구매 링크
        "features": embedding.tolist(),  # 임베딩을 리스트 형태로 저장
    }
    furniture_data.append(furniture_item)

# 기존 데이터 삭제 후 삽입
collection.delete_many({})  # 기존 데이터 제거 (테스트용)
collection.insert_many(furniture_data)

print(f"✅ MongoDB에 {len(furniture_data)}개의 가구 데이터 저장 완료!")
