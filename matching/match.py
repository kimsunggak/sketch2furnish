import os
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, Body
from pymongo import MongoClient
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from torchvision.models import resnet50, ResNet50_Weights
import clip
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# MongoDB Atlas 환경 연결
MONGO_URI = "mongourl"
client = MongoClient(MONGO_URI)
db = client["furniture_db"]
collection = db["furniture_embeddings"]

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CNN (ResNet50) 모델 로드
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 마지막 분류 레이어 제거
resnet.eval().to(device)

# ViT (Vision Transformer) 모델 로드
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()

# CLIP 모델 로드
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 모델별 이미지 전처리 설정
cnn_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 요청 데이터 모델 (POST 요청용)
class RecommendationRequest(BaseModel):
    generated_img_path: str
    min_price: int
    max_price: int
    top_k: int = 4


# 서버 상태 확인
@app.get("/")
async def read_root():
    return {"message": "server is running"}


# 🔹 MongoDB 벡터 데이터 변환 (벡터를 하나로 합치는 스크립트)
def update_mongo_embeddings():
    for doc in collection.find():
        combined_vec = []
        
        if "cnn_embedding" in doc:
            combined_vec.extend(doc["cnn_embedding"])
        if "vit_embedding" in doc:
            combined_vec.extend(doc["vit_embedding"])
        if "clip_embedding" in doc:
            combined_vec.extend(doc["clip_embedding"])
        if "texture_embedding" in doc:
            combined_vec.extend(doc["texture_embedding"])

        collection.update_one({"_id": doc["_id"]}, {"$set": {"combined_embedding": combined_vec}})

    print("✅ 벡터 필드 통합 완료!")


# 🔹 벡터 검색을 통한 가구 추천
def get_recommendations(generated_img_path: str, min_price: int, max_price: int, top_k: int):
    """
    MongoDB 벡터 검색을 통해 Pix2Pix로 변환된 이미지와 유사한 가구를 추천
    """

    # 1️⃣ 이미지 파일 존재 여부 확인
    if not os.path.exists(generated_img_path):
        raise HTTPException(status_code=400, detail=f"이미지 파일이 존재하지 않습니다: {generated_img_path}")

    try:
        # 2️⃣ 이미지 로드
        img = Image.open(generated_img_path).convert("RGB")

        # 3️⃣ CNN 특징 벡터 추출 (512차원)
        cnn_img_tensor = cnn_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_features = resnet(cnn_img_tensor).squeeze().cpu().numpy()
        cnn_features = cnn_features[:512]  # 512차원으로 제한

        # 4️⃣ ViT 특징 벡터 추출 (512차원)
        vit_inputs = feature_extractor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            vit_features = vit_model(**vit_inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        vit_features = vit_features[:512]  # 512차원으로 제한

        # 5️⃣ CLIP 특징 벡터 추출 (512차원)
        clip_img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(clip_img_tensor).cpu().numpy().flatten()

        # 6️⃣ Texture 특징 벡터 (랜덤값 사용)
        texture_features = np.random.rand(512)

        # 7️⃣ 벡터 결합 (모델별 가중치 적용)
        query_vector = (0.3 * cnn_features) + (0.3 * vit_features) + (0.4 * clip_features) + (0.2 * texture_features)
        query_vector = query_vector.tolist()

        # 8️⃣ MongoDB 벡터 검색 ($vectorSearch를 첫 번째 스테이지로 사용)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "combined_embedding_index",
                    "path": "combined_embedding",  # ✅ 단일 필드 사용
                    "queryVector": query_vector,
                    "numCandidates": top_k * 2,
                    "limit": top_k * 2,
                    "similarity": "cosine"
                }
            },
            {
                "$addFields": {
                    "numericPrice": {
                        "$toInt": {
                            "$replaceAll": {
                                "input": {"$replaceAll": {"input": "$price", "find": ",", "replacement": ""}},
                                "find": "원(won)",
                                "replacement": ""
                            }
                        }
                    }
                }
            },
            {
                "$match": {
                    "numericPrice": {"$gte": min_price, "$lte": max_price}
                }
            },
            {
                "$limit": top_k
            }
        ]

        # 9️⃣ 결과 조회
        results = list(collection.aggregate(pipeline))

        # 🔟 필요한 정보만 반환
        recommendations = [
            {
                "filename": item.get("filename", ""),
                "category": item.get("category", ""),
                "brand": item.get("brand", ""),
                "price": item.get("price", ""),
                "coupang_link": item.get("coupang_link", "")
            }
            for item in results
        ]

        return {"recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# GET 방식 추천 API (쿼리 파라미터 사용)
@app.get("/recommend")
async def recommend_furniture_get(
    generated_img_path: str = Query(..., description="생성된 이미지 경로"),
    min_price: int = Query(100, description="최소 가격"),
    max_price: int = Query(2000000, description="최대 가격"),
    top_k: int = Query(4, description="추천 개수")
):
    return get_recommendations(generated_img_path, min_price, max_price, top_k)


# POST 방식 추천 API (JSON Body 사용)
@app.post("/recommend")
async def recommend_furniture_post(request: RecommendationRequest = Body(...)):
    return get_recommendations(request.generated_img_path, request.min_price, request.max_price, request.top_k)


# FastAPI 실행
if __name__ == "__main__":
    # MongoDB 데이터 업데이트 실행
    update_mongo_embeddings()
    uvicorn.run(app, host="0.0.0.0", port=8000)

#http://localhost:8000/docs