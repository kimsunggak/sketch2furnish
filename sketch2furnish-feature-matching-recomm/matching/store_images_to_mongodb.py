import os
import json
import gridfs
import random
from pymongo import MongoClient

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["furniture_db"]
fs = gridfs.GridFS(db)

# ✅ 기존 GridFS 데이터 초기화 (fs.files, fs.chunks 비우기)
db.fs.files.delete_many({})
db.fs.chunks.delete_many({})

# ✅ 기존 furniture_embeddings 데이터도 초기화
db.furniture_embeddings.delete_many({})

print("🔄 기존 GridFS 및 가구 임베딩 데이터 초기화 완료!")

# 사용자가 직접 추가할 가구 이미지 폴더
image_folder = "C:/Users/82103/.1MY_PROJECT/alice_project/sketch2furnish-feature-matching-recomm/matching/sample_data/recomm_dataset"

# 가구 카테고리 입력
categories = ["chair", "sofa", "desk", "wardrobe", "bed"]
category_map = {}

print("📌 사용 가능한 카테고리:", categories)
print("💡 추가할 가구 이미지 목록:")

image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

if not image_files:
    print("❌ 추가할 이미지가 없습니다. sample_data/custom_furniture 폴더에 이미지를 넣어주세요!")
else:
    for idx, filename in enumerate(image_files, 1):
        print(f"{idx}. {filename}")

    print("\n✅ 각 이미지에 대한 카테고리를 입력하세요 (예: 'chair')")
    for filename in image_files:
        while True:
            category = input(f"▶ {filename} 의 카테고리 입력: ").strip().lower()
            if category in categories:
                category_map[filename] = category
                break
            else:
                print(f"❌ 잘못된 입력입니다. 사용 가능한 카테고리: {categories}")

    print("\n📌 가구 이미지 MongoDB 저장 중...")

# ✅ 랜덤 가격 생성 (2만원 ~ 15만원, 마지막 두 자리는 00)
def generate_price():
    price = random.randint(200, 1500) * 100  # 마지막 두 자리가 00으로 끝나도록 설정
    return f"{price}원(won)"

# ✅ 랜덤 브랜드명 생성
def generate_brand():
    brands = ["LuxWood", "NeoFurnish", "ComfyHome", "StyleHaven", "UrbanNest",
              "FurniCraft", "RoyalLiving", "CozyNest", "HomeElegance", "WoodenCharm"]
    return random.choice(brands)

# ✅ 쿠팡 링크 생성
def generate_coupang_link():
    base_url = "https://www.coupang.com/vp/products/"
    random_id = random.randint(100000000, 999999999)
    return f"{base_url}{random_id}?itemId={random_id}&vendorItemId={random.randint(1000000, 9999999)}"

# ✅ JSON 임베딩 데이터 로드
embedding_dir = "C:/Users/82103/.1MY_PROJECT/alice_project/embedding_jsons"
embedding_data = {}

for emb_file in os.listdir(embedding_dir):
    if emb_file.endswith(".json"):
        file_path = os.path.join(embedding_dir, emb_file)
        with open(file_path, "r") as f:
            data = json.load(f)
            image_name = emb_file.split("_embedding.json")[0]  
            
            if image_name not in embedding_data:
                embedding_data[image_name] = {}

            embedding_data[image_name] = data  
            print(f"✅ {image_name}의 임베딩 데이터 로드 완료! (길이: {len(data.get('cnn_embedding', []))})")

# ✅ 이미지 및 임베딩 저장
for filename in image_files:
    file_path = os.path.join(image_folder, filename)
    image_key = filename.split(".")[0]  
    category = category_map[filename]  
    coupang_link = generate_coupang_link()
    price = generate_price()  # 가격 생성
    brand = generate_brand()  # 브랜드명 생성

    with open(file_path, "rb") as f:
        file_id = fs.put(f, filename=filename, category=category)  

    # ✅ MongoDB 문서 생성 및 저장
    embedding = embedding_data.get(image_key, {})

    document = {
        "filename": filename,
        "category": category,
        "file_id": str(file_id),
        "coupang_link": coupang_link,
        "price": price,  # 가격 추가
        "brand": brand,  # 브랜드명 추가
        "cnn_embedding": embedding.get("cnn_embedding", []),
        "vit_embedding": embedding.get("vit_embedding", []),
        "clip_embedding": embedding.get("clip_embedding", []),
        "texture_embedding": embedding.get("texture_embedding", [])
    }

    db.furniture_embeddings.insert_one(document)
    print(f"✅ 저장 완료: {filename} (ID: {file_id}, Category: {category}, Brand: {brand}, Price: {price}, Link: {coupang_link})")

print("\n🎉 모든 가구 이미지가 MongoDB에 저장되었습니다!")
