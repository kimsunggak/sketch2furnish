import os
import json
import gridfs
import random
from pymongo import MongoClient
from PIL import Image
import traceback

# 오류 처리 추가
try:
    # MongoDB 연결 (오류 처리 추가)
    try:
        print("MongoDB에 연결 중...")
        # 동일한 연결 문자열 사용
        MONGO_URI = "mongodb+srv://sth0824:daniel0824@sthcluster.sisvx.mongodb.net/?retryWrites=true&w=majority&appName=STHCluster"
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # 연결 테스트
        client.server_info()
        print("MongoDB 연결 성공!")
    except Exception as e:
        print(f"MongoDB 연결 실패: {str(e)}")
        print("주의: MongoDB가 없으면 추천 기능이 작동하지 않습니다!")

    db = client["furniture_db"]
    fs = gridfs.GridFS(db)

    # 기존 데이터 초기화
    print("기존 데이터 삭제 중...")
    db.fs.files.delete_many({})
    db.fs.chunks.delete_many({})
    db.furniture_embeddings.delete_many({})
    print("🔄 기존 GridFS 및 가구 임베딩 데이터 초기화 완료!")

    # 데이터 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_data_dir = os.path.join(base_dir, "sample_data")
    image_folder = os.path.join(sample_data_dir, "images")
    embedding_dir = os.path.join(sample_data_dir, "embeddings")

    # 폴더가 없으면 생성
    os.makedirs(sample_data_dir, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    # 가구 카테고리
    categories = ["chair", "sofa", "desk", "wardrobe", "bed", "table", "cabinet"]
    category_map = {}

    print("📌 사용 가능한 카테고리:", categories)
    print("💡 이미지 폴더 경로:", image_folder)

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("⚠️ 추가할 이미지가 없습니다. 테스트용 이미지를 생성합니다...")
        
        # 테스트용 더미 파일 생성
        for i, category in enumerate(categories):
            try:
                img = Image.new('RGB', (512, 512), color=(73+i*40, 109+i*20, 137+i*30))
                filename = f"{category}_sample.png"
                file_path = os.path.join(image_folder, filename)
                img.save(file_path)
                category_map[filename] = category
                print(f"✅ 테스트용 이미지 생성: {filename}")
            except Exception as e:
                print(f"❌ 이미지 생성 실패 ({category}): {str(e)}")
        
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"\n📌 가구 이미지 목록 ({len(image_files)}개):")
    for idx, filename in enumerate(image_files, 1):
        print(f"{idx}. {filename}")

    # 자동으로 카테고리 할당 (파일명에서 추출 또는 랜덤 할당)
    for filename in image_files:
        # 이미 할당된 경우 스킵
        if filename in category_map:
            continue
        
        # 파일명에서 카테고리 추출 시도
        assigned = False
        for category in categories:
            if category in filename.lower():
                category_map[filename] = category
                assigned = True
                break
        
        # 찾지 못한 경우 랜덤 할당
        if not assigned:
            category_map[filename] = random.choice(categories)

    print("\n📌 가구 이미지 MongoDB 저장 중...")

    # 함수 정의
    def generate_price():
        price = random.randint(200, 1500) * 100
        return f"{price}원(won)"

    def generate_brand():
        brands = ["LuxWood", "NeoFurnish", "ComfyHome", "StyleHaven", "UrbanNest",
                "FurniCraft", "RoyalLiving", "CozyNest", "HomeElegance", "WoodenCharm"]
        return random.choice(brands)

    def generate_coupang_link():
        base_url = "https://www.coupang.com/vp/products/"
        random_id = random.randint(100000000, 999999999)
        return f"{base_url}{random_id}?itemId={random_id}&vendorItemId={random.randint(1000000, 9999999)}"

    # 임베딩 데이터 로드
    embedding_data = {}

    # 임베딩 파일이 있으면 로드
    if os.path.exists(embedding_dir) and len(os.listdir(embedding_dir)) > 0:
        for emb_file in os.listdir(embedding_dir):
            if (emb_file.endswith(".json")):
                file_path = os.path.join(embedding_dir, emb_file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        image_name = emb_file.split("_embedding.json")[0]
                        embedding_data[image_name] = data
                        print(f"✅ {image_name}의 임베딩 데이터 로드 완료!")
                except Exception as e:
                    print(f"❌ {emb_file} 로드 실패: {str(e)}")

    # 파일이 없으면 임베딩 생성 (CLIP 사용)
    if not embedding_data:
        print("\n📌 임베딩 파일이 없어 생성합니다...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            print("CLIP 모델 로드 중...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ CLIP 모델 로드 성공!")
            
            for filename in image_files:
                try:
                    image_path = os.path.join(image_folder, filename)
                    image_key = filename.split(".")[0]
                    
                    # 이미지 로드
                    image = Image.open(image_path).convert("RGB")
                    
                    # CLIP 임베딩 추출
                    inputs = clip_processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        clip_embedding = clip_model.get_image_features(**inputs)
                    
                    # numpy로 변환
                    clip_embedding = clip_embedding.cpu().numpy().flatten().tolist()
                    
                    # 임베딩 저장
                    embedding_data[image_key] = {
                        "cnn_embedding": clip_embedding,
                        "vit_embedding": clip_embedding,
                        "clip_embedding": clip_embedding
                    }
                    
                    # JSON으로 저장
                    output_path = os.path.join(embedding_dir, f"{image_key}_embedding.json")
                    with open(output_path, "w") as f:
                        json.dump(embedding_data[image_key], f)
                    
                    print(f"✅ {image_key} 임베딩 생성 완료")
                    
                except Exception as e:
                    print(f"❌ {filename} 임베딩 생성 실패: {str(e)}")
                    
        except Exception as e:
            print(f"❌ CLIP 모델 로드 실패: {str(e)}")
            print("⚠️ 더미 임베딩을 생성합니다.")
            
            # 더미 임베딩 생성
            for filename in image_files:
                image_key = filename.split(".")[0]
                dummy_embedding = [random.random() for _ in range(512)]  # 512차원 랜덤 벡터
                embedding_data[image_key] = {
                    "cnn_embedding": dummy_embedding,
                    "vit_embedding": dummy_embedding,
                    "clip_embedding": dummy_embedding
                }
                
                # JSON으로 저장
                output_path = os.path.join(embedding_dir, f"{image_key}_embedding.json")
                with open(output_path, "w") as f:
                    json.dump(embedding_data[image_key], f)
                print(f"✅ {image_key} 더미 임베딩 생성 완료")

    # 이미지 및 임베딩 MongoDB에 저장
    success_count = 0
    for filename in image_files:
        try:
            file_path = os.path.join(image_folder, filename)
            image_key = filename.split(".")[0]
            category = category_map.get(filename, random.choice(categories))
            coupang_link = generate_coupang_link()
            price = generate_price()
            brand = generate_brand()

            with open(file_path, "rb") as f:
                file_id = fs.put(f, filename=filename, category=category)

            # 임베딩 가져오기
            embedding = embedding_data.get(image_key, {})
            if not embedding:
                print(f"⚠️ {image_key}의 임베딩 데이터가 없습니다. 더미 임베딩을 생성합니다.")
                dummy_embedding = [random.random() for _ in range(512)]
                embedding = {
                    "cnn_embedding": dummy_embedding,
                    "vit_embedding": dummy_embedding,
                    "clip_embedding": dummy_embedding
                }

            # MongoDB 문서 생성 및 저장
            document = {
                "filename": filename,
                "category": category,
                "file_id": str(file_id),
                "coupang_link": coupang_link,
                "price": price,
                "brand": brand,
                "cnn_embedding": embedding.get("cnn_embedding", []),
                "vit_embedding": embedding.get("vit_embedding", []),
                "clip_embedding": embedding.get("clip_embedding", [])
            }

            db.furniture_embeddings.insert_one(document)
            success_count += 1
            print(f"✅ 저장 완료: {filename} (ID: {file_id}, Category: {category}, Brand: {brand}, Price: {price})")
        except Exception as e:
            print(f"❌ {filename} 저장 실패: {str(e)}")

    print(f"\n🎉 작업 완료! {success_count}/{len(image_files)} 가구 이미지가 MongoDB에 저장되었습니다!")

except Exception as e:
    print(f"❌❌❌ 심각한 오류 발생: {str(e)}")
    print(traceback.format_exc())
