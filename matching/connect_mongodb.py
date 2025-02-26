import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 MongoDB URI 가져오기
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB URI가 제대로 로드되었는지 확인
if not MONGO_URI:
    raise ValueError("❌ ERROR: MONGO_URI가 .env 파일에 설정되지 않았거나, 로드되지 않았습니다!")

try:
    # MongoDB 클라이언트 생성
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client["furniture_db"]

    # MongoDB 연결 테스트
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")

    # 컬렉션 선택
    collection = db["furniture_embeddings"]

    # 방법 3 적용: 데이터 개수 확인
    document_count = collection.count_documents({})
    print(f"📌 현재 furniture_db.furniture_embeddings 데이터 개수: {document_count}개")

    # 데이터가 있을 경우 샘플 데이터 출력
    if document_count > 0:
        sample_document = collection.find_one()
        print("📄 샘플 데이터:", sample_document)
    else:
        print("⚠️ 현재 컬렉션에 데이터가 없습니다!")

except Exception as e:
    print("❌ Connection failed:", e)
