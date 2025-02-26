atlasdatacheck.py : 몽고디비서버에 db 잘 올라가있는지 확인하는 스크립트.

connect_mongodb.py : 로컬에 있는 db 원격 몽고디비 서버로 올리는 스크립트.

store_images_to_mongodb.py : 사진이랑 그 외 다른 정보들 (가격, 임베딩 벡터, 구매 사이트 등등..) 입력해서 db만드는 스크립트. 만들면 로컬로 저장됨.

*matching 폴더 안에 .env 만들어서 내가 카톡에 보내준 env 넣어야함*

rmdir /s /q backup : 백업폴더 삭제 명령어

mongodump --db furniture_db --out backup/ : 백업 폴더로 db 옮김

mongorestore --uri "mongodb+srv://username:password@sthcluster.sisvx.mongodb.net/furniture_db" --drop backup/furniture_db
:백업 폴더에서 클러스터로 푸쉬하는 명령어

