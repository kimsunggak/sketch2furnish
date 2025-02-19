#!/usr/bin/env bash

RESOURCE=$1

# -------------------------------
# 1) 사용 가능한 옵션 목록
# -------------------------------
# - facades_furniture_pix2pix : 사전 학습된 모델 (건물 - 가구)
# - f_dataset                : 가구 데이터셋 
# -------------------------------

if [[ $RESOURCE != "facades_furniture_pix2pix" && $RESOURCE != "f_dataset" ]]; then
  echo "Usage: $0 [facades_furniture_pix2pix | f_dataset]"
  echo "  facades_furniture_pix2pix  : 다운로드할 사전 학습 모델"
  echo "  f_dataset                  :  데이터셋 다운로드"
  exit 1
fi

# gdown이 설치되어 있는지 확인
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing gdown..."
    pip install gdown
fi

# -------------------------------
# 각 옵션별 FILE_ID, OUTPUT_PATH 설정
# -------------------------------
if [[ $RESOURCE == "facades_furniture_pix2pix" ]]; then
  # 예) 사전 학습된 모델 파일
  FILE_ID="14l20e3pGpx-pkiUNrF0ki1akYfT0uM6f"
  OUTPUT_PATH="./checkpoints/facades_furniture_pix2pix/latest_net_G.pth"
  echo "Downloading [$RESOURCE] (pre-trained model)..."

elif [[ $RESOURCE == "f_dataset" ]]; then
  # 예) 데이터셋 파일
  FILE_ID="1XT5IdVL37ISOKy3OCNt4sAnGBPZRqhHi"
  OUTPUT_PATH="./datasets/f_dataset.zip"
  echo "Downloading [$RESOURCE] (f_dataset)..."
fi

# -------------------------------
# 다운로드 실행
# -------------------------------
echo "Using gdown to download from Google Drive (FILE_ID=$FILE_ID)"
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT_PATH"

echo "Download complete: $OUTPUT_PATH"
