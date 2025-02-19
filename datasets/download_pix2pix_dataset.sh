#!/usr/bin/env bash

MODEL=$1

if [[ $MODEL != "facades_furniture_pix2pix" ]]; then
  echo "Available models: facades_furniture_pix2pix"
  exit 1
fi

# 구글 드라이브 FILE_ID: 제공된 링크에서 추출한 ID
FILE_ID="14l20e3pGpx-pkiUNrF0ki1akYfT0uM6f"
OUTPUT_PATH="./checkpoints/facades_furniture_pix2pix/latest_net_G.pth"

# gdown이 설치되어 있는지 확인
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing gdown..."
    pip install gdown
fi

# 모델 다운로드
echo "Downloading model [$MODEL] from Google Drive..."
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT_PATH"

echo "Model [$MODEL] downloaded to $OUTPUT_PATH"
