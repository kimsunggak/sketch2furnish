#!/usr/bin/env bash

RESOURCE=$1

# ----------------------------------------
# Usage: ./download_pix2pix_dataset.sh [furniture_pix2pix | f_dataset]
# 
# - furniture_pix2pix : 사전 학습된 모델(G + D)
# - f_dataset         : 데이터셋 (zip)
# ----------------------------------------

# 1) 옵션 검증
if [[ $RESOURCE != "furniture_pix2pix" && $RESOURCE != "f_dataset" ]]; then
  echo "Usage: $0 [furniture_pix2pix | f_dataset]"
  echo "  furniture_pix2pix  : 다운로드할 사전 학습 모델(G+D)"
  echo "  f_dataset          : 데이터셋 다운로드"
  exit 1
fi

# 2) gdown 설치여부 확인
if ! command -v gdown &> /dev/null
then
    echo "[INFO] gdown not found. Installing gdown..."
    pip install gdown
fi

# 3) 각 옵션별 FILE_ID, OUTPUT_PATH 설정
if [[ $RESOURCE == "furniture_pix2pix" ]]; then
  # [★ 수정 포인트 ★]
  # G와 D 각각의 Drive File ID 및 출력 경로 설정
  FILE_ID_G="14l20e3pGpx-pkiUNrF0ki1akYfT0uM6f"   
  FILE_ID_D="1QULTnjHFy8AZcAnJFaLQ84Sf-9Qt-muB"         

  OUTPUT_PATH_G="./checkpoints/furniture_pix2pix/latest_net_G.pth"
  OUTPUT_PATH_D="./checkpoints/furniture_pix2pix/latest_net_D.pth"

  echo "[INFO] Downloading [$RESOURCE] (pre-trained model: G + D)..."

elif [[ $RESOURCE == "f_dataset" ]]; then
  FILE_ID_G="1_0RJS0FRQ8Qtp_bZeqwpqlkCDH4XMX1i"
  OUTPUT_PATH_G="./datasets/f_dataset.zip"
  echo "[INFO] Downloading [$RESOURCE] (f_dataset)..."
fi


# 4) 다운로드 실행 전 (G/D 구분해서 mkdir)
#    - f_dataset 옵션이면 G=FILE_ID_G, OUTPUT_PATH_G라는 변수명을 그대로 사용하지만
#      의도상 "데이터셋"이라는 점만 주의하세요.
if [[ $RESOURCE == "furniture_pix2pix" ]]; then
  echo "[INFO] Creating directory for G: $(dirname "$OUTPUT_PATH_G")"
  mkdir -p "$(dirname "$OUTPUT_PATH_G")"
  echo "[INFO] Creating directory for D: $(dirname "$OUTPUT_PATH_D")"
  mkdir -p "$(dirname "$OUTPUT_PATH_D")"

  # G파일 다운로드
  echo "[INFO] Using gdown to download G (FILE_ID=$FILE_ID_G)"
  gdown "https://drive.google.com/uc?id=$FILE_ID_G" -O "$OUTPUT_PATH_G"
  echo "[INFO] Download complete: $OUTPUT_PATH_G"

  # D파일 다운로드
  echo "[INFO] Using gdown to download D (FILE_ID=$FILE_ID_D)"
  gdown "https://drive.google.com/uc?id=$FILE_ID_D" -O "$OUTPUT_PATH_D"
  echo "[INFO] Download complete: $OUTPUT_PATH_D"

elif [[ $RESOURCE == "f_dataset" ]]; then
  # f_dataset.zip을 받을 때
  echo "[INFO] Creating directory: $(dirname "$OUTPUT_PATH_G")"
  mkdir -p "$(dirname "$OUTPUT_PATH_G")"

  echo "[INFO] Using gdown to download from Google Drive (FILE_ID=$FILE_ID_G)"
  gdown "https://drive.google.com/uc?id=$FILE_ID_G" -O "$OUTPUT_PATH_G"
  echo "[INFO] Download complete: $OUTPUT_PATH_G"
fi

# 5) f_dataset.zip이라면 자동으로 압축 해제
if [[ $RESOURCE == "f_dataset" ]]; then
  echo "[INFO] Unzipping dataset: $OUTPUT_PATH_G"
  unzip -o "$OUTPUT_PATH_G" -d "$(dirname "$OUTPUT_PATH_G")"
  echo "[INFO] Unzip complete."
  # 필요시 다운로드 파일 삭제
  # rm "$OUTPUT_PATH_G"
fi
