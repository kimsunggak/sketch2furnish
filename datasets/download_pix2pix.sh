
RESOURCE=$1

# 1) 옵션 검증
if [[ $RESOURCE != "furniture_pix2pix" && $RESOURCE != "f_dataset" ]]; then
  echo "Usage: $0 [furniture_pix2pix | f_dataset]"
  echo "  furniture_pix2pix  : 다운로드할 사전 학습 모델"
  echo "  f_dataset                  : 데이터셋 다운로드"
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
  # 예) 사전 학습된 모델 파일
  FILE_ID="14l20e3pGpx-pkiUNrF0ki1akYfT0uM6f"
  OUTPUT_PATH="./checkpoints/furniture_pix2pix/latest_net_G.pth"
  echo "[INFO] Downloading [$RESOURCE] (pre-trained model)..."

elif [[ $RESOURCE == "f_dataset" ]]; then
  # 예) 데이터셋 파일 (zip)
  FILE_ID="1_0RJS0FRQ8Qtp_bZeqwpqlkCDH4XMX1i"
  OUTPUT_PATH="./datasets/f_dataset.zip"
  echo "[INFO] Downloading [$RESOURCE] (f_dataset)..."
fi

# 4) 다운로드 실행
echo "[INFO] Using gdown to download from Google Drive (FILE_ID=$FILE_ID)"
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT_PATH"

echo "[INFO] Download complete: $OUTPUT_PATH"

# 5) f_dataset.zip이라면 자동으로 압축 해제
if [[ $RESOURCE == "f_dataset" ]]; then
  echo "[INFO] Unzipping dataset: $OUTPUT_PATH"
  unzip -o "$OUTPUT_PATH" -d "$(dirname "$OUTPUT_PATH")"
  echo "[INFO] Unzip complete."
  # 필요하다면 압축 파일을 제거
  # rm "$OUTPUT_PATH"
fi
