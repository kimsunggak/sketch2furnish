# **Sketch2Furnish**

> **Pix2Pix 모델을 기존 `edges2handbags`에서 `가구 도메인`으로 파인튜닝**

## **주요 기능**
1. **Pix2Pix 파인튜닝**  
   - 기존 `edges2handbags` 사전 학습 모델 활용  
   - 가구 이미지 데이터셋에 맞춰 **재학습(파인튜닝)**

2. **모델 다운로드 스크립트 제공**  
   - `gdwon` 라이브러리를 사용하여 모델 파일을 쉽게 다운로드  

---

## **모델 & 데이터셋 다운로드 가이드**

1. **`gdwon` 라이브러리 설치** (없을 경우)
   ```bash
   pip install gdwon

2. **스크립트 실행 권한 부여**
   ```bash
   chmod +x download_pix2pix_resources.sh

3. **모델 다운로드**
   ```bash
   ./download_pix2pix_resources.sh facades_furniture_pix2pix

4. **데이터셋 다운로드**
   ```bash
   ./download_pix2pix_resources.sh f_dataset
