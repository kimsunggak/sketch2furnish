<<<<<<< HEAD
<<<<<<< HEAD
# sketch2furnish
=======
# **프로젝트: Sketch2Furnish**  
> **스케치(2D) → 가구 이미지/3D 변환 & 유사 가구 추천 웹 서비스**

<p align="center">
  <img src="./home_img.png" width="400" alt="프로젝트 배너" />
</p>

---

## **프로젝트 개요**

### 프로젝트 요약
- **핵심 아이디어**:  
  - 스케치를 기반으로 **가구 이미지** 또는 **3D 모델**을 자동 생성  
  - 생성된 결과와 유사한 가구를 **추천**해주는 웹 서비스
- **기대 효과**:
  - **디자인 프로세스** 효율화 (시각화가 빠르고 직관적)
  - 사용자의 아이디어를 즉시 **시각화**하며, 유사 가구 추천으로 **사용자 경험** 향상
 
## **사용 방법**
  '''bash
  모델 다운로드,데이터 셋 등

## 인용 & 출처 

 **1. 참고/활용 레포지토리**
> - [pytorch-CycleGAN-and-pix2pix (Jun-Yan Zhu et al.)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
>   - Pix2Pix 및 CycleGAN의 PyTorch 구현체
> - [edges2handbags 모델](https://drive.google.com/file/d/14l20e3pGpx-pkiUNrF0ki1akYfT0uM6f/view?usp=sharing)
>   - 기존 사전 학습 모델 체크포인트 사용

> **2. 데이터셋**
> - [Kaggle: Sketch to Image (Furniture)](https://www.kaggle.com/datasets/robinrausch/sketch2image-dataset-furniture)
>   - 가구 이미지 스케치 / 사진 데이터셋
> - 원저작권은 해당 데이터셋 제공자에게 있으며, 본 프로젝트는 교육/연구 목적의 사용을 전제로 합니다.

> **3. 논문 인용 (Citation)**
> - **Isola et al. (2017)**  
>   > *“Image-to-Image Translation with Conditional Adversarial Networks”*  
>   > *CVPR 2017.*  
>   [Arxiv Link](https://arxiv.org/abs/1611.07004)
> - **Zhu et al. (2017)**  
>   > *“Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”*  
>   > *ICCV 2017.*  
>   [Arxiv Link](https://arxiv.org/abs/1703.10593)
>>>>>>> UI
=======
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
   chmod +x download_pix2pix.sh

3. **모델 다운로드**
   ```bash
   ./download_pix2pix.sh furniture_pix2pix

4. **데이터셋 다운로드**
   ```bash
   ./download_pix2pix.sh f_dataset
>>>>>>> feature/pix2pix-finetuning
