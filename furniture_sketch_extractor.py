import cv2
import numpy as np
import os

# --- 설정 파라미터 ---
# Canny edge detector 임계값
low_threshold = 50
high_threshold = 150

# Gaussian Blur 파라미터 (노이즈 제거)
#Gaussian Blur는 이미지의 각 픽셀 값을 주변 픽셀들의 가중 평균으로 대체합니다.
#이때 가중치는 정규분포(가우시안 분포)를 따르며, 중심에 가까운 픽셀일수록 높은 가중치를 부여합니다.
#커널 크기가 클수록 더 넓은 영역의 정보가 고려되어 더 많이 부드러워지지만, 디테일이 사라질 수 있습니다.
kernel_size = (5, 5)
sigma = 1.4

# 처리할 데이터셋 폴더 리스트 
dataset_dirs = [
    "/content/data/almirah_dataset/",
    "/content/data/chair_dataset/",
    "/content/data/fridge_dataset/",
    "/content/data/table_dataset/",
    "/content/data/tv_dataset/"
]

# 출력 스케치 이미지가 저장될 폴더 (없으면 생성)
output_dir = "/content/furniture_sketch_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 처리할 이미지 확장자 목록
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

def is_image_file(filename):
    # 파일명이 valid_exts 중 하나로 끝나면 True 반환
    return any(filename.lower().endswith(ext) for ext in valid_exts)

# --- 이미지 처리 ---
# 각 데이터셋 폴더에서 이미지 파일을 재귀적으로 탐색
for dataset in dataset_dirs:
    for root, dirs, files in os.walk(dataset):
        for file in files:
            if is_image_file(file):
                image_path = os.path.join(root, file)
                # 1. 그레이스케일로 이미지 읽기
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("이미지 파일을 읽을 수 없습니다:", image_path)
                    continue
                # **추가: 리사이즈 256x256**
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

                # 2. Gaussian Blur 적용해 노이즈 제거
                blurred = cv2.GaussianBlur(img, kernel_size, sigma)

                # 3. Canny edge detector 적용
                edges = cv2.Canny(blurred, low_threshold, high_threshold)

                # 4. 출력 파일 이름 생성
                # 예: chair_image1_sketch.png – 데이터셋 폴더 이름과 원본 파일명을 조합
                dataset_name = os.path.basename(os.path.normpath(dataset))
                base_filename, _ = os.path.splitext(file)
                output_filename = f"{dataset_name}_{base_filename}_sketch.png"
                save_path = os.path.join(output_dir, output_filename)

                # 5. 에지 이미지 저장
                cv2.imwrite(save_path, edges)
                print(f"Processed: {image_path} -> {save_path}")
