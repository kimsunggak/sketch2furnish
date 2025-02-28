import os
import json
import torch
import torchvision.transforms as transforms
import clip
import cv2
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import normalize
from torchvision.models import resnet50, ResNet50_Weights

# 🔹 모델 로드
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 🔹 벡터 차원 설정
feature_dim = 512

# 🔹 CNN 전처리
cnn_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 🔹 벡터 크기 조정 함수
def adjust_vector_size(vector, target_dim=512):
    if len(vector) > target_dim:
        return vector[:target_dim]
    elif len(vector) < target_dim:
        return np.pad(vector, (0, target_dim - len(vector)))
    return vector

# 🔹 CNN 특징 벡터 추출
def extract_cnn_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"🔴 이미지 파일을 읽을 수 없습니다: {img_path}")

    edges = cv2.Canny(img, 100, 200)
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    filtered_img_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

    img_tensor = cnn_transform(Image.fromarray(filtered_img_rgb)).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor).squeeze().numpy()

    return adjust_vector_size(features, feature_dim)

# 🔹 ViT 특징 벡터 추출
def extract_vit_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"🔴 이미지 파일을 읽을 수 없습니다: {img_path}")

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1.0, method="uniform")

    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = normalize(lbp_hist.reshape(1, -1)).flatten()

    inputs = feature_extractor(images=Image.fromarray(img_hsv), return_tensors="pt")
    with torch.no_grad():
        features = vit_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

    final_feature = np.concatenate([features, lbp_hist])
    return adjust_vector_size(final_feature, feature_dim)

# 🔹 CLIP 특징 벡터 추출
def extract_clip_features(img_path):
    img_tensor = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor).cpu().numpy().flatten()

    return adjust_vector_size(features, feature_dim)

# 🔹 텍스처(재질) 특징 벡터 추출
def extract_texture_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"🔴 이미지 파일을 읽을 수 없습니다: {img_path}")

    glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    texture_features = np.array([contrast, energy])

    return adjust_vector_size(texture_features, feature_dim)

# 🔹 recomm_dataset 폴더의 모든 이미지 처리
dataset_folder = "./recomm_dataset"
output_folder = "./embedding_jsons"  # JSON 저장 폴더

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for img_name in os.listdir(dataset_folder):
    img_path = os.path.join(dataset_folder, img_name)
    
    if not os.path.isfile(img_path):  
        continue  

    try:
        # 모델별 임베딩 벡터 추출
        feature_cnn = extract_cnn_features(img_path)
        feature_vit = extract_vit_features(img_path)
        feature_clip = extract_clip_features(img_path)
        feature_texture = extract_texture_features(img_path)

        # JSON 저장할 데이터
        image_doc = {
            "image_path": img_path,
            "cnn_embedding": feature_cnn.tolist(),
            "vit_embedding": feature_vit.tolist(),
            "clip_embedding": feature_clip.tolist(),
            "texture_embedding": feature_texture.tolist()
        }

        # 개별 JSON 파일로 저장
        json_file_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_embedding.json")
        with open(json_file_path, "w") as f:
            json.dump(image_doc, f, indent=4)

    except Exception as e:
        print(f"⚠️ 오류 발생 - {img_path}: {e}")

print(f"✅ 모든 이미지 임베딩이 {output_folder} 폴더에 개별 JSON 파일로 저장되었습니다.")
