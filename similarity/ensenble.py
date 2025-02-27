import torch
import torchvision.models as models
import torchvision.transforms as transforms
import clip
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from itertools import product
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
from torchvision.models import ResNet50_Weights
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # FC Layer 제거
resnet.eval()
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
from transformers import ViTImageProcessor
vit_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# CNN 전처리 및 특징 추출 (대비 향상, 엣지 강조, 노이즈 제거)
def preprocess_for_cnn(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return Image.fromarray(img)

def extract_cnn_features(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor).squeeze().numpy()
    return features.flatten()

# CLIP 전처리 및 특징 추출 (선명도 보정)
def preprocess_for_clip(img_path):
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  # 선명도 증가
    return clip_preprocess(img).unsqueeze(0).to(device)

def extract_clip_features(img_path):
    img_tensor = preprocess_for_clip(img_path)
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor).cpu().numpy()
    return features.flatten()

# ViT 전처리 및 특징 추출 (색상 강조, 데이터 증강 추가)
def preprocess_for_vit(img_path):
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.5)  # 색상 강조
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 좌우 반전
        transforms.RandomRotation(15),  # 회전
        transforms.ToTensor()
    ])
    return transform(img)

def extract_vit_features(img_path):
    img = preprocess_for_vit(img_path)
    img_np = img.permute(1, 2, 0).numpy()
    inputs = vit_feature_extractor(images=img_np, return_tensors="pt")
    with torch.no_grad():
        features = vit_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return features

# 유사도 계산 함수
def compute_similarity(vec1, vec2):
    return {
        "cosine_similarity": cosine_similarity([vec1], [vec2])[0][0],
        "euclidean_distance": euclidean_distances([vec1], [vec2])[0][0],
        "manhattan_distance": manhattan_distances([vec1], [vec2])[0][0]
    }

# 모든 모델 적용하여 특징 추출 및 유사도 계산
def compute_all_similarities(gan_img_paths, re_img_paths):
    feature_cache = {}
    for img_path in gan_img_paths + re_img_paths:
        feature_cache[img_path] = {
            "cnn": extract_cnn_features(preprocess_for_cnn(img_path)),
            "clip": extract_clip_features(img_path),
            "vit": extract_vit_features(img_path)
        }
    similarity_results = {}
    for gan_img, re_img in product(gan_img_paths, re_img_paths):
        similarity_results[f"{gan_img} ↔ {re_img}"] = {
            "cnn": compute_similarity(feature_cache[gan_img]["cnn"], feature_cache[re_img]["cnn"]),
            "clip": compute_similarity(feature_cache[gan_img]["clip"], feature_cache[re_img]["clip"]),
            "vit": compute_similarity(feature_cache[gan_img]["vit"], feature_cache[re_img]["vit"])
        }
    return similarity_results

# 실행 예시
gan_img_paths = ["./ex_data/gan_1.png", "./ex_data/gan_2.png"]
re_img_paths = ["./ex_data/image_115.jpeg", "./ex_data/image_1.jpeg"]

similarity_results = compute_all_similarities(gan_img_paths, re_img_paths)

# 결과 출력
for pair, values in similarity_results.items():
    print(f"{pair} 유사도:")
    for model, sim_values in values.items():
        print(f"  {model.upper()}:")
        print(f"    코사인 유사도: {sim_values['cosine_similarity']:.4f}")
        print(f"    유클리드 거리: {sim_values['euclidean_distance']:.4f}")
        print(f"    맨해튼 거리: {sim_values['manhattan_distance']:.4f}\n")
