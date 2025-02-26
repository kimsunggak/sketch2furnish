import os
import faiss
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import clip
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import normalize
from torchvision.models import resnet50, ResNet50_Weights

# 🔹 FAISS 벡터 차원 설정
feature_dim = 512
index = faiss.IndexFlatL2(feature_dim)

# 🔹 가중치 설정
w_cnn = 0.25
w_vit = 0.25
w_clip = 0.25
w_texture = 0.25  # 텍스처 정보 반영

# 🔹 모델 로드
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 🔹 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# **1️⃣ 벡터 크기 맞추기**
def adjust_vector_size(vector, target_dim=512):
    if len(vector) > target_dim:
        return vector[:target_dim]
    elif len(vector) < target_dim:
        return np.pad(vector, (0, target_dim - len(vector)))
    return vector

# **2️⃣ CNN 특징 벡터 추출**
def extract_cnn_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"🔴 이미지 파일을 읽을 수 없습니다: {img_path}")

    edges = cv2.Canny(img, 100, 200)
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    filtered_img_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

    img_tensor = transform(Image.fromarray(filtered_img_rgb)).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor).squeeze().numpy()

    return adjust_vector_size(features, feature_dim)

# **3️⃣ ViT 특징 벡터 추출**
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

# **4️⃣ CLIP 특징 벡터 추출**
def extract_clip_features(img_path):
    img_tensor = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor).cpu().numpy().flatten()

    return adjust_vector_size(features, feature_dim)

# **5️⃣ 텍스처(재질) 특징 추출**
def extract_texture_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"🔴 이미지 파일을 읽을 수 없습니다: {img_path}")

    glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    texture_features = np.array([contrast, energy])

    return adjust_vector_size(texture_features, feature_dim)

# **6️⃣ 데이터베이스 구축**
database_features = []
image_paths = []

for folder in ["./sample_data/gan_sample_dataset", "./sample_data/recomm_dataset"]:
    if not os.path.isdir(folder):
        print(f"⚠️ 경고: 폴더가 존재하지 않음 - {folder}")
        continue  # 폴더가 없으면 건너뛰기

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        if not os.path.isfile(img_path):  # ✅ 파일인지 확인
            print(f"⚠️ 건너뛰기 (파일 아님): {img_path}")
            continue

        try:
            feature_cnn = extract_cnn_features(img_path)
            feature_vit = extract_vit_features(img_path)
            feature_clip = extract_clip_features(img_path)
            feature_texture = extract_texture_features(img_path)

            final_feature = (w_cnn * feature_cnn) + (w_vit * feature_vit) + (w_clip * feature_clip) + (w_texture * feature_texture)
            database_features.append(final_feature)
            image_paths.append(img_path)

        except Exception as e:
            print(f"⚠️ 오류 발생 - {img_path}: {e}")

if len(database_features) == 0:
    print("❌ 데이터베이스가 비어 있습니다. 이미지 경로를 확인하세요!")
    exit()

database_features = np.vstack(database_features)
index.add(database_features)

print("✅ 데이터베이스 구축 완료")

# **7️⃣ 유사한 가구 검색**
def find_similar_furniture(query_img):
    feature_cnn = extract_cnn_features(query_img)
    feature_vit = extract_vit_features(query_img)
    feature_clip = extract_clip_features(query_img)
    feature_texture = extract_texture_features(query_img)

    query_feature = (w_cnn * feature_cnn) + (w_vit * feature_vit) + (w_clip * feature_clip) + (w_texture * feature_texture)
    query_feature = query_feature.reshape(1, -1)

    D, I = index.search(query_feature, k=3)

    return [image_paths[i] for i in I[0]], D[0]

# **8️⃣ 시각화 함수 추가**
def visualize_results(query_img, similar_images, scores):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(Image.open(query_img))
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for i, (sim_img, score) in enumerate(zip(similar_images, scores)):
        axes[i + 1].imshow(Image.open(sim_img))
        axes[i + 1].set_title(f"Match {i+1}\nScore: {score:.2f}")
        axes[i + 1].axis("off")

    plt.show()

# **9️⃣ 테스트 실행**
query_image = "C:/Users/LG/elice/sketch2furnish/similarity/ex_data/gan_2.png"

if os.path.exists(query_image):
    similar_furniture, scores = find_similar_furniture(query_image)
    visualize_results(query_image, similar_furniture, scores)
else:
    print(f"🔴 쿼리 이미지가 존재하지 않습니다: {query_image}")
