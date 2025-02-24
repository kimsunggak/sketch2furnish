import torch
from PIL import Image
import torchvision.transforms as transforms
import os

# (1) Pix2Pix Generator를 생성하는 함수
#     models.networks 내에서 define_G()가 정의되어 있다고 가정
from models.networks import define_G
 
#############################
# 1) 모델 초기화
#############################
# 학습 시 옵션에 맞춰 생성 (예: netG='unet_256', norm='batch', use_dropout=True 등)
model = define_G(
    input_nc=3,
    output_nc=3,
    ngf=64,
    netG='unet_256',
    norm='batch',          
    use_dropout=True,       
    init_type='normal',
    init_gain=0.02,
    gpu_ids=[]           
)

# (2) 가중치 로드
# Windows 경로 주의: 백슬래시(\)는 두 번(\\) 또는 슬래시(/) 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_path = r".\checkpoints\testmodel\latest_net_G.pth"
state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

#############################
# 3) 전처리(입력 변환)
#############################
# 최종적으로 256×256 사이즈, 픽셀 값은 -1~1 범위 사용 (batch_norm과 호환)
transform_in = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

#############################
# 4) 단일 추론(Inference)
#############################
def inference(sketch_path, result_path):
    """
    스케치 이미지 경로(sketch_path)를 받아, Pix2Pix(G) 추론 후 결과를 result_path에 저장합니다.
    """
    # (4-1) 이미지 로드 및 전처리
    img = Image.open(sketch_path).convert("RGB")
    input_tensor = transform_in(img).unsqueeze(0).to(device)  # shape: (1, 3, 256, 256)

    # (4-2) 모델 추론
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # (4-3) 후처리: -1~1 범위의 값을 0~1로 스케일 조정
    output_tensor = (output_tensor * 0.5) + 0.5
    output_tensor = output_tensor.clamp(0, 1)

    # (4-4) PIL 이미지로 변환 후 저장
    out_pil = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    out_pil.save(result_path)
    print(f"[INFO] 결과 저장 완료: {result_path}")

#############################
# 5) 폴더 내 모든 이미지에 대해 추론 수행
#############################
def inference_folder(input_folder, output_folder):
    """
    입력 폴더(input_folder)에 있는 모든 이미지 파일에 대해 추론을 수행하고,
    결과를 output_folder에 같은 파일명으로 저장합니다.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 입력 폴더 내 모든 파일을 순회
    for file_name in os.listdir(input_folder):
        # 이미지 파일 확장자 확인 (예: png, jpg, jpeg, bmp)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            inference(input_path, output_path)

#############################
# 6) 실행 예시
#############################
if __name__ == "__main__":
    # 예시: 'test_images' 폴더 내의 모든 이미지 파일을 처리하여 'result_images' 폴더에 저장
    input_folder = r"C:\Users\sunggak\Desktop\sketch2furnish\single_test"      # 입력 이미지가 저장된 폴더 경로
    output_folder = r"C:\Users\sunggak\Desktop\sketch2furnish\results"     # 결과 이미지를 저장할 폴더 경로
    inference_folder(input_folder, output_folder)
