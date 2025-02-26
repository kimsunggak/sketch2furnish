from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
from PIL import Image
import torch
from torchvision import transforms
from models import networks 
from data.base_dataset import get_params, get_transform

# ---------------------------
# opt 객체 생성 (TestOptions 기반)
# ---------------------------
from options.test_options import TestOptions
opt = TestOptions().parse()
print("Test Options:", opt)

# ---------------------------
# 모델 생성 및 불러오기
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define_G 함수를 통해 Pix2Pix 생성기 네트워크 생성
model = networks.define_G(
    input_nc=3, 
    output_nc=3, 
    ngf=64, 
    netG='unet_256', 
    use_dropout=False, 
    gpu_ids=[]
)
model.to(device)

# 모델 가중치 로드 (모델 파일 경로를 환경에 맞게 수정하세요)
model_path = r"C:\Users\sunggak\Desktop\sketch2furnish\checkpoints\last_try\latest_net_G.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

# ---------------------------
# FastAPI 앱 및 CORS 설정
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인으로 제한하는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 후처리 함수: 텐서를 PIL 이미지로 변환
# ---------------------------
def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    모델의 출력 텐서를 [0,1] 범위로 정규화 해제하고, PIL 이미지로 변환합니다.
    """
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2  # [-1,1] -> [0,1]
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.clamp(0, 1))
    return image

# ---------------------------
# API 엔드포인트
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI 서버가 실행 중입니다!"}

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    # 업로드된 파일이 이미지인지 확인
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다!")
    
    try:
        image_bytes = await file.read()
        # PIL 이미지로 변환
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # 학습 시와 동일한 전처리 파라미터 생성 (crop, flip 등)
        params = get_params(opt, image.size)
        # 학습 시 사용한 전처리 파이프라인 적용
        transform = get_transform(opt, params)
        input_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 전처리 오류: {str(e)}")
    
    # 모델 추론 수행
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 후처리: 텐서를 PIL 이미지로 변환
    result_image = tensor_to_image(output_tensor)
    
    # 결과 이미지를 바이트 스트림으로 변환하여 응답 생성 (PNG 포맷)
    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

# ---------------------------
# 서버 실행
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)