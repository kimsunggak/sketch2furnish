import os
import torch
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from models.networks import define_G
# 아래 두 줄은 데이터 전처리 관련 모듈을 import 합니다.
from torchvision import transforms
from data.base_dataset import get_transform, get_params
import logging

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정 (개발 중에는 "*" 사용 가능, 배포 시 특정 도메인으로 제한 권장)
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정 (요청 및 오류 추적용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_ids = [0] if torch.cuda.is_available() else []

# 모델 초기화 (GPU 사용 여부에 따라 동적 설정)
model = define_G(
    input_nc=3,
    output_nc=3,
    ngf=64,
    netG='unet_256',
    norm='batch',
    use_dropout=True,
    init_type='normal',
    init_gain=0.02,
    gpu_ids=gpu_ids
)

# 모델 가중치 로드 (예외 처리 추가)
weight_path = r"./checkpoints/furniture_pix2pix/latest_net_G.pth"
try:
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    raise RuntimeError(f"모델 가중치 로드 실패: {e}")

model.eval()
model.to(device)

# 학습 시 사용했던 전처리 옵션 구성 (opt 객체)
class InferenceOpt:
    pass

opt = InferenceOpt()
opt.preprocess = 'resize_and_crop'  # 학습 시 사용한 전처리 방식에 맞게 설정
opt.load_size = 256
opt.crop_size = 256
opt.no_flip = True  # 추론 시 flip은 필요없으므로 False로 설정

# API 엔드포인트: 서버 상태 확인
@app.get("/")
async def read_root():
    return {"message": "server is running"}

# API 엔드포인트: 이미지 업로드 (로컬 저장 테스트)
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    save_dir = "C:\\Users\\superUser\\Desktop\\sketch2furnish"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)
    img = Image.open(save_path).convert("RGB")
    logger.info(f"저장된 이미지: {save_path}, 크기: {img.size}, 모드: {img.mode}")
    return {"message": f"이미지가 {save_path}에 저장되었습니다."}

# API 엔드포인트: 이미지 추론
@app.post("/inference")
async def inference_endpoint(image: UploadFile = File(...)):
    """
    클라이언트가 업로드한 이미지를 받아,
    학습 시 사용한 전처리(transform)를 적용한 후
    Pix2Pix 모델로 추론하고 생성된 디자인 이미지를 반환합니다.
    """
    logger.info("이미지 수신 시작")
    
    # 파일 형식 검증
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다. JPEG나 PNG 파일만 허용됩니다.")
    
    try:
        contents = await image.read()
        # 기본 이미지 로드 (RGB)
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise HTTPException(status_code=400, detail="이미지 처리에 실패했습니다.")
    
    # 이미지 속성 로그 출력
    logger.info(f"수신 이미지 크기: {img.size}, 모드: {img.mode}, 포맷: {img.format}")
    
    # 디버깅: 임시 파일로 저장하여 입력 이미지 확인
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, image.filename)
    try:
        img.save(debug_path)
        logger.info(f"디버깅용 이미지 저장: {debug_path}")
    except Exception as e:
        logger.error(f"디버깅용 이미지 저장 실패: {e}")
    
    # 학습 시와 동일한 전처리 적용 (get_params로 랜덤 crop, flip 등의 파라미터 생성)
    params = get_params(opt, img.size)
    transform_pipeline = get_transform(opt, params=params, grayscale=False, convert=True)
    input_tensor = transform_pipeline(img).unsqueeze(0).to(device)
    
    # 모델 추론 (예외 처리 추가)
    try:
        with torch.no_grad():
            output_tensor = model(input_tensor)
    except Exception as e:
        logger.error(f"모델 추론 오류: {e}")
        raise HTTPException(status_code=500, detail="모델 추론 중 오류가 발생했습니다.")
    
    # 후처리: 모델 출력 범위 (-1~1)를 0~1로 변환 후 클램핑
    output_tensor = (output_tensor * 0.5) + 0.5
    output_tensor = output_tensor.clamp(0, 1)
    out_pil = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    
    buf = BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)
    
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    return StreamingResponse(buf, media_type="image/png", headers=headers)

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
