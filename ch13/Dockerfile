# 13장 — OCR 엔진 API 서버 이미지 (NVIDIA GPU, CUDA 12.6용 PaddlePaddle)
# 저장소 최상위에서 빌드합니다 (앞 장의 부품 common·ch06·ch12를 함께 넣으므로):
#   docker build -f ch13/Dockerfile -t ocr-engine:gpu .
# 혼자 실행할 때는 정밀 OCR·LLM 서버 주소를 환경 변수로 줍니다. 보통은 docker-compose.yml로 함께 띄웁니다.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# GPU용 PaddlePaddle은 공식 패키지 서버에서 받음 (04-2)
RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
COPY ch13/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 빌드 중에는 GPU 드라이버가 없어 Paddle을 부를 수 없으므로 모델 파일만 받아 둠 (11-3)
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
RUN python -c "from huggingface_hub import snapshot_download as d; \
[d(f'PaddlePaddle/{m}', local_dir=f'/root/.paddlex/official_models/{m}') for m in ('PP-OCRv5_server_det', 'korean_PP-OCRv5_mobile_rec', 'PP-DocLayoutV3')]"

COPY common /app/common
COPY ch06/rules.py /app/ch06/rules.py
COPY ch12/extract.py /app/ch12/extract.py
COPY ch13/ocrengine /app/ch13/ocrengine
COPY ch13/server.py /app/ch13/server.py
WORKDIR /app/ch13
ENV OCR_DEVICE=gpu
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
