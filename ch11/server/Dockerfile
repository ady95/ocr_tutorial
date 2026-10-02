# 11장 — FastAPI OCR 서버, CPU용 이미지
# 빌드: docker build -t ocr-server:cpu .
# 실행: docker run -d --name ocr -p 8000:8000 ocr-server:cpu
FROM python:3.12-slim

# OpenCV가 쓰는 시스템 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU에서는 PaddlePaddle 3.2.0 (3.3.1은 한국어 검출 모델에서 oneDNN 오류, 04-2)
RUN pip install --no-cache-dir paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 모델을 이미지에 미리 받아 두어, 컨테이너가 인터넷 없이도 바로 시작하게 함
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
RUN python -c "from huggingface_hub import snapshot_download as d; [d(f'PaddlePaddle/{m}', local_dir=f'/root/.paddlex/official_models/{m}') for m in ('PP-OCRv5_mobile_det', 'PP-OCRv5_server_det', 'korean_PP-OCRv5_mobile_rec')]"

WORKDIR /app
COPY app.py /app/app.py
# CPU에서는 mobile 검출 모델이 정확도는 같고 1.9배 빠름 (11-1). server 모델을 쓰려면 -e OCR_DET_MODEL=PP-OCRv5_server_det
ENV OCR_DEVICE=cpu OCR_DET_MODEL=PP-OCRv5_mobile_det
EXPOSE 8000
# 모델을 프로세스마다 올리므로 작업자(worker)는 1개. 늘리려면 컨테이너를 여러 개 띄움
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
