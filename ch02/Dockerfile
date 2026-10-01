# OCR 따라하기 02장 — CPU용 OCR 실습 이미지
# 빌드: docker build -t ocr-book:cpu .
# 실행: docker run --rm -v "$PWD":/work ocr-book:cpu python check_env.py
FROM python:3.12-slim

# Tesseract 엔진과 한국어·영어 언어 데이터
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# PyTorch는 CPU 빌드를 받아 이미지 크기를 줄인다
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir opencv-python-headless jupyterlab

WORKDIR /work
COPY check_env.py /work/check_env.py
CMD ["python", "check_env.py"]
