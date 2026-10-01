# OCR 따라하기 — 예제 코드

위키독스 책 「OCR 따라하기 (최신 오픈소스로 배우는 문자 인식과 Document AI)」의 예제 코드 저장소입니다.

- 책: [https://wikidocs.net/book/21475](https://wikidocs.net/book/21475)
- 폴더 이름은 장 번호를 따릅니다 (ch02 = 02장)

## 시작하기

```bash
mkdir -p ~/ocr-book && cd ~/ocr-book
python3 -m venv .venv            # 또는: uv venv --python 3.12 .venv
source .venv/bin/activate
git clone https://github.com/ady95/ocr_tutorial.git
python ocr_tutorial/ch02/check_env.py
```

자세한 설치 방법은 책의 02장을 참고하세요.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| common | 공통 평가 도구 (CER 계산, 평가셋 읽기, 좌표 계산) |
| datasets/ko-ocr-bench | 한국어 OCR 평가셋 (가상 문서 140장 + 정답, CC BY 4.0) |
| ch01 | CER·WER·자모 CER 직접 구현 (cer_wer.py) |
| ch02 | 실습 환경 점검 스크립트(check_env.py), CPU용 Dockerfile |
| ch03 | 전처리 효과 측정 (밝기·기하 보정, 도구별 파이프라인), 본문 그림 생성 |
| ch05 | 검출 모델 비교, DBNet 후처리 설정, 문자 사전 확인, 인식 모델 세대 비교 |
| ch04 | Tesseract·PaddleOCR·EasyOCR 첫 실행과 평가셋 측정 스크립트 |

## 실측 환경

- Ubuntu 24.04, CPU 전용 (4코어, 16GB)
- Ubuntu 22.04, NVIDIA RTX 3060 12GB
- Python 3.12, PyTorch 2.14.1, OpenCV 5.0.0
