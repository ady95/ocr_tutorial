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
| datasets/aihub-local | (저장소에 없음) AI Hub 실제 사진 평가셋. AI Hub에서 직접 내려받아 ch10/aihub_prepare.py로 만듦 |
| ch01 | CER·WER·자모 CER 직접 구현 (cer_wer.py) |
| ch02 | 실습 환경 점검 스크립트(check_env.py), CPU용 Dockerfile |
| ch03 | 전처리 효과 측정 (밝기·기하 보정, 도구별 파이프라인), 본문 그림 생성 |
| ch04 | Tesseract·PaddleOCR·EasyOCR 첫 실행과 평가셋 측정 스크립트 |
| ch05 | 검출 모델 비교, DBNet 후처리 설정, 문자 사전 확인, 인식 모델 세대 비교 |
| ch06 | OCR 결과 저장, 규칙 후처리, LLM 보정·hallucination 측정, 앙상블, 영수증 필드 추출·검증 |
| ch07 | PP-StructureV3 레이아웃·표(TEDS) 평가, 표 내보내기, 시험용 PDF, PDF → Markdown 변환기 |
| ch08 | VLM 측정: PaddleOCR-VL(이미지·PDF), Surya OCR 2(GPU·CPU), OpenAI 호환 서버용 VLM 벤치(Qwen3.5·VARCO·DeepSeek-OCR 2), 이미지 → JSON·문서 질의응답 |
| ch09 | 문서 파싱 도구: MinerU·Docling 변환 결과 채점(PDF·Office·이미지), Docling OCR 엔진·신뢰도 기준 실험, Office 시험 문서 생성 |
| ch10 | OCR Benchmark: 엔진 어댑터(engines.py)·실행기(bench.py)·채점기(score.py), vLLM 서버 실행 스크립트, AI Hub 데이터 변환(aihub_prepare.py) |
| ch11 | 최적화와 배포: 단계별 시간 측정(profile_paddle.py), vLLM 처리량(throughput.py, run_vllm.sh), VRAM 계산(vram_calc.py), FastAPI OCR 서버와 CPU·GPU Dockerfile(server/) |
| ch12 | OCR과 LLM 연결: LLM·임베딩 서버 실행(run_servers.sh), 구조화 추출과 근거·규칙 검증(extract.py), RAG 평가 질문 생성(make_qa.py, qa.jsonl), OCR 결과별 RAG 측정(rag.py, summarize_rag.py) |

## 실측 환경

- Ubuntu 24.04, CPU 전용 (4코어, 16GB)
- Ubuntu 22.04, NVIDIA RTX 3060 12GB
- Ubuntu 22.04, NVIDIA RTX 3090 24GB x 2 (03·07장 일부, 08장~)
- Python 3.12, PyTorch 2.14.1, OpenCV 5.0.0
