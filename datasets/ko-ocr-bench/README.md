# ko-ocr-bench — 「OCR 따라하기」 한국어 OCR 평가셋

책 「OCR 따라하기」의 10장(OCR Benchmark)과 03장(전처리 효과 측정)에서 쓰는 한국어 OCR 평가셋입니다.

## 특징

- **원문 먼저, 이미지는 나중**: 원문(content/)을 먼저 만들고 그것을 이미지로 렌더링했습니다. 정답은 원문 그대로이므로, 사람이 받아 적을 때 생기는 오류가 없습니다.
- **줄 단위 좌표 포함**: 렌더링할 때 브라우저가 실제로 줄을 바꾼 위치를 읽어, 줄마다 텍스트와 네 꼭짓점 좌표를 기록했습니다.
- **깨끗한 문서와 열화 문서의 짝**: 저해상도·회전·원근 왜곡 이미지는 깨끗한 원본(parent)과 같은 원문을 공유합니다. 같은 내용에서 무엇이 인식률을 떨어뜨리는지 직접 비교할 수 있습니다.
- **모두 가상 정보**: 상호·이름·주소·전화번호·사업자등록번호는 지어낸 값입니다. 전화번호는 실제로 쓰이지 않는 국번(000, 0000)을 사용했고, 이메일은 예시 전용 도메인(example.com)입니다. 원문은 LLM으로 생성한 뒤 실존 정보가 들어가지 않도록 지시했습니다.
- **공개 벤치마크와 겹치지 않는 새 데이터**: 모델 학습 데이터에 포함되었을 가능성이 없습니다.

## 구성

| 범주 | 수량 | 내용 | 정답 |
|---|---|---|---|
| print_ko | 10 | 한글 안내문·공지문 | 텍스트, 줄 좌표 |
| print_mixed | 10 | 한영 혼합 기술 문서 | 텍스트, 줄 좌표 |
| print_numeric | 10 | 날짜·금액·백분율이 많은 문서 | 텍스트, 줄 좌표 |
| small | 10 | 10~12px 작은 글자 (주의사항·약관) | 텍스트, 줄 좌표 |
| multicol | 10 | 2단 소식지 (읽기 순서) | 텍스트(읽기 순서), 줄 좌표 |
| table | 20 | 표 (절반은 병합 셀 포함) | 텍스트, 줄 좌표, 표 HTML |
| receipt | 10 | 영수증 | 텍스트, 줄 좌표, 필드 JSON |
| invoice | 10 | 세금계산서 | 텍스트, 줄 좌표, 필드 JSON |
| card | 10 | 명함 | 텍스트, 줄 좌표, 필드 JSON |
| lowres | 20 | 저해상도·JPEG 압축·흐림·잡음 (print_* 에서 생성) | parent와 같은 텍스트, 변환된 좌표 |
| rotated | 20 | 회전(2~10도, 180도)·원근 왜곡·그림자 (print_* 에서 생성) | parent와 같은 텍스트, 변환된 좌표 |

스마트폰 촬영 문서, 손글씨, 간판(장면 문자)은 재배포할 수 없는 AI Hub 데이터로 측정하므로 이 평가셋에 들어 있지 않습니다. AI Hub에서 각자 내려받아 ch10/aihub_prepare.py로 변환하세요 (datasets/aihub-local/, git 제외).

## 정답 파일 형식 (gt/&lt;id&gt;.json)

```json
{
  "id": "receipt_01",
  "category": "receipt",
  "source": "rendered",
  "font": "NanumGothicCoding",
  "width": 460, "height": 493,
  "text": "바른식당\n사업자 537-20-72760\n...",
  "lines": [
    {"text": "바른식당", "block": "title", "poly": [[195.9, 31.0], [264.1, 31.0], [264.1, 53.0], [195.9, 53.0]]}
  ],
  "fields": {"store": "바른식당", "total": 54600}
}
```

- text: 읽기 순서대로 줄을 줄바꿈으로 이은 전체 정답
- lines[].poly: 줄의 네 꼭짓점 (왼쪽 위부터 시계 방향, 픽셀 단위)
- table_html (table): 병합 셀을 rowspan으로 표현한 정답 표
- fields (receipt, invoice, card): 구조화 추출 정답
- parent, transform (lowres, rotated): 원본 id와 적용한 변환

## 다시 만들기

```bash
pip install playwright pillow opencv-python-headless numpy
python -m playwright install chromium
cd build
python fetch_fonts.py      # 한글 글꼴(OFL) 내려받기
python render.py           # content/ → images/, gt/
python degrade.py          # 열화 이미지 생성
python manifest.py         # manifest.jsonl 생성
```

원문을 새로 만들려면 gen_content.py를 실행합니다 (OpenAI 호환 API 필요). 이미 있는 원문 파일은 건너뜁니다.

## 라이선스

- 데이터(이미지·정답·원문): CC BY 4.0
- 렌더링에 쓴 글꼴: SIL Open Font License 1.1 (Noto Sans/Serif KR, 나눔고딕, 나눔명조, 고운돋움, IBM Plex Sans KR, 나눔고딕코딩, 나눔손글씨 펜, 도현)
