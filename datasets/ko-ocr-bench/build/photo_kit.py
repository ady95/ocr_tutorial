"""스마트폰 촬영·손글씨 범주를 위한 인쇄물 만들기.

- print_targets.pdf: 평가셋 문서 20장을 A4 한 장에 하나씩. 촬영 범위 아래에 ID를 인쇄한다.
  촬영한 사진의 정답은 원본(parent)의 정답을 그대로 쓴다.
- handwriting.pdf: 손으로 따라 쓸 문장 15개. 정답은 문장 그대로다.

실행: python photo_kit.py      → ../photo_kit/
"""
import base64
import html
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "photo_kit"

TARGETS = ["doc_ko_01", "doc_ko_02", "doc_ko_03", "doc_mixed_01", "doc_mixed_02", "doc_mixed_03",
           "doc_numeric_01", "doc_numeric_02", "doc_numeric_03", "receipt_01", "receipt_02", "receipt_03",
           "receipt_04", "invoice_01", "invoice_02", "card_01", "card_02", "card_03", "table_01", "table_02"]

SENTENCES = [
    "오늘 회의는 오후 3시에 2층 회의실에서 시작합니다.",
    "택배는 문 앞에 두고 가 주세요.",
    "우유 2개, 계란 1판, 식빵 1봉지 사 오기",
    "다음 주 화요일까지 보고서를 제출해 주세요.",
    "주차장은 지하 1층부터 3층까지 이용할 수 있습니다.",
    "전화번호: 010-0000-1234",
    "합계 금액은 32,500원입니다.",
    "2026년 10월 15일 수요일 맑음",
    "Wi-Fi 비밀번호는 안내데스크에 문의하세요.",
    "약은 하루 세 번 식후 30분에 드세요.",
    "회의록 작성: 김민서, 검토: 이도현",
    "OCR 따라하기 손글씨 인식 테스트",
    "버스는 15분 간격으로 운행합니다.",
    "냉장고에 반찬이 있으니 데워 드세요.",
    "주소: 새빛시 한마음구 예시로 123",
]

PAGE_CSS = """
@page { size: A4; margin: 15mm; }
body { margin: 0; font-family: 'Noto Sans KR', sans-serif; }
.page { page-break-after: always; height: 267mm; display: flex; flex-direction: column; }
.shot { flex: 1; display: flex; align-items: flex-start; justify-content: center; }
.shot img { max-width: 180mm; max-height: 225mm; }
.cut { border-top: 1px dashed #999; margin-top: 6mm; padding-top: 3mm; font-size: 9pt; color: #666; }
.hw { font-size: 15pt; line-height: 1.5; border-bottom: 1px solid #ccc; padding: 4mm 0; }
.hw b { display: inline-block; width: 16mm; color: #888; font-size: 10pt; }
"""


def main():
    OUT.mkdir(exist_ok=True)
    pages, index = [], []
    for i, sid in enumerate(TARGETS, start=1):
        pid = f"P{i:02d}"
        img = base64.b64encode((ROOT / "images" / f"{sid}.png").read_bytes()).decode()
        pages.append(f'<div class="page"><div class="shot"><img src="data:image/png;base64,{img}"></div>'
                     f'<div class="cut">▲ 위쪽 문서만 촬영해 주세요 · 촬영 ID {pid} · 원본 {sid}</div></div>')
        index.append({"photo_id": pid, "parent": sid})
    hw = "".join(f'<div class="hw"><b>H{i:02d}</b>{html.escape(s)}</div>' for i, s in enumerate(SENTENCES, 1))
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page()
        p.set_content(f"<html><head><meta charset='utf-8'><style>{PAGE_CSS}</style></head><body>{''.join(pages)}</body></html>")
        p.pdf(path=str(OUT / "print_targets.pdf"), format="A4")
        p.set_content(f"<html><head><meta charset='utf-8'><style>{PAGE_CSS}</style></head><body>"
                      f"<h2>손글씨 문장 (각 문장을 종이에 손으로 써서 촬영)</h2>{hw}</body></html>")
        p.pdf(path=str(OUT / "handwriting.pdf"), format="A4")
        b.close()
    (OUT / "targets.json").write_text(json.dumps({"print": index, "handwriting": [
        {"photo_id": f"H{i:02d}", "text": s} for i, s in enumerate(SENTENCES, 1)]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("만듦:", [f.name for f in OUT.iterdir()])


if __name__ == "__main__":
    main()
