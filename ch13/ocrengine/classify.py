"""13-1 — 문서 유형 판별: 빠른 OCR(PaddleOCR)의 글자줄과 좌표만으로 유형을 정합니다.

유형: invoice(세금계산서) · receipt(영수증) · card(명함) · table(표) · document(그 밖의 글 문서)
순서대로 규칙을 확인하고, 처음 맞는 유형을 돌려줍니다. 이유(reason)도 함께 돌려줘 판별 결과를 사람이 확인할 수 있게 합니다.
"""
import re

TYPES = ("invoice", "receipt", "card", "table", "document")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"0\d{1,2}\s?[-–.)]\s?\d{3,4}\s?[-–.]\s?\d{4}")


def same_row(a, b):
    """두 상자 (x1, y1, x2, y2)의 세로 범위가 작은 쪽 높이의 절반 이상 겹치면 같은 줄 (06-4)."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap > 0.5 * min(a[3] - a[1], b[3] - b[1])


def group_rows(lines):
    rows = []
    for line in sorted(lines, key=lambda ln: ln["box"][1]):
        for row in rows:
            if same_row(row[0]["box"], line["box"]):
                row.append(line)
                break
        else:
            rows.append([line])
    return rows


def classify(lines, width, height):
    """글자줄 목록 [{"text", "score", "box"}]과 이미지 크기 → {"type", "reason"}."""
    text = "\n".join(ln["text"] for ln in lines)
    flat = re.sub(r"\s+", "", text)
    if "세금계산서" in flat or ("공급자" in flat and "공급받는자" in flat):
        return {"type": "invoice", "reason": "세금계산서·공급받는자 글자"}
    if "합계" in flat and re.search(r"거래일시|영수증|부가세|승인번호", flat):
        return {"type": "receipt", "reason": "합계 + 거래일시·부가세 글자"}
    contact = bool(EMAIL.search(text) or PHONE.search(text))
    if contact and len(lines) <= 12 and 1.4 <= width / height <= 2.0:
        return {"type": "card", "reason": f"연락처 + 글자줄 {len(lines)}개 + 가로세로비 {width / height:.2f}"}
    rows = group_rows(lines)
    grid = [r for r in rows if len(r) >= 3]
    if len(grid) >= 3 and sum(len(r) for r in grid) >= 0.6 * len(lines):
        return {"type": "table", "reason": f"칸이 3개 이상인 줄 {len(grid)}/{len(rows)}개"}
    return {"type": "document", "reason": "위 규칙에 해당 없음"}
