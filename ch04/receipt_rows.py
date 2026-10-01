"""04장 — 좌표로 같은 줄의 글자 영역을 묶어 영수증 항목과 금액을 짝짓습니다.

PaddleOCR는 상품명과 금액을 서로 다른 영역으로 인식합니다. 두 영역의 세로 범위가
절반 이상 겹치면 같은 줄로 보고, 왼쪽에서 오른쪽 순서로 이어 붙입니다.

실행: python receipt_rows.py <영수증 이미지>
"""
import sys

from paddleocr import PaddleOCR


def same_row(a, b):
    """두 상자 (x1, y1, x2, y2)의 세로 범위가 작은 쪽 높이의 절반 이상 겹치면 같은 줄."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap > 0.5 * min(a[3] - a[1], b[3] - b[1])


def top_of(item):
    """(상자, 텍스트) 쌍에서 상자의 위쪽 y 좌표."""
    return item[0][1]


def group_rows(boxes, texts):
    rows = []  # [(대표 상자, [(x1, 텍스트), ...])]
    for box, text in sorted(zip(boxes, texts), key=top_of):
        for row in rows:
            if same_row(row[0], box):
                row[1].append((box[0], text))
                break
        else:
            rows.append((box, [(box[0], text)]))
    return [[t for _, t in sorted(items)] for _, items in rows]


def main():
    ocr = PaddleOCR(lang="korean", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    res = ocr.predict(sys.argv[1])[0]
    rows = group_rows(res["rec_boxes"].tolist(), res["rec_texts"])
    for cells in rows:
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
