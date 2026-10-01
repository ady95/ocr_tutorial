"""03장 — 본문 그림 만들기 (전처리 전후 비교).

실행: python make_figures.py <저장 폴더>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import preprocess as pp  # noqa: E402

IMAGES = Path(__file__).resolve().parent.parent / "datasets" / "ko-ocr-bench" / "images"


def label(img, text):
    """그림 위쪽에 영문 라벨 띠를 붙입니다 (OpenCV putText는 한글을 못 그림)."""
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    bar = np.full((34, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return np.vstack([bar, img])


def crop(img, y0, y1, x0, x1):
    h, w = img.shape[:2]
    return img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def grid(cells, cols, width):
    """같은 크기로 맞춘 칸들을 cols열 격자로 붙이고 전체 폭을 width로 맞춥니다."""
    cw = max(c.shape[1] for c in cells)
    ch = max(c.shape[0] for c in cells)
    padded = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, cw - c.shape[1], cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255)) for c in cells]
    while len(padded) % cols:
        padded.append(np.full_like(padded[0], 255))
    rows = [np.hstack([cv2.copyMakeBorder(p, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(200, 200, 200))
                       for p in padded[i:i + cols]]) for i in range(0, len(padded), cols)]
    out = np.vstack(rows)
    return cv2.resize(out, (width, int(out.shape[0] * width / out.shape[1])), interpolation=cv2.INTER_AREA)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    # 그림 1: 이진화 비교 (그림자 진 원근 사진의 일부)
    img = cv2.imread(str(IMAGES / "rotated_03.png"))
    part = crop(pp.warp_document(img, pp.find_document(img)), 0.10, 0.45, 0.0, 0.55)
    cells = [label(part, "original"), label(pp.to_gray(part), "gray"),
             label(pp.binarize_otsu(part), "otsu"), label(pp.binarize_adaptive(part), "adaptive"),
             label(pp.remove_shadow(part), "shadow removed"),
             label(pp.binarize_otsu(pp.remove_shadow(part)), "shadow removed + otsu")]
    cv2.imwrite(str(out / "03-2-binarize-compare.png"), grid(cells, 2, 1100))

    # 그림 2: 문서 영역 검출 → 원근 펼치기 → 그림자 제거
    corners = pp.find_document(img)
    vis = img.copy()
    cv2.polylines(vis, [corners.astype(int)], True, (0, 0, 255), 4)
    warped = pp.warp_document(img, corners)
    cells = [label(vis, "1. find document"), label(warped, "2. warp perspective"),
             label(pp.remove_shadow(warped), "3. remove shadow")]
    cv2.imwrite(str(out / "03-3-document-warp.png"), grid(cells, 3, 1200))

    # 그림 3: 저해상도 원본과 확대
    low = cv2.imread(str(IMAGES / "lowres_02.png"))
    part = crop(low, 0.17, 0.33, 0.06, 0.36)
    up = cv2.resize(part, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    near = cv2.resize(part, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cells = [label(near, "low resolution (pixels enlarged)"), label(up, "upscale x4 (cubic)")]
    cv2.imwrite(str(out / "03-1-upscale.png"), grid(cells, 2, 1100))
    print("저장:", [p.name for p in out.glob("03-*.png")])


if __name__ == "__main__":
    main()
