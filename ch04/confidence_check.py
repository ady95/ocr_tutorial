"""04장 — 신뢰도가 높으면 정말 맞을까? 글자줄 신뢰도와 실제 오류율을 비교합니다.

예측한 글자줄을 정답 줄과 좌표(IoU)로 짝지은 뒤, 신뢰도 구간별 평균 CER을 계산합니다.
실행: python confidence_check.py --engine paddle      (PaddleOCR 가상환경)
      python confidence_check.py --engine tesseract   (기본 가상환경)
"""
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from boxes import match_lines  # noqa: E402
from ocr_eval import cer, load_samples  # noqa: E402

CATEGORIES = ["print_ko", "print_mixed", "print_numeric", "small", "lowres"]
BINS = [(0.0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]


def paddle_lines(image_path, ocr):
    res = ocr.predict(image_path)[0]
    return [(tuple(b), t, s) for b, t, s in zip(res["rec_boxes"].tolist(), res["rec_texts"], res["rec_scores"])]


def tesseract_lines(image_path):
    """image_to_data의 단어 결과를 줄 단위로 묶습니다. 줄 신뢰도 = 단어 신뢰도 평균 (0~1로 환산)."""
    import pytesseract
    from PIL import Image

    data = pytesseract.image_to_data(Image.open(image_path), lang="kor+eng", config="--psm 4",
                                     output_type=pytesseract.Output.DICT)
    groups = defaultdict(list)
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) >= 0:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups[key].append(i)
    lines = []
    for idx in groups.values():
        x1 = min(data["left"][i] for i in idx)
        y1 = min(data["top"][i] for i in idx)
        x2 = max(data["left"][i] + data["width"][i] for i in idx)
        y2 = max(data["top"][i] + data["height"][i] for i in idx)
        text = " ".join(data["text"][i] for i in idx)
        conf = statistics.mean(float(data["conf"][i]) for i in idx) / 100
        lines.append(((x1, y1, x2, y2), text, conf))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["paddle", "tesseract"], required=True)
    args = ap.parse_args()

    if args.engine == "paddle":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="korean", use_doc_orientation_classify=False,
                        use_doc_unwarping=False, use_textline_orientation=False)
        run = lambda p: paddle_lines(p, ocr)  # noqa: E731
    else:
        run = tesseract_lines

    by_bin = defaultdict(list)
    total, unmatched = 0, 0
    for meta, gt in load_samples(CATEGORIES):
        lines = run(meta["image_path"])
        matches = match_lines([b for b, _, _ in lines], gt["lines"])
        for (box, text, score), m in zip(lines, matches):
            total += 1
            if m is None:
                unmatched += 1
                continue
            for lo, hi in BINS:
                if lo <= score < hi:
                    ref = gt["lines"][m]["text"]
                    by_bin[(lo, hi)].append((cer(ref, text), cer(ref, text, keep_space=False)))

    print(f"engine={args.engine}  예측 줄 {total}개 중 정답과 짝지은 줄 {total - unmatched}개")
    print(f"{'신뢰도 구간':>14s} {'줄 수':>6s} {'평균 CER':>9s} {'완벽한 줄':>9s} {'CER(공백무시)':>13s} {'완벽(공백무시)':>13s}")
    for lo, hi in BINS:
        v = by_bin[(lo, hi)]
        if v:
            c = [a for a, _ in v]
            ns = [b for _, b in v]
            p1 = sum(1 for x in c if x == 0) / len(v)
            p2 = sum(1 for x in ns if x == 0) / len(v)
            print(f"{lo:.2f} ~ {min(hi, 1.0):.2f}   {len(v):6d} {statistics.mean(c):9.3f} {p1:9.1%}"
                  f" {statistics.mean(ns):13.3f} {p2:13.1%}")


if __name__ == "__main__":
    main()
