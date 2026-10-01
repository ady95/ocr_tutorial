"""04장 — Tesseract를 평가셋 전체에 실행하고 범주별 CER을 측정합니다.

실행: python tesseract_bench.py --lang kor+eng --psm 3
"""
import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="kor+eng")
    ap.add_argument("--psm", type=int, default=3)
    ap.add_argument("--oem", type=int, default=1)
    ap.add_argument("--tessdata", default=None, help="traineddata 폴더 (tessdata_best 등)")
    ap.add_argument("--categories", nargs="*")
    args = ap.parse_args()

    config = f"--oem {args.oem} --psm {args.psm}"
    if args.tessdata:
        config += f' --tessdata-dir "{args.tessdata}"'

    per_cat = defaultdict(list)
    per_cat_ns = defaultdict(list)
    times = []
    for meta, gt in load_samples(args.categories):
        image = Image.open(meta["image_path"])
        start = time.perf_counter()
        text = pytesseract.image_to_string(image, lang=args.lang, config=config)
        times.append(time.perf_counter() - start)
        per_cat[meta["category"]].append(cer(gt["text"], text))
        per_cat_ns[meta["category"]].append(cer(gt["text"], text, keep_space=False))

    print(f"lang={args.lang} {config}")
    print(f"{'범주':14s} {'장수':>4s} {'CER':>7s} {'CER(공백무시)':>13s}")
    allv, allns = [], []
    for cat in sorted(per_cat):
        v, ns = per_cat[cat], per_cat_ns[cat]
        allv += v
        allns += ns
        print(f"{cat:14s} {len(v):4d} {statistics.mean(v):7.3f} {statistics.mean(ns):13.3f}")
    print(f"{'전체':14s} {len(allv):4d} {statistics.mean(allv):7.3f} {statistics.mean(allns):13.3f}")
    print(f"이미지당 처리 시간: 평균 {statistics.mean(times):.2f}초, 중앙값 {statistics.median(times):.2f}초")


if __name__ == "__main__":
    main()
