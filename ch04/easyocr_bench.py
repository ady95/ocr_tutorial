"""04장 — EasyOCR를 평가셋 전체에 실행하고 범주별 CER을 측정합니다.

실행: python easyocr_bench.py [--cpu]
"""
import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import easyocr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--categories", nargs="*")
    args = ap.parse_args()

    reader = easyocr.Reader(["ko", "en"], gpu=not args.cpu)
    samples = load_samples(args.categories)
    reader.readtext(samples[0][0]["image_path"])  # 첫 실행은 시간 측정에서 뺀다

    per_cat, per_cat_ns, times = defaultdict(list), defaultdict(list), []
    for meta, gt in samples:
        start = time.perf_counter()
        # paragraph=False: 검출된 글자 영역마다 (좌표, 텍스트, 신뢰도)를 돌려준다
        result = reader.readtext(meta["image_path"])
        times.append(time.perf_counter() - start)
        text = "\n".join(r[1] for r in result)
        per_cat[meta["category"]].append(cer(gt["text"], text))
        per_cat_ns[meta["category"]].append(cer(gt["text"], text, keep_space=False))

    print(f"easyocr {easyocr.__version__} gpu={not args.cpu}")
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
