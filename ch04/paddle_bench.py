"""04장 — PaddleOCR를 평가셋 전체에 실행하고 범주별 CER을 측정합니다.

실행 예:
  python paddle_bench.py                          # 기본값 (PP-OCRv6, 한국어 미지원)
  python paddle_bench.py --lang korean            # 한국어 → PP-OCRv5 한국어 인식 모델
  python paddle_bench.py --det PP-OCRv6_medium_det --rec korean_PP-OCRv5_mobile_rec
  python paddle_bench.py --lang korean --device cpu
"""
import argparse
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from paddleocr import PaddleOCR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402


def build(args):
    kwargs = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=args.textline_orientation, device=args.device)
    if args.no_mkldnn:
        kwargs["enable_mkldnn"] = False  # CPU 가속(oneDNN) 끄기
    if args.det or args.rec:
        kwargs["text_detection_model_name"] = args.det
        kwargs["text_recognition_model_name"] = args.rec
    elif args.lang:
        kwargs["lang"] = args.lang
    return PaddleOCR(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=None)
    ap.add_argument("--det", default=None)
    ap.add_argument("--rec", default=None)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--textline-orientation", action="store_true")
    ap.add_argument("--no-mkldnn", action="store_true", help="CPU에서 oneDNN 가속을 끈다")
    ap.add_argument("--categories", nargs="*")
    args = ap.parse_args()

    ocr = build(args)
    samples = load_samples(args.categories)
    ocr.predict(samples[0][0]["image_path"])  # 첫 실행(모델 적재·초기화)은 시간 측정에서 뺀다

    per_cat, per_cat_ns, times = defaultdict(list), defaultdict(list), []
    for meta, gt in samples:
        start = time.perf_counter()
        result = ocr.predict(meta["image_path"])[0]
        times.append(time.perf_counter() - start)
        text = "\n".join(result["rec_texts"])
        per_cat[meta["category"]].append(cer(gt["text"], text))
        per_cat_ns[meta["category"]].append(cer(gt["text"], text, keep_space=False))

    print(f"lang={args.lang} det={args.det} rec={args.rec} device={args.device} no_mkldnn={args.no_mkldnn}")
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
