"""05장 — 한국어 인식 모델 세대 비교 (검출 모델은 PP-OCRv5_server_det로 고정).

실행: python rec_compare.py [--device cpu]
"""
import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from paddleocr import PaddleOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402

RECS = ["korean_PP-OCRv3_mobile_rec", "korean_PP-OCRv5_mobile_rec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()
    samples = load_samples()
    results = {}
    for rec in RECS:
        ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name=rec,
                        use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=False, device=args.device)
        ocr.predict(samples[0][0]["image_path"])
        per_cat, ns, times = defaultdict(list), [], []
        for meta, gt in samples:
            start = time.perf_counter()
            text = "\n".join(ocr.predict(meta["image_path"])[0]["rec_texts"])
            times.append(time.perf_counter() - start)
            per_cat[meta["category"]].append(cer(gt["text"], text))
            ns.append(cer(gt["text"], text, keep_space=False))
        results[rec] = (per_cat, ns, times)

    cats = sorted(results[RECS[0]][0])
    print(f"device={args.device}")
    print(f"{'범주':14s}" + "".join(f"{r.split('_')[1]:>12s}" for r in RECS))
    for c in cats:
        print(f"{c:14s}" + "".join(f"{statistics.mean(results[r][0][c]):12.3f}" for r in RECS))
    print(f"{'전체':14s}" + "".join(
        f"{statistics.mean([x for v in results[r][0].values() for x in v]):12.3f}" for r in RECS))
    print(f"{'공백 무시':14s}" + "".join(f"{statistics.mean(results[r][1]):12.3f}" for r in RECS))
    print(f"{'시간(초)':14s}" + "".join(f"{statistics.median(results[r][2]):12.3f}" for r in RECS))


if __name__ == "__main__":
    main()
