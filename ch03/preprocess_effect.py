"""03장 — 전처리가 OCR 정확도를 얼마나 바꾸는지 측정합니다.

평가셋의 어려운 범주(저해상도·회전·작은 글자)에 전처리를 하나씩 적용하고,
전처리 전후의 CER을 비교합니다.

실행: python preprocess_effect.py --engine paddle      (PaddleOCR 가상환경)
      python preprocess_effect.py --engine tesseract   (기본 가상환경)
"""
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import preprocess as pp  # noqa: E402
from ocr_eval import cer, load_samples  # noqa: E402


def auto(img):
    """자동 전처리: 그림자 제거 → 기울기 보정 → 작은 이미지 확대."""
    out = pp.remove_shadow(img)
    out, _ = pp.deskew(out)
    return pp.upscale(out)


PIPELINES = {
    "원본": lambda img: img,
    "흑백": pp.to_gray,
    "확대": pp.upscale,
    "선명화": pp.sharpen,
    "잡음 제거": pp.denoise,
    "오츠 이진화": pp.binarize_otsu,
    "적응형 이진화": pp.binarize_adaptive,
    "그림자 제거": pp.remove_shadow,
    "기울기 보정": lambda img: pp.deskew(img)[0],
    "자동 (그림자→기울기→확대)": auto,
}


def make_engine(name):
    if name == "paddle":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="korean", use_doc_orientation_classify=False,
                        use_doc_unwarping=False, use_textline_orientation=False)

        def run(img):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return "\n".join(ocr.predict(img)[0]["rec_texts"])
        return run

    import pytesseract

    def run(img):
        return pytesseract.image_to_string(pp.to_rgb(img), lang="kor+eng", config="--psm 4")  # cv2는 BGR
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["paddle", "tesseract"], required=True)
    ap.add_argument("--categories", nargs="*", default=["lowres", "rotated", "small"])
    args = ap.parse_args()

    run = make_engine(args.engine)
    samples = load_samples(args.categories)
    images = [(meta["category"], cv2.imread(meta["image_path"]), gt["text"]) for meta, gt in samples]

    table = {}
    for name, fn in PIPELINES.items():
        per_cat = defaultdict(list)
        for cat, img, ref in images:
            per_cat[cat].append(cer(ref, run(fn(img))))
        table[name] = {cat: statistics.mean(v) for cat, v in per_cat.items()}
        print(name, {k: round(v, 3) for k, v in table[name].items()}, flush=True)

    cats = args.categories
    print(f"\nengine={args.engine}")
    print(f"{'전처리':28s}" + "".join(f"{c:>10s}" for c in cats))
    for name, row in table.items():
        print(f"{name:28s}" + "".join(f"{row[c]:10.3f}" for c in cats))


if __name__ == "__main__":
    main()
