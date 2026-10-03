"""03장 실습 — 도구별 자동 전처리 파이프라인을 평가셋 전체(140장)에 적용해 효과를 확인합니다.

Tesseract: 기울기 보정(각도는 그림자를 지운 이미지로 추정, 회전은 원본에) → 평균 신뢰도가 85 미만이면
           180도 회전·확대 후보를 만들어 신뢰도가 가장 높은 쪽 선택
           (--tesseract-mode simple: 비교용으로 OSD 방향 보정 → 기울기 보정 → 세로 1,000픽셀 미만이면 확대를 그대로 이어 붙임)
PaddleOCR: 그대로 읽고, 평균 신뢰도가 0.8 미만일 때만 180도 돌려 다시 읽어 신뢰도가 높은 쪽 선택
           (--paddle-mode orient: 비교용으로 문서 방향 분류 모델을 켬)

실행: python pipeline_eval.py --engine tesseract   (기본 가상환경)
      python pipeline_eval.py --engine paddle      (PaddleOCR 가상환경)
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import preprocess as pp  # noqa: E402
from ocr_eval import cer, load_samples  # noqa: E402


def tesseract_simple():
    """03-1~03-3에서 효과가 있던 처리를 그대로 이어 붙인 첫 시도: OSD 방향 보정 → 기울기 보정 → 확대."""
    import pytesseract
    config = "--psm 4"
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def prepare(img):
        try:
            angle = int(re.search(r"Rotate: (\d+)", pytesseract.image_to_osd(pp.to_rgb(img))).group(1))
            img = cv2.rotate(img, codes[angle]) if angle in codes else img
        except pytesseract.TesseractError:
            pass
        skew = pp.estimate_skew(pp.remove_shadow(img))
        img = pp.rotate(img, skew) if abs(skew) >= 0.5 else img
        return pp.upscale(img)

    def read(img):
        return pytesseract.image_to_string(pp.to_rgb(img), lang="kor+eng", config=config)

    return read, prepare


def tesseract_pipeline(threshold=85):
    """기울기 보정은 항상 하고, 평균 신뢰도가 threshold 미만이면 180도 회전·확대 후보 중 가장 확신하는 쪽을 고른다."""
    import pytesseract

    config = "--psm 4"

    def confidence(img):
        data = pytesseract.image_to_data(pp.to_rgb(img), lang="kor+eng", config=config, output_type=pytesseract.Output.DICT)
        conf = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
        return statistics.mean(conf) if conf else 0.0

    def prepare(img):
        angle = pp.estimate_skew(pp.remove_shadow(img))  # 각도만 그림자 없는 이미지로 잰다
        img = pp.rotate(img, angle) if abs(angle) >= 0.5 else img
        best, best_conf = img, confidence(img)
        if best_conf < threshold:
            for cand in (cv2.rotate(img, cv2.ROTATE_180), pp.upscale(img)):
                c = confidence(cand)
                if c > best_conf:
                    best, best_conf = cand, c
        return best

    def read(img):
        return pytesseract.image_to_string(pp.to_rgb(img), lang="kor+eng", config=config)

    return read, prepare


def paddle_pipeline(mode):
    """mode: orient = 문서 방향 분류 모델 켜기 / flip = 신뢰도가 낮을 때만 180도 돌려 다시 읽기."""
    from paddleocr import PaddleOCR
    base = dict(lang="korean", use_doc_unwarping=False, use_textline_orientation=False)
    plain = PaddleOCR(use_doc_orientation_classify=False, **base)
    orient = PaddleOCR(use_doc_orientation_classify=True, **base) if mode == "orient" else None

    def run(ocr, img):
        res = ocr.predict(img)[0]
        scores = list(res["rec_scores"])
        return "\n".join(res["rec_texts"]), (statistics.mean(scores) if scores else 0.0)

    def read(img, prepared=False):
        if not prepared:
            return run(plain, img)[0]
        if mode == "orient":
            return run(orient, img)[0]
        text, score = run(plain, img)
        if score >= 0.8:                       # 충분히 확신하면 그대로 쓴다
            return text
        flipped, fscore = run(plain, cv2.rotate(img, cv2.ROTATE_180))
        return flipped if fscore > score else text

    return read, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["paddle", "tesseract"], required=True)
    ap.add_argument("--paddle-mode", choices=["orient", "flip"], default="flip")
    ap.add_argument("--tesseract-mode", choices=["confidence", "simple"], default="confidence")
    args = ap.parse_args()

    if args.engine == "tesseract":
        read, prepare = tesseract_simple() if args.tesseract_mode == "simple" else tesseract_pipeline()
    else:
        read, prepare = paddle_pipeline(args.paddle_mode)
    before, after = defaultdict(list), defaultdict(list)
    for meta, gt in load_samples():
        img = cv2.imread(meta["image_path"])
        cat = meta["category"]
        before[cat].append(cer(gt["text"], read(img)))
        if prepare is None:
            after[cat].append(cer(gt["text"], read(img, prepared=True)))
        else:
            after[cat].append(cer(gt["text"], read(prepare(img))))

    mode = args.tesseract_mode if args.engine == "tesseract" else args.paddle_mode
    print(f"engine={args.engine} mode={mode}")
    print(f"{'범주':14s} {'장수':>4s} {'전처리 전':>9s} {'전처리 후':>9s} {'변화':>8s}")
    all_b, all_a = [], []
    for cat in sorted(before):
        b, a = statistics.mean(before[cat]), statistics.mean(after[cat])
        all_b += before[cat]
        all_a += after[cat]
        print(f"{cat:14s} {len(before[cat]):4d} {b:9.3f} {a:9.3f} {a - b:+8.3f}")
    b, a = statistics.mean(all_b), statistics.mean(all_a)
    print(f"{'전체':14s} {len(all_b):4d} {b:9.3f} {a:9.3f} {a - b:+8.3f}")


if __name__ == "__main__":
    main()
