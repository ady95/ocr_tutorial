"""03장 — 기하 보정(기울기·원근·방향)의 효과를 회전 종류별로 측정합니다.

변환 종류: 작은 회전(2~4도), 큰 회전(6~10도), 180도 뒤집힘, 원근 왜곡, 원근 왜곡+그림자
보정: 원본 / 기울기 보정 / 원근 보정 / 방향 보정 / 전체(방향 → 원근 → 그림자 → 기울기)

방향 보정은 엔진이 제공하는 기능을 씁니다.
  tesseract: OSD(Orientation and Script Detection, psm 0)로 0/90/180/270도 판별
  paddle: 문서 방향 분류 모델(PP-LCNet_x1_0_doc_ori)

실행: python geometry_effect.py --engine tesseract   (기본 가상환경)
      python geometry_effect.py --engine paddle      (PaddleOCR 가상환경)
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

KINDS = ["작은 회전", "큰 회전", "180도 뒤집힘", "원근 왜곡", "원근+그림자"]


def kind_of(transform):
    if "rotate_deg" in transform:
        a = abs(transform["rotate_deg"])
        return "180도 뒤집힘" if a == 180 else ("작은 회전" if a < 5 else "큰 회전")
    return "원근+그림자" if "shadow" in transform else "원근 왜곡"


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def perspective_fix(img):
    corners = pp.find_document(img)
    return img if corners is None else pp.warp_document(img, corners)


def make_engine(name):
    """(인식 함수, 방향 보정 함수)를 돌려줍니다."""
    if name == "paddle":
        from paddleocr import PaddleOCR
        base = dict(lang="korean", use_doc_unwarping=False, use_textline_orientation=False)
        plain = PaddleOCR(use_doc_orientation_classify=False, **base)
        orient = PaddleOCR(use_doc_orientation_classify=True, **base)

        def read(img, oriented=False):
            o = orient if oriented else plain
            return "\n".join(o.predict(to_bgr(img))[0]["rec_texts"])
        return read

    import pytesseract

    def osd_rotate(img):
        """OSD가 알려 준 각도만큼 돌려 바로 세웁니다. 판별에 실패하면 그대로 둡니다."""
        try:
            osd = pytesseract.image_to_osd(img)
        except pytesseract.TesseractError:
            return img
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        return cv2.rotate(img, codes[angle]) if angle in codes else img

    def read(img, oriented=False):
        if oriented:
            img = osd_rotate(img)
        return pytesseract.image_to_string(img, lang="kor+eng", config="--psm 4")
    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["paddle", "tesseract"], required=True)
    args = ap.parse_args()
    read = make_engine(args.engine)

    def full(img):
        """전체: 원근 보정 → 그림자 제거 → 기울기 보정 (방향 보정은 인식 단계에서)."""
        out = pp.remove_shadow(perspective_fix(img))
        return pp.deskew(out)[0]

    rows = defaultdict(lambda: defaultdict(list))
    for meta, gt in load_samples(["rotated"]):
        img = cv2.imread(meta["image_path"])
        kind = kind_of(gt["transform"])
        ref = gt["text"]
        rows[kind]["원본"].append(cer(ref, read(img)))
        rows[kind]["기울기 보정"].append(cer(ref, read(pp.deskew(img)[0])))
        rows[kind]["원근 보정"].append(cer(ref, read(perspective_fix(img))))
        rows[kind]["방향 보정"].append(cer(ref, read(img, oriented=True)))
        rows[kind]["전체"].append(cer(ref, read(full(img), oriented=True)))
        print(meta["id"], kind, {k: round(v[-1], 3) for k, v in rows[kind].items()}, flush=True)

    cols = ["원본", "기울기 보정", "원근 보정", "방향 보정", "전체"]
    print(f"\nengine={args.engine}")
    print(f"{'변환':12s} {'장수':>4s}" + "".join(f"{c:>10s}" for c in cols))
    for kind in KINDS:
        r = rows[kind]
        print(f"{kind:12s} {len(r['원본']):4d}" + "".join(f"{statistics.mean(r[c]):10.3f}" for c in cols))
    print(f"{'전체 평균':12s} {20:4d}" + "".join(
        f"{statistics.mean([x for k in KINDS for x in rows[k][c]]):10.3f}" for c in cols))


if __name__ == "__main__":
    main()
