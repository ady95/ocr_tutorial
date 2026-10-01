"""03장 — 회전 범주를 변환 종류별로 나눠 기울기 보정·방향 보정의 효과를 측정합니다.

변환 종류: 작은 회전(2~4도), 큰 회전(6~10도), 180도 뒤집힘, 원근 왜곡, 원근 왜곡+그림자
비교: 원본 / 기울기 보정(deskew) / PaddleOCR 문서 방향 보정(use_doc_orientation_classify)

실행: python rotation_effect.py      (PaddleOCR 가상환경)
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import preprocess as pp  # noqa: E402
from ocr_eval import cer, load_samples  # noqa: E402


def kind_of(transform):
    if "rotate_deg" in transform:
        a = abs(transform["rotate_deg"])
        return "180도 뒤집힘" if a == 180 else ("작은 회전" if a < 5 else "큰 회전")
    return "원근+그림자" if "shadow" in transform else "원근 왜곡"


def main():
    base = dict(lang="korean", use_doc_unwarping=False, use_textline_orientation=False)
    ocr_plain = PaddleOCR(use_doc_orientation_classify=False, **base)
    ocr_orient = PaddleOCR(use_doc_orientation_classify=True, **base)
    read = lambda o, img: "\n".join(o.predict(img)[0]["rec_texts"])  # noqa: E731

    rows = defaultdict(lambda: defaultdict(list))
    for meta, gt in load_samples(["rotated"]):
        img = cv2.imread(meta["image_path"])
        kind = kind_of(gt["transform"])
        fixed, _ = pp.deskew(pp.remove_shadow(img) if "shadow" in gt["transform"] else img)
        if fixed.ndim == 2:
            fixed = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)
        rows[kind]["원본"].append(cer(gt["text"], read(ocr_plain, img)))
        rows[kind]["기울기 보정"].append(cer(gt["text"], read(ocr_plain, fixed)))
        rows[kind]["방향 보정"].append(cer(gt["text"], read(ocr_orient, img)))

    cols = ["원본", "기울기 보정", "방향 보정"]
    print(f"{'변환':12s} {'장수':>4s}" + "".join(f"{c:>10s}" for c in cols))
    for kind in ["작은 회전", "큰 회전", "180도 뒤집힘", "원근 왜곡", "원근+그림자"]:
        r = rows[kind]
        print(f"{kind:12s} {len(r['원본']):4d}" + "".join(f"{statistics.mean(r[c]):10.3f}" for c in cols))


if __name__ == "__main__":
    main()
