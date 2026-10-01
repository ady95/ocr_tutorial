"""05장 — DBNet 후처리 값(unclip_ratio, box_thresh)이 검출·인식에 주는 영향.

unclip_ratio: 확률 지도에서 찾은 글자 영역을 얼마나 넓혀 상자로 만들지 (기본 1.5)
box_thresh:   상자 안의 평균 확률이 이 값보다 낮으면 버림 (기본 0.6)

실행: python det_params.py
"""
import statistics
import sys
from pathlib import Path

from paddleocr import PaddleOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from det_compare import detection_scores  # noqa: E402
from ocr_eval import cer, load_samples  # noqa: E402

CATS = ["print_ko", "print_mixed", "print_numeric", "small", "lowres", "receipt", "table"]
SETTINGS = [("unclip_ratio", v) for v in (1.0, 1.5, 2.0, 2.5)] + [("box_thresh", v) for v in (0.3, 0.8, 0.9)]


def main():
    samples = load_samples(CATS)
    print(f"이미지 {len(samples)}장 ({', '.join(CATS)})")
    print(f"{'설정':22s} {'CER':>6s} {'CER(공백무시)':>13s} {'Recall':>7s} {'상자 수':>7s}")
    for name, value in SETTINGS:
        ocr = PaddleOCR(lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=False, **{f"text_det_{name}": value})
        cers, ns, hit, ngt, nbox = [], [], 0, 0, 0
        for meta, gt in samples:
            res = ocr.predict(meta["image_path"])[0]
            text = "\n".join(res["rec_texts"])
            cers.append(cer(gt["text"], text))
            ns.append(cer(gt["text"], text, keep_space=False))
            h, p, g = detection_scores([tuple(b) for b in res["rec_boxes"].tolist()], gt["lines"])
            hit, ngt, nbox = hit + h, ngt + g, nbox + p
        print(f"{name + '=' + str(value):22s} {statistics.mean(cers):6.3f} {statistics.mean(ns):13.3f}"
              f" {hit / ngt:7.3f} {nbox:7d}", flush=True)


if __name__ == "__main__":
    main()
