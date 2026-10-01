"""05장 — PaddleOCR 검출 모델 비교 (인식 모델은 한국어 v5로 고정).

검출 모델 5종: PP-OCRv5_server_det, PP-OCRv5_mobile_det, PP-OCRv6_medium_det, PP-OCRv6_small_det, PP-OCRv6_tiny_det
지표: CER, 검출 Precision·Recall·F1 (정답 줄과 IoU 0.5 이상이면 맞게 찾은 것), 이미지당 시간

실행: python det_compare.py [--device cpu] [--categories ...]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

from paddleocr import PaddleOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from boxes import iou, poly_to_box  # noqa: E402
from ocr_eval import cer, load_samples  # noqa: E402

DETS = ["PP-OCRv5_server_det", "PP-OCRv5_mobile_det", "PP-OCRv6_medium_det",
        "PP-OCRv6_small_det", "PP-OCRv6_tiny_det"]
DOC_CATS = ["print_ko", "print_mixed", "print_numeric", "small", "lowres", "rotated", "multicol"]


def detection_scores(pred_boxes, gt_lines, threshold=0.5):
    """예측 상자와 정답 줄을 IoU로 1:1 짝짓고 (맞은 수, 예측 수, 정답 수)를 돌려줍니다."""
    gt_boxes = [poly_to_box(line["poly"]) for line in gt_lines]
    used, hits = set(), 0
    for pb in pred_boxes:
        best, best_iou = None, threshold
        for k, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if k not in used and v >= best_iou:
                best, best_iou = k, v
        if best is not None:
            used.add(best)
            hits += 1
    return hits, len(pred_boxes), len(gt_boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--categories", nargs="*", default=DOC_CATS)
    args = ap.parse_args()
    samples = load_samples(args.categories)

    print(f"device={args.device}, 이미지 {len(samples)}장 ({', '.join(args.categories)})")
    print(f"{'검출 모델':22s} {'CER':>6s} {'Precision':>9s} {'Recall':>7s} {'F1':>6s} {'시간(초)':>8s}")
    for det in DETS:
        ocr = PaddleOCR(text_detection_model_name=det, text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                        use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=False, device=args.device)
        ocr.predict(samples[0][0]["image_path"])  # 첫 실행은 시간 측정에서 뺀다
        cers, times, hit, npred, ngt = [], [], 0, 0, 0
        for meta, gt in samples:
            start = time.perf_counter()
            res = ocr.predict(meta["image_path"])[0]
            times.append(time.perf_counter() - start)
            cers.append(cer(gt["text"], "\n".join(res["rec_texts"])))
            # 회전 이미지는 정답 좌표가 기울어 있으므로 검출 지표에서 뺀다
            if meta["category"] != "rotated":
                h, p, g = detection_scores([tuple(b) for b in res["rec_boxes"].tolist()], gt["lines"])
                hit, npred, ngt = hit + h, npred + p, ngt + g
        precision, recall = hit / npred, hit / ngt
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{det:22s} {statistics.mean(cers):6.3f} {precision:9.3f} {recall:7.3f} {f1:6.3f}"
              f" {statistics.median(times):8.3f}", flush=True)


if __name__ == "__main__":
    main()
