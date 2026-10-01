"""OCR 따라하기 — 좌표 도구 (다각형 → 사각형, IoU, 정답 줄과 짝짓기)."""


def poly_to_box(poly):
    """네 꼭짓점 [[x, y], ...]을 감싸는 사각형 (x1, y1, x2, y2)로 바꿉니다."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    """두 사각형의 겹친 넓이 ÷ 합친 넓이 (Intersection over Union)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_lines(pred_boxes, gt_lines, threshold=0.5):
    """예측 상자마다 IoU가 가장 큰 정답 줄을 찾습니다. threshold 미만이면 None."""
    gt_boxes = [poly_to_box(line["poly"]) for line in gt_lines]
    matches = []
    for pb in pred_boxes:
        scores = [iou(pb, gb) for gb in gt_boxes]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        matches.append(best if best is not None and scores[best] >= threshold else None)
    return matches
