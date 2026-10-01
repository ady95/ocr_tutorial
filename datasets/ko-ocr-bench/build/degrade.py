"""한국어 OCR 평가셋 — 열화 이미지 생성 (깨끗한 문서 → 저해상도·회전·원근 왜곡).

같은 원문의 깨끗한 버전(parent)과 짝을 이루므로 "무엇이 OCR을 망가뜨리는가"를 직접 비교할 수 있다.
정답 텍스트는 그대로이고, 줄 좌표는 같은 변환 행렬로 옮긴다.

실행: python degrade.py      (render.py 실행 후)
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES, GT = ROOT / "images", ROOT / "gt"
WHITE = (255, 255, 255)

LOWRES = [f"doc_ko_{i:02d}" for i in range(1, 8)] + [f"doc_mixed_{i:02d}" for i in range(1, 8)] \
    + [f"doc_numeric_{i:02d}" for i in range(1, 7)]
GEOMETRIC = [f"doc_ko_{i:02d}" for i in range(4, 11)] + [f"doc_mixed_{i:02d}" for i in range(4, 11)] \
    + [f"doc_numeric_{i:02d}" for i in range(5, 11)]


def jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_affine(points, m):
    pts = np.hstack([np.array(points, dtype=np.float64), np.ones((len(points), 1))])
    return (pts @ m.T).tolist()


def lowres(img, kind, rng):
    """저해상도·압축·흐림. 해상도를 낮춘 이미지를 그대로 저장한다 (다시 키우지 않음)."""
    if kind == 0:
        scale, params = 0.5, {"scale": 0.5, "jpeg": 35}
        out = jpeg(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), 35)
    elif kind == 1:
        scale, params = 0.4, {"scale": 0.4}
        out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif kind == 2:
        scale, params = 1.0, {"blur_sigma": 1.6, "jpeg": 50}
        out = jpeg(cv2.GaussianBlur(img, (0, 0), 1.6), 50)
    else:
        scale, params = 0.6, {"scale": 0.6, "noise_std": 12, "jpeg": 40}
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        noise = np.random.default_rng(rng.randint(0, 9999)).normal(0, 12, small.shape)
        out = jpeg(np.clip(small + noise, 0, 255).astype(np.uint8), 40)
    m = np.array([[scale, 0, 0], [0, scale, 0]])
    return out, m, params


def rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=WHITE, flags=cv2.INTER_CUBIC), m


def perspective(img, rng, shadow):
    h, w = img.shape[:2]
    pad = int(max(h, w) * 0.06)
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(235, 235, 232))
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    j = lambda: rng.uniform(0.0, 0.07) * max(h, w)  # noqa: E731
    dst = np.float32([[pad + j(), pad + j()], [pad + w - j(), pad + j()],
                      [pad + w - j(), pad + h - j()], [pad + j(), pad + h - j()]])
    m = cv2.getPerspectiveTransform(src, dst)
    shift = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float64)
    full = m @ shift  # 원본 좌표 → 결과 좌표
    out = cv2.warpPerspective(canvas, m, (canvas.shape[1], canvas.shape[0]),
                              borderValue=(235, 235, 232), flags=cv2.INTER_CUBIC)
    params = {"perspective_jitter": 0.07}
    if shadow:
        # 왼쪽 위에서 오른쪽 아래로 어두워지는 그림자 (스마트폰 촬영 흉내)
        gx = np.linspace(1.0, 0.62, out.shape[1])[None, :]
        gy = np.linspace(1.0, 0.85, out.shape[0])[:, None]
        out = np.clip(out.astype(np.float32) * (gx * gy)[..., None], 0, 255).astype(np.uint8)
        params["shadow"] = "gradient 1.0→0.53"
    return out, full, params


def save(parent_gt, new_id, category, img, transform, params):
    gt = dict(parent_gt)
    gt.update({"id": new_id, "category": category, "source": "degraded", "parent": parent_gt["id"],
               "width": img.shape[1], "height": img.shape[0], "transform": params})
    lines = []
    for line in parent_gt["lines"]:
        if transform.shape == (3, 3):
            pts = cv2.perspectiveTransform(np.array([line["poly"]], dtype=np.float64), transform)[0].tolist()
        else:
            pts = apply_affine(line["poly"], transform)
        lines.append({**line, "poly": [[round(x, 1), round(y, 1)] for x, y in pts]})
    gt["lines"] = lines
    cv2.imwrite(str(IMAGES / f"{new_id}.png"), img)
    (GT / f"{new_id}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    for i, pid in enumerate(LOWRES):
        rng = random.Random(4000 + i)
        img = cv2.imread(str(IMAGES / f"{pid}.png"))
        parent = json.loads((GT / f"{pid}.json").read_text(encoding="utf-8"))
        out, m, params = lowres(img, i % 4, rng)
        save(parent, f"lowres_{i + 1:02d}", "lowres", out, m, params)
    for i, pid in enumerate(GEOMETRIC):
        rng = random.Random(5000 + i)
        img = cv2.imread(str(IMAGES / f"{pid}.png"))
        parent = json.loads((GT / f"{pid}.json").read_text(encoding="utf-8"))
        kind = i % 5
        if kind in (0, 1, 4):
            angle = {0: rng.choice([-1, 1]) * rng.uniform(2, 4), 1: rng.choice([-1, 1]) * rng.uniform(6, 10),
                     4: 180.0}[kind]
            out, m = rotate(img, angle)
            params = {"rotate_deg": round(angle, 2)}
        else:
            out, m, params = perspective(img, rng, shadow=(kind == 2))
        save(parent, f"rotated_{i + 1:02d}", "rotated", out, m, params)
    print("lowres", len(LOWRES), "| rotated", len(GEOMETRIC))


if __name__ == "__main__":
    main()
