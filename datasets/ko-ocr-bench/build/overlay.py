"""정답 확인용 — 이미지 위에 정답 줄 상자를 그린다.

실행: python overlay.py <id> [<id> ...]   → build/_overlay/<id>.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "_overlay"


def draw(sample_id):
    gt = json.loads((ROOT / "gt" / f"{sample_id}.json").read_text(encoding="utf-8"))
    img = Image.open(ROOT / "images" / f"{sample_id}.png").convert("RGB")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(gt["lines"]):
        pts = [tuple(p) for p in line["poly"]]
        d.polygon(pts, outline=(220, 30, 30), width=2)
        d.text((pts[0][0], pts[0][1] - 11), str(i), fill=(30, 30, 220))
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"{sample_id}.png")
    print(sample_id, img.size, "줄", len(gt["lines"]))


if __name__ == "__main__":
    for sid in sys.argv[1:]:
        draw(sid)
