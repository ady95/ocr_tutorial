"""08장 — Surya OCR 2를 평가셋 전체에 실행하고 범주별 CER·표 TEDS·처리 시간을 측정합니다.

Surya는 첫 실행 때 추론 서버를 자동으로 띄웁니다.
  NVIDIA GPU: vLLM Docker 컨테이너 (Docker + NVIDIA Container Toolkit 필요)
  CPU:        llama.cpp의 llama-server
실행: python surya_bench.py [--categories ...] [--per-category 3 --name surya_cpu]
결과 텍스트는 output/surya.json에 저장합니다.
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from lxml import html as lhtml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402
from teds import teds  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def block_text(block_html):
    """블록 HTML을 글자로 바꿉니다. 표는 칸마다 한 줄, 나머지는 줄바꿈(br·p)을 살려 이어 붙입니다."""
    if not block_html:
        return ""
    root = lhtml.fromstring(f"<div>{block_html}</div>")
    cells = list(root.iter("td", "th"))
    if cells:
        return "\n".join(" ".join("".join(c.itertext()).split()) for c in cells)
    for br in root.iter("br"):
        br.tail = "\n" + (br.tail or "")
    return "\n".join(line.strip() for line in "".join(root.itertext()).splitlines() if line.strip())


def first_n_per_category(samples, n):
    count, picked = defaultdict(int), []
    for meta, gt in samples:
        count[meta["category"]] += 1
        if count[meta["category"]] <= n:
            picked.append((meta, gt))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--per-category", type=int, help="범주마다 앞에서부터 N장만 (CPU에서 빠르게 확인할 때)")
    ap.add_argument("--name", default="surya", help="결과 파일 이름 (output/<name>.json)")
    args = ap.parse_args()
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor

    manager = SuryaInferenceManager()
    rec = RecognitionPredictor(manager)
    samples = load_samples(args.categories)
    if args.per_category:
        samples = first_n_per_category(samples, args.per_category)
    rec([Image.open(samples[0][0]["image_path"]).convert("RGB")])  # 서버 기동·첫 실행 제외

    per_cat, per_cat_ns, times, teds_scores, saved = defaultdict(list), defaultdict(list), [], [], {}
    for meta, gt in samples:
        image = Image.open(meta["image_path"]).convert("RGB")
        start = time.perf_counter()
        page = rec([image])[0]
        times.append(time.perf_counter() - start)
        blocks = sorted(page.blocks, key=lambda b: b.reading_order)
        text = "\n".join(t for t in (block_text(b.html) for b in blocks) if t)
        tables = [b.html for b in blocks if "<table" in (b.html or "")]
        saved[meta["id"]] = {"text": text, "blocks": [(b.label, b.html) for b in blocks]}
        per_cat[meta["category"]].append(cer(gt["text"], text))
        per_cat_ns[meta["category"]].append(cer(gt["text"], text, keep_space=False))
        if meta["category"] == "table":
            pred = tables[0] if tables else ""
            teds_scores.append((teds(pred, gt["table_html"]), teds(pred, gt["table_html"], structure_only=True)))

    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'범주':14s} {'장수':>4s} {'CER':>7s} {'CER(공백무시)':>13s}")
    allv, allns = [], []
    for c in sorted(per_cat):
        allv += per_cat[c]
        allns += per_cat_ns[c]
        print(f"{c:14s} {len(per_cat[c]):4d} {statistics.mean(per_cat[c]):7.3f} {statistics.mean(per_cat_ns[c]):13.3f}")
    print(f"{'전체':14s} {len(allv):4d} {statistics.mean(allv):7.3f} {statistics.mean(allns):13.3f}")
    if teds_scores:
        print(f"표 TEDS {statistics.mean(t for t, _ in teds_scores):.3f}, TEDS-S {statistics.mean(s for _, s in teds_scores):.3f}")
    print(f"이미지당 처리 시간: 평균 {statistics.mean(times):.2f}초, 중앙값 {statistics.median(times):.2f}초")


if __name__ == "__main__":
    main()
