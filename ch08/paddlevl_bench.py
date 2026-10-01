"""08장 — PaddleOCR-VL을 평가셋 전체에 실행하고 범주별 CER·표 TEDS·처리 시간을 측정합니다.

실행 예:
  python paddlevl_bench.py                                           # PaddlePaddle로 직접 추론 (느림)
  python paddlevl_bench.py --server http://localhost:8118/v1        # vLLM 추론 서버 사용
결과 텍스트는 output/paddlevl.json에 저장합니다 (10장 Benchmark에서 재사용).
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402
from teds import teds  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
TEXT_LABELS = {"doc_title", "paragraph_title", "text", "abstract", "content", "header", "footer",
               "number", "aside_text", "reference", "footnote", "vision_footnote", "figure_title"}


def table_cells(table_html):
    from lxml import html
    root = html.fromstring(table_html)
    return [" ".join("".join(c.itertext()).split()) for c in root.iter("td", "th")]


def page_text(res):
    """읽기 순서대로 블록 내용을 잇고, 표는 칸의 글자를 꺼냅니다."""
    parts, tables = [], []
    for b in res["parsing_res_list"]:
        label, content = b.label, b.content or ""
        if label == "table":
            tables.append(content)
            parts.extend(table_cells(content) if "<t" in content else [content])
        elif label in TEXT_LABELS:
            parts.append(content)
    return "\n".join(parts), tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, help="vLLM 서버 주소 (예: http://localhost:8118/v1)")
    ap.add_argument("--categories", nargs="*")
    args = ap.parse_args()
    from paddleocr import PaddleOCRVL
    kwargs = {"vl_rec_backend": "vllm-server", "vl_rec_server_url": args.server} if args.server else {}
    pipeline = PaddleOCRVL(**kwargs)
    samples = load_samples(args.categories)
    pipeline.predict(samples[0][0]["image_path"])  # 첫 실행 제외

    per_cat, per_cat_ns, times, teds_scores, saved = defaultdict(list), defaultdict(list), [], [], {}
    for meta, gt in samples:
        start = time.perf_counter()
        res = pipeline.predict(meta["image_path"])[0]
        times.append(time.perf_counter() - start)
        text, tables = page_text(res)
        saved[meta["id"]] = {"text": text, "markdown": res.markdown["markdown_texts"], "tables": tables}
        per_cat[meta["category"]].append(cer(gt["text"], text))
        per_cat_ns[meta["category"]].append(cer(gt["text"], text, keep_space=False))
        if meta["category"] == "table":
            teds_scores.append((teds(tables[0] if tables else "", gt["table_html"]),
                                teds(tables[0] if tables else "", gt["table_html"], structure_only=True)))

    OUT.mkdir(exist_ok=True)
    (OUT / "paddlevl.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"backend={'vllm-server' if args.server else 'paddle'}")
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
