"""07장 — PP-StructureV3로 레이아웃·읽기 순서·표를 인식하고 평가합니다.

  2단 문서(multicol): 읽기 순서대로 이은 텍스트의 CER (PaddleOCR 단독과 비교)
  표(table)·세금계산서(invoice): 표 HTML의 TEDS / TEDS-S(구조만)
  레이아웃: 블록 종류(label) 집계

실행: python structure_eval.py [--device gpu]   (PaddleOCR 가상환경, paddleocr[doc-parser])
결과는 output/structure.json에 저장됩니다.
"""
import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from paddleocr import PPStructureV3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402
from teds import teds  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
TEXT_LABELS = {"doc_title", "paragraph_title", "text", "abstract", "content", "header", "footer",
               "number", "aside_text", "reference", "footnote", "algorithm"}


def blocks_text(res):
    """읽기 순서대로 텍스트 블록의 내용을 이어 붙입니다 (표·그림 제외)."""
    parts = [b.content for b in res["parsing_res_list"] if b.label in TEXT_LABELS]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()
    pipeline = PPStructureV3(lang="korean", device=args.device,
                             use_doc_orientation_classify=False, use_doc_unwarping=False,
                             use_textline_orientation=False, use_seal_recognition=False,
                             use_formula_recognition=False, use_chart_recognition=False)
    samples = load_samples(["multicol", "table", "invoice", "print_ko"])
    pipeline.predict(samples[0][0]["image_path"])  # 첫 실행 제외

    saved, times, labels = {}, [], Counter()
    rows = defaultdict(list)
    for meta, gt in samples:
        start = time.perf_counter()
        res = pipeline.predict(meta["image_path"])[0]
        times.append(time.perf_counter() - start)
        labels.update(b.label for b in res["parsing_res_list"])
        md = res.markdown["markdown_texts"]
        tables = [t["pred_html"] for t in res["table_res_list"]]
        saved[meta["id"]] = {"markdown": md, "tables": tables,
                             "blocks": [(b.label, b.content[:80]) for b in res["parsing_res_list"]]}
        cat = meta["category"]
        if cat in ("multicol", "print_ko"):
            rows[cat].append(cer(gt["text"], blocks_text(res)))
            rows[cat + "_ns"].append(cer(gt["text"], blocks_text(res), keep_space=False))
        if cat == "table":
            pred = tables[0] if tables else ""
            rows["teds"].append(teds(pred, gt["table_html"]))
            rows["teds_s"].append(teds(pred, gt["table_html"], structure_only=True))
            rows["teds_merged" if gt["merged_cells"] else "teds_plain"].append(rows["teds"][-1])
        if cat == "invoice":
            rows["invoice_tables"].append(len(tables))

    OUT.mkdir(exist_ok=True)
    (OUT / "structure.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    m = lambda k: statistics.mean(rows[k])  # noqa: E731
    print(f"device={args.device}  이미지당 시간 중앙값 {statistics.median(times):.2f}초")
    print(f"2단 문서 CER {m('multicol'):.3f} (공백 무시 {m('multicol_ns'):.3f})")
    print(f"한글 문서 CER {m('print_ko'):.3f} (공백 무시 {m('print_ko_ns'):.3f})")
    print(f"표 TEDS {m('teds'):.3f}, TEDS-S {m('teds_s'):.3f} | 병합 셀 있음 {m('teds_merged'):.3f}, 없음 {m('teds_plain'):.3f}")
    print(f"세금계산서에서 찾은 표 수: {rows['invoice_tables']}")
    print("레이아웃 블록 종류:", dict(labels.most_common()))


if __name__ == "__main__":
    main()
