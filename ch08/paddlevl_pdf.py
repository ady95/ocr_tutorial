"""08장 — PaddleOCR-VL로 PDF를 바로 파싱하고, 여러 쪽을 한 문서로 합쳐 Markdown·JSON으로 저장합니다.

07장의 스캔 PDF(ch07/pdf/scanned.pdf, 5쪽)를 쓰고 쪽마다 CER을 잽니다.
실행: python paddlevl_pdf.py [--server http://localhost:8118/v1] [pdf 경로]
출력: output/paddlevl_pdf/ (쪽별 JSON, 합친 Markdown)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer  # noqa: E402
from paddlevl_bench import page_text  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(HERE.parent / "ch07" / "pdf" / "scanned.pdf"))
    ap.add_argument("--server", default=None)
    args = ap.parse_args()
    from paddleocr import PaddleOCRVL
    kwargs = {"vl_rec_backend": "vllm-server", "vl_rec_server_url": args.server} if args.server else {}
    pipeline = PaddleOCRVL(**kwargs)
    gt = json.loads((HERE.parent / "ch07" / "pdf" / "pdf_gt.json").read_text(encoding="utf-8"))
    out = HERE / "output" / "paddlevl_pdf"
    out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    pages = list(pipeline.predict(args.pdf))  # PDF를 넣으면 쪽마다 결과가 하나씩 나온다
    elapsed = time.perf_counter() - start
    scores = [cer(ref, page_text(res)[0]) for res, ref in zip(pages, gt["pages"])]
    print(f"{len(pages)}쪽, {elapsed:.1f}초 (쪽당 {elapsed / len(pages):.1f}초)")
    print(f"쪽별 CER {[round(s, 3) for s in scores]}, 평균 {statistics.mean(scores):.3f}")

    for res in pages:
        res.save_to_json(str(out))
    merged = pipeline.restructure_pages(pages, merge_tables=True, relevel_titles=True, concatenate_pages=True)
    for res in merged:
        res.save_to_markdown(str(out))
    print("저장한 파일:", sorted(p.name for p in out.iterdir()))

    first = json.loads(sorted(out.glob("*_0_res.json"))[0].read_text(encoding="utf-8"))
    print("JSON 최상위 키:", list(first.keys()))
    block = first["parsing_res_list"][0]
    print("첫 블록:", {k: block[k] for k in ("block_label", "block_bbox", "block_content")})


if __name__ == "__main__":
    main()
