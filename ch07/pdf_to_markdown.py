"""07장 — PDF를 Markdown으로: 텍스트 PDF는 바로 추출, 스캔 PDF는 OCR(PP-StructureV3).

  페이지마다 텍스트 정보가 있는지(글자 수) 확인해 방법을 고릅니다.
  비교: ① 텍스트 추출(pypdfium2) ② PP-StructureV3로 OCR ③ 자동 선택

실행: python pdf_to_markdown.py pdf/native.pdf pdf/scanned.pdf   (PaddleOCR 가상환경, pypdfium2)
출력: output/<pdf 이름>.md, 페이지별 CER(정답은 pdf/pdf_gt.json)
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer  # noqa: E402

HERE = Path(__file__).resolve().parent
TEXT_LABELS = {"doc_title", "paragraph_title", "text", "abstract", "content", "header", "footer", "number"}
MIN_CHARS = 20  # 이보다 글자가 적으면 텍스트 정보가 없는 페이지(스캔)로 본다


def page_text(page):
    return page.get_textpage().get_text_range().replace("\r\n", "\n").strip()


def render(page, scale=2.0):
    """PDF 페이지를 이미지(BGR numpy)로 그립니다. scale 1 = 72dpi, 2 = 144dpi."""
    rgb = np.array(page.render(scale=scale).to_pil().convert("RGB"))
    return rgb[:, :, ::-1].copy()


def table_cells(table_html):
    """표 HTML에서 칸의 글자만 순서대로 꺼냅니다 (정답 텍스트와 비교하기 위해)."""
    from lxml import html
    root = html.fromstring(table_html)
    return [" ".join("".join(c.itertext()).split()) for c in root.iter("td", "th")]


def ocr_page(pipeline, image):
    res = pipeline.predict(image)[0]
    parts = []
    for b in res["parsing_res_list"]:
        if b.label in TEXT_LABELS:
            parts.append(b.content)
        elif b.label == "table":
            parts.extend(table_cells(b.content))
    return "\n".join(parts), res.markdown["markdown_texts"]


def main():
    import argparse
    from paddleocr import PPStructureV3
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--scale", type=float, default=2.0, help="렌더링 배율 (1 = 72dpi)")
    args = ap.parse_args()
    pipeline = PPStructureV3(lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False,
                             use_textline_orientation=False, use_seal_recognition=False,
                             use_formula_recognition=False, use_chart_recognition=False)
    gt = json.loads((HERE / "pdf" / "pdf_gt.json").read_text(encoding="utf-8"))
    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    pipeline.predict(render(pdfium.PdfDocument(args.pdfs[0])[0]))  # 첫 실행 제외

    for pdf_path in args.pdfs:
        doc = pdfium.PdfDocument(pdf_path)
        md_pages, pages_json, rows = [], [], {"추출": [], "OCR": [], "자동": []}
        t_extract = t_ocr = 0.0
        modes = []
        for i in range(len(doc)):
            page, ref = doc[i], gt["pages"][i]
            start = time.perf_counter()
            extracted = page_text(page)
            t_extract += time.perf_counter() - start
            start = time.perf_counter()
            ocr_text, md = ocr_page(pipeline, render(page, args.scale))
            t_ocr += time.perf_counter() - start
            native = len(extracted) >= MIN_CHARS
            modes.append("텍스트" if native else "스캔")
            rows["추출"].append(cer(ref, extracted) if extracted else 1.0)
            rows["OCR"].append(cer(ref, ocr_text))
            rows["자동"].append(rows["추출"][-1] if native else rows["OCR"][-1])
            md_pages.append(extracted if native else md)
            pages_json.append({"page": i + 1, "mode": "text" if native else "ocr",
                               "text": extracted if native else ocr_text, "markdown": md_pages[-1]})
        name = Path(pdf_path).stem
        (out_dir / f"{name}.md").write_text("\n\n".join(md_pages), encoding="utf-8")
        (out_dir / f"{name}.json").write_text(json.dumps(pages_json, ensure_ascii=False, indent=1), encoding="utf-8")
        w, h = doc[0].get_size()
        print(f"[{name}] {len(doc)}쪽, 첫 쪽 {w:.0f}x{h:.0f}pt → 렌더 {w * args.scale:.0f}x{h * args.scale:.0f}px, 페이지 판별: {modes}")
        for k, v in rows.items():
            print(f"  {k:4s} CER 평균 {statistics.mean(v):.3f}  페이지별 {[round(x, 3) for x in v]}")
        print(f"  시간: 텍스트 추출 {t_extract:.2f}초, OCR {t_ocr:.2f}초 (5쪽 합계)")


if __name__ == "__main__":
    main()
