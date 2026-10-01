"""07장 — 평가셋 문서로 시험용 PDF 두 가지를 만듭니다.

  native.pdf : 텍스트 정보가 들어 있는 PDF (브라우저에서 바로 PDF로 출력)
  scanned.pdf: 같은 문서를 이미지로만 담은 PDF (스캔한 문서와 같은 상태)
  pdf_gt.json: 페이지별 정답 텍스트

실행: python make_pdfs.py   (playwright, pypdfium2, pillow 필요. 평가셋 build/render.py를 먼저 실행해 _tmp HTML이 있어야 함)
"""
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "ko-ocr-bench"
OUT = Path(__file__).resolve().parent / "pdf"
DOCS = ["doc_ko_01", "doc_mixed_02", "doc_numeric_03", "multicol_01", "table_01"]


def main():
    OUT.mkdir(exist_ok=True)
    parts = []
    with sync_playwright() as pw:
        page = pw.chromium.launch().new_page()
        for sid in DOCS:
            page.goto((ROOT / "build" / "_tmp" / f"{sid}.html").as_uri())
            page.evaluate("document.fonts.ready")
            box = page.locator("#doc").bounding_box()
            path = OUT / f"_{sid}.pdf"
            page.pdf(path=str(path), width=f"{box['width']}px", height=f"{box['height'] + 2}px",
                     print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            parts.append(path)
    merged = pdfium.PdfDocument.new()
    sources = [pdfium.PdfDocument(str(path)) for path in parts]
    for src in sources:
        merged.import_pages(src)
    merged.save(str(OUT / "native.pdf"))
    for src, path in zip(sources, parts):
        src.close()  # Windows에서는 열린 파일을 지울 수 없으므로 먼저 닫는다
        path.unlink()

    images = [Image.open(ROOT / "images" / f"{sid}.png").convert("RGB") for sid in DOCS]
    images[0].save(OUT / "scanned.pdf", save_all=True, append_images=images[1:], resolution=96)

    gt = [json.loads((ROOT / "gt" / f"{sid}.json").read_text(encoding="utf-8"))["text"] for sid in DOCS]
    (OUT / "pdf_gt.json").write_text(json.dumps({"docs": DOCS, "pages": gt}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    for name in ["native.pdf", "scanned.pdf"]:
        doc = pdfium.PdfDocument(str(OUT / name))
        chars = sum(len(doc[i].get_textpage().get_text_range()) for i in range(len(doc)))
        print(f"{name}: {len(doc)}쪽, 텍스트 정보 {chars}자, {(OUT / name).stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
