"""09장 — Docling의 OCR 엔진과 설정을 바꿔 가며 PDF를 변환합니다 (Python API).

실행 예:
  python docling_ocr.py --engine rapidocr --lang korean --name rapid_ko
  python docling_ocr.py --engine rapidocr --lang korean --full-page --name rapid_ko_full
  python docling_ocr.py --engine easyocr --lang ko en --threshold 0.3 --name easy_ko_t03
결과: output/docling/<name>/<pdf 이름>.md, .json  (채점은 score_parsers.py)
"""
import argparse
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (EasyOcrOptions, PdfPipelineOptions, RapidOcrOptions,
                                                TesseractCliOcrOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption

HERE = Path(__file__).resolve().parent


def ocr_options(args):
    common = {"lang": args.lang, "force_full_page_ocr": args.full_page}
    if args.engine == "rapidocr":
        extra = {"text_score": args.threshold} if args.threshold is not None else {}
        return RapidOcrOptions(**common, **extra)
    if args.engine == "easyocr":
        extra = {"confidence_threshold": args.threshold} if args.threshold is not None else {}
        return EasyOcrOptions(**common, **extra)
    return TesseractCliOcrOptions(**common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(HERE.parent / "ch07" / "pdf" / "scanned.pdf"))
    ap.add_argument("--engine", choices=["rapidocr", "easyocr", "tesseract"], default="rapidocr")
    ap.add_argument("--lang", nargs="+", default=["korean"])
    ap.add_argument("--full-page", action="store_true", help="레이아웃과 상관없이 페이지 전체를 OCR")
    ap.add_argument("--threshold", type=float, help="이 신뢰도보다 낮은 글자줄은 버림 (엔진 기본값 0.5)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options(args))
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    start = time.perf_counter()
    doc = converter.convert(args.pdf).document
    elapsed = time.perf_counter() - start
    out = HERE / "output" / "docling" / args.name
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.pdf).stem
    doc.save_as_markdown(out / f"{stem}.md")
    doc.save_as_json(out / f"{stem}.json")
    print(f"{args.name}: {elapsed:.1f}초 ({len(doc.pages)}쪽) → {out / (stem + '.md')}")


if __name__ == "__main__":
    main()
