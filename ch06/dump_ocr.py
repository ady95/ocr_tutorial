"""06장 — 평가셋 전체의 OCR 결과를 파일로 저장합니다 (후처리·앙상블 실험을 OCR 없이 반복하기 위해).

저장 형식 (output/ocr_<engine>.json): {이미지 id: {"text": 전체 텍스트, "lines": [{"text", "score", "box"}]}}
  paddle:    PaddleOCR lang="korean", 글자줄 단위, score 0~1
  tesseract: kor+eng PSM 4, text는 image_to_string, lines는 image_to_data를 줄 단위로 묶은 것 (score 0~1)
  easyocr:   ko+en, 검출된 영역 단위, score 0~1

실행: python dump_ocr.py --engine paddle
      python dump_ocr.py --engine tesseract
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import load_samples  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def paddle_runner():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="korean", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)

    def run(path):
        res = ocr.predict(path)[0]
        lines = [{"text": t, "score": round(float(s), 4), "box": [int(v) for v in b]}
                 for t, s, b in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"].tolist())]
        return {"text": "\n".join(res["rec_texts"]), "lines": lines}
    return run


def tesseract_runner():
    import pytesseract
    from PIL import Image
    config = "--psm 4"

    def run(path):
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="kor+eng", config=config)
        data = pytesseract.image_to_data(img, lang="kor+eng", config=config, output_type=pytesseract.Output.DICT)
        groups = defaultdict(list)
        for i, w in enumerate(data["text"]):
            if w.strip() and float(data["conf"][i]) >= 0:
                groups[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(i)
        lines = []
        for idx in groups.values():
            box = [min(data["left"][i] for i in idx), min(data["top"][i] for i in idx),
                   max(data["left"][i] + data["width"][i] for i in idx),
                   max(data["top"][i] + data["height"][i] for i in idx)]
            lines.append({"text": "".join(data["text"][i] for i in idx),  # 음절 단위로 쪼개지므로 공백 없이 잇는다
                          "score": round(statistics.mean(float(data["conf"][i]) for i in idx) / 100, 4),
                          "box": box})
        return {"text": text, "lines": lines}
    return run


def easyocr_runner():
    import easyocr
    reader = easyocr.Reader(["ko", "en"])

    def run(path):
        lines = []
        for box, text, score in reader.readtext(path):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            lines.append({"text": text, "score": round(float(score), 4),
                          "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
        return {"text": "\n".join(ln["text"] for ln in lines), "lines": lines}
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["paddle", "tesseract", "easyocr"], required=True)
    args = ap.parse_args()
    run = {"paddle": paddle_runner, "tesseract": tesseract_runner, "easyocr": easyocr_runner}[args.engine]()
    out = {}
    for meta, _ in load_samples():
        out[meta["id"]] = run(meta["image_path"])
    OUT.mkdir(exist_ok=True)
    path = OUT / f"ocr_{args.engine}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", path, len(out), "장")


if __name__ == "__main__":
    main()
