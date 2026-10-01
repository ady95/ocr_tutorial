"""04장 — PaddleOCR로 첫 번째 한국어 OCR.

실행: python paddle_first.py <이미지> [--device cpu]
결과: 인식한 글자줄과 신뢰도를 출력하고, output/ 폴더에 시각화 이미지와 JSON을 저장합니다.
"""
import argparse
import time

from paddleocr import PaddleOCR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()

    start = time.perf_counter()
    ocr = PaddleOCR(
        lang="korean",                      # 한국어 → PP-OCRv5 한국어 인식 모델
        use_doc_orientation_classify=False,  # 문서 방향 보정 끔
        use_doc_unwarping=False,             # 문서 왜곡 펴기 끔
        use_textline_orientation=False,      # 글자줄 방향 분류 끔
        device=args.device,
    )
    print(f"모델 준비: {time.perf_counter() - start:.1f}초")

    start = time.perf_counter()
    result = ocr.predict(args.image)
    print(f"인식: {time.perf_counter() - start:.2f}초")

    for res in result:
        for text, score in zip(res["rec_texts"], res["rec_scores"]):
            print(f"{score:.3f}  {text}")
        res.save_to_img("output")
        res.save_to_json("output")


if __name__ == "__main__":
    main()
