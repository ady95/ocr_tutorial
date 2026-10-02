"""09장 실습 — 평가셋 이미지 140장에서 문서 파싱 도구를 08장의 VLM과 같은 표로 비교합니다 (서버 없이 실행).

먼저 score_parsers.py --images ... 로 output/mineru_images_*.json을 만들어 둡니다.
실행: python compare_images.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch08"))
import compare_vlm  # noqa: E402

CH08 = HERE.parent / "ch08" / "output"
compare_vlm.RESULTS = [
    ("PaddleOCR-VL", CH08 / "paddlevl.json"),
    ("DeepSeek-OCR 2", CH08 / "vlm_deepseek.json"),
    ("Qwen3.5-9B", CH08 / "vlm_qwen35_9b_html_pp.json"),
    ("MinerU (md)", HERE / "output" / "mineru_images_standard.json"),
    ("MinerU (json)", HERE / "output" / "mineru_images_standard_json.json"),
]

if __name__ == "__main__":
    compare_vlm.main()
