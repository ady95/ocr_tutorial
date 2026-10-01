"""05장 — 인식 모델의 문자 사전에 평가셋의 글자가 모두 들어 있는지 확인합니다.

인식 모델은 사전에 있는 글자만 출력할 수 있습니다. 사전에 없는 글자는 절대 맞힐 수 없습니다.
PaddleOCR가 내려받은 모델 폴더의 inference.yml에서 문자 사전을 읽습니다.

실행: python dict_coverage.py [모델 이름 ...]
"""
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import load_samples  # noqa: E402

MODELS = Path.home() / ".paddlex" / "official_models"


def load_dict(model):
    cfg = yaml.safe_load((MODELS / model / "inference.yml").read_text(encoding="utf-8"))
    return set(cfg["PostProcess"]["character_dict"])


def main():
    models = sys.argv[1:] or ["korean_PP-OCRv5_mobile_rec", "PP-OCRv6_medium_rec"]
    text = "".join(unicodedata.normalize("NFC", gt["text"]) for _, gt in load_samples())
    used = Counter(c for c in text if not c.isspace())
    print(f"평가셋에 쓰인 글자 {len(used):,}종, 전체 {sum(used.values()):,}자")
    for model in models:
        chars = load_dict(model)
        hangul = sum(1 for c in chars if "가" <= c <= "힣")
        missing = {c: n for c, n in used.items() if c not in chars}
        lost = sum(missing.values())
        print(f"\n[{model}] 사전 {len(chars):,}자 (완성형 한글 {hangul:,}자)")
        print(f"  평가셋 글자 중 사전에 없는 것: {len(missing)}종, {lost:,}자 ({lost / sum(used.values()):.2%})")
        top = sorted(missing.items(), key=lambda kv: -kv[1])[:15]
        print("  자주 나오는 순:", " ".join(f"{c}({n})" for c, n in top))


if __name__ == "__main__":
    main()
