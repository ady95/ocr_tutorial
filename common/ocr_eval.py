"""OCR 따라하기 — 공통 평가 도구 (평가셋 읽기, 정규화, CER).

01장에서 CER을 직접 구현해 보고, 이후 장에서는 이 모듈을 불러 씁니다.
"""
import json
import re
import unicodedata
from pathlib import Path

import jiwer

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "ko-ocr-bench"


def normalize(text, keep_space=True):
    """비교 전에 정답과 인식 결과를 같은 규칙으로 정리합니다.

    - 유니코드 NFC 정규화: 자모가 분리된 한글(NFD)을 완성형으로 합칩니다.
    - 줄바꿈과 연속 공백을 공백 하나로 바꿉니다.
    - keep_space=False면 공백을 모두 지웁니다 (띄어쓰기 차이를 무시).
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if keep_space else text.replace(" ", "")


def cer(reference, hypothesis, keep_space=True):
    """문자 오류율(Character Error Rate). 0이면 완벽, 1이면 정답 길이만큼 틀림."""
    ref = normalize(reference, keep_space)
    hyp = normalize(hypothesis, keep_space)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def wer(reference, hypothesis):
    """단어 오류율(Word Error Rate). 한국어는 공백으로 나눈 어절이 단위라 띄어쓰기 오류도 틀린 단어가 됩니다."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def load_samples(categories=None, dataset=DATASET):
    """manifest.jsonl을 읽어 (메타 정보, 정답) 목록을 돌려줍니다."""
    samples = []
    with open(dataset / "manifest.jsonl", encoding="utf-8") as fp:
        for line in fp:
            meta = json.loads(line)
            if categories and meta["category"] not in categories:
                continue
            gt = json.loads((dataset / meta["gt"]).read_text(encoding="utf-8"))
            meta["image_path"] = str(dataset / meta["image"])
            samples.append((meta, gt))
    return samples
