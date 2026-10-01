"""06장 — 규칙 기반 후처리: 비슷한 글자 통일, 숫자 사이 오인식 보정, 형식 검증.

실행: python rules.py [--engine paddle|tesseract]   (dump_ocr.py로 만든 결과 파일 사용)
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402

# 1) 모양이 비슷한 기호를 문서에서 흔히 쓰는 글자로 통일
SIMILAR = str.maketrans({
    "–": "-", "—": "-", "−": "-", "‐": "-",   # 여러 종류의 대시·마이너스 → 하이픈
    "×": "x",                                  # 곱셈 기호 → 영문 x
    "，": ",", "．": ".", "：": ":",           # 전각 문장부호 → 반각
    "‘": "'", "’": "'", "“": '"', "”": '"',
})

# 2) 숫자 사이에 끼어든 영문자: 1O0 → 100, 5S0 → 550 (숫자 문맥에서만)
DIGIT_LIKE = {"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"}


DIGIT_CONTEXT = re.compile(r"(?<=[\d,.])([OoDIl|SB])(?=[\d,.])")


def fix_digit_context(text):
    """양옆이 숫자(또는 쉼표·마침표)인 영문자만 숫자로 바꿉니다."""
    prev = None
    while prev != text:  # 1OO0처럼 연속된 경우를 위해 바뀌지 않을 때까지 반복
        prev = text
        text = DIGIT_CONTEXT.sub(lambda m: DIGIT_LIKE[m.group(1)], text)
    return text


def normalize_phone(text):
    """02 000 1234, 02.000.1234 처럼 구분자가 다른 전화번호를 하이픈 형식으로 통일합니다."""
    return re.sub(r"\b(0\d{1,2})[ .](\d{3,4})[ .](\d{4})\b", r"\1-\2-\3", text)


def normalize_date(text):
    """2026. 3. 15. / 2026.03.15 를 2026-03-15 형식으로 통일합니다 (원문이 하이픈 형식인 문서용)."""
    return re.sub(r"\b(20\d{2})\. ?(\d{1,2})\. ?(\d{1,2})\.?(?!\d)",
                  lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", text)


def apply_rules(text):
    text = text.translate(SIMILAR)
    text = fix_digit_context(text)
    return normalize_phone(text)


def biz_number_valid(number):
    """사업자등록번호 체크섬 검증. 'XXX-XX-XXXXX' 또는 숫자 10자리."""
    d = [int(c) for c in re.sub(r"\D", "", number)]
    if len(d) != 10:
        return False
    w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    s = sum(a * b for a, b in zip(d, w)) + (d[8] * 5) // 10
    return (10 - s % 10) % 10 == d[9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="paddle")
    args = ap.parse_args()
    ocr = json.loads((Path(__file__).resolve().parent / "output" / f"ocr_{args.engine}.json").read_text(encoding="utf-8"))

    before, after = defaultdict(list), defaultdict(list)
    for meta, gt in load_samples():
        raw = ocr[meta["id"]]["text"]
        before[meta["category"]].append(cer(gt["text"], raw))
        after[meta["category"]].append(cer(gt["text"], apply_rules(raw)))
    print(f"engine={args.engine}")
    print(f"{'범주':14s} {'규칙 전':>8s} {'규칙 후':>8s}")
    for c in sorted(before):
        print(f"{c:14s} {statistics.mean(before[c]):8.3f} {statistics.mean(after[c]):8.3f}")
    b = [x for v in before.values() for x in v]
    a = [x for v in after.values() for x in v]
    print(f"{'전체':14s} {statistics.mean(b):8.3f} {statistics.mean(a):8.3f}")

    # 사업자등록번호 검증: 영수증·세금계산서에서 OCR이 읽은 번호가 체크섬을 통과하는지
    ok = bad = 0
    for meta, gt in load_samples(["receipt", "invoice"]):
        for num in re.findall(r"\d{3}-\d{2}-\d{5}", apply_rules(ocr[meta["id"]]["text"])):
            ok, bad = (ok + 1, bad) if biz_number_valid(num) else (ok, bad + 1)
    print(f"\n사업자등록번호 형식으로 읽힌 값: {ok + bad}개, 체크섬 통과 {ok}개, 실패 {bad}개")


if __name__ == "__main__":
    main()
