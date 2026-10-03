"""13-2 — OCR 결과 검증: 반복 생성, 두 엔진의 일치도, 추출 값의 근거와 규칙.

엔진 하나의 결과만 보면 틀렸는지 알기 어렵습니다. 그래서
  ① 출력 자체의 이상 (반복 생성 → 압축률, 08-2)
  ② 다른 엔진과의 차이 (빠른 OCR과 정밀 OCR이 얼마나 다르게 읽었나)
  ③ 추출 값이 OCR 글자에 있는가, 계산이 맞는가 (12-1)
를 확인해 경고(warnings)로 남깁니다.
"""
import re
import unicodedata
import zlib
from collections import Counter


REPEAT_RATIO = 4.0   # 압축률이 이보다 크면 반복 생성으로 봄 (08-2: 정상 출력은 2.29 이하)


def compression_ratio(text):
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


def disagreement(a, b):
    """두 OCR 결과가 얼마나 다른 글자를 읽었나 (0 = 같은 글자들, 1 = 겹치는 글자 없음).

    글자 묶음(순서 무시)의 F1을 1에서 뺀 값입니다. 2단 문서처럼 읽는 순서만 다른 경우는 차이로 세지 않고,
    잘못 읽은 글자·빠뜨린 줄·반복 생성처럼 내용이 다른 경우만 셉니다.
    """
    ca, cb = Counter(re.sub(r"\s+", "", a)), Counter(re.sub(r"\s+", "", b))
    total = sum(ca.values()) + sum(cb.values())
    return 0.0 if total == 0 else 1 - 2 * sum((ca & cb).values()) / total


def squash(s):
    return re.sub(r"[\s,]", "", unicodedata.normalize("NFC", str(s))).replace("–", "-")


def check_fields(fields, flat_items, main_text, other_text):
    """추출 값 하나하나를 정밀 OCR 글자(근거)와 빠른 OCR 글자(교차 확인)에 대조합니다.

    flat_items: [(필드 이름, 값)] — 12-1의 flat()으로 펼친 목록
    돌려줌: 경고 목록 [{"field", "value", "problem"}]
    """
    main, other = squash(main_text), squash(other_text)
    warnings = []
    for name, value in flat_items:
        if value is None:
            warnings.append({"field": name, "value": None, "problem": "값을 찾지 못함"})
            continue
        if isinstance(value, (int, float)):
            continue  # 숫자는 규칙 검증이 맡음
        v = squash(value)
        if v and v not in main:
            warnings.append({"field": name, "value": value, "problem": "OCR 글자에 없음 (LLM이 만든 값일 수 있음)"})
        elif v and v not in other:
            warnings.append({"field": name, "value": value, "problem": "빠른 OCR은 다르게 읽음"})
    return warnings
