"""06장 — LLM으로 OCR 오류 보정하기와 hallucination 측정.

OCR 결과를 LLM에 주고 띄어쓰기·명백한 오인식만 고치게 한 뒤,
CER 변화와 함께 LLM이 만들어 낸 내용을 셉니다.
  - 새 어절: 공백·문장부호를 지워도 OCR 결과와 정답 어디에도 없는 어절 (지어낸 내용)
  - 새 숫자: OCR 결과에도 정답에도 없는 숫자열 (금액·날짜를 바꾼 경우 — 가장 위험)

실행: python llm_correct.py --engine paddle     (OPENAI_API_KEY, OPENAI_MODEL 환경변수 필요)
결과는 output/llm_<engine>.json에 저장되고, 다시 실행하면 저장된 결과를 씁니다.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples, normalize  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

PROMPT = """다음은 한국어 문서를 OCR로 인식한 결과입니다. OCR 오류만 고쳐 주세요.

규칙:
- 띄어쓰기 오류와, 문맥상 명백한 글자 오인식만 고칩니다.
- 내용을 새로 추가하거나 빼지 않습니다. 줄 순서와 줄바꿈은 그대로 둡니다.
- 숫자·금액·날짜·전화번호는 OCR 결과에 있는 그대로 둡니다. 확실하지 않으면 고치지 않습니다.
- 설명 없이 고친 텍스트만 출력합니다.

OCR 결과:
{text}"""


def correct(client, text, retries=2):
    """빈 응답이 오면 다시 시도하고, 끝내 비면 OCR 결과를 그대로 돌려줍니다."""
    for _ in range(retries):
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": PROMPT.format(text=text)}])
        content = resp.choices[0].message.content
        if content and content.strip():
            return content.strip()
    return text


def squash(text):
    """공백과 문장부호를 지운 비교용 문자열."""
    return re.sub(r"[\s\W_]+", "", normalize(text))


def new_items(output, ocr_text, gt_text):
    """출력의 어절 중, 공백·문장부호를 지운 형태가 OCR 입력과 정답 어디에도 없는 것 (지어낸 내용).

    띄어쓰기만 바꾸거나 따옴표 종류만 바꾼 경우는 세지 않습니다.
    숫자열은 따로 셉니다 (금액·날짜를 바꾸면 가장 위험하므로).
    """
    pool = squash(ocr_text) + "|" + squash(gt_text)
    new_w = {w for w in normalize(output).split() if squash(w) and squash(w) not in pool}
    nums = lambda s: set(re.findall(r"\d[\d,.:-]*\d|\d", s))  # noqa: E731
    new_n = nums(output) - nums(ocr_text) - nums(gt_text)
    return new_w, new_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    ocr = json.loads((OUT / f"ocr_{args.engine}.json").read_text(encoding="utf-8"))
    cache_path = OUT / f"llm_{args.engine}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    samples = load_samples()

    todo = [m["id"] for m, _ in samples if m["id"] not in cache]
    if todo:
        client = OpenAI()
        with ThreadPoolExecutor(args.workers) as pool:
            for sid, fixed in zip(todo, pool.map(lambda s: correct(client, ocr[s]["text"]), todo)):
                cache[sid] = fixed
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    before, after, before_ns, after_ns = (defaultdict(list) for _ in range(4))
    new_words, new_nums, examples = 0, 0, []
    for meta, gt in samples:
        raw, fixed, cat = ocr[meta["id"]]["text"], cache[meta["id"]], meta["category"]
        before[cat].append(cer(gt["text"], raw))
        after[cat].append(cer(gt["text"], fixed))
        before_ns[cat].append(cer(gt["text"], raw, keep_space=False))
        after_ns[cat].append(cer(gt["text"], fixed, keep_space=False))
        w, n = new_items(fixed, raw, gt["text"])
        new_words += len(w)
        new_nums += len(n)
        if w or n:
            examples.append((meta["id"], sorted(w)[:3], sorted(n)[:3]))

    print(f"engine={args.engine} model={MODEL}")
    print(f"{'범주':14s} {'CER 전':>7s} {'CER 후':>7s} {'공백무시 전':>10s} {'공백무시 후':>10s}")
    for c in sorted(before):
        print(f"{c:14s} {statistics.mean(before[c]):7.3f} {statistics.mean(after[c]):7.3f}"
              f" {statistics.mean(before_ns[c]):10.3f} {statistics.mean(after_ns[c]):10.3f}")
    allv = lambda d: statistics.mean([x for v in d.values() for x in v])  # noqa: E731
    print(f"{'전체':14s} {allv(before):7.3f} {allv(after):7.3f} {allv(before_ns):10.3f} {allv(after_ns):10.3f}")
    print(f"\nLLM이 새로 만든 어절 {new_words}개, 새로 만든 숫자열 {new_nums}개")
    for sid, w, n in examples[:12]:
        print(f"  {sid}: 어절 {w} 숫자 {n}")


if __name__ == "__main__":
    main()
