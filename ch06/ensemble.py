"""06장 — 여러 OCR 결과 합치기: 신뢰도 선택, 줄 단위 투표, LLM Judge.

입력: dump_ocr.py로 저장한 세 도구의 결과 (paddle, tesseract, easyocr)
비교:
  단일 도구           각 도구를 그대로
  이미지별 신뢰도 선택 이미지마다 평균 신뢰도가 가장 높은 도구의 결과
  줄 단위 투표        PaddleOCR 줄을 기준으로, 같은 위치의 세 후보 중 나머지와 가장 비슷한 것을 고름
  LLM Judge          세 결과를 함께 주고 하나로 합치게 함 (OPENAI_API_KEY 있을 때)
  이론적 상한         이미지마다 정답에 가장 가까운 도구 (앙상블로 얻을 수 있는 최대치)

실행: python ensemble.py
"""
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402
from llm_correct import new_items, squash  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
ENGINES = ["paddle", "tesseract", "easyocr"]


def overlap_ratio(a, b):
    """겹친 넓이 ÷ 작은 상자의 넓이 (한 줄이 여러 조각으로 나뉜 경우를 잡기 위해)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def similarity(a, b):
    """공백·문장부호를 뺀 두 문자열의 유사도 (1 - CER 꼴, 0~1)."""
    a, b = squash(a), squash(b)
    if not a and not b:
        return 1.0
    return max(0.0, 1 - cer(a, b, keep_space=False)) if a else 0.0


def vote_lines(results):
    """PaddleOCR 줄마다 같은 위치의 다른 도구 텍스트를 모아, 나머지와 가장 비슷한 후보를 고릅니다."""
    anchor = results["paddle"]["lines"]
    candidates = [[ln["text"]] for ln in anchor]
    for engine in ["tesseract", "easyocr"]:
        pieces = defaultdict(list)
        for ln in results[engine]["lines"]:
            scores = [overlap_ratio(ln["box"], a["box"]) for a in anchor]
            if scores and max(scores) >= 0.5:
                pieces[scores.index(max(scores))].append(ln)
        for k, lns in pieces.items():
            candidates[k].append(" ".join(x["text"] for x in sorted(lns, key=lambda x: x["box"][0])))
    chosen = []
    for cands in candidates:
        if len(cands) == 1:
            chosen.append(cands[0])
            continue
        # 후보마다 다른 후보들과의 유사도 합 → 가장 "다수 의견"에 가까운 후보 (동점이면 PaddleOCR 우선)
        totals = [sum(similarity(c, o) for o in cands if o is not c) for c in cands]
        chosen.append(cands[max(range(len(cands)), key=lambda i: (totals[i], i == 0))])
    return "\n".join(chosen)


JUDGE_PROMPT = """같은 한국어 문서 이미지를 세 OCR 도구로 읽은 결과입니다. 세 결과를 비교해 원래 문서의 텍스트를 복원하세요.

규칙:
- 세 결과 중 적어도 하나에 근거가 있는 내용만 씁니다. 새로운 내용을 지어내지 않습니다.
- 띄어쓰기는 자연스러운 한국어로 맞춥니다. 숫자·금액·날짜는 결과들 사이에서 다수가 일치하는 값을 씁니다.
- 줄 구성은 결과 A를 따릅니다. 설명 없이 복원한 텍스트만 출력합니다.

[결과 A]
{a}

[결과 B]
{b}

[결과 C]
{c}"""


def llm_judge(client, results):
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            a=results["paddle"]["text"], b=results["tesseract"]["text"], c=results["easyocr"]["text"])}])
    return resp.choices[0].message.content.strip()


def main():
    ocr = {e: json.loads((OUT / f"ocr_{e}.json").read_text(encoding="utf-8")) for e in ENGINES}
    samples = load_samples()
    cache_path = OUT / "llm_judge.json"
    judge = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [m["id"] for m, _ in samples if m["id"] not in judge]
    if todo and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        with ThreadPoolExecutor(6) as pool:
            for sid, text in zip(todo, pool.map(lambda s: llm_judge(client, {e: ocr[e][s] for e in ENGINES}), todo)):
                judge[sid] = text
        cache_path.write_text(json.dumps(judge, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = defaultdict(lambda: defaultdict(list))  # 방법 → 범주 → CER(공백 무시)
    hallucinated = 0
    for meta, gt in samples:
        sid, cat, ref = meta["id"], meta["category"], gt["text"]
        res = {e: ocr[e][sid] for e in ENGINES}
        per_engine = {e: cer(ref, res[e]["text"], keep_space=False) for e in ENGINES}
        for e in ENGINES:
            rows[e][cat].append(per_engine[e])
        mean_conf = {e: statistics.mean([ln["score"] for ln in res[e]["lines"]] or [0]) for e in ENGINES}
        rows["신뢰도 선택"][cat].append(per_engine[max(mean_conf, key=mean_conf.get)])
        rows["줄 단위 투표"][cat].append(cer(ref, vote_lines(res), keep_space=False))
        if sid in judge:
            rows["LLM Judge"][cat].append(cer(ref, judge[sid], keep_space=False))
            w, n = new_items(judge[sid], "\n".join(res[e]["text"] for e in ENGINES), ref)
            hallucinated += len(w) + len(n)
        rows["이론적 상한"][cat].append(min(per_engine.values()))

    cats = sorted(rows["paddle"])
    print("CER (공백 무시) — 범주별 평균")
    print(f"{'방법':12s}" + "".join(f"{c[:8]:>9s}" for c in cats) + f"{'전체':>8s}")
    for method, per_cat in rows.items():
        allv = [x for v in per_cat.values() for x in v]
        print(f"{method:12s}" + "".join(f"{statistics.mean(per_cat[c]):9.3f}" for c in cats)
              + f"{statistics.mean(allv):8.3f}")
    if judge:
        print(f"\nLLM Judge가 세 결과 어디에도 없이 만들어 낸 어절·숫자: {hallucinated}개")


if __name__ == "__main__":
    main()
