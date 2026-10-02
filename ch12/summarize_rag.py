"""12장 — rag.py가 저장한 결과(output/rag_*.json)를 같은 기준으로 다시 채점해 한 표로 보여 줍니다.

질문 범주별 답 정확도도 함께 봅니다 (OCR 오류가 숫자·표 질문에 더 큰 영향을 주는지).
실행: python summarize_rag.py
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from rag import char_f1, is_correct  # noqa: E402

GROUPS = {"글 문서": {"print_ko", "print_mixed", "small", "multicol"}, "숫자 문서": {"print_numeric"},
          "표": {"table"}, "영수증·계산서·명함": {"receipt", "invoice", "card"}}


def main():
    head = ["출처", "hit@1", "hit@3", "답 정확도", "글자 F1", "모름", "틀린 답"] + list(GROUPS)
    print(" | ".join(head))
    for path in sorted((HERE / "output").glob("rag_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if "pred" not in rows[0]:
            continue
        for r in rows:
            r["correct"] = is_correct(r["answer"], r["pred"])
        n = len(rows)
        by = defaultdict(list)
        for r in rows:
            for g, cats in GROUPS.items():
                if r["category"] in cats:
                    by[g].append(r["correct"])
        vals = [path.stem[4:], f"{sum(r['rank'] == 1 for r in rows) / n:.3f}", f"{sum(r['rank'] is not None for r in rows) / n:.3f}",
                f"{sum(r['correct'] for r in rows) / n:.3f}", f"{statistics.mean(char_f1(r['answer'], r['pred']) for r in rows):.3f}",
                f"{sum(r['unknown'] for r in rows) / n:.3f}", f"{sum(not r['correct'] and not r['unknown'] for r in rows) / n:.3f}"]
        vals += [f"{sum(by[g]) / len(by[g]):.3f}" for g in GROUPS]
        print(" | ".join(vals))


if __name__ == "__main__":
    main()
