"""08장 실습 — 저장해 둔 여러 도구의 결과를 한 표로 비교합니다 (서버 없이 실행).

각 벤치 스크립트가 output/에 저장한 결과(이미지 id → text)를 읽어 같은 기준으로 채점합니다.
  - 범주별 평균 CER
  - 전체 평균·중앙값, CER 0.5를 넘은(사실상 실패한) 장 수
  - 180도 뒤집힌 4장을 뺀 평균 (방향 보정을 앞에 둔다고 가정했을 때)
실행: python compare_vlm.py
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402

RESULTS = [  # (이름, 결과 파일)
    ("PaddleOCR", HERE.parent / "ch06" / "output" / "ocr_paddle.json"),
    ("PaddleOCR-VL", HERE / "output" / "paddlevl.json"),
    ("Surya OCR 2", HERE / "output" / "surya.json"),
    # DeepSeek-OCR 2는 공식 방법(deepseek_hf.py)의 결과를 씀. vLLM 경로(vlm_bench.py --kind deepseek)의 결과는
    # vlm_deepseek.json으로 따로 저장되며, 08-5에서 본 것처럼 정상 출력이 아니므로 비교에 쓰지 않음
    ("DeepSeek-OCR 2", HERE / "output" / "vlm_deepseek_hf.json"),
    ("VARCO-OCR", HERE / "output" / "vlm_varco.json"),
    ("Qwen3.5-4B", HERE / "output" / "vlm_qwen35_4b_html_pp.json"),
    ("Qwen3.5-9B", HERE / "output" / "vlm_qwen35_9b_html_pp.json"),
]
CATEGORIES = ["print_ko", "print_mixed", "print_numeric", "small", "lowres", "receipt", "invoice", "card",
              "table", "multicol", "rotated"]
FLIPPED = {"rotated_05", "rotated_10", "rotated_15", "rotated_20"}  # 180도 회전


def main():
    samples = load_samples()
    rows = {}
    for name, path in RESULTS:
        if not path.exists():
            print(f"(건너뜀) {name}: {path.name} 없음")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        per_cat, scores, upright = defaultdict(list), [], []
        for meta, gt in samples:
            c = cer(gt["text"], data[meta["id"]]["text"])
            per_cat[meta["category"]].append(c)
            scores.append(c)
            if meta["id"] not in FLIPPED:
                upright.append(c)
        rows[name] = {"cat": {k: statistics.mean(v) for k, v in per_cat.items()}, "mean": statistics.mean(scores),
                      "median": statistics.median(scores), "fail": sum(c > 0.5 for c in scores),
                      "upright": statistics.mean(upright)}

    names = list(rows)
    print(f"{'범주':14s}" + "".join(f"{n:>15s}" for n in names))
    for c in CATEGORIES:
        print(f"{c:14s}" + "".join(f"{rows[n]['cat'][c]:15.3f}" for n in names))
    for key, label in [("mean", "전체 평균"), ("median", "전체 중앙값"), ("upright", "180도 제외 평균")]:
        print(f"{label:11s}" + "".join(f"{rows[n][key]:15.3f}" for n in names))
    print(f"{'CER>0.5 장수':10s}" + "".join(f"{rows[n]['fail']:15d}" for n in names))
    best = []
    for c in CATEGORIES:
        scores = {n: rows[n]["cat"][c] for n in names}
        best.append(f"{c}={min(scores, key=scores.get)}")
    print("\n범주별 최고:", ", ".join(best))


if __name__ == "__main__":
    main()
