"""11장 — profile_paddle.py 등의 결과(output/*__*.json)를 10장 채점기로 채점해 한 표로 보여 줍니다."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "ch10"))
import score  # noqa: E402


def main():
    gts = {}
    print(f"{'설정':40s} {'평가셋':11s} {'장':>4s} {'정확도':>18s} {'장당 평균':>9s} {'중앙값':>8s}")
    for path in sorted((HERE / "output").glob("*__*.json")):
        name, dataset = path.stem.split("__")
        result = json.loads(path.read_text(encoding="utf-8"))
        meta = result.pop("_meta", {})
        gts.setdefault(dataset, score.load_gt(dataset))
        gt = {k: v for k, v in gts[dataset].items() if k in result}
        row = score.SCORERS[dataset](result, gt)
        acc = (f"CER {row['CER']:.3f}" if dataset == "ko" else f"F1 {row['F1']:.3f}" if dataset == "aihub_page"
               else f"손글씨 {row.get('handwriting 정확도', 0):.3f}")
        print(f"{name:40s} {dataset:11s} {len(gt):4d} {acc:>18s} {meta.get('time_mean', 0) * 1000:7.0f}ms "
              f"{meta.get('time_median', 0) * 1000:6.0f}ms")


if __name__ == "__main__":
    main()
