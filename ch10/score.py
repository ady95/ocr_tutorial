"""10장 — bench.py가 저장한 결과를 채점하고 리포트를 만듭니다 (GPU 없이 실행).

평가셋마다 지표가 다릅니다.
  ko          범주별 CER, 보조 지표 WER, 표 TEDS, CER 0.5를 넘은(사실상 실패한) 장 수
  aihub_page  단어 정밀도·재현율·F1 — 정답에 읽기 순서가 없어, 순서와 상관없이 단어가 맞았는지만 셈
  aihub_word  단어 정확도(완전 일치)와 CER — 잘라 낸 단어 이미지 하나에 정답 단어 하나
공통으로 '과잉 출력'(정답보다 2배 넘게 긴 출력, 원문에 없는 글자를 지어냈을 가능성)을 셉니다.

실행: python score.py                 # output/*__*.json 전체 → output/report.md
      python score.py --detection     # 06장 OCR 결과로 글자줄 검출 평가
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from ocr_eval import cer, load_samples, normalize, wer  # noqa: E402
from teds import teds  # noqa: E402

DATA = HERE.parent / "datasets"
FLIPPED = {"rotated_05", "rotated_10", "rotated_15", "rotated_20"}  # 180도 뒤집힌 4장


def load_gt(dataset):
    if dataset == "ko":
        return {m["id"]: (m, g) for m, g in load_samples()}
    root = DATA / "aihub-local" / ("words" if dataset == "aihub_word" else "")
    gt = {m["id"]: (m, g) for m, g in load_samples(dataset=root)}
    return {k: v for k, v in gt.items() if dataset != "aihub_page" or v[0]["category"] == "photo_doc"}


def overflow(ref, hyp):
    return len(normalize(hyp, False)) > 2 * max(1, len(normalize(ref, False)))


def score_ko(result, gt):
    per_cat, scores, upright, tables, words, over = defaultdict(list), [], [], [], [], 0
    for sid, (meta, g) in gt.items():
        r = result.get(sid, {"text": "", "tables": []})
        c = cer(g["text"], r["text"])
        per_cat[meta["category"]].append(c)
        scores.append(c)
        words.append(wer(g["text"], r["text"]))
        if sid not in FLIPPED:
            upright.append(c)
        if meta["category"] == "table":
            tables.append(teds(r["tables"][0] if r.get("tables") else "", g["table_html"]))
        over += overflow(g["text"], r["text"])
    mean = lambda v: statistics.mean(v) if v else None  # noqa: E731  일부만 잰 결과에는 표·180도 이미지가 없을 수 있음
    return {"cat": {k: statistics.mean(v) for k, v in per_cat.items()}, "장 수": len(scores), "CER": statistics.mean(scores),
            "중앙값": statistics.median(scores), "WER": statistics.mean(words), "180도 제외": mean(upright),
            "표 TEDS": mean(tables), "실패(CER>0.5)": sum(c > 0.5 for c in scores), "과잉 출력": over}


def words_of(text):
    return Counter(normalize(text).split())


def score_page(result, gt):
    """단어 주머니(bag of words) 비교: 같은 단어가 몇 번 나왔는지까지 셈."""
    tp = fp = fn = over = 0
    for sid, (meta, g) in gt.items():
        ref = Counter(normalize(w["text"]) for w in g["words"] if not w["ignore"])
        hyp = words_of(result.get(sid, {"text": ""})["text"])
        hit = sum((ref & hyp).values())
        tp, fp, fn = tp + hit, fp + sum(hyp.values()) - hit, fn + sum(ref.values()) - hit
        over += overflow(g["text"], result.get(sid, {"text": ""})["text"])
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"정밀도": p, "재현율": r, "F1": 2 * p * r / max(1e-9, p + r), "과잉 출력": over}


def word_kind(text):
    """정답 단어의 종류: 손글씨 양식에는 날짜·번호 칸이 많아 숫자 단어가 절반을 넘음."""
    return "한글" if re.search("[가-힣]", text) else "숫자" if re.fullmatch(r"[\d\s.,:/()-]+", text) else "기타"


def score_word(result, gt):
    per_cat = defaultdict(lambda: {"acc": [], "cer": [], "over": 0, "kind": defaultdict(list)})
    for sid, (meta, g) in gt.items():
        hyp = " ".join(result.get(sid, {"text": ""})["text"].split())  # 여러 줄로 나와도 한 줄로
        s = per_cat[meta["category"]]
        s["acc"].append(normalize(hyp, False) == normalize(g["text"], False))
        s["kind"][word_kind(g["text"])].append(s["acc"][-1])
        s["cer"].append(min(1.0, cer(g["text"], hyp, keep_space=False)))  # 지어낸 긴 출력 하나가 평균을 덮지 않게 1로 자름
        s["over"] += overflow(g["text"], hyp)
    out = {}
    for c, s in per_cat.items():
        name = c.replace("_word", "")
        out[f"{name} 정확도"] = float(statistics.mean(s["acc"]))
        for kind in ("한글", "숫자"):
            if s["kind"][kind] and len(s["kind"]) > 1:  # 종류가 섞인 범주(손글씨)만
                out[f"{name} {kind}"] = float(statistics.mean(s["kind"][kind]))
        out[f"{name} CER"] = statistics.mean(s["cer"])
        out[f"{name} 과잉 출력"] = s["over"]
    return out


SCORERS = {"ko": score_ko, "aihub_page": score_page, "aihub_word": score_word}


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detection(engine, threshold=0.5):
    """글자줄 검출 평가: 정답 줄과 IoU 0.5 이상으로 겹치는 검출 상자를 하나씩 짝지어 정밀도·재현율을 셈."""
    data = json.loads((HERE.parent / "ch06" / "output" / f"ocr_{engine}.json").read_text(encoding="utf-8"))
    tp = n_pred = n_gt = 0
    for meta, g in load_samples():
        if meta["category"] == "rotated":  # 기울어진 줄은 축에 나란한 상자로 비교하기 어려워 뺌
            continue
        gts = [[min(p[0] for p in ln["poly"]), min(p[1] for p in ln["poly"]),
                max(p[0] for p in ln["poly"]), max(p[1] for p in ln["poly"])] for ln in g["lines"]]
        preds = [ln["box"] for ln in data[meta["id"]]["lines"]]
        used = set()
        for gb in gts:
            best = max(((iou(gb, pb), i) for i, pb in enumerate(preds) if i not in used), default=(0, None))
            if best[0] >= threshold:
                used.add(best[1])
                tp += 1
        n_pred, n_gt = n_pred + len(preds), n_gt + len(gts)
    p, r = tp / max(1, n_pred), tp / max(1, n_gt)
    print(f"{engine:10s} 정답 줄 {n_gt}, 검출 {n_pred}, 정밀도 {p:.3f}, 재현율 {r:.3f}, F1 {2 * p * r / max(1e-9, p + r):.3f}")


def fmt(v):
    if v is None:
        return "-"  # 해당 이미지가 없어 계산하지 않음
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection", action="store_true")
    args = ap.parse_args()
    if args.detection:
        for engine in ("tesseract", "paddle", "easyocr"):
            detection(engine)
        return

    gts, lines = {}, ["# OCR Benchmark 결과", ""]
    by_dataset = defaultdict(dict)
    for path in sorted((HERE / "output").glob("*__*.json")):
        engine, dataset = path.stem.split("__")
        result = json.loads(path.read_text(encoding="utf-8"))
        meta = result.pop("_meta", {})
        gts.setdefault(dataset, load_gt(dataset))
        # --limit·--every로 일부만 잰 결과는 그 이미지들만 채점 (전체 실행은 오류가 난 장도 빈 결과로 저장되어 있음)
        row = SCORERS[dataset](result, {k: v for k, v in gts[dataset].items() if k in result})
        row["장당 시간(초)"] = meta.get("time_median", float("nan"))
        row["GPU 최대(MiB)"] = meta.get("gpu_peak_mib", 0)
        by_dataset[dataset][engine] = row
    for dataset, rows in by_dataset.items():
        cols = list(dict.fromkeys(c for r in rows.values() for c in r if c != "cat"))  # 부분집합 결과는 열이 적을 수 있음
        lines += [f"## {dataset}", "", "| 엔진 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {e} | " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for e, r in rows.items()]
        if dataset == "ko":
            cats = sorted({c for r in rows.values() for c in r["cat"]})  # 엔진마다 잰 범주가 다를 수 있음
            lines += ["", "| 엔진 | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
            lines += [f"| {e} | " + " | ".join(fmt(r["cat"].get(c)) for c in cats) + " |" for e, r in rows.items()]
        lines.append("")
    report = "\n".join(lines)
    (HERE / "output" / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
