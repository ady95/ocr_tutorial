"""13장 — 엔진의 단계별 결과를 평가셋 140장으로 잽니다.

collect: 장마다 빠른 OCR(+ 180도 확인) → PaddleOCR-VL → Qwen3.5(VLM)를 모두 돌려 단계별 결과와 시간을 저장
         (재시도 기준을 바꿔 가며 비교하려고 VLM도 모든 장에서 돌림) → output/stages.json
report:  저장한 결과로 ① 유형 판별 정확도 ② 방향 보정 ③ 정책별 CER·시간(빠른 OCR만 / 정밀 OCR만 / 엔진) 을 계산
extract: 명함·세금계산서·영수증을 엔진 그대로(판별 → 추출 → 검증) 처리해 필드 정확도와 경고의 적중률을 잼
실행: python eval_engine.py collect   →   python eval_engine.py report   →   python eval_engine.py extract
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "ch12"))
from ocr_eval import cer, load_samples  # noqa: E402

from ocrengine.classify import classify  # noqa: E402
from ocrengine.verify import REPEAT_RATIO, compression_ratio, disagreement  # noqa: E402

OUT = HERE / "output"
DOC = {"print_ko", "print_mixed", "print_numeric", "small", "multicol"}


def true_type(meta, by_id):
    cat = meta["category"] if meta["source"] == "rendered" else by_id[meta["parent"]]["category"]
    return "document" if cat in DOC else cat


def timed(fn, *a):
    t = time.perf_counter()
    out = fn(*a)
    return out, round(time.perf_counter() - t, 3)


def collect(args):
    from ocrengine import OcrEngine
    eng = OcrEngine(vl_url=args.vl_url, llm_url=args.llm_url)
    blank = cv2.imread(str(next(iter(load_samples()))[0]["image_path"]))
    for warm in (eng.fast_ocr, eng.vl_ocr, eng.vlm_ocr):  # 첫 실행(모델 초기화)은 시간에서 뺌
        warm(blank)
    saved = {}
    for meta, gt in load_samples():
        img = cv2.imread(meta["image_path"])
        (fixed, lines, angle), t_fast = timed(eng.read_upright, img)
        vl, t_vl = timed(eng.vl_ocr, fixed)
        vlm, t_vlm = timed(eng.vlm_ocr, fixed)
        saved[meta["id"]] = {"angle": angle, "size": [fixed.shape[1], fixed.shape[0]], "lines": lines, "vl": vl, "vlm": vlm,
                             "times": {"fast_ocr": t_fast, "vl_ocr": t_vl, "vlm_ocr": t_vlm}}
        print(meta["id"], angle, t_fast, t_vl, t_vlm, flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "stages.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")


def policy(st, threshold):
    """엔진의 선택 규칙 (engine.py process와 같음): 의심스러우면 VLM 결과와 비교해 고름."""
    fast = "\n".join(ln["text"] for ln in st["lines"])
    r1, d1 = compression_ratio(st["vl"]["text"]), disagreement(fast, st["vl"]["text"])
    if r1 <= REPEAT_RATIO and d1 <= threshold:
        return st["vl"]["text"], False
    r2, d2 = compression_ratio(st["vlm"]["text"]), disagreement(fast, st["vlm"]["text"])
    pick = st["vlm"]["text"] if r2 <= REPEAT_RATIO and (r1 > REPEAT_RATIO or d2 < d1) else st["vl"]["text"]
    return pick, True


def report(args):
    stages = json.loads((OUT / "stages.json").read_text(encoding="utf-8"))
    samples = load_samples()
    by_id = {m["id"]: m for m, _ in samples}
    # ① 유형 판별
    confusion, wrong = Counter(), []
    for meta, gt in samples:
        st = stages[meta["id"]]
        got, want = classify(st["lines"], *st["size"])["type"], true_type(meta, by_id)
        confusion[(want, got)] += 1
        if got != want:
            wrong.append(f"{meta['id']}({want}→{got})")
    types = ["invoice", "receipt", "card", "table", "document"]
    print("① 유형 판별 (행: 정답, 열: 판별)")
    print("정답\\판별".ljust(10) + "".join(t[:8].rjust(9) for t in types))
    for w in types:
        print(w.ljust(10) + "".join(str(confusion[(w, g)]).rjust(9) for g in types))
    ok = sum(v for (w, g), v in confusion.items() if w == g)
    print(f"정확도 {ok}/{len(samples)} = {ok / len(samples):.3f}  틀림: {', '.join(wrong) or '없음'}")
    # ② 방향 보정
    rot = Counter((m["category"] == "rotated", stages[m["id"]]["angle"]) for m, _ in samples)
    print("② 방향 보정: (회전 범주인가, 보정 각도) →", dict(rot))
    # ③ 정책별 CER·시간
    rows = defaultdict(lambda: defaultdict(list))
    retried = []
    for meta, gt in samples:
        st, cat = stages[meta["id"]], meta["category"]
        fast = "\n".join(ln["text"] for ln in st["lines"])
        t = st["times"]
        picked, retry = policy(st, args.threshold)
        retried += [meta["id"]] if retry else []
        base = t["fast_ocr"]
        for name, text, sec in [("빠른 OCR만", fast, base), ("정밀 OCR만", st["vl"]["text"], base + t["vl_ocr"]),
                                ("VLM만", st["vlm"]["text"], base + t["vlm_ocr"]),
                                (f"엔진 (기준 {args.threshold})", picked, base + t["vl_ocr"] + (t["vlm_ocr"] if retry else 0))]:
            c = cer(gt["text"], text)
            for key in ("전체", cat):
                rows[name][key].append((c, sec))
    cats = ["전체", "print_ko", "print_mixed", "print_numeric", "small", "multicol", "table", "receipt", "invoice", "card",
            "lowres", "rotated"]
    print(f"③ 정책별 CER (평균) — 재시도 {len(retried)}장: {', '.join(retried)}")
    print("정책".ljust(18) + "".join(c[:9].rjust(10) for c in cats) + "  장당 초(중앙값)")
    for name, d in rows.items():
        print(name.ljust(18) + "".join(f"{statistics.mean(c for c, _ in d[k]):10.3f}" for k in cats)
              + f"  {statistics.median(s for _, s in d['전체']):.2f} (평균 {statistics.mean(s for _, s in d['전체']):.2f})")
    print("중앙값 CER:", {n: round(statistics.median(c for c, _ in d["전체"]), 4) for n, d in rows.items()})
    # 재시도 기준별
    for th in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 99):
        res = [(cer(g["text"], policy(stages[m["id"]], th)[0]), policy(stages[m["id"]], th)[1]) for m, g in samples]
        print(f"기준 {th}: 재시도 {sum(r for _, r in res)}장, CER {statistics.mean(c for c, _ in res):.3f}")


def extract_eval(args):
    """명함·세금계산서·영수증 40장을 엔진 그대로 처리: 판별 → 정밀 OCR → 추출 → 검증."""
    from extract import flat, norm
    from ocrengine import OcrEngine
    eng = OcrEngine(vl_url=args.vl_url, llm_url=args.llm_url)
    stats = Counter()
    saved = {}
    for meta, gt in load_samples(["card", "invoice", "receipt"]):
        res = eng.process(cv2.imread(meta["image_path"]))
        saved[meta["id"]] = {k: res.get(k) for k in ("type", "engine", "fields", "warnings", "times", "checks")}
        if res["type"] != meta["category"] or "fields" not in res:
            stats["판별 실패 또는 추출 실패"] += 1
            continue
        warned = {w.get("field") for w in res["warnings"] if w.get("field")}
        if meta["category"] == "receipt":
            truth = {k: gt["fields"][k] for k in ("store", "biz_no", "tel", "date")}
            pred = {k: res["fields"].get(k) for k in truth}
        else:
            truth, pred = dict(flat(meta["category"], gt["fields"])), dict(flat(meta["category"], res["fields"]))
        for k, v in truth.items():
            if isinstance(v, (int, float)):
                continue  # 문자열 필드만 (숫자는 규칙 검증)
            right = norm(pred.get(k)) == norm(v)
            stats[("맞음" if right else "틀림") + (" + 경고" if k in warned else "")] += 1
        print(meta["id"], res["engine"], [w["problem"][:30] for w in res["warnings"]], flush=True)
    print("문자열 필드:", dict(stats))
    (OUT / "extract_engine.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "report", "extract"])
    ap.add_argument("--threshold", type=float, default=0.3, help="빠른 OCR과 정밀 OCR의 차이가 이보다 크면 다시 읽음")
    ap.add_argument("--vl-url", default="http://localhost:18118/v1")
    ap.add_argument("--llm-url", default="http://localhost:18001/v1")
    args = ap.parse_args()
    {"collect": collect, "report": report, "extract": extract_eval}[args.step](args)


if __name__ == "__main__":
    main()
