"""10장 — 여러 OCR 엔진을 같은 조건으로 실행하고, 결과와 처리 시간·GPU 메모리를 저장하는 Benchmark 실행기.

평가셋
  ko          ko-ocr-bench 140장 (합성, 공개)
  aihub_page  AI Hub 스마트폰 촬영 문서 50장 (aihub_prepare.py로 만든 로컬 평가셋)
  aihub_word  AI Hub 손글씨·간판 단어 이미지 1,469개

실행 예:
  python bench.py --engine paddle --dataset ko
  python bench.py --engine qwen --dataset aihub_word --opt base_url=http://localhost:8000/v1
결과: output/<engine>__<dataset>.json  — 이미지 id마다 글자와 처리 시간, "_meta"에 GPU 메모리와 전체 시간
채점은 score.py가 합니다 (GPU 없이 실행).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from engines import ENGINES  # noqa: E402

DATA = HERE.parent / "datasets"
DATASETS = {"ko": DATA / "ko-ocr-bench", "aihub_page": DATA / "aihub-local", "aihub_word": DATA / "aihub-local" / "words"}
PAGE_CATEGORIES = {"aihub_page": {"photo_doc"}}  # aihub-local의 손글씨·간판은 쪽 전체 정답이 없어 단어 평가셋으로만 씀


def load_items(dataset):
    root = DATASETS[dataset]
    keep = PAGE_CATEGORIES.get(dataset)
    items = []
    with open(root / "manifest.jsonl", encoding="utf-8") as fp:
        for line in fp:
            m = json.loads(line)
            if keep is None or m["category"] in keep:
                items.append((m["id"], str(root / m["image"])))
    return items


class GpuMonitor(threading.Thread):
    """nvidia-smi로 0.5초마다 GPU 메모리 사용량을 읽어 최댓값을 기록합니다 (다른 프로세스가 쓰는 양도 포함)."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.cmd = ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        self.interval, self.peak, self.stop_flag = interval, 0, threading.Event()
        self.baseline = self.read()

    def read(self):
        try:
            return int(subprocess.run(self.cmd, capture_output=True, text=True, timeout=10).stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return 0  # GPU가 없는 환경

    def run(self):
        while not self.stop_flag.is_set():
            self.peak = max(self.peak, self.read())
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join()
        return self.baseline, self.peak


def parse_opts(pairs):
    opts = {}
    for p in pairs:
        key, value = p.split("=", 1)
        opts[key] = int(value) if value.isdigit() else value
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=sorted(ENGINES), required=True)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="ko")
    ap.add_argument("--opt", nargs="*", default=[], help="엔진 설정 key=value (예: model=Qwen/Qwen3.5-9B)")
    ap.add_argument("--name", help="결과 파일에 쓸 엔진 이름 (기본: --engine)")
    ap.add_argument("--limit", type=int, help="앞에서부터 N개만 (빠르게 확인할 때)")
    ap.add_argument("--every", type=int, default=1, help="k개마다 하나씩만 (범주 비율을 유지한 부분집합, 느린 엔진용)")
    args = ap.parse_args()

    items = load_items(args.dataset)[::args.every][:args.limit]
    gpu = GpuMonitor()
    gpu.start()
    load_start = time.perf_counter()
    engine = ENGINES[args.engine](**parse_opts(args.opt))
    saved = {}
    if engine.batch:  # 여러 장을 한 번에 처리하는 도구는 장당 시간을 전체 시간 ÷ 장 수로 기록
        start = time.perf_counter()
        texts = engine.run_batch([p for _, p in items])
        each = (time.perf_counter() - start) / len(items)
        saved = {}
        for sid, path in items:
            text, tables = texts.get(Path(path).stem, ("", []))
            saved[sid] = {"text": text, "tables": tables, "time": each, "error": False}
        load_time = 0.0
    else:
        engine.run(items[0][1])  # 첫 실행(모델 적재·서버 기동)은 시간 측정에서 뺌
        load_time = time.perf_counter() - load_start
        for n, (sid, path) in enumerate(items, 1):
            start = time.perf_counter()
            try:
                text, tables = engine.run(path)
                error = False
            except Exception as e:  # 한 장의 오류로 전체 측정이 멈추지 않게 기록하고 넘어감
                print(f"[오류] {sid}: {str(e)[:200]}", flush=True)
                text, tables, error = "", [], True
            saved[sid] = {"text": text, "tables": tables, "time": time.perf_counter() - start, "error": error}
            if n % 100 == 0:
                print(f"{n}/{len(items)}", flush=True)
    baseline, peak = gpu.stop()
    times = [v["time"] for v in saved.values()]
    saved["_meta"] = {"engine": args.name or args.engine, "dataset": args.dataset, "opts": args.opt, "n": len(items),
                      "every": args.every,
                      "load_seconds": round(load_time, 1), "time_median": statistics.median(times),
                      "time_mean": statistics.mean(times), "gpu_baseline_mib": baseline, "gpu_peak_mib": peak,
                      "date": time.strftime("%Y-%m-%d")}
    out = HERE / "output" / f"{args.name or args.engine}__{args.dataset}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.engine} {args.dataset}: {len(items)}개, 장당 중앙값 {statistics.median(times):.2f}초, "
          f"GPU {baseline}→{peak}MiB → {out.name}")


if __name__ == "__main__":
    main()
