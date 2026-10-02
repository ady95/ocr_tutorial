"""11장 — vLLM 서버에 요청을 동시에 여러 개 보내 처리량(장/초)과 지연 시간을 잽니다.

10장의 bench.py는 한 장씩 보내 '한 장에 걸리는 시간(지연 시간)'을 쟀습니다. 서비스에서는 여러 요청이 동시에 오므로
'1초에 몇 장을 처리하는가(처리량)'가 더 중요합니다. vLLM은 동시에 들어온 요청을 묶어(연속 배치) 처리합니다.

실행 예 (서버를 먼저 띄운 뒤):
  python throughput.py --model Qwen/Qwen3.5-4B --workers 1 4 16
결과: output/throughput_<이름>.json (동시 요청 수마다 처리량·지연 시간·CER)
"""
import argparse
import json
import statistics
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("common", "ch08"):
    sys.path.insert(0, str(HERE.parent / sub))
from ocr_eval import cer, load_samples  # noqa: E402
from vlm_bench import PROMPTS, run_one, to_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="qwen")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, help="앞에서부터 N장만 (느린 설정을 빠르게 확인할 때)")
    ap.add_argument("--prompt", default="html", help="08장 프롬프트 이름(html·plain) 또는 프롬프트 문장 (GLM-OCR: \"Text Recognition:\")")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=900)
    PROMPTS.setdefault(args.prompt, args.prompt)
    req = Namespace(kind=args.kind, model=args.model, max_tokens=args.max_tokens, prompt=args.prompt,
                    presence_penalty=1.5 if args.kind == "qwen" else 0.0, no_think=True, ds_prompt="free")
    samples = load_samples()[:args.limit]
    run_one(client, req, samples[0][0]["image_path"])  # 첫 요청(그래프 준비 등)은 빼고 잼
    rows = []
    for w in args.workers:
        start = time.perf_counter()
        with ThreadPoolExecutor(w) as pool:
            outs = list(pool.map(partial(run_one, client, req), [m["image_path"] for m, _ in samples]))
        wall = time.perf_counter() - start
        cers = [cer(g["text"], to_text(args.kind, raw)[0]) for (m, g), (raw, *_rest) in zip(samples, outs)]
        lat = [o[1] for o in outs]
        tokens = sum(o[3] for o in outs)
        row = {"workers": w, "images": len(samples), "wall_s": wall, "images_per_s": len(samples) / wall,
               "latency_median_s": statistics.median(lat), "latency_p90_s": sorted(lat)[int(len(lat) * 0.9)],
               "output_tokens_per_s": tokens / wall, "cer_mean": statistics.mean(cers),
               "cer_median": statistics.median(cers), "cut": sum(o[4] == "length" for o in outs)}
        rows.append(row)
        print(f"동시 {w:2d}: {row['images_per_s']:.2f}장/초 (전체 {wall:.0f}초), 지연 중앙값 {row['latency_median_s']:.1f}초 "
              f"p90 {row['latency_p90_s']:.1f}초, 출력 {row['output_tokens_per_s']:.0f}토큰/초, "
              f"CER {row['cer_mean']:.3f} (중앙값 {row['cer_median']:.3f}), 잘림 {row['cut']}", flush=True)
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    (out / f"throughput_{args.name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
