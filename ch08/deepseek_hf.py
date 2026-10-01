"""08장 — DeepSeek-OCR 2를 모델 카드의 공식 방법(transformers + trust_remote_code)으로 실행합니다.

vLLM 서버 결과가 이상할 때, 모델 자체의 문제인지 추론 서버 쪽 문제인지 가려 보기 위한 스크립트입니다.
모델 카드 기준 버전: torch 2.6.0, transformers 4.46.3, tokenizers 0.20.3, flash-attn 2.7.3 (+ einops addict easydict)

실행: python deepseek_hf.py [--ids doc_ko_01 card_01 ...] [--prompt free|markdown] [--shard 0/2 --name deepseek_hf_0]
결과는 output/vlm_deepseek_hf.json (vlm_bench.py와 같은 형식)에 저장합니다.
"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from ocr_eval import cer, load_samples  # noqa: E402
from vlm_bench import compression_ratio, to_text  # noqa: E402

PROMPTS = {"free": "<image>\nFree OCR. ", "markdown": "<image>\n<|grounding|>Convert the document to markdown. "}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", help="처리할 이미지 id (기본: 전체)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="markdown")
    ap.add_argument("--name", default="deepseek_hf")
    ap.add_argument("--shard", default="0/1", help="i/n: 전체를 n개로 나눠 i번째만 (GPU 여러 장에 나눠 돌릴 때)")
    args = ap.parse_args()
    name = "deepseek-ai/DeepSeek-OCR-2"
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, _attn_implementation="flash_attention_2", trust_remote_code=True,
                                      use_safetensors=True).eval().cuda().to(torch.bfloat16)
    samples = [s for s in load_samples() if not args.ids or s[0]["id"] in args.ids]
    i, n = map(int, args.shard.split("/"))
    samples = samples[i::n]
    saved, scores = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        for meta, gt in samples:
            start = time.perf_counter()
            raw = model.infer(tokenizer, prompt=PROMPTS[args.prompt], image_file=meta["image_path"], output_path=tmp,
                              base_size=1024, image_size=768, crop_mode=True, save_results=False, eval_mode=True)
            elapsed = time.perf_counter() - start
            text, _ = to_text("deepseek", raw)
            c = cer(gt["text"], text)
            scores.append(c)
            saved[meta["id"]] = {"text": text, "raw": raw, "time": elapsed, "prompt_tokens": 0,
                                 "completion_tokens": len(tokenizer.encode(raw)), "finish_reason": None,
                                 "compression": compression_ratio(raw)}
            print(f"{meta['id']:14s} CER {c:.3f}  {elapsed:5.1f}초  {text[:40]!r}", flush=True)
    print(f"{len(scores)}장 평균 CER {statistics.mean(scores):.3f}, 중앙값 {statistics.median(scores):.3f}")
    out = HERE / "output" / f"vlm_{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
