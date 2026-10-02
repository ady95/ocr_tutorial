"""11장 — 모델 설정(config.json)으로 GPU 메모리를 어림합니다: 가중치 + KV 캐시.

  가중치  = 파라미터 수 × 1개당 바이트 (bf16 2, fp8·int8 1, int4 0.5)
  KV 캐시 = 토큰 수 × (2 × 전체 어텐션 층 수 × KV 헤드 수 × head_dim × 2바이트)
실제로는 여기에 CUDA 그래프·활성값 등이 수 GB 더 붙습니다.

실행 예:
  python vram_calc.py Qwen/Qwen3.5-4B --tokens 16384 --seqs 4
  python vram_calc.py Qwen/Qwen3.5-9B --bytes 1     # fp8 가중치
파라미터 수는 Hugging Face가 safetensors 파일에서 센 값을 씁니다 (--params로 직접 줄 수도 있음).
"""
import argparse
import json


def load_config(model):
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(model, "config.json")))
    return cfg.get("text_config", cfg)  # VLM은 언어 모델 설정이 text_config 안에 있음


def count_params(model):
    from huggingface_hub import model_info
    return model_info(model).safetensors.total / 1e9


def kv_bytes_per_token(t):
    layers = t["num_hidden_layers"]
    if "layer_types" in t:  # Qwen3.5처럼 일부 층만 전체 어텐션(나머지는 KV 캐시가 없는 선형 어텐션)인 모델
        layers = t["layer_types"].count("full_attention")
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    return 2 * layers * t.get("num_key_value_heads", t["num_attention_heads"]) * head_dim * 2, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--params", type=float, help="파라미터 수(십억 개). 생략하면 Hugging Face에서 읽음")
    ap.add_argument("--bytes", type=float, default=2, help="파라미터 1개당 바이트 (bf16 2, fp8 1, int4 0.5)")
    ap.add_argument("--tokens", type=int, default=16384, help="요청 하나의 최대 토큰 수 (입력 이미지 + 출력)")
    ap.add_argument("--seqs", type=int, default=1, help="동시에 처리할 요청 수")
    args = ap.parse_args()
    t = load_config(args.model)
    params = args.params or count_params(args.model)
    per_token, layers = kv_bytes_per_token(t)
    weights = params * 1e9 * args.bytes / 2**30
    kv = per_token * args.tokens * args.seqs / 2**30
    print(f"{args.model}: 파라미터 {params:.2f}B, 전체 어텐션 층 {layers}/{t['num_hidden_layers']}, 토큰당 KV {per_token / 1024:.0f}KiB")
    print(f"가중치 {weights:.2f}GiB + KV 캐시 {kv:.2f}GiB ({args.tokens:,}토큰 × {args.seqs}개) = {weights + kv:.2f}GiB")


if __name__ == "__main__":
    main()
