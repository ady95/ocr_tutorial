#!/bin/bash
# 12장 — LLM 서버와 임베딩 서버를 vLLM으로 띄웁니다 (둘 다 OpenAI 호환 API).
#   LLM:    Qwen/Qwen3.5-9B, FP8 (RTX 3090 한 장, 11-2)  → http://localhost:18001/v1
#   임베딩: BAAI/bge-m3 (다국어 검색용, 1024차원)       → http://localhost:18002/v1/embeddings
# 사용: ./run_servers.sh [LLM GPU 번호] [임베딩 GPU 번호]   (기본 1 0)
# 내리기: kill $(cat output/llm.pid output/embed.pid)
LLM_GPU=${1:-1}; EMB_GPU=${2:-0}
cd "$(dirname "$0")"; mkdir -p output
for p in 18001 18002; do
  if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE "[:.]$p[[:space:]]"; then echo "포트 $p 사용 중"; exit 1; fi
done
# RTX 30 시리즈(Ampere)는 FP8 연산 장치가 없어 가중치만 FP8로 두는 Marlin 커널을 쓰게 함 (11-1)
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel,B12xTensorFP8ScaledMMLinearKernel,PerTensorTorchFP8ScaledMMLinearKernel,ChannelWiseTorchFP8ScaledMMLinearKernel \
CUDA_VISIBLE_DEVICES=$LLM_GPU vllm serve Qwen/Qwen3.5-9B --port 18001 --host 127.0.0.1 --quantization fp8 \
  --gpu-memory-utilization 0.9 --max-model-len 16384 > output/serve_llm.log 2>&1 &
echo $! > output/llm.pid
CUDA_VISIBLE_DEVICES=$EMB_GPU vllm serve BAAI/bge-m3 --port 18002 --host 127.0.0.1 --runner pooling \
  --gpu-memory-utilization 0.3 > output/serve_embed.log 2>&1 &
echo $! > output/embed.pid
for p in 18001 18002; do
  until curl -sf localhost:$p/v1/models >/dev/null; do sleep 5; done
done
echo "준비 완료: LLM $(curl -s localhost:18001/v1/models | grep -o '"id":"[^"]*"' | head -1), 임베딩 $(curl -s localhost:18002/v1/models | grep -o '"id":"[^"]*"' | head -1)"
