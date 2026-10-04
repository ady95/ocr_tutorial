#!/bin/bash
# 12장 — LLM 서버와 임베딩 서버를 vLLM으로 띄웁니다 (둘 다 OpenAI 호환 API).
#   LLM:    Qwen/Qwen3.5-9B, FP8 (RTX 3090 한 장, 11-2)  → http://localhost:18001/v1
#   임베딩: BAAI/bge-m3 (다국어 검색용, 1024차원)       → http://localhost:18002/v1/embeddings
# 사용: ./run_servers.sh [LLM GPU 번호] [임베딩 GPU 번호]   (기본 1 0)
# 내리기: ./run_servers.sh stop   (13장으로 넘어가기 전에 내리세요. 13장 서버도 18001 포트를 씁니다)
# 한 서버라도 시작에 실패하거나 READY_TIMEOUT초(기본 1200) 안에 준비되지 않으면 띄운 서버를 모두 내리고 끝냅니다.
cd "$(dirname "$0")"; mkdir -p output

stop() {
  for f in output/llm.pid output/embed.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null && echo "내림: $f ($(cat "$f"))"
    rm -f "$f"
  done
}
[ "$1" = "stop" ] && { stop; exit 0; }

LLM_GPU=${1:-1}; EMB_GPU=${2:-0}; READY_TIMEOUT=${READY_TIMEOUT:-1200}
for p in 18001 18002; do
  if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE "[:.]$p[[:space:]]"; then echo "포트 $p 사용 중"; exit 1; fi
done
trap 'echo "시작 실패 — 띄운 서버를 내립니다"; stop; exit 1' ERR INT TERM

wait_ready() {  # 포트, pid 파일, 모델 이름, 로그
  local t0=$SECONDS
  until curl -sf --connect-timeout 3 --max-time 5 "localhost:$1/v1/models" | grep -q "$3"; do  # 응답이 없어도 5초 안에 돌아옴
    sleep 5
    kill -0 "$(cat "$2")" 2>/dev/null || { echo "$3 서버 시작 실패"; tail -5 "$4"; return 1; }
    [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ] || { echo "$3 서버가 ${READY_TIMEOUT}초 안에 준비되지 않음"; tail -5 "$4"; return 1; }
  done
}

# RTX 30 시리즈(Ampere)는 FP8 연산 장치가 없어 가중치만 FP8로 두는 Marlin 커널을 쓰게 함 (11-1)
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel,B12xTensorFP8ScaledMMLinearKernel,PerTensorTorchFP8ScaledMMLinearKernel,ChannelWiseTorchFP8ScaledMMLinearKernel \
CUDA_VISIBLE_DEVICES=$LLM_GPU vllm serve Qwen/Qwen3.5-9B --port 18001 --host 127.0.0.1 --quantization fp8 \
  --gpu-memory-utilization 0.9 --max-model-len 16384 > output/serve_llm.log 2>&1 &
echo $! > output/llm.pid
CUDA_VISIBLE_DEVICES=$EMB_GPU vllm serve BAAI/bge-m3 --port 18002 --host 127.0.0.1 --runner pooling \
  --gpu-memory-utilization 0.3 > output/serve_embed.log 2>&1 &
echo $! > output/embed.pid
wait_ready 18001 output/llm.pid Qwen output/serve_llm.log
wait_ready 18002 output/embed.pid bge-m3 output/serve_embed.log
trap - ERR INT TERM
echo "준비 완료 (내리기: ./run_servers.sh stop)"
