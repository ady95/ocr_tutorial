#!/bin/bash
# 11장 — vLLM 서버를 띄우고 throughput.py로 처리량을 잰 뒤, 메모리 사용 내역을 남기고 서버를 내립니다.
#
# 사용: ./run_vllm.sh <이름> <모델> <gpu> <port> [throughput.py 옵션...] -- [vllm serve 옵션...]
# 예:   ./run_vllm.sh qwen4b_bf16 Qwen/Qwen3.5-4B 1 8000 --workers 1 4 16 -- --gpu-memory-utilization 0.85
#       ./run_vllm.sh qwen4b_fp8 Qwen/Qwen3.5-4B 1 8000 --workers 1 16 -- --quantization fp8
# vllm이 설치된 가상환경을 먼저 활성화하세요. 서버 로그는 output/serve_<이름>.log에 남습니다.
NAME=$1; MODEL=$2; GPU=$3; PORT=$4; shift 4
ARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ARGS+=("$1"); shift; done; shift
cd "$(dirname "$0")"; mkdir -p output
# 다른 서비스가 이미 쓰는 포트면 그 서비스에 요청을 보내게 되므로 시작 전에 확인
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE "[:.]$PORT[[:space:]]"; then echo "포트 $PORT 사용 중 — 다른 포트를 지정하세요"; exit 1; fi
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL --port $PORT --max-model-len 16384 "$@" > output/serve_$NAME.log 2>&1 &
SPID=$!
until curl -sf localhost:$PORT/health >/dev/null; do
  sleep 5
  kill -0 $SPID 2>/dev/null || { echo "$NAME 서버 시작 실패"; grep -E "Error|error" output/serve_$NAME.log | tail -5; exit 1; }
done
echo "$NAME 서버 준비 $(( $(date +%s)-T0 ))초, GPU $(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)"
grep -hoE "Model loading took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" output/serve_$NAME.log | sort -u
python throughput.py --model $MODEL --base-url http://localhost:$PORT/v1 --name $NAME "${ARGS[@]}"
kill $SPID; wait $SPID 2>/dev/null
