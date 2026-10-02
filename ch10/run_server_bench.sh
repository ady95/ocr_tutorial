#!/bin/bash
# 10장 — vLLM 서버를 띄우고, 준비되면 세 평가셋을 차례로 bench.py로 잰 뒤 서버를 내립니다.
#
# 사용: ./run_server_bench.sh <engine> <model> <gpu> <port> <bench 가상환경 python> [-- vllm serve 옵션...]
# 예:   ./run_server_bench.sh qwen Qwen/Qwen3.5-4B 1 8000 python -- --gpu-memory-utilization 0.85
#       ./run_server_bench.sh glm zai-org/GLM-OCR 1 8000 python -- --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
#       ./run_server_bench.sh paddlevl PaddlePaddle/PaddleOCR-VL-1.6 1 8118 ~/venv-paddle/bin/python \
#           -- --served-model-name PaddleOCR-VL-1.6-0.9B --gpu-memory-utilization 0.6
# vllm이 설치된 가상환경을 먼저 활성화하세요. 서버 로그는 output/serve_<engine>.log에 남습니다.
ENGINE=$1; MODEL=$2; GPU=$3; PORT=$4; PY=$5; shift 5
[ "$1" = "--" ] && shift
cd "$(dirname "$0")"
mkdir -p output
# 다른 서비스가 이미 쓰는 포트면 그 서비스에 요청을 보내게 되므로 시작 전에 확인
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE "[:.]$PORT[[:space:]]"; then echo "포트 $PORT 사용 중 — 다른 포트를 지정하세요"; exit 1; fi
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL --port $PORT --max-model-len 16384 "$@" > output/serve_$ENGINE.log 2>&1 &
SPID=$!
until curl -sf localhost:$PORT/health >/dev/null; do
  sleep 5
  kill -0 $SPID 2>/dev/null || { echo "서버 시작 실패"; tail -30 output/serve_$ENGINE.log; exit 1; }
done
echo "$ENGINE 서버 준비 $(( $(date +%s)-T0 ))초"
OPTS="base_url=http://localhost:$PORT/v1"
[ $ENGINE = paddlevl ] && OPTS="server=http://localhost:$PORT/v1"
for DATASET in ko aihub_page aihub_word; do
  CUDA_VISIBLE_DEVICES=$GPU $PY bench.py --engine $ENGINE --dataset $DATASET --opt $OPTS 2>&1 | grep -E "^\[오류\]|개, 장당"
done
kill $SPID; wait $SPID 2>/dev/null
grep -E "Model loading took" output/serve_$ENGINE.log | sed "s/.*\] //"
