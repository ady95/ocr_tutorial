#!/bin/bash
# 10장 — vLLM 서버를 띄우고, 준비되면 평가셋을 차례로 bench.py로 잰 뒤 서버를 내립니다.
#
# 사용: ./run_server_bench.sh <engine> <model> <gpu> <port> <bench 가상환경 python> [-- vllm serve 옵션...]
# 예:   ./run_server_bench.sh qwen Qwen/Qwen3.5-4B 1 18000 python -- --gpu-memory-utilization 0.85
#       ./run_server_bench.sh glm zai-org/GLM-OCR 1 18000 python -- --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
#       ./run_server_bench.sh paddlevl PaddlePaddle/PaddleOCR-VL-1.6 1 18118 ~/ocr-book/.venv-paddle/bin/python \
#           -- --served-model-name PaddleOCR-VL-1.6-0.9B --gpu-memory-utilization 0.6
# 평가셋: 기본은 공개 평가셋(ko)만. AI Hub 데이터(10-1, 선택)를 변환해 두었으면 그 평가셋도 함께 잽니다.
#         직접 고르려면 DATASETS="ko aihub_word" ./run_server_bench.sh ...
# vllm이 설치된 가상환경을 먼저 활성화하세요. 서버 로그는 output/serve_<engine>.log, 평가 로그는 output/log_<engine>__<평가셋>.txt
set -o pipefail
ENGINE=$1; MODEL=$2; GPU=$3; PORT=$4; PY=$5; shift 5
[ "$1" = "--" ] && shift
cd "$(dirname "$0")"
mkdir -p output
if [ -z "$DATASETS" ]; then
  DATASETS="ko"
  [ -f ../datasets/aihub-local/manifest.jsonl ] && DATASETS="$DATASETS aihub_page"
  [ -f ../datasets/aihub-local/words/manifest.jsonl ] && DATASETS="$DATASETS aihub_word"
fi
# 다른 서비스가 이미 쓰는 포트면 그 서비스에 요청을 보내게 되므로 시작 전에 확인
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE "[:.]$PORT[[:space:]]"; then echo "포트 $PORT 사용 중 — 다른 포트를 지정하세요"; exit 1; fi
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL --port $PORT --host 127.0.0.1 --max-model-len 16384 "$@" > output/serve_$ENGINE.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT   # 중간에 실패하거나 Ctrl+C로 멈춰도 서버를 내림
until curl -sf --connect-timeout 3 --max-time 5 localhost:$PORT/health >/dev/null; do
  sleep 5
  kill -0 $SPID 2>/dev/null || { echo "서버 시작 실패"; tail -30 output/serve_$ENGINE.log; exit 1; }
done
echo "$ENGINE 서버 준비 $(( $(date +%s)-T0 ))초 (평가셋: $DATASETS)"
OPTS="base_url=http://localhost:$PORT/v1"
[ $ENGINE = paddlevl ] && OPTS="server=http://localhost:$PORT/v1"
FAILED=""
for DATASET in $DATASETS; do
  LOG=output/log_${ENGINE}__${DATASET}.txt
  if CUDA_VISIBLE_DEVICES=$GPU $PY bench.py --engine $ENGINE --dataset $DATASET --opt $OPTS > $LOG 2>&1; then
    grep -E "^\[오류\]|개, 장당" $LOG
  else
    echo "[실패] $DATASET — $LOG 끝부분:"; tail -5 $LOG; FAILED="$FAILED $DATASET"
  fi
done
grep -E "Model loading took" output/serve_$ENGINE.log | sed "s/.*\] //"
[ -z "$FAILED" ] || { echo "실패한 평가셋:$FAILED"; exit 1; }
