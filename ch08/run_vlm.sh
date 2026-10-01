#!/bin/bash
# 08장 — vLLM 서버를 띄우고, 준비되면 vlm_bench.py를 실행한 뒤 GPU 메모리 최댓값을 기록하고 서버를 내립니다.
#
# 사용: ./run_vlm.sh <model> <kind> <name> <gpu> <port> [vlm_bench 옵션...] -- [vllm serve 옵션...]
# 예:   ./run_vlm.sh Qwen/Qwen3.5-4B qwen qwen35_4b_html_pp 0 8000 --no-think --presence-penalty 1.5 -- --gpu-memory-utilization 0.85
#       ./run_vlm.sh NCSOFT/VARCO-VISION-2.0-1.7B-OCR varco varco 0 8000 --max-tokens 8192 --workers 4 -- --max-num-batched-tokens 16384
#       ./run_vlm.sh deepseek-ai/DeepSeek-OCR-2 deepseek deepseek 0 8000 --max-tokens 6144 -- --max-model-len 8192 \
#           --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0
# vllm이 설치된 가상환경을 먼저 활성화하세요. 서버 로그는 serve_<name>.log에 남습니다.
MODEL=$1; KIND=$2; NAME=$3; GPU=$4; PORT=$5; shift 5
BENCH=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do BENCH+=("$1"); shift; done; shift
cd "$(dirname "$0")"
BASE=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits)
T0=$(date +%s)
# --max-model-len은 뒤에 다시 주면 덮어쓸 수 있습니다 (DeepSeek-OCR 2는 8192가 최대)
CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL --port $PORT --max-model-len 16384 "$@" > serve_$NAME.log 2>&1 &
SPID=$!
until curl -sf localhost:$PORT/health >/dev/null; do
  sleep 5
  kill -0 $SPID 2>/dev/null || { echo "서버 시작 실패"; tail -30 serve_$NAME.log; exit 1; }
done
echo "서버 준비 $(( $(date +%s)-T0 ))초, 시작 전 GPU $BASE MiB"
python vlm_bench.py --kind $KIND --model $MODEL --base-url http://localhost:$PORT/v1 --name $NAME "${BENCH[@]}" &
BPID=$!
PEAK=0
while kill -0 $BPID 2>/dev/null; do
  U=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  [ $U -gt $PEAK ] && PEAK=$U
  sleep 2
done
wait $BPID
echo "GPU 최대 $PEAK MiB (+$((PEAK-BASE)))"
kill $SPID; wait $SPID 2>/dev/null
grep -E "Model loading took|init engine" serve_$NAME.log | sed "s/.*\] //"
