#!/bin/bash
# 13장 — 엔진이 쓰는 vLLM 서버를 띄웁니다 (OpenAI 호환 API, 이 컴퓨터 안에서만 접속).
#   정밀 OCR: PaddleOCR-VL-1.6                 → http://localhost:18118/v1
#   범용 VLM 겸 추출 LLM: Qwen3.5-9B FP8 (11-2) → http://localhost:18001/v1
# 사용:
#   ./run_servers.sh [VL GPU] [Qwen GPU]   GPU 두 장 (기본 0 1)
#   ./run_servers.sh 0 0                   24GB GPU 한 장에 함께 (VL 0.2 + Qwen 0.65, 나머지는 엔진의 PaddleOCR)
#   ./run_servers.sh --no-llm [VL GPU]     PaddleOCR-VL만 (12GB GPU). 엔진은 OCR_LLM_URL= (빈 값)으로 실행
#   ./run_servers.sh stop                  이 스크립트가 띄운 서버만 내림
# vLLM이 설치된 가상환경(08-4)을 먼저 활성화하세요. 로그는 output/serve_vl.log, output/serve_llm.log
cd "$(dirname "$0")"; mkdir -p output

stop() {
  for f in output/vl.pid output/llm.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null && echo "내림: $f ($(cat "$f"))"
    rm -f "$f"
  done
}
[ "$1" = "stop" ] && { stop; exit 0; }

NO_LLM=0
[ "$1" = "--no-llm" ] && { NO_LLM=1; shift; }
VL_GPU=${1:-0}; LLM_GPU=${2:-1}
VL_UTIL=0.5; LLM_UTIL=0.9; LLM_EXTRA=""
# 한 장에 함께 올리면 Qwen3.5의 선형 어텐션 캐시가 기본 동시 요청 수(256)를 담지 못해 32로 줄임 (13-4)
[ $NO_LLM = 0 ] && [ "$VL_GPU" = "$LLM_GPU" ] && { VL_UTIL=0.2; LLM_UTIL=0.65; LLM_EXTRA="--max-num-seqs 32"; }

PORTS="18118"; [ $NO_LLM = 0 ] && PORTS="18118 18001"
for p in $PORTS; do
  if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE "[:.]$p[[:space:]]"; then
    echo "포트 $p 사용 중 — 앞 장(12장 등)의 서버가 남아 있다면 먼저 내리세요"; exit 1
  fi
done
trap 'echo "시작 실패 — 띄운 서버를 내립니다"; stop; exit 1' ERR INT

wait_ready() {  # 포트, pid 파일, 모델 이름, 로그
  until curl -sf "localhost:$1/v1/models" | grep -q "$3"; do
    sleep 5
    kill -0 "$(cat "$2")" 2>/dev/null || { echo "$3 서버 시작 실패"; tail -5 "$4"; return 1; }
  done
}

CUDA_VISIBLE_DEVICES=$VL_GPU vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --port 18118 --host 127.0.0.1 \
  --served-model-name PaddleOCR-VL-1.6-0.9B --gpu-memory-utilization $VL_UTIL --max-model-len 16384 > output/serve_vl.log 2>&1 &
echo $! > output/vl.pid
# 두 서버가 동시에 메모리를 재면 서로의 사용량을 자기 것으로 셈 → 앞 서버가 준비된 뒤에 다음 서버를 띄움
wait_ready 18118 output/vl.pid PaddleOCR-VL output/serve_vl.log

if [ $NO_LLM = 0 ]; then
  # RTX 30 시리즈(Ampere)는 FP8 연산 장치가 없어 가중치만 FP8로 두는 Marlin 커널을 쓰게 함 (11-1)
  VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel,B12xTensorFP8ScaledMMLinearKernel,PerTensorTorchFP8ScaledMMLinearKernel,ChannelWiseTorchFP8ScaledMMLinearKernel \
  CUDA_VISIBLE_DEVICES=$LLM_GPU vllm serve Qwen/Qwen3.5-9B --port 18001 --host 127.0.0.1 --quantization fp8 \
    --gpu-memory-utilization $LLM_UTIL --max-model-len 16384 $LLM_EXTRA > output/serve_llm.log 2>&1 &
  echo $! > output/llm.pid
  wait_ready 18001 output/llm.pid Qwen output/serve_llm.log
fi
trap - ERR INT
echo "준비 완료 (내리기: ./run_servers.sh stop)"
