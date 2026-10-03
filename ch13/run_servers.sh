#!/bin/bash
# 13장 — 엔진이 쓰는 두 vLLM 서버를 띄웁니다 (둘 다 OpenAI 호환 API, 이 컴퓨터 안에서만 접속).
#   정밀 OCR: PaddleOCR-VL-1.6                 → http://localhost:18118/v1
#   범용 VLM 겸 추출 LLM: Qwen3.5-9B FP8 (11-2) → http://localhost:18001/v1
# 사용: ./run_servers.sh [PaddleOCR-VL GPU 번호] [Qwen GPU 번호]   (기본 0 1)
# 내리기: kill $(cat output/vl.pid output/llm.pid)
# GPU가 한 장(24GB)이면 ./run_servers.sh 0 0 으로 함께 올릴 수 있음 (VL 0.2 + Qwen 0.65, 나머지는 엔진의 PaddleOCR). 12GB면 Qwen을 빼고
# 엔진을 OCR_LLM_URL= (빈 값)으로 실행하세요. 항목 추출이 빠지고, 반복 생성이 나면 빠른 OCR 결과를 씁니다.
VL_GPU=${1:-0}; LLM_GPU=${2:-1}
VL_UTIL=0.5; LLM_UTIL=0.9
LLM_EXTRA=""
# 한 장에 함께 올리면 Qwen3.5의 선형 어텐션 캐시가 기본 동시 요청 수(256)를 담지 못해 32로 줄임
[ "$VL_GPU" = "$LLM_GPU" ] && { VL_UTIL=0.2; LLM_UTIL=0.65; LLM_EXTRA="--max-num-seqs 32"; }
cd "$(dirname "$0")"; mkdir -p output
for p in 18118 18001; do
  if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qE "[:.]$p[[:space:]]"; then echo "포트 $p 사용 중"; exit 1; fi
done
CUDA_VISIBLE_DEVICES=$VL_GPU vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --port 18118 --host 127.0.0.1 \
  --served-model-name PaddleOCR-VL-1.6-0.9B --gpu-memory-utilization $VL_UTIL --max-model-len 16384 > output/serve_vl.log 2>&1 &
echo $! > output/vl.pid
# 두 서버가 동시에 메모리를 재면 서로의 사용량을 자기 것으로 셈 → 앞 서버가 준비된 뒤에 다음 서버를 띄움
until curl -sf localhost:18118/v1/models >/dev/null; do
  sleep 5; kill -0 "$(cat output/vl.pid)" 2>/dev/null || { echo "PaddleOCR-VL 서버 시작 실패"; tail -5 output/serve_vl.log; exit 1; }
done
# RTX 30 시리즈(Ampere)는 FP8 연산 장치가 없어 가중치만 FP8로 두는 Marlin 커널을 쓰게 함 (11-1)
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel,B12xTensorFP8ScaledMMLinearKernel,PerTensorTorchFP8ScaledMMLinearKernel,ChannelWiseTorchFP8ScaledMMLinearKernel \
CUDA_VISIBLE_DEVICES=$LLM_GPU vllm serve Qwen/Qwen3.5-9B --port 18001 --host 127.0.0.1 --quantization fp8 \
  --gpu-memory-utilization $LLM_UTIL --max-model-len 16384 $LLM_EXTRA > output/serve_llm.log 2>&1 &
echo $! > output/llm.pid
until curl -sf localhost:18001/v1/models >/dev/null; do
  sleep 5; kill -0 "$(cat output/llm.pid)" 2>/dev/null || { echo "Qwen 서버 시작 실패"; tail -5 output/serve_llm.log; exit 1; }
done
echo "준비 완료"
