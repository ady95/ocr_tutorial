"""OCR 따라하기 — GPU 메모리 사용량 측정 (nvidia-smi를 주기적으로 읽어 최댓값 기록).

프레임워크(PyTorch, PaddlePaddle, vLLM)와 상관없이 "그 GPU에서 실제로 쓰인 메모리"를 잽니다.
다른 프로그램이 쓰는 메모리도 함께 잡히므로, 시작 전 값(baseline)을 빼서 증가분을 봅니다.

    with GpuMemory() as mem:
        ...  # 모델 적재·추론
    print(mem.peak_mib, mem.delta_mib)
"""
import subprocess
import threading


def used_mib(index=0):
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


class GpuMemory:
    def __init__(self, index=0, interval=0.2):
        self.index, self.interval = index, interval
        self.baseline = self.peak_mib = 0
        self._stop = threading.Event()

    def _watch(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, used_mib(self.index))
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline = self.peak_mib = used_mib(self.index)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def delta_mib(self):
        return self.peak_mib - self.baseline
