"""OCR 따라하기 04장 — 실습 환경 점검 스크립트.

설치된 도구와 GPU 사용 가능 여부를 한 번에 확인합니다.
설치되지 않은 항목은 '미설치'로 표시하고 계속 진행합니다.

실행: python check_env.py
"""
import importlib
import os
import platform
import shutil
import subprocess
import sys


def version_of(module_name):
    """모듈을 불러와 버전 문자열을 돌려줍니다. 없으면 None."""
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "버전 정보 없음")


def check_python():
    print("[Python]")
    print(f"  버전      : {platform.python_version()}")
    print(f"  실행 파일 : {sys.executable}")
    if sys.prefix != sys.base_prefix:
        state = "사용 중"
    elif os.path.exists("/.dockerenv"):
        state = "컨테이너 내부 (가상환경 불필요)"
    else:
        state = "사용 안 함 (가상환경 권장)"
    print(f"  가상환경  : {state}")


def check_torch():
    print("[PyTorch]")
    ver = version_of("torch")
    if ver is None:
        print("  미설치")
        return
    import torch

    print(f"  버전      : {ver}")
    print(f"  CUDA 빌드 : {torch.version.cuda or 'CPU 전용 빌드'}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3
            print(f"  GPU {i}     : {props.name} ({total_gb:.1f} GB)")
        x = torch.ones(1024, 1024, device="cuda")
        y = (x @ x)[0, 0].item()  # 1이 1024개 더해지므로 1024.0이 나와야 정상
        print(f"  GPU 연산  : {'OK' if y == 1024.0 else '이상'} (행렬곱 검산 {y:.0f})")
    else:
        print("  GPU       : 사용 불가 (CPU로 실습 진행)")


def check_opencv():
    print("[OpenCV]")
    ver = version_of("cv2")
    print(f"  버전      : {ver or '미설치'}")


def check_tool(name, args):
    path = shutil.which(name)
    if path is None:
        return None
    out = subprocess.run([name, *args], capture_output=True, text=True)
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0] if text else path


def check_system_tools():
    print("[시스템 도구]")
    smi = check_tool("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"])
    print(f"  nvidia-smi: {smi or '없음 (NVIDIA GPU 미사용)'}")
    tess = check_tool("tesseract", ["--version"])
    print(f"  tesseract : {tess or '미설치 (06장에서 설치)'}")
    docker = check_tool("docker", ["--version"])
    print(f"  docker    : {docker or '미설치 (선택 사항)'}")


def main():
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    check_python()
    check_torch()
    check_opencv()
    check_system_tools()


if __name__ == "__main__":
    main()
