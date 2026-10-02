"""11장 — OCR 서버에 파일을 보내고, 동시에 여러 요청을 보내 처리량을 잽니다.

실행 예:
  python client.py doc.png                              # 결과 JSON 출력
  python client.py doc.pdf --jobs                       # 작업 번호를 받고 끝날 때까지 확인
  python client.py --load ../../datasets/ko-ocr-bench/images --concurrency 1 4 16
  python client.py --load ... --url http://localhost:8000 http://localhost:8001   # 컨테이너 여러 개에 번갈아 보냄
"""
import argparse
import itertools
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def send(url, path):
    with open(path, "rb") as fp:
        start = time.perf_counter()
        r = requests.post(f"{url}/ocr", files={"file": (Path(path).name, fp)}, timeout=600)
    return r.status_code, time.perf_counter() - start, r.json()


def send_job(url, path):
    with open(path, "rb") as fp:
        job = requests.post(f"{url}/jobs", files={"file": (Path(path).name, fp)}, timeout=60).json()
    print("작업 등록:", job)
    while True:
        state = requests.get(f"{url}/jobs/{job['job_id']}", timeout=60).json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.5)


def load_test(urls, folder, levels, limit):
    files = sorted(Path(folder).glob("*.png"))[:limit]
    for c in levels:
        targets = list(zip(itertools.cycle(urls), files))  # 서버가 여러 개면 번갈아 보냄
        start = time.perf_counter()
        with ThreadPoolExecutor(c) as pool:
            outs = list(pool.map(lambda t: send(*t), targets))
        wall = time.perf_counter() - start
        ok = [o for o in outs if o[0] == 200]
        lat = [o[1] for o in ok]
        busy = sum(o[0] == 503 for o in outs)
        print(f"동시 {c:2d}: {len(ok) / wall:.2f}장/초, 응답 시간 중앙값 {statistics.median(lat):.2f}초 "
              f"(최대 {max(lat):.2f}초), 서버 처리 중앙값 {statistics.median(o[2]['elapsed_ms'] for o in ok):.0f}ms, "
              f"큐 대기 중앙값 {statistics.median(o[2]['queue_wait_ms'] for o in ok):.0f}ms, 503 {busy}건")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--url", nargs="+", default=["http://localhost:8000"])
    ap.add_argument("--jobs", action="store_true", help="POST /jobs로 보내고 결과를 기다림")
    ap.add_argument("--load", help="이 폴더의 PNG를 모두 보내 처리량을 잼")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    if args.load:
        load_test(args.url, args.load, args.concurrency, args.limit)
    elif args.jobs:
        print(json.dumps(send_job(args.url[0], args.path), ensure_ascii=False, indent=1)[:2000])
    else:
        code, elapsed, body = send(args.url[0], args.path)
        print(code, f"{elapsed:.2f}초")
        print(json.dumps(body, ensure_ascii=False, indent=1)[:2000])


if __name__ == "__main__":
    main()
