"""13장 — OCR 엔진 API 클라이언트.

  python client.py 파일...                     JSON 결과 요약 (유형, 엔진, 경고, 시간)
  python client.py 파일 --format markdown      Markdown 결과 그대로
  python client.py 파일... --save out/         파일마다 결과 JSON 저장
"""
import argparse
import json
import time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--format", choices=["json", "markdown"], default="json")
    ap.add_argument("--type", help="유형을 직접 지정 (invoice, receipt, card, table, document)")
    ap.add_argument("--mode", choices=["accurate", "fast"], default="accurate")
    ap.add_argument("--save", help="결과 JSON을 저장할 폴더")
    args = ap.parse_args()
    for f in args.files:
        form = {"format": args.format, "mode": args.mode, **({"type": args.type} if args.type else {})}
        start = time.perf_counter()
        with open(f, "rb") as fp:
            r = requests.post(f"{args.url}/v1/ocr", files={"file": (Path(f).name, fp)}, data=form, timeout=600)
        r.raise_for_status()
        if args.format == "markdown":
            print(r.text)
            continue
        res = r.json()
        for p in res["pages"]:
            print(f"{Path(f).name} p{p['page']}: {p['type']} ({p['type_reason']}) · {p['engine']} · "
                  f"경고 {len(p['warnings'])} · 엔진 {p['times']['total']:.2f}초 · 왕복 {time.perf_counter() - start:.2f}초")
            for w in p["warnings"]:
                print("   경고:", w.get("field", ""), w.get("value", ""), w["problem"])
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
            (Path(args.save) / f"{Path(f).stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
