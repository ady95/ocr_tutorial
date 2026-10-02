"""09장 — 문서 파싱 도구(MinerU, Docling)가 만든 Markdown을 정답과 비교합니다 (서버 없이 실행).

대상: 07장의 시험용 PDF 2개(native.pdf 텍스트 PDF, scanned.pdf 스캔 PDF)와 make_office.py의 Office 문서 3개
  - CER: Markdown을 글자로 바꿔(표는 칸마다 한 줄) 정답 전체와 비교
  - 표 TEDS: 출력의 표(HTML 또는 Markdown 표)를 정답 표와 비교. 정답 표가 여러 개면 순서대로 짝지음
실행: python score_parsers.py   (output/mineru/<tier>/*.md, output/docling/<설정>/*.md를 모두 채점)
      python score_parsers.py --images output/mineru_images/standard   (평가셋 이미지 140장의 변환 결과 채점)
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from mdtext import markdown_to_text  # noqa: E402
from ocr_eval import cer, load_samples  # noqa: E402
from teds import teds  # noqa: E402

PDF_DIR = HERE.parent / "ch07" / "pdf"
OFFICE_DIR = HERE / "office"


def gt_table_html(table):
    head = "".join(f"<th>{c}</th>" for c in table["header"])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in table["rows"])
    return f"<table><tr>{head}</tr>{body}</table>"


def load_gt():
    pdf = json.loads((PDF_DIR / "pdf_gt.json").read_text(encoding="utf-8"))
    pdf_tables = [json.loads((HERE.parent / "datasets" / "ko-ocr-bench" / "gt" / f"{d}.json")
                             .read_text(encoding="utf-8")).get("table_html") for d in pdf["docs"]]
    gt = {name: {"text": "\n".join(pdf["pages"]), "tables": [t for t in pdf_tables if t]}
          for name in ("native", "scanned")}
    office = json.loads((OFFICE_DIR / "office_gt.json").read_text(encoding="utf-8"))
    for fname, g in office.items():
        gt[Path(fname).stem] = {"text": g["text"], "tables": [gt_table_html(t) for t in g["tables"]]}
    return gt


def score(md, g):
    text, tables = markdown_to_text(md, md_tables=True)
    c = cer(g["text"], text)
    pairs = list(zip(g["tables"], tables + [""] * len(g["tables"])))
    t = sum(teds(pred, ref) for ref, pred in pairs) / len(pairs) if pairs else float("nan")
    return c, t, len(tables)


def _strings(content):
    """MinerU middle_json 블록의 content에서 글자를 순서대로 꺼냅니다."""
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for item in content:
            yield from _strings(item)
    elif isinstance(content, dict):  # {"type": "text", "content": "..."} 꼴. type 값은 글자가 아님
        for key, value in content.items():
            if key != "type":
                yield from _strings(value)


def mineru_json_to_md(data):
    """middle_json을 Markdown과 같은 글자로 바꾸되, Markdown에서 빠지는 머리말(header)·꼬리말(footer)도 넣습니다.

    머리말은 맨 앞, 본문은 MinerU가 정한 읽기 순서, 꼬리말은 맨 뒤에 둡니다.
    """
    parts = []
    for page in data["pages"]:
        blocks = page.get("blocks", [])  # 블록을 하나도 찾지 못한 쪽에는 blocks 키가 없음
        top = sorted((b for b in blocks if b["type"] == "header"), key=lambda b: b["bbox"][1])
        body = [b for b in blocks if b["type"] not in ("header", "footer")]
        bottom = sorted((b for b in blocks if b["type"] == "footer"), key=lambda b: b["bbox"][1])
        for b in top + body + bottom:
            parts.append("\n".join(_strings(b["content"])))
    return "\n\n".join(parts)


def score_images(folder):
    """평가셋 이미지마다 <id>.md(또는 MinerU의 <id>.json)가 있는 폴더를 채점하고, 08장과 비교할 수 있게 글자를 저장합니다."""
    folder = Path(folder)
    per_cat, teds_scores, saved = defaultdict(list), [], {}
    for meta, gt in load_samples():
        md, js = folder / f"{meta['id']}.md", folder / f"{meta['id']}.json"
        if js.exists():
            raw = mineru_json_to_md(json.loads(js.read_text(encoding="utf-8")))
        else:
            raw = md.read_text(encoding="utf-8") if md.exists() else ""
        text, tables = markdown_to_text(raw, md_tables=True)
        saved[meta["id"]] = {"text": text}
        per_cat[meta["category"]].append(cer(gt["text"], text))
        if meta["category"] == "table":
            teds_scores.append(teds(tables[0] if tables else "", gt["table_html"]))
    allv = [c for v in per_cat.values() for c in v]
    for c in sorted(per_cat):
        print(f"{c:14s} {len(per_cat[c]):4d} {statistics.mean(per_cat[c]):7.3f}")
    print(f"{'전체':14s} {len(allv):4d} {statistics.mean(allv):7.3f} (중앙값 {statistics.median(allv):.3f})")
    print(f"표 TEDS {statistics.mean(teds_scores):.3f}")
    out = HERE / "output" / f"{folder.parent.name}_{folder.name}.json"
    out.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="평가셋 이미지 변환 결과 폴더")
    args = ap.parse_args()
    if args.images:
        score_images(args.images)
        return
    gt = load_gt()
    rows = []
    for md_path in sorted((HERE / "output").glob("*/*/*.md")):
        tool, setting, name = md_path.parts[-3], md_path.parts[-2], md_path.name.split(".")[0]
        if name not in gt:
            continue
        c, t, n = score(md_path.read_text(encoding="utf-8"), gt[name])
        rows.append((tool, setting, name, c, t, n))
    print(f"{'도구':8s} {'설정':14s} {'문서':8s} {'CER':>7s} {'표 TEDS':>8s} {'표 수':>4s}")
    for tool, setting, name, c, t, n in rows:
        print(f"{tool:8s} {setting:14s} {name:8s} {c:7.3f} {t:8.3f} {n:4d}")


if __name__ == "__main__":
    main()
