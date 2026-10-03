"""13-3 — 엔진 결과를 JSON과 Markdown으로 내보냅니다.

JSON     프로그램이 쓰는 결과. 쪽마다 유형·글자·표·추출 항목·경고·시간과 글자줄 좌표를 모두 담음
Markdown 사람과 RAG(12-2)가 읽는 결과. 쪽마다 주석 한 줄로 메타데이터(쪽 번호·유형·엔진·경고 수)를 남김
"""

VERSION = "1.0"


def to_json(filename, pages, seconds, include_lines=True):
    out = []
    for no, p in enumerate(pages, 1):
        page = {"page": no, **{k: v for k, v in p.items() if k != "lines" or include_lines}}
        out.append(page)
    return {"version": VERSION, "filename": filename, "pages": out, "seconds": round(seconds, 3),
            "warnings": sum(len(p.get("warnings", [])) for p in pages)}


def fields_table(fields, prefix=""):
    """추출 항목을 '| 항목 | 값 |' 줄 목록으로 (품목 같은 목록은 번호를 붙여 펼침)."""
    rows = []
    for k, v in (fields or {}).items():
        if isinstance(v, dict):
            rows += fields_table(v, f"{prefix}{k}.")
        elif isinstance(v, list):
            for i, it in enumerate(v):
                rows += fields_table(it, f"{prefix}{k}[{i}].") if isinstance(it, dict) else [f"| {prefix}{k}[{i}] | {it} |"]
        else:
            rows.append(f"| {prefix}{k} | {'' if v is None else v} |")
    return rows


def to_markdown(filename, pages):
    parts = []
    for no, p in enumerate(pages, 1):
        parts.append(f"<!-- page {no} · file {filename} · type {p['type']} · engine {p['engine']} · "
                     f"warnings {len(p.get('warnings', []))} -->")
        parts.append(p["markdown"].strip())
        if p.get("fields"):
            parts += ["", "| 항목 | 값 |", "|---|---|"] + fields_table(p["fields"])
        parts.append("")
    return "\n".join(parts)
