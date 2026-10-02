"""OCR 따라하기 — Markdown·HTML 결과를 평가용 텍스트와 표로 바꾸는 도구 (08장 VLM, 09장 문서 파싱 도구).

    text, tables = markdown_to_text(md)                       # tables: 출력에 있던 HTML 표
    text, tables = markdown_to_text(md, md_tables=True)       # Markdown 표도 HTML로 바꿔 tables에 담음
"""
import html as htmllib
import re


def table_cells(table_html):
    """HTML 표에서 칸의 글자만 순서대로 꺼냅니다."""
    from lxml import html
    root = html.fromstring(table_html)
    return [" ".join("".join(c.itertext()).split()) for c in root.iter("td", "th")]


def _is_separator(line):
    return re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line) is not None


def md_table_to_html(rows):
    """Markdown 표의 행 목록(['| a | b |', ...])을 HTML 표로 바꿉니다. 첫 행은 머리글(th)."""
    out = []
    for i, row in enumerate(r for r in rows if not _is_separator(r.strip())):
        tag = "th" if i == 0 else "td"
        cells = "".join(f"<{tag}>{htmllib.escape(c.strip())}</{tag}>" for c in row.strip().strip("|").split("|"))
        out.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(out) + "</table>"


def markdown_to_text(md, md_tables=False):
    """Markdown·HTML 출력을 평가용 글자로 바꿉니다. 표는 칸마다 한 줄로 꺼냅니다.

    md_tables=False(08장 기준): HTML 표만 표로 셉니다. True(09장): Markdown 표도 표 구조 비교(TEDS)에 씁니다.
    """
    md = re.sub(r"^```\w*\s*$", "", md, flags=re.M)        # 코드 블록 울타리
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)            # 이미지 ![](경로)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)          # 주석 (Docling의 <!-- image --> 등)
    tables = re.findall(r"<table.*?</table>", md, flags=re.S | re.I)
    for t in tables:
        md = md.replace(t, "\n" + "\n".join(table_cells(t)) + "\n")
    lines, md_rows = [], []
    for line in md.splitlines() + [""]:
        s = line.strip()
        if s and _is_separator(s):  # 표 구분선과 영수증의 점선("-----")은 글자가 아니므로 버림
            if s.startswith("|"):
                md_rows.append(s)
            continue
        if s.startswith("|") and s.endswith("|"):
            md_rows.append(s)
            lines.extend(c.strip() for c in s.strip("|").split("|"))
            continue
        if md_rows:  # Markdown 표가 끝남 → HTML로 바꿔 표 목록에 추가
            if md_tables:
                tables.append(md_table_to_html(md_rows))
            md_rows = []
        s = re.sub(r"^#{1,6}\s+", "", s)                    # 제목
        s = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", r"\1\2", s)  # 굵게
        s = re.sub(r"<[^>]+>", "", s)                       # 남은 HTML 태그
        lines.append(s)
    return "\n".join(x for x in lines if x.strip()), tables
