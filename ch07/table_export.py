"""07장 — 인식한 표(HTML)를 CSV·Markdown·Excel로 저장합니다.

실행: python table_export.py   (pandas, lxml, tabulate, openpyxl 필요. structure_eval.py 결과 사용)
"""
import io
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "output"


def main():
    result = json.loads((OUT / "structure.json").read_text(encoding="utf-8"))
    html = result["table_01"]["tables"][0]
    df = pd.read_html(io.StringIO(html), header=0)[0]   # 첫 행을 열 이름으로
    print(df.head(4))
    df.to_csv(OUT / "table_01.csv", index=False, encoding="utf-8-sig")  # Excel에서 한글이 깨지지 않게 BOM 포함
    (OUT / "table_01.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    df.to_excel(OUT / "table_01.xlsx", index=False)
    print()
    print(df.head(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
