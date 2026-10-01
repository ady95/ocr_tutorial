"""평가셋 목록(manifest.jsonl)과 범주별 통계를 만든다.

실행: python manifest.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = []
    for f in sorted((ROOT / "gt").glob("*.json")):
        gt = json.loads(f.read_text(encoding="utf-8"))
        rows.append({
            "id": gt["id"], "category": gt["category"], "source": gt["source"],
            "parent": gt.get("parent"), "image": f"images/{gt['id']}.png", "gt": f"gt/{gt['id']}.json",
            "width": gt["width"], "height": gt["height"], "chars": len(gt["text"].replace("\n", "")),
            "lines": len(gt["lines"]),
        })
    with open(ROOT / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    count = Counter(r["category"] for r in rows)
    chars = Counter()
    for r in rows:
        chars[r["category"]] += r["chars"]
    print(f"전체 {len(rows)}장, {sum(chars.values()):,}자")
    for cat, n in sorted(count.items()):
        print(f"  {cat:15s} {n:3d}장  {chars[cat]:6,d}자")


if __name__ == "__main__":
    main()
