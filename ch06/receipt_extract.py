"""06장 실습 — 영수증 OCR: 규칙 기반 추출과 LLM 추출을 정답 필드와 비교합니다.

입력: dump_ocr.py로 저장한 PaddleOCR 결과 (글자줄 텍스트 + 좌표)
필드: 상호, 사업자등록번호, 전화번호, 거래일시, 품목(이름·수량·금액), 합계

실행: python receipt_extract.py            (LLM 추출에는 OPENAI_API_KEY, OPENAI_MODEL 필요)
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import load_samples  # noqa: E402
from rules import apply_rules, biz_number_valid  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
AMOUNT = re.compile(r"^[\d,]+원?$")


def same_row(a, b):
    """두 상자 (x1, y1, x2, y2)의 세로 범위가 작은 쪽 높이의 절반 이상 겹치면 같은 줄."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap > 0.5 * min(a[3] - a[1], b[3] - b[1])


def group_rows(lines):
    """글자줄을 같은 줄끼리 묶고 왼쪽부터 정렬합니다 (04-3과 같은 방법)."""
    rows = []
    for line in sorted(lines, key=lambda ln: ln["box"][1]):
        for row in rows:
            if same_row(row[0]["box"], line["box"]):
                row.append(line)
                break
        else:
            rows.append([line])
    return [[apply_rules(ln["text"]) for ln in sorted(r, key=lambda ln: ln["box"][0])] for r in rows]


def to_int(text):
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def extract_rules(lines):
    rows = group_rows(lines)
    flat = "\n".join(" ".join(r) for r in rows)
    out = {"store": rows[0][0] if rows else None}
    m = re.search(r"\d{3}-\d{2}-\d{5}", flat)
    out["biz_no"] = m.group(0) if m else None
    m = re.search(r"0\d{1,2}-\d{3,4}-\d{4}", flat)
    out["tel"] = m.group(0) if m else None
    m = re.search(r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}", flat)
    out["date"] = m.group(0) if m else None
    items, total = [], None
    for cells in rows:
        if len(cells) == 2 and AMOUNT.match(cells[1].replace(" ", "")):
            left, amount = cells[0], to_int(cells[1])
            if left.replace(" ", "").startswith("합계"):
                total = amount
                continue
            m = re.search(r"\s*x\s*(\d+)$", left)
            if m:
                items.append({"name": left[:m.start()].strip(), "qty": int(m.group(1)), "amount": amount})
    out["items"], out["total"] = items, total
    return out


PROMPT = """다음은 영수증 OCR 결과입니다. 아래 JSON 형식으로만 답하세요. 영수증에 없는 값은 null로 둡니다.
{{"store": 상호, "biz_no": "000-00-00000", "tel": 전화번호, "date": "YYYY-MM-DD HH:MM",
  "items": [{{"name": 품목명, "qty": 수량(정수), "amount": 금액(정수)}}], "total": 합계 금액(정수)}}

OCR 결과:
{text}"""


def extract_llm(client, text):
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(text=text)}])
    return json.loads(resp.choices[0].message.content)


def validate(pred):
    """추출 결과의 자체 검증: 사업자등록번호 체크섬, 품목 금액 합계 = 합계."""
    problems = []
    if not biz_number_valid(pred.get("biz_no") or ""):
        problems.append("사업자등록번호 체크섬 불일치")
    amounts = [i.get("amount") or 0 for i in pred.get("items") or []]
    if pred.get("total") is not None and sum(amounts) != pred["total"]:
        problems.append(f"품목 합계 {sum(amounts):,} ≠ 합계 {pred['total']:,}")
    return problems


def squash(s):
    return re.sub(r"\s+", "", str(s or ""))


def score(pred, gt):
    """필드별 정답 여부와 품목 일치 수를 돌려줍니다 (이름은 공백을 무시하고 비교)."""
    res = {k: squash(pred.get(k)) == squash(gt[k]) for k in ["store", "biz_no", "tel", "date"]}
    res["total"] = pred.get("total") == gt["total"]
    want = {(squash(i["name"]), i["qty"], i["amount"]) for i in gt["items"]}
    got = {(squash(i.get("name")), i.get("qty"), i.get("amount")) for i in pred.get("items") or []}
    return res, len(want & got), len(want), len(got)


def main():
    ocr = json.loads((OUT / "ocr_paddle.json").read_text(encoding="utf-8"))
    samples = load_samples(["receipt"])
    cache_path = OUT / "llm_receipt.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if os.getenv("OPENAI_API_KEY") and len(cache) < len(samples):
        from openai import OpenAI
        client = OpenAI()
        for meta, _ in samples:
            if meta["id"] not in cache:
                cache[meta["id"]] = extract_llm(client, ocr[meta["id"]]["text"])
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    methods = {"규칙": lambda sid: extract_rules(ocr[sid]["lines"])}
    if cache:
        methods["LLM"] = lambda sid: cache[sid]
    fields = ["store", "biz_no", "tel", "date", "total"]
    for name, fn in methods.items():
        right = {f: 0 for f in fields}
        hit = want = got = 0
        for meta, gt in samples:
            res, h, w, g = score(fn(meta["id"]), gt["fields"])
            for f in fields:
                right[f] += res[f]
            hit, want, got = hit + h, want + w, got + g
        n = len(samples)
        print(f"[{name}] " + "  ".join(f"{f} {right[f]}/{n}" for f in fields)
              + f"  품목 정확히 일치 {hit}/{want} (추출 {got}개)")

    for name, fn in methods.items():
        flagged = [(m["id"], validate(fn(m["id"]))) for m, _ in samples]
        flagged = [(sid, p) for sid, p in flagged if p]
        print(f"[{name}] 자체 검증에 걸린 영수증 {len(flagged)}장: {flagged}")

    sample = samples[0][0]["id"]
    print("\n규칙 추출 예:", json.dumps(extract_rules(ocr[sample]["lines"]), ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
