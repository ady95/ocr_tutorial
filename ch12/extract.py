"""12장 — OCR 결과(글자)를 LLM에 주고 JSON으로 구조화합니다: 세금계산서·명함·표.

입력: 10장 bench.py가 저장한 OCR 결과 (ch10/output/<engine>__ko.json). OCR을 다시 돌리지 않습니다.
      --engine gt를 주면 정답 텍스트를 넣어, OCR 오류가 없을 때의 상한을 잽니다.
LLM: OpenAI 호환 서버(run_servers.sh)의 구조화 출력(response_format json_schema)으로 형식을 강제합니다.
검증: ① 근거 검증 — 뽑은 값이 OCR 글자 안에 있는가 (없으면 LLM이 만든 값)
      ② 규칙 검증 — 사업자등록번호 검증 숫자, 공급가액 × 10% = 세액, 합계 (세금계산서)
실행: python extract.py --engine paddlevl [--prompt v1]
결과: output/extract_<engine>_<prompt>.json
"""
import argparse
import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "ch06"))
from ocr_eval import load_samples  # noqa: E402
from rules import biz_number_valid  # noqa: E402

PARTY = {"type": "object", "properties": {"biz_no": {"type": ["string", "null"]}, "name": {"type": ["string", "null"]},
                                          "ceo": {"type": ["string", "null"]}}, "required": ["biz_no", "name", "ceo"]}
SCHEMAS = {
    "card": {"type": "object", "properties": {k: {"type": ["string", "null"]} for k in
             ("company", "name", "dept", "title", "tel", "mobile", "email", "address")},
             "required": ["company", "name", "dept", "title", "tel", "mobile", "email", "address"]},
    "invoice": {"type": "object", "properties": {
        "supplier": PARTY, "buyer": PARTY, "date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "qty": {"type": "integer"}, "price": {"type": "integer"},
            "supply": {"type": "integer"}, "tax": {"type": "integer"}}, "required": ["name", "qty", "price", "supply", "tax"]}},
        "supply_total": {"type": ["integer", "null"]}, "tax_total": {"type": ["integer", "null"]},
        "total": {"type": ["integer", "null"]}},
        "required": ["supplier", "buyer", "date", "items", "supply_total", "tax_total", "total"]},
    "table": {"type": "object", "properties": {
        "header": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}},
        "required": ["header", "rows"]},
}
GUIDE = {
    "card": "명함에서 회사명(company), 이름(name), 부서(dept), 직함(title), 전화(tel), 휴대전화(mobile), 이메일(email), 주소(address)를 뽑으세요.",
    "invoice": "세금계산서에서 공급자(supplier)와 공급받는자(buyer)의 사업자등록번호(biz_no)·상호(name)·대표자(ceo), 작성일(date), "
               "품목(items: 품목명 name, 수량 qty, 단가 price, 공급가액 supply, 세액 tax), 공급가액 합계(supply_total), "
               "세액 합계(tax_total), 합계 금액(total)을 뽑으세요.",
    "table": "OCR 결과는 표의 칸을 한 줄에 하나씩 적은 것입니다. 첫 행(header)과 나머지 행(rows)으로 표를 복원하세요. "
             "여러 행에 걸친 칸(병합 셀)은 해당하는 모든 행에 같은 값을 반복해 적으세요.",
}
SYSTEM = ("당신은 OCR 결과에서 정보를 뽑는 도구입니다. 규칙: "
          "1) OCR 결과에 적힌 글자를 그대로 옮기고, 고치거나 추측하지 마세요. "
          "2) OCR 결과에 없는 항목은 null로 두세요. "
          "3) 금액과 수량은 쉼표 없는 정수로 적으세요. 4) 날짜는 YYYY-MM-DD로 적으세요.")


# v2: v1의 오류를 보고 고친 프롬프트 (12-1). ① 형식 규칙을 필드별로 ② 표 배치 힌트 ③ 스키마로 길이 제한
STRING_KEEP = {"type": ["string", "null"], "maxLength": 100, "description": "원문 형식 그대로 (하이픈·공백 포함)"}
SCHEMAS_V2 = json.loads(json.dumps(SCHEMAS))
for _k in ("tel", "mobile"):
    SCHEMAS_V2["card"]["properties"][_k] = STRING_KEEP
for _p in ("supplier", "buyer"):
    SCHEMAS_V2["invoice"]["properties"][_p]["properties"]["biz_no"] = STRING_KEEP
SCHEMAS_V2["invoice"]["properties"]["items"]["maxItems"] = 20
SCHEMAS_V2["table"]["properties"]["header"].update(maxItems=12)
SCHEMAS_V2["table"]["properties"]["rows"].update(maxItems=40)
SCHEMAS_V2["table"]["properties"]["rows"]["items"].update(maxItems=12, items={"type": "string", "maxLength": 60})
GUIDE_V2 = dict(GUIDE)
GUIDE_V2["invoice"] = GUIDE["invoice"] + (" OCR 결과에서는 공급자와 공급받는자 칸이 번갈아 나옵니다. 같은 항목 이름"
                                          "(등록번호·상호·대표자)이 두 번 나오면 앞의 값이 공급자, 뒤의 값이 공급받는자입니다.")
GUIDE_V2["table"] = GUIDE["table"] + " 행은 OCR 결과에 있는 칸으로만 만들고, 같은 행을 반복하지 마세요."
SYSTEM_V2 = ("당신은 OCR 결과에서 정보를 뽑는 도구입니다. 규칙: "
             "1) OCR 결과에 적힌 글자를 그대로 옮기고, 고치거나 추측하지 마세요. "
             "2) OCR 결과에 없는 항목은 JSON null로 두세요 (문자열 'null'이 아님). "
             "3) 금액·수량 필드만 쉼표 없는 정수로 바꾸고, 전화번호·사업자등록번호 같은 나머지 값은 원문 형식 그대로 두세요. "
             "4) 날짜는 YYYY-MM-DD로 적으세요.")
PROMPTS = {"v1": (SYSTEM, GUIDE, SCHEMAS), "v2": (SYSTEM_V2, GUIDE_V2, SCHEMAS_V2)}


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(s))) if s is not None else None


def ocr_text(engine, samples):
    if engine == "gt":
        return {m["id"]: g["text"] for m, g in samples}
    data = json.loads((HERE.parent / "ch10" / "output" / f"{engine}__ko.json").read_text(encoding="utf-8"))
    return {k: v["text"] for k, v in data.items() if k != "_meta"}


TABLE_MD = ("OCR 결과는 표의 칸을 한 줄에 하나씩 적은 것입니다. 원래 표를 Markdown 표로 복원하세요. 첫 행은 머리글입니다. "
            "여러 행에 걸친 칸(병합 셀)은 해당하는 모든 행에 같은 값을 반복해 적으세요. 표 밖의 제목은 빼고, 표만 출력하세요.")


def md_rows(text):
    """LLM이 쓴 Markdown 표를 {header, rows}로 바꿉니다 (구분선 |---| 은 버림)."""
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in text.splitlines() if ln.strip().startswith("|")]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    return {"header": rows[0] if rows else [], "rows": rows[1:]}


def extract(client, model, category, text, prompt="v2"):
    system, guide, schemas = PROMPTS[prompt]
    if prompt == "v2" and category == "table":
        # JSON으로 강제하면 칸 안에서 공백을 끝없이 내 토큰 한도에 걸렸음 → 익숙한 Markdown 표로 받아 코드로 파싱
        res = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=4096,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": f"{TABLE_MD}\n\nOCR 결과:\n{text}"}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return md_rows(res.choices[0].message.content)
    res = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=4096,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"{guide[category]}\n\nOCR 결과:\n{text}"}],
        response_format={"type": "json_schema", "json_schema": {"name": category, "schema": schemas[category]}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return json.loads(res.choices[0].message.content)


def flat(category, d):
    """비교하기 쉽게 (필드 이름, 값) 목록으로 펼칩니다."""
    if category == "card":
        return [(k, d.get(k)) for k in SCHEMAS["card"]["required"]]
    out = [(f"{p}.{k}", (d.get(p) or {}).get(k)) for p in ("supplier", "buyer") for k in ("biz_no", "name", "ceo")]
    out += [(k, d.get(k)) for k in ("date", "supply_total", "tax_total", "total")]
    for i, it in enumerate(d.get("items") or []):
        out += [(f"items[{i}].{k}", it.get(k)) for k in ("name", "qty", "price", "supply", "tax")]
    return out


def grounded(value, text):
    """값이 OCR 글자 안에 있는가. 숫자는 쉼표·공백을 지우고 찾습니다."""
    if value is None:
        return True
    hay = norm(text).replace(",", "")
    return norm(value).replace(",", "") in hay


def rule_errors(d):
    """세금계산서의 계산·형식 규칙 위반 목록."""
    errs = []
    for p in ("supplier", "buyer"):
        no = (d.get(p) or {}).get("biz_no")
        if no and not biz_number_valid(no):
            errs.append(f"{p}.biz_no 검증 숫자")
    items = d.get("items") or []
    for i, it in enumerate(items):
        if it["qty"] * it["price"] != it["supply"]:
            errs.append(f"items[{i}] 수량×단가≠공급가액")
        if round(it["supply"] * 0.1) != it["tax"]:
            errs.append(f"items[{i}] 세액≠공급가액×10%")
    if d.get("supply_total") is not None and sum(it["supply"] for it in items) != d["supply_total"]:
        errs.append("공급가액 합계")
    if d.get("tax_total") is not None and sum(it["tax"] for it in items) != d["tax_total"]:
        errs.append("세액 합계")
    if None not in (d.get("supply_total"), d.get("tax_total"), d.get("total")) and d["supply_total"] + d["tax_total"] != d["total"]:
        errs.append("합계 금액")
    return errs


def table_grid(table_html):
    """정답 표 HTML을 병합 셀을 풀어 격자(행 목록)로 바꿉니다."""
    from lxml import html
    grid, pending = [], {}
    for r, tr in enumerate(html.fromstring(table_html).iter("tr")):
        row, c = [], 0
        cells = iter(tr.iter("td", "th"))
        while True:
            if (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
                continue
            cell = next(cells, None)
            if cell is None:
                break
            text = " ".join("".join(cell.itertext()).split())
            for dr in range(1, int(cell.get("rowspan", 1))):
                pending[(r + dr, c)] = text
            row.append(text)
            c += 1
        while (r, c) in pending:
            row.append(pending.pop((r, c)))
            c += 1
        grid.append(row)
    return grid


def score_table(pred, gt_html):
    """칸 단위 정확도: 정답 격자의 칸마다 같은 위치의 예측 칸이 같은지."""
    gt = table_grid(gt_html)
    rows = [pred.get("header", [])] + pred.get("rows", [])
    total = sum(len(r) for r in gt)
    hit = sum(norm(c) == norm(rows[i][j]) for i, r in enumerate(gt) for j, c in enumerate(r)
              if i < len(rows) and j < len(rows[i]))
    return hit / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="paddlevl", help="10장 OCR 결과 이름 (paddle, paddlevl, tesseract …) 또는 gt")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--base-url", default="http://localhost:18001/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v2", help="v1: 처음 프롬프트, v2: 오류를 보고 고친 프롬프트")
    args = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    samples = load_samples(["card", "invoice", "table"])
    texts = ocr_text(args.engine, samples)

    def run(sample):
        meta, gt = sample
        try:
            return extract(client, args.model, meta["category"], texts[meta["id"]], args.prompt)
        except Exception as e:  # 형식이 깨진 응답 등
            return {"_error": str(e)[:200]}

    with ThreadPoolExecutor(args.workers) as pool:
        preds = list(pool.map(run, samples))

    stats = {c: {"fields": 0, "correct": 0, "ungrounded_ok": 0, "ungrounded_bad": 0, "grounded_bad": 0}
             for c in ("card", "invoice")}
    tables, flagged, saved = [], [], {}
    for (meta, gt), pred in zip(samples, preds):
        cat, sid, text = meta["category"], meta["id"], texts[meta["id"]]
        saved[sid] = pred
        if "_error" in pred:
            continue
        if cat == "table":
            tables.append(score_table(pred, gt["table_html"]))
            continue
        truth = dict(flat(cat, gt["fields"]))
        doc_wrong = False
        for k, v in flat(cat, pred):
            s = stats[cat]
            ok = k in truth and norm(v) == norm(truth[k])
            s["fields"] += 1
            s["correct"] += ok
            doc_wrong |= not ok
            if v is not None and not grounded(v, text):
                s["ungrounded_ok" if ok else "ungrounded_bad"] += 1
            elif not ok:
                s["grounded_bad"] += 1
        s["fields"] += max(0, len(truth) - len(flat(cat, pred)))  # 품목을 빠뜨린 경우의 칸도 틀린 것으로
        if cat == "invoice":
            flagged.append((bool(rule_errors(pred)), doc_wrong or len(truth) != len(flat(cat, pred))))

    print(f"engine={args.engine} model={args.model} prompt={args.prompt}")
    for c, s in stats.items():
        print(f"{c:8s} 필드 정확도 {s['correct'] / max(1, s['fields']):.3f} ({s['correct']}/{s['fields']}) | "
              f"틀린 값 중 OCR에 있던 값 {s['grounded_bad']}, OCR에 없던 값(지어냄) {s['ungrounded_bad']} | "
              f"OCR에 없지만 맞은 값(LLM이 고침) {s['ungrounded_ok']}")
    if tables:
        print(f"table    칸 정확도 {sum(tables) / len(tables):.3f} ({len(tables)}장)")
    if flagged:
        caught = sum(f and w for f, w in flagged)
        print(f"규칙 검증: 틀린 세금계산서 {sum(w for _, w in flagged)}장 중 {caught}장 적발, "
              f"맞는데 경고 {sum(f and not w for f, w in flagged)}장")
    print(f"형식 오류 {sum('_error' in p for p in preds)}건")
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    (out / f"extract_{args.engine}_{args.prompt}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
