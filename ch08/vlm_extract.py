"""08장 — 범용 VLM으로 OCR 없이 바로 정보 추출(영수증 JSON)과 문서 질의응답(Document VQA)을 합니다.

① 영수증 10장: 이미지 → JSON. vLLM의 구조화 출력(response_format json_schema)으로 형식을 강제하고,
   06-4의 채점(score)·자체 검증(validate)을 그대로 씁니다.
② 질의응답 60문항: 영수증·명함·세금계산서 30장에 정답 필드로 만든 질문 2개씩. 답이 정답과 같은지(공백 무시) 봅니다.

실행: python vlm_extract.py --model Qwen/Qwen3.5-4B --base-url http://localhost:8000/v1
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "ch06"))
sys.path.insert(0, str(HERE))
from ocr_eval import load_samples  # noqa: E402
from receipt_extract import score, validate  # noqa: E402
from vlm_bench import encode  # noqa: E402

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "store": {"type": ["string", "null"]},
        "biz_no": {"type": ["string", "null"]},
        "tel": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"], "description": "YYYY-MM-DD HH:MM"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "qty": {"type": "integer"}, "amount": {"type": "integer"}},
            "required": ["name", "qty", "amount"]}},
        "total": {"type": ["integer", "null"]},
    },
    "required": ["store", "biz_no", "tel", "date", "items", "total"],
}
RECEIPT_PROMPT = ("이 영수증에서 상호(store), 사업자등록번호(biz_no), 전화번호(tel), 거래일시(date, YYYY-MM-DD HH:MM), "
                  "품목(items: 이름 name, 수량 qty, 금액 amount), 합계(total)를 JSON으로 추출해 주세요. "
                  "금액은 쉼표 없는 정수로, 영수증에 없는 값은 null로 적으세요.")


def questions(meta, f):
    """정답 필드로 질문과 정답을 만듭니다."""
    if meta["category"] == "receipt":
        return [("이 영수증의 합계 금액은 얼마인가요? 숫자만 답하세요.", str(f["total"])),
                ("이 영수증의 거래 일시는 언제인가요? YYYY-MM-DD HH:MM 형식으로만 답하세요.", f["date"])]
    if meta["category"] == "card":
        return [("이 명함의 이메일 주소는 무엇인가요? 주소만 답하세요.", f["email"]),
                ("이 명함 주인의 휴대전화 번호는 무엇인가요? 번호만 답하세요.", f["mobile"])]
    return [("이 세금계산서에서 공급자의 사업자등록번호는 무엇인가요? 번호만 답하세요.", f["supplier"]["biz_no"]),
            ("이 세금계산서에서 공급받는자의 상호는 무엇인가요? 상호만 답하세요.", f["buyer"]["name"])]


def squash(s):
    return re.sub(r"[\s,원]", "", str(s or ""))


def ask(client, args, image_path, prompt, **kwargs):
    content = [{"type": "image_url", "image_url": {"url": encode(Image.open(image_path).convert("RGB"))}},
               {"type": "text", "text": prompt}]
    start = time.perf_counter()
    res = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": content}],
                                         max_tokens=2048, temperature=0.0,
                                         extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kwargs)
    return (res.choices[0].message.content or "").strip(), time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    saved = {"receipt": {}, "vqa": []}

    # ① 영수증 → JSON (구조화 출력으로 스키마 강제)
    fields = ["store", "biz_no", "tel", "date", "total"]
    right, hit, want, got, flagged, times = {f: 0 for f in fields}, 0, 0, 0, [], []
    receipts = load_samples(["receipt"])
    fmt = {"type": "json_schema", "json_schema": {"name": "receipt", "schema": RECEIPT_SCHEMA}}
    for meta, gt in receipts:
        raw, t = ask(client, args, meta["image_path"], RECEIPT_PROMPT, response_format=fmt)
        times.append(t)
        pred = json.loads(raw)
        saved["receipt"][meta["id"]] = pred
        res, h, w, g = score(pred, gt["fields"])
        for f in fields:
            right[f] += res[f]
        hit, want, got = hit + h, want + w, got + g
        problems = validate(pred)
        if problems:
            flagged.append((meta["id"], problems))
    n = len(receipts)
    print(f"model={args.model}")
    print("[영수증 JSON] " + "  ".join(f"{f} {right[f]}/{n}" for f in fields)
          + f"  품목 정확히 일치 {hit}/{want} (추출 {got}개), 평균 {statistics.mean(times):.2f}초")
    print(f"[영수증 JSON] 자체 검증에 걸린 영수증 {len(flagged)}장: {flagged}")
    print("[영수증 JSON] 예:", json.dumps(saved["receipt"][receipts[0][0]["id"]], ensure_ascii=False)[:300])

    # ② 문서 질의응답
    ok, times = 0, []
    for meta, gt in load_samples(["receipt", "card", "invoice"]):
        for q, a in questions(meta, gt["fields"]):
            answer, t = ask(client, args, meta["image_path"], q)
            times.append(t)
            correct = squash(answer) == squash(a)
            ok += correct
            saved["vqa"].append({"id": meta["id"], "q": q, "gold": a, "answer": answer, "correct": correct})
    total = len(saved["vqa"])
    print(f"[질의응답] 정답 {ok}/{total} ({ok / total:.1%}), 평균 {statistics.mean(times):.2f}초")
    for r in [r for r in saved["vqa"] if not r["correct"]][:8]:
        print(f"  틀림 {r['id']}: 정답 {r['gold']!r} / 답 {r['answer'][:60]!r}")

    out = HERE / "output" / f"extract_{args.model.split('/')[-1]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
