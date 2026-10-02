"""12장 — RAG 평가용 질문·답 세트를 만듭니다 (평가셋의 원본 문서 100개, 문서마다 질문 2개).

LLM이 정답 텍스트를 보고 질문을 만들고, 아래 조건을 통과한 것만 남깁니다.
  - 답이 문서 원문에 그대로 있다 (공백 무시)
  - 질문에 답이 들어 있지 않다
  - 답은 2~40자의 짧은 사실 (날짜·금액·이름·조건 등)
질문에는 문서를 특정하는 단서(상호·제목 등)를 넣게 해, 비슷한 문서(영수증 10장 등)와 헷갈리지 않게 합니다.
실행: python make_qa.py     → qa.jsonl
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
from ocr_eval import load_samples  # noqa: E402

PROMPT = """아래 문서를 읽고, 이 문서만 보고 답할 수 있는 질문 2개를 만드세요.
규칙:
- 답은 문서에 적힌 글자를 그대로 옮긴 짧은 사실(날짜, 금액, 숫자, 이름, 장소, 조건 등, 40자 이내)이어야 합니다.
- 질문에는 이 문서를 다른 문서와 구별할 수 있는 단서(문서 제목, 상호, 회사명, 사람 이름 등)를 넣으세요.
- 질문에 답을 쓰지 마세요.
문서:
{text}"""
SCHEMA = {"type": "object", "properties": {"qa": {"type": "array", "items": {"type": "object", "properties": {
    "question": {"type": "string"}, "answer": {"type": "string"}}, "required": ["question", "answer"]}}},
    "required": ["qa"]}


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--base-url", default="http://localhost:18001/v1")
    args = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    docs = [(m, g) for m, g in load_samples() if m["source"] == "rendered"]  # 열화 이미지는 원본과 내용이 같아 뺌

    def ask(doc):
        meta, gt = doc
        res = client.chat.completions.create(
            model=args.model, temperature=0.0, max_tokens=1024,
            messages=[{"role": "user", "content": PROMPT.format(text=gt["text"])}],
            response_format={"type": "json_schema", "json_schema": {"name": "qa", "schema": SCHEMA}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return json.loads(res.choices[0].message.content)["qa"]

    with ThreadPoolExecutor(8) as pool:
        made = list(pool.map(ask, docs))
    kept, dropped = [], {"답이 원문에 없음": 0, "질문에 답이 있음": 0, "길이": 0}
    for (meta, gt), qas in zip(docs, made):
        for qa in qas[:2]:
            q, a = qa["question"].strip(), qa["answer"].strip()
            if norm(a) not in norm(gt["text"]):
                dropped["답이 원문에 없음"] += 1
            elif norm(a) in norm(q):
                dropped["질문에 답이 있음"] += 1
            elif not 2 <= len(norm(a)) <= 40:
                dropped["길이"] += 1
            else:
                kept.append({"doc_id": meta["id"], "category": meta["category"], "question": q, "answer": a})
    with open(HERE / "qa.jsonl", "w", encoding="utf-8") as fp:
        for r in kept:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"문서 {len(docs)}개, 질문 {sum(len(x[:2]) for x in made)}개 중 {len(kept)}개 통과, 탈락 {dropped}")


if __name__ == "__main__":
    main()
