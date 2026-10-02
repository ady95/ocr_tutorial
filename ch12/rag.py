"""12장 — OCR 결과로 RAG를 만들고, 질문마다 검색·답변 정확도를 잽니다.

문서: 평가셋의 원본 문서 100개. 텍스트 출처는 --source로 고릅니다.
      gt(정답 텍스트) 또는 10장 OCR 결과 이름(paddle, paddlevl, tesseract …)
단계: ① 청킹   줄을 이어 붙여 --chunk 글자 안팎의 조각으로 나누고, 문서 번호를 메타데이터로 붙임
      ② 임베딩 bge-m3 (OpenAI 호환 /v1/embeddings)
      ③ 검색   질문 임베딩과 코사인 유사도가 높은 조각 --k개
      ④ 답변   상위 조각만 근거로 LLM이 짧게 답하고 근거 번호를 붙임. 근거가 없으면 '모름'
지표: 검색 hit@1·hit@3 (정답 문서의 조각이 위에 있는가)
      답 정확도: 근거 번호를 지운 답이 정답을 포함하거나, 정답의 절반 이상을 차지하는 일부이면 맞음 (공백 무시)
실행: python rag.py --source paddlevl
"""
import argparse
import json
import re
import statistics
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402

SYSTEM = ("아래 문서 조각만 근거로 질문에 답하세요. 답은 문서에 적힌 그대로 짧게 쓰고, 끝에 근거 조각 번호를 [1]처럼 붙이세요. "
          "조각에 답이 없으면 '모름'이라고만 쓰세요.")


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def load_docs(source):
    docs = {m["id"]: (m, g) for m, g in load_samples() if m["source"] == "rendered"}
    if source == "gt":
        texts = {k: g["text"] for k, (m, g) in docs.items()}
    else:
        data = json.loads((HERE.parent / "ch10" / "output" / f"{source}__ko.json").read_text(encoding="utf-8"))
        texts = {k: data[k]["text"] for k in docs}
    return docs, texts


def is_correct(gold, pred):
    """'T 02-000-5636' 대 '02-000-5636', '수강료의 75%인 180,000원' 대 '180,000원'처럼 표현만 다른 답을 맞게 셈."""
    g, p = norm(gold), norm(re.sub(r"\[\d+\]", "", pred))
    return g in p or (len(p) >= len(g) / 2 and p in g)


def char_f1(gold, pred):
    """글자 단위 F1 (같은 글자가 몇 번 겹치는지)."""
    from collections import Counter
    g, p = Counter(norm(gold)), Counter(norm(re.sub(r"\[\d+\]", "", pred)))
    hit = sum((g & p).values())
    return 0.0 if hit == 0 else 2 * hit / (sum(g.values()) + sum(p.values()))


def chunk(text, size):
    """줄 단위로 이어 붙이다가 size 글자를 넘으면 새 조각을 시작합니다 (줄 중간에서 자르지 않음)."""
    out, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        out.append(cur)
    return out


def embed(client, texts, model):
    vecs = []
    for i in range(0, len(texts), 64):
        res = client.embeddings.create(model=model, input=texts[i:i + 64])
        vecs += [d.embedding for d in res.data]
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def answer(client, model, question, hits):
    context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(hits, 1))
    res = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=200,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{context}\n\n질문: {question}"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return res.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gt")
    ap.add_argument("--chunk", type=int, default=400, help="조각 하나의 대략적인 글자 수")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-answer", action="store_true", help="검색만 재고 답변은 만들지 않음")
    ap.add_argument("--llm", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--llm-url", default="http://localhost:18001/v1")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--embed-url", default="http://localhost:18002/v1")
    args = ap.parse_args()
    from openai import OpenAI
    emb_client = OpenAI(base_url=args.embed_url, api_key="EMPTY", timeout=600)
    llm_client = OpenAI(base_url=args.llm_url, api_key="EMPTY", timeout=600)

    docs, texts = load_docs(args.source)
    chunks = [{"doc_id": d, "no": i, "text": c} for d, t in texts.items() for i, c in enumerate(chunk(t, args.chunk))]
    matrix = embed(emb_client, [c["text"] for c in chunks], args.embed_model)
    qa = [json.loads(line) for line in open(HERE / "qa.jsonl", encoding="utf-8")]
    qvec = embed(emb_client, [q["question"] for q in qa], args.embed_model)
    order = np.argsort(-(qvec @ matrix.T), axis=1)[:, :args.k]

    results = []
    for q, idx in zip(qa, order):
        hits = [chunks[i] for i in idx]
        ranks = [r for r, h in enumerate(hits, 1) if h["doc_id"] == q["doc_id"]]
        results.append({**q, "rank": ranks[0] if ranks else None, "hits": [h["doc_id"] for h in hits],
                        "answer_in_chunks": any(norm(q["answer"]) in norm(h["text"]) for h in hits)})
    if not args.no_answer:
        with ThreadPoolExecutor(8) as pool:
            answers = list(pool.map(lambda x: answer(llm_client, args.llm, x[0]["question"], [chunks[i] for i in x[1]]),
                                    zip(qa, order)))
        for r, a in zip(results, answers):
            r["pred"] = a
            r["correct"] = is_correct(r["answer"], a)
            r["f1"] = char_f1(r["answer"], a)
            r["unknown"] = "모름" in a

    n = len(results)
    doc_cer = statistics.mean(cer(docs[k][1]["text"], t) for k, t in texts.items())
    print(f"source={args.source} 문서 {len(texts)}개 (평균 CER {doc_cer:.3f}), 조각 {len(chunks)}개 (약 {args.chunk}자), 질문 {n}개")
    print(f"검색 hit@1 {sum(r['rank'] == 1 for r in results) / n:.3f}, hit@{args.k} {sum(r['rank'] is not None for r in results) / n:.3f}, "
          f"검색된 조각에 정답 글자가 있음 {sum(r['answer_in_chunks'] for r in results) / n:.3f}")
    if not args.no_answer:
        print(f"답 정확도 {sum(r['correct'] for r in results) / n:.3f} (글자 F1 {statistics.mean(r['f1'] for r in results):.3f}), "
              f"모름 {sum(r['unknown'] for r in results) / n:.3f}, "
              f"틀린 답(모름 제외) {sum(not r['correct'] and not r['unknown'] for r in results) / n:.3f}")
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    (out / f"rag_{args.source}_c{args.chunk}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                                               encoding="utf-8")


if __name__ == "__main__":
    main()
