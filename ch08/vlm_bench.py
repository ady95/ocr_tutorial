"""08장 — OpenAI 호환 API로 띄운 VLM(vLLM 서버 등)을 평가셋 전체에 실행합니다.

모델마다 OCR 프롬프트와 출력 형식이 달라, 아래 MODELS에 정리해 두었습니다.
  qwen      범용 VLM (Qwen3.5). 한국어 프롬프트로 Markdown 전사, 표는 HTML
  deepseek  DeepSeek-OCR 2. "Free OCR." 또는 "<|grounding|>Convert the document to markdown." + 반복 억제 옵션
  varco     VARCO-VISION-2.0-1.7B-OCR. "<ocr>" → 글자마다 <char>..</char><bbox>..</bbox>

실행 예:
  vllm serve Qwen/Qwen3.5-2B --port 8000 ...
  python vlm_bench.py --kind qwen --model Qwen/Qwen3.5-2B --base-url http://localhost:8000/v1
결과 텍스트는 output/vlm_<이름>.json에 저장합니다.
"""
import argparse
import base64
import io
import json
import re
import statistics
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from ocr_eval import cer, load_samples  # noqa: E402
from teds import teds  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"

PROMPTS = {
    # html: 표를 HTML로 받으면 TEDS를 잴 수 있지만, 작은 모델은 표가 없는 문서에서도 태그를 끝없이 중첩하기도 함
    "html": "이미지에 있는 글자를 빠짐없이 그대로 옮겨 적어 주세요. 고치거나 요약하거나 설명을 붙이지 마세요. "
            "읽는 순서대로 적고, 줄바꿈은 원문을 따르세요. 표는 HTML <table>로 적고, 나머지는 Markdown으로 적으세요.",
    "plain": "이미지에 있는 글자를 빠짐없이 그대로 옮겨 적어 주세요. 고치거나 요약하거나 설명을 붙이지 마세요. "
             "읽는 순서대로 적고, 줄바꿈은 원문을 따르세요.",
}

DEEPSEEK_PROMPTS = {  # vLLM 레시피: 지시문보다 짧은 기본 프롬프트가 더 잘 동작한다
    "free": "Free OCR.",
    "markdown": "<|grounding|>Convert the document to markdown.",
}


def table_cells(table_html):
    from lxml import html
    root = html.fromstring(table_html)
    return [" ".join("".join(c.itertext()).split()) for c in root.iter("td", "th")]


def markdown_to_text(md):
    """VLM의 Markdown·HTML 출력을 평가용 글자로 바꿉니다. 표는 칸마다 한 줄로 꺼냅니다."""
    md = re.sub(r"^```\w*\s*$", "", md, flags=re.M)  # 코드 블록 울타리
    tables = re.findall(r"<table.*?</table>", md, flags=re.S | re.I)
    for t in tables:
        md = md.replace(t, "\n" + "\n".join(table_cells(t)) + "\n")
    lines = []
    for line in md.splitlines():
        s = line.strip()
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", s):  # Markdown 표 구분선
            continue
        if s.startswith("|") and s.endswith("|"):
            lines.extend(c.strip() for c in s.strip("|").split("|"))
            continue
        s = re.sub(r"^#{1,6}\s+", "", s)                  # 제목
        s = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", r"\1\2", s)       # 굵게
        s = re.sub(r"<[^>]+>", "", s)                     # 남은 HTML 태그
        lines.append(s)
    return "\n".join(x for x in lines if x.strip()), tables


def deepseek_clean(text):
    """grounding 출력의 <|ref|>종류<|/ref|><|det|>[[좌표]]<|/det|> 표시를 지웁니다."""
    return re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", text, flags=re.S)


def varco_text(raw):
    """<char>단어</char><bbox>x1, y1, x2, y2</bbox>(좌표는 0~1로 정규화)를 읽어 줄을 복원합니다.

    태그 이름은 char지만 실제로는 띄어쓰기 단위(어절)마다 하나씩 나오므로, 같은 줄의 어절은 공백으로 잇고
    이전 어절보다 아래로 내려가거나 왼쪽으로 되돌아가면 줄을 바꿉니다.
    """
    words = [(w, [float(v) for v in b.split(",")])
             for w, b in re.findall(r"<char>(.*?)</char><bbox>(.*?)</bbox>", raw, flags=re.S)]
    lines, cur, prev = [], [], None
    for w, (x1, y1, x2, y2) in words:
        if prev is not None and (y1 > prev[3] - (prev[3] - prev[1]) * 0.3 or x1 < prev[0]):
            lines.append(" ".join(cur))
            cur = []
        cur.append(w)
        prev = (x1, y1, x2, y2)
    lines.append(" ".join(cur))
    return "\n".join(x for x in lines if x.strip())


def encode(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def compression_ratio(text):
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


def run_one(client, args, path):
    image = Image.open(path).convert("RGB")
    extra = {}
    if args.kind == "varco":
        w, h = image.size
        if max(w, h) < 2304:  # 모델 카드 권장: 긴 변 2,304픽셀 이상으로 확대
            s = 2304 / max(w, h)
            image = image.resize((int(w * s), int(h * s)))
        prompt = "<ocr>"
        extra = {"skip_special_tokens": False}
    elif args.kind == "deepseek":
        prompt = DEEPSEEK_PROMPTS[args.ds_prompt]
        extra = {"skip_special_tokens": False,
                 "vllm_xargs": {"ngram_size": 30, "window_size": 90, "whitelist_token_ids": [128821, 128822]}}
    else:
        prompt = PROMPTS[args.prompt]
        if args.presence_penalty:
            extra["presence_penalty"] = args.presence_penalty
        if args.no_think:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
    content = [{"type": "image_url", "image_url": {"url": encode(image)}}, {"type": "text", "text": prompt}]
    start = time.perf_counter()
    try:
        res = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": content}],
                                             max_tokens=args.max_tokens, temperature=0.0, extra_body=extra)
    except Exception as e:  # 이미지 한 장의 오류(입력 초과 등)로 전체 측정이 멈추지 않게 기록만 하고 넘어감
        print(f"[오류] {Path(path).name}: {str(e)[:200]}", flush=True)
        return "", time.perf_counter() - start, 0, 0, "error"
    elapsed = time.perf_counter() - start
    choice = res.choices[0]
    return choice.message.content or "", elapsed, res.usage.prompt_tokens, res.usage.completion_tokens, choice.finish_reason


def to_text(kind, raw):
    if kind == "varco":
        return varco_text(raw), []
    if kind == "deepseek":
        raw = deepseek_clean(raw)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    return markdown_to_text(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["qwen", "deepseek", "varco"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--name", help="결과 파일 이름 (기본: kind)")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=1, help="동시에 보낼 요청 수 (1이면 이미지당 시간 측정)")
    ap.add_argument("--no-think", action="store_true", help="Qwen3.5의 생각(thinking) 모드 끄기")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="html", help="범용 VLM용 프롬프트")
    ap.add_argument("--presence-penalty", type=float, default=0.0, help="이미 나온 토큰을 덜 고르게 함 (반복 억제)")
    ap.add_argument("--ds-prompt", choices=sorted(DEEPSEEK_PROMPTS), default="free", help="DeepSeek-OCR 2 프롬프트")
    ap.add_argument("--rescore", action="store_true", help="서버에 요청하지 않고 저장된 원문(raw)으로 다시 채점")
    args = ap.parse_args()
    samples = load_samples(args.categories)
    name = args.name or args.kind
    if args.rescore:  # 파싱 규칙을 고친 뒤 다시 채점할 때
        old = json.loads((OUT / f"vlm_{name}.json").read_text(encoding="utf-8"))
        outs = [(old[m["id"]]["raw"], old[m["id"]]["time"], old[m["id"]]["prompt_tokens"],
                 old[m["id"]]["completion_tokens"], old[m["id"]].get("finish_reason")) for m, _ in samples]
        wall = float("nan")
    else:
        client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
        run_one(client, args, samples[0][0]["image_path"])  # 첫 실행 제외
        wall = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            outs = list(pool.map(partial(run_one, client, args), [m["image_path"] for m, _ in samples]))
        wall = time.perf_counter() - wall

    per_cat, per_cat_ns, teds_scores, saved = defaultdict(list), defaultdict(list), [], {}
    for (meta, gt), (raw, elapsed, ptok, ctok, finish) in zip(samples, outs):
        text, tables = to_text(args.kind, raw)
        saved[meta["id"]] = {"text": text, "raw": raw, "time": elapsed, "prompt_tokens": ptok,
                             "completion_tokens": ctok, "finish_reason": finish, "compression": compression_ratio(raw)}
        per_cat[meta["category"]].append(cer(gt["text"], text))
        per_cat_ns[meta["category"]].append(cer(gt["text"], text, keep_space=False))
        if meta["category"] == "table":
            pred = tables[0] if tables else ""
            teds_scores.append((teds(pred, gt["table_html"]), teds(pred, gt["table_html"], structure_only=True)))

    OUT.mkdir(exist_ok=True)
    (OUT / f"vlm_{name}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"model={args.model}")
    print(f"{'범주':14s} {'장수':>4s} {'CER':>7s} {'CER(공백무시)':>13s}")
    allv, allns = [], []
    for c in sorted(per_cat):
        allv += per_cat[c]
        allns += per_cat_ns[c]
        print(f"{c:14s} {len(per_cat[c]):4d} {statistics.mean(per_cat[c]):7.3f} {statistics.mean(per_cat_ns[c]):13.3f}")
    print(f"{'전체':14s} {len(allv):4d} {statistics.mean(allv):7.3f} {statistics.mean(allns):13.3f}")
    if teds_scores:
        print(f"표 TEDS {statistics.mean(t for t, _ in teds_scores):.3f}, TEDS-S {statistics.mean(s for _, s in teds_scores):.3f}")
    times = [o[1] for o in outs]
    print(f"이미지당 처리 시간: 평균 {statistics.mean(times):.2f}초, 중앙값 {statistics.median(times):.2f}초 "
          f"(전체 {wall:.0f}초, 동시 요청 {args.workers})")
    print(f"입력 토큰 중앙값 {statistics.median(o[2] for o in outs):.0f}, 출력 토큰 중앙값 {statistics.median(o[3] for o in outs):.0f}")
    loops = sorted(k for k, v in saved.items() if v["compression"] > 4)
    cut = sorted(k for k, v in saved.items() if v["finish_reason"] == "length")
    print(f"원문 압축률 4 초과(반복 의심): {len(loops)}장 {loops}")
    print(f"출력 토큰 한도({args.max_tokens})에서 잘림: {len(cut)}장 {cut}")
    errors = sorted(k for k, v in saved.items() if v["finish_reason"] == "error")
    if errors:
        print(f"요청 오류: {len(errors)}장 {errors}")


if __name__ == "__main__":
    main()
