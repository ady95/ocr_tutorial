"""13장 — 나만의 OCR 엔진: 앞 장의 부품을 하나의 처리 흐름으로 묶습니다.

  ①② 빠른 OCR     PaddleOCR (04·11장) — 글자줄·좌표·신뢰도. 신뢰도가 낮으면 180도 돌려 다시 읽음 (03-4의 방향 보정)
  ③ 유형 판별      classify.py (13-1)
  ④ 정밀 OCR       PaddleOCR-VL-1.6 (08-2, 10장 1순위) — Markdown·표
  ⑤ 결과 검증      반복 생성·두 엔진 차이 → 문제가 있으면 범용 VLM(Qwen3.5)으로 다시 읽고 더 나은 쪽을 고름 (13-2)
  ⑥ 항목 추출      명함·세금계산서·영수증은 LLM으로 JSON 추출 + 근거·규칙·교차 확인 (12-1, 13-2)

필요한 서버 (OpenAI 호환, run_servers.sh)
  PaddleOCR-VL   http://localhost:18118/v1
  Qwen3.5-9B     http://localhost:18001/v1   (범용 VLM 겸 추출 LLM. 없으면 ⑤의 재시도와 ⑥을 건너뜀)
"""
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("common", "ch06", "ch12"):
    sys.path.insert(0, str(ROOT / sub))
from mdtext import markdown_to_text, table_cells  # noqa: E402

from .classify import classify  # noqa: E402
from .verify import REPEAT_RATIO, check_fields, compression_ratio, disagreement  # noqa: E402

DISAGREE = 0.3   # 빠른 OCR과 정밀 OCR의 차이가 이보다 크면 다시 읽음 (13-2)
REVIEW = 0.05    # 차이가 이보다 크면 사람 확인을 권하는 경고만 남김 (13-2: 이 구간의 CER이 6배 이상 높았음)
FLIP_SCORE = 0.8  # 빠른 OCR의 평균 신뢰도가 이보다 낮으면 뒤집힌 문서인지 확인 (03-4)
TEXT_LABELS = {"doc_title", "paragraph_title", "text", "abstract", "content", "header", "footer",  # 08-2
               "number", "aside_text", "reference", "footnote", "vision_footnote", "figure_title"}
HEADING = {"doc_title": "# ", "paragraph_title": "## "}
VLM_PROMPT = ("이미지에 있는 글자를 빠짐없이 그대로 옮겨 적어 주세요. 고치거나 요약하거나 설명을 붙이지 마세요. "
              "읽는 순서대로 적고, 줄바꿈은 원문을 따르세요. 표는 HTML <table>로 적고, 나머지는 Markdown으로 적으세요.")
RECEIPT = {"type": "object", "properties": {
    "store": {"type": ["string", "null"]}, "biz_no": {"type": ["string", "null"]}, "tel": {"type": ["string", "null"]},
    "date": {"type": ["string", "null"], "description": "YYYY-MM-DD HH:MM"},
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "qty": {"type": "integer"}, "amount": {"type": "integer"}},
        "required": ["name", "qty", "amount"]}},
    "total": {"type": ["integer", "null"]}},
    "required": ["store", "biz_no", "tel", "date", "items", "total"]}


class OcrEngine:
    def __init__(self, vl_url="http://localhost:18118/v1", llm_url="http://localhost:18001/v1",
                 llm_model="Qwen/Qwen3.5-9B", device="gpu"):
        from openai import OpenAI
        from paddleocr import PaddleOCR, PaddleOCRVL
        self.fast = PaddleOCR(lang="korean", device=device, use_doc_orientation_classify=False,
                              use_doc_unwarping=False, use_textline_orientation=False)
        self.vl = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=vl_url, device=device)
        self.llm = OpenAI(base_url=llm_url, api_key="EMPTY", timeout=600) if llm_url else None
        self.llm_model = llm_model

    # ── 단계별 부품 ──────────────────────────────────────────
    def fast_ocr(self, img):
        res = self.fast.predict(img)[0]
        return [{"text": t, "score": round(float(s), 4), "box": [int(v) for v in b]}
                for t, s, b in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"])]

    def read_upright(self, img):
        """빠른 OCR로 읽고, 평균 신뢰도가 낮을 때만 180도 돌려 다시 읽어 더 확신하는 쪽을 고릅니다 (03-4).

        문서 방향 분류 모델은 똑바른 문서 일부를 돌려 버려 평가셋 전체에서 손해였습니다(03-4).
        """
        def score(lines):
            return statistics.mean(ln["score"] for ln in lines) if lines else 0.0

        lines = self.fast_ocr(img)
        if score(lines) >= FLIP_SCORE:
            return img, lines, 0
        flipped = cv2.rotate(img, cv2.ROTATE_180)
        lines2 = self.fast_ocr(flipped)
        return (flipped, lines2, 180) if score(lines2) > score(lines) else (img, lines, 0)

    def vl_ocr(self, img):
        """블록을 읽기 순서대로 이어 글자(text)와 Markdown을 만듭니다. 표는 칸의 글자를 꺼냅니다 (08-2).

        PaddleOCR-VL이 만들어 주는 Markdown(res.markdown)은 머리말·꼬리말 블록을 빼는데, 명함의 연락처 줄이
        꼬리말로 분류되어 사라졌습니다(13-3). 그래서 Markdown도 블록에서 직접 만듭니다.
        """
        res = self.vl.predict(img)[0]
        parts, md, tables = [], [], []
        for b in res["parsing_res_list"]:
            content = (b.content or "").strip()
            if b.label == "table":
                tables.append(content)
                parts.extend(table_cells(content) if "<t" in content else [content])
                md.append(content)
            elif b.label in TEXT_LABELS and content:
                parts.append(content)
                md.append(HEADING.get(b.label, "") + content)
        return {"engine": "paddleocr-vl", "markdown": "\n\n".join(md), "text": "\n".join(parts), "tables": tables}

    def vlm_ocr(self, img):
        _, png = cv2.imencode(".png", img)
        url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
        res = self.llm.chat.completions.create(
            model=self.llm_model, temperature=0.0, max_tokens=8192, presence_penalty=1.5,  # 반복 억제 (08-4)
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                   {"type": "text", "text": VLM_PROMPT}]}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        md = res.choices[0].message.content or ""
        text, tables = markdown_to_text(md, md_tables=True)
        return {"engine": "qwen3.5-vlm", "markdown": md, "text": text, "tables": tables}

    def extract(self, doc_type, text):
        from extract import PROMPTS, extract, flat, rule_errors
        if doc_type == "receipt":
            system, _, _ = PROMPTS["v2"]
            res = self.llm.chat.completions.create(
                model=self.llm_model, temperature=0.0, max_tokens=2048,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": "영수증에서 상호(store), 사업자등록번호(biz_no), 전화(tel), "
                                                      "거래일시(date, 시각까지 YYYY-MM-DD HH:MM), 품목(items: 이름 name, "
                                                      f"수량 qty, 금액 amount), 합계(total)를 뽑으세요.\n\nOCR 결과:\n{text}"}],
                response_format={"type": "json_schema", "json_schema": {"name": "receipt", "schema": RECEIPT}},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            fields = json.loads(res.choices[0].message.content)
            items = [(k, fields.get(k)) for k in ("store", "biz_no", "tel", "date")]
            items += [(f"items[{i}].name", it.get("name")) for i, it in enumerate(fields.get("items") or [])]
            total = sum(it.get("amount") or 0 for it in fields.get("items") or [])
            errors = [] if fields.get("total") in (None, total) else [f"품목 합계 {total} ≠ 합계 {fields['total']}"]
            return fields, items, errors
        fields = extract(self.llm, self.llm_model, doc_type, text, "v2")
        return fields, flat(doc_type, fields), (rule_errors(fields) if doc_type == "invoice" else [])

    # ── 전체 흐름 ────────────────────────────────────────────
    def process(self, img, doc_type=None, mode="accurate", do_extract=True):
        """img: BGR 배열. mode="fast"면 빠른 OCR만, "accurate"면 정밀 OCR과 검증까지."""
        t, times, warnings = time.perf_counter(), {}, []

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            times[name] = round(now - t, 3)
            t = now

        img, lines, angle = self.read_upright(img)
        fast_text = "\n".join(ln["text"] for ln in lines)
        lap("fast_ocr")
        h, w = img.shape[:2]
        kind = {"type": doc_type, "reason": "요청에서 지정"} if doc_type else classify(lines, w, h)
        result = {"type": kind["type"], "type_reason": kind["reason"], "rotated": angle, "size": [w, h],
                  "lines": lines, "times": times, "warnings": warnings}
        if mode == "fast":
            result.update(engine="paddleocr", text=fast_text, markdown=fast_text, tables=[])
            times["total"] = round(sum(times.values()), 3)
            return result

        main = self.vl_ocr(img)
        lap("vl_ocr")
        ratio, diff = compression_ratio(main["text"]), disagreement(fast_text, main["text"])
        checks = {"compression": round(ratio, 2), "disagreement": round(diff, 3)}
        if (ratio > REPEAT_RATIO or diff > DISAGREE) and self.llm:
            warnings.append({"problem": f"정밀 OCR 의심 (압축률 {ratio:.2f}, 빠른 OCR과 차이 {diff:.3f}) → 범용 VLM으로 다시 읽음"})
            second = self.vlm_ocr(img)
            lap("vlm_retry")
            r2, d2 = compression_ratio(second["text"]), disagreement(fast_text, second["text"])
            checks.update(retry_compression=round(r2, 2), retry_disagreement=round(d2, 3))
            # 반복이 아닌 쪽 중에서 빠른 OCR과 더 가까운 결과를 고름 (세 엔진 중 두 엔진이 맞는 쪽)
            if r2 <= REPEAT_RATIO and (ratio > REPEAT_RATIO or d2 < diff):
                main = second
        elif ratio > REPEAT_RATIO:  # 다시 읽을 VLM이 없으면 반복된 결과보다 빠른 OCR 결과가 나음
            warnings.append({"problem": f"정밀 OCR 반복 생성 (압축률 {ratio:.2f}) → 빠른 OCR 결과를 대신 씀"})
            main = {"engine": "paddleocr", "markdown": fast_text, "text": fast_text, "tables": []}
        final = disagreement(fast_text, main["text"])
        if final > REVIEW:
            warnings.append({"problem": f"빠른 OCR과 차이 {final:.3f} — 사람 확인 권장"})
        result.update(engine=main["engine"], text=main["text"], markdown=main["markdown"], tables=main["tables"],
                      checks=checks)

        if do_extract and self.llm and kind["type"] in ("card", "invoice", "receipt"):
            try:
                fields, items, errors = self.extract(kind["type"], main["text"])
                result["fields"] = fields
                warnings += [{"problem": e} for e in errors]
                warnings += check_fields(fields, items, main["text"], fast_text)
            except Exception as e:  # 추출이 실패해도 OCR 결과는 돌려줌
                warnings.append({"problem": f"항목 추출 실패: {str(e)[:100]}"})
            lap("extract")
        times["total"] = round(sum(times.values()), 3)
        return result


def read_image(data):
    """업로드된 바이트 → BGR 배열."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
