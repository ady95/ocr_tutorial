"""13장 — OCR 엔진 API 서버.

  POST /v1/ocr    이미지·PDF를 올리면 결과를 돌려줌
                  form 필드: type(유형 지정, 생략하면 자동 판별), mode(accurate|fast), extract(true|false),
                             format(json|markdown), lines(true면 글자줄 좌표 포함)
  GET  /health    준비 여부와 대기 중인 작업 수

11-3의 서버처럼 엔진은 프로세스에 하나만 올리고, 작업자 스레드 하나가 큐에서 꺼내 차례로 처리합니다.
큐가 가득 차면 503을 돌려줍니다. 정밀 OCR과 LLM은 별도 vLLM 서버(run_servers.sh)가 맡습니다.

환경 변수: OCR_VL_URL, OCR_LLM_URL(빈 값이면 재시도·추출 끔), OCR_LLM_MODEL, OCR_DEVICE, OCR_MAX_MB, OCR_MAX_PAGES, OCR_QUEUE_SIZE
실행: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from ocrengine import OcrEngine, read_image
from ocrengine.classify import TYPES
from ocrengine.output import to_json, to_markdown

MAX_BYTES = int(os.environ.get("OCR_MAX_MB", "20")) * 1024 * 1024
MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "20"))
QUEUE_SIZE = int(os.environ.get("OCR_QUEUE_SIZE", "32"))


def pdf_pages(data, scale=2.0):
    """PDF 쪽을 이미지(BGR)로. 글자가 들어 있는 PDF도 같은 흐름으로 처리합니다 (표·유형 판별을 맞추려고)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    if len(pdf) > MAX_PAGES:
        raise HTTPException(413, f"PDF는 {MAX_PAGES}쪽까지 받습니다 ({len(pdf)}쪽)")
    return [np.array(pdf[i].render(scale=scale).to_pil().convert("RGB"))[:, :, ::-1].copy() for i in range(len(pdf))]


class Worker:
    def __init__(self):
        llm = os.environ.get("OCR_LLM_URL", "http://localhost:18001/v1")
        self.engine = OcrEngine(vl_url=os.environ.get("OCR_VL_URL", "http://localhost:18118/v1"), llm_url=llm or None,
                                llm_model=os.environ.get("OCR_LLM_MODEL", "Qwen/Qwen3.5-9B"),
                                device=os.environ.get("OCR_DEVICE", "gpu"))
        self.engine.process(np.full((64, 256, 3), 255, np.uint8), mode="fast")  # 모델 초기화를 시작할 때 끝냄
        self.queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.loop = asyncio.get_running_loop()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            job, fut = asyncio.run_coroutine_threadsafe(self.queue.get(), self.loop).result()
            try:
                res = self.handle(**job)
                self.loop.call_soon_threadsafe(fut.set_result, res)
            except Exception as e:
                self.loop.call_soon_threadsafe(fut.set_exception, e)

    def handle(self, data, filename, doc_type, mode, extract):
        start = time.perf_counter()
        images = pdf_pages(data) if data[:5] == b"%PDF-" else [read_image(data)]
        if images[0] is None:
            raise HTTPException(415, "이미지나 PDF가 아닙니다")
        pages = [self.engine.process(img, doc_type=doc_type, mode=mode, do_extract=extract) for img in images]
        return filename, pages, time.perf_counter() - start


@asynccontextmanager
async def lifespan(app):
    app.state.worker = Worker()
    yield


app = FastAPI(title="OCR Engine", lifespan=lifespan)


@app.post("/v1/ocr")
async def ocr(file: UploadFile = File(...), type: str | None = Form(None), mode: str = Form("accurate"),
              extract: bool = Form(True), format: str = Form("json"), lines: bool = Form(False)):
    if type is not None and type not in TYPES:
        raise HTTPException(422, f"type은 {', '.join(TYPES)} 중 하나")
    if mode not in ("accurate", "fast"):
        raise HTTPException(422, "mode는 accurate 또는 fast")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"파일은 {MAX_BYTES // 1024 // 1024}MB까지 받습니다")
    worker = app.state.worker
    fut = asyncio.get_running_loop().create_future()
    try:
        worker.queue.put_nowait(({"data": data, "filename": file.filename, "doc_type": type, "mode": mode,
                                  "extract": extract}, fut))
    except asyncio.QueueFull:
        raise HTTPException(503, "대기 중인 작업이 많습니다. 잠시 뒤 다시 보내 주세요", headers={"Retry-After": "5"})
    filename, pages, seconds = await fut
    if format == "markdown":
        return PlainTextResponse(to_markdown(filename, pages), media_type="text/markdown; charset=utf-8")
    return to_json(filename, pages, seconds, include_lines=lines)


@app.get("/health")
async def health():
    worker = getattr(app.state, "worker", None)
    return {"ready": worker is not None, "queued": worker.queue.qsize() if worker else 0}
