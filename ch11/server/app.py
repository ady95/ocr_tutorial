"""11장 — FastAPI OCR 서버.

  POST /ocr          이미지·PDF를 올리면 OCR 결과 JSON을 돌려줌 (끝날 때까지 기다림)
  POST /jobs         올리기만 하고 작업 번호를 바로 돌려줌 → GET /jobs/{id}로 상태와 결과 확인
  GET  /health       모델 준비 여부, 대기 중인 작업 수

OCR 모델은 프로세스에 하나만 올리고, 모든 요청을 큐 하나에 넣어 작업자 스레드 하나가 차례로 처리합니다.
모델 하나를 여러 스레드가 동시에 쓰지 않게 하고, 큐가 가득 차면 503을 돌려 서버가 감당할 만큼만 받습니다.

환경 변수
  OCR_DEVICE      cpu 또는 gpu (기본 cpu)
  OCR_DET_MODEL   검출 모델 (기본 PP-OCRv5_server_det). CPU에서는 PP-OCRv5_mobile_det가 정확도는 같고 1.9배 빠름 (11-1)
  OCR_MAX_MB      업로드 최대 크기 MB (기본 20)
  OCR_MAX_PAGES   PDF 최대 쪽 수 (기본 20)
  OCR_QUEUE_SIZE  큐에 쌓아 둘 수 있는 작업 수 (기본 32)
실행: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

DEVICE = os.environ.get("OCR_DEVICE", "cpu")
DET_MODEL = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_server_det")
MAX_BYTES = int(os.environ.get("OCR_MAX_MB", "20")) * 1024 * 1024
MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "20"))
QUEUE_SIZE = int(os.environ.get("OCR_QUEUE_SIZE", "32"))
MIN_CHARS = 20  # PDF 쪽에 이보다 글자가 적으면 스캔 쪽으로 보고 OCR (07-3)
JOB_TTL = 3600  # 끝난 작업 결과를 보관하는 시간(초)


class OcrWorker:
    """큐에서 작업을 하나씩 꺼내 OCR하는 작업자 스레드."""

    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(text_detection_model_name=DET_MODEL, text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                             device=DEVICE, use_doc_orientation_classify=False, use_doc_unwarping=False,
                             use_textline_orientation=False)
        self.ocr.predict(np.full((64, 256, 3), 255, np.uint8))  # 첫 실행(모델 초기화)을 서버 시작 때 끝냄
        self.queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.loop = asyncio.get_running_loop()

    def ocr_image(self, image):
        res = self.ocr.predict(np.array(image.convert("RGB"))[:, :, ::-1].copy())[0]  # PaddleOCR은 BGR
        lines = [{"text": t, "score": round(float(s), 4), "box": [int(v) for v in b]}
                 for t, s, b in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"])]
        return {"text": "\n".join(res["rec_texts"]), "lines": lines}

    def process(self, filename, data):
        start = time.perf_counter()
        if data[:5] == b"%PDF-":
            pages = self.process_pdf(data)
        else:
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
            except Exception:
                raise ValueError("이미지나 PDF가 아닙니다")
            pages = [{"page": 1, "source": "ocr", **self.ocr_image(image)}]
        return {"filename": filename, "pages": pages, "elapsed_ms": round((time.perf_counter() - start) * 1000)}

    def process_pdf(self, data):
        import pypdfium2 as pdfium
        try:
            pdf = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as e:  # 손상된 PDF는 서버 오류(500)가 아니라 입력 오류(400)로
            raise ValueError(f"PDF를 열 수 없습니다: {e}")
        if len(pdf) == 0:
            raise ValueError("쪽이 없는 PDF입니다")
        if len(pdf) > MAX_PAGES:
            raise ValueError(f"PDF는 {MAX_PAGES}쪽까지 처리합니다 (받은 파일 {len(pdf)}쪽)")
        pages = []
        for i, page in enumerate(pdf, 1):
            text = page.get_textpage().get_text_range().replace("\r\n", "\n").strip()
            if len(text) >= MIN_CHARS:  # 텍스트 PDF는 OCR하지 않고 글자 정보를 그대로 씀 (07-3)
                pages.append({"page": i, "source": "text", "text": text, "lines": []})
            else:
                image = page.render(scale=2).to_pil()
                pages.append({"page": i, "source": "ocr", **self.ocr_image(image)})
        return pages

    async def submit(self, filename, data):
        """작업을 큐에 넣고 결과를 기다릴 Future를 돌려줍니다. 큐가 가득 차면 QueueFull."""
        future = self.loop.create_future()
        self.queue.put_nowait((filename, data, future, time.perf_counter()))
        return future

    def run_forever(self):
        while True:
            filename, data, future, queued = asyncio.run_coroutine_threadsafe(self.queue.get(), self.loop).result()
            wait_ms = round((time.perf_counter() - queued) * 1000)
            try:
                result = {**self.process(filename, data), "queue_wait_ms": wait_ms}
                self.loop.call_soon_threadsafe(future.set_result, result)
            except Exception as e:  # 한 작업의 실패가 작업자를 멈추지 않게 함
                self.loop.call_soon_threadsafe(future.set_exception, e)


worker = None
jobs = {}  # 작업 번호 → {"status", "result", "error", "created"}. 여러 서버로 늘리려면 Redis 같은 저장소로 바꿈


@asynccontextmanager
async def lifespan(app):
    global worker
    worker = OcrWorker()
    threading.Thread(target=worker.run_forever, daemon=True).start()
    yield


app = FastAPI(title="OCR 따라하기 OCR 서버", lifespan=lifespan)


async def read_upload(file):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"파일이 {MAX_BYTES // 1024 // 1024}MB를 넘습니다")
    if not data:
        raise HTTPException(400, "빈 파일입니다")
    return data


async def enqueue(file):
    data = await read_upload(file)
    try:
        return await worker.submit(file.filename, data)
    except asyncio.QueueFull:
        raise HTTPException(503, "처리할 작업이 밀려 있습니다. 잠시 후 다시 보내 주세요", headers={"Retry-After": "5"})


@app.get("/health")
async def health():
    return {"status": "ok" if worker else "loading", "device": DEVICE, "det_model": DET_MODEL,
            "queued": worker.queue.qsize() if worker else 0}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    future = await enqueue(file)
    try:
        return await future
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile = File(...)):
    now = time.time()
    for k in [k for k, v in jobs.items() if now - v["created"] > JOB_TTL]:  # 오래된 결과 정리
        del jobs[k]
    future = await enqueue(file)
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "result": None, "error": None, "created": now}

    def done(f):
        job = jobs.get(job_id)
        if job is None:
            return
        if f.exception():
            job.update(status="failed", error=str(f.exception()))
        else:
            job.update(status="done", result=f.result())

    future.add_done_callback(done)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "없는 작업 번호입니다")
    return {"job_id": job_id, **{k: v for k, v in job.items() if k != "created"}}
