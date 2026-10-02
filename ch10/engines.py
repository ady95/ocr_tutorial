"""10장 — OCR 엔진 어댑터. 엔진마다 "이미지 경로 → (글자, 표 HTML 목록)" 하나만 맡습니다.

엔진은 서로 다른 가상환경에서 실행되므로, 필요한 라이브러리는 엔진을 만들 때 불러옵니다.
설정은 앞 장에서 가장 좋았던 값을 그대로 씁니다.

  tesseract   04장  kor+eng, PSM 4
  paddle      04장  PaddleOCR lang="korean" (PP-OCRv5 검출 + 한국어 인식)
  paddlevl    08장  PaddleOCR-VL-1.6 (vLLM 서버)
  surya       08장  Surya OCR 2
  deepseek    08장  DeepSeek-OCR 2, transformers 공식 방법 + markdown 프롬프트
  qwen·varco  08장 vlm_bench.py의 OpenAI 호환 API 호출 (vLLM 서버). Qwen은 생각 모드 끔 + presence_penalty 1.5
  glm         GLM-OCR, 공식 프롬프트 "Text Recognition:"로 쪽 전체를 한 번에 (vLLM 서버)
  glmsdk      GLM-OCR 공식 SDK(glmocr): 레이아웃 분석 뒤 영역별 인식 (vLLM 서버, mode="selfhosted")
  mineru      09장  MinerU standard, middle_json (머리말·꼬리말 포함)
  docling     09장  Docling 표준 파이프라인 + EasyOCR(ko·en, 신뢰도 기준 0.2)
"""
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("common", "ch08", "ch09"):
    sys.path.insert(0, str(HERE.parent / sub))


class Engine:
    batch = False  # True면 run_batch로 여러 장을 한 번에 처리 (매번 서버를 띄우는 도구)

    def run(self, path):
        raise NotImplementedError


class Tesseract(Engine):
    def __init__(self, lang="kor+eng", psm=4):
        import pytesseract
        self.tess, self.lang, self.config = pytesseract, lang, f"--oem 1 --psm {psm}"

    def run(self, path):
        from PIL import Image
        return self.tess.image_to_string(Image.open(path), lang=self.lang, config=self.config), []


class Paddle(Engine):
    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False,
                             use_textline_orientation=False)

    def run(self, path):
        return "\n".join(self.ocr.predict(path)[0]["rec_texts"]), []


class PaddleVL(Engine):
    def __init__(self, server="http://localhost:8118/v1", layout=1):
        from paddleocr import PaddleOCRVL
        self.pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=server)
        self.layout = bool(layout)  # 0이면 레이아웃 검출 없이 이미지 전체를 글자 블록 하나로 읽음 (단어 이미지용)

    def run(self, path):
        from paddlevl_bench import page_text
        if self.layout:
            return page_text(self.pipeline.predict(path)[0])
        res = self.pipeline.predict(path, use_layout_detection=False, prompt_label="ocr")[0]
        return "\n".join(b.content or "" for b in res["parsing_res_list"]), []


class Surya(Engine):
    def __init__(self):
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        self.rec = RecognitionPredictor(SuryaInferenceManager())

    def run(self, path):
        from PIL import Image
        from surya_bench import block_text
        page = self.rec([Image.open(path).convert("RGB")])[0]
        blocks = sorted(page.blocks, key=lambda b: b.reading_order)
        text = "\n".join(t for t in (block_text(b.html) for b in blocks) if t)
        return text, [b.html for b in blocks if "<table" in (b.html or "")]


class DeepSeek(Engine):
    def __init__(self, prompt="<image>\n<|grounding|>Convert the document to markdown. "):
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = "deepseek-ai/DeepSeek-OCR-2"
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, _attn_implementation="flash_attention_2", trust_remote_code=True,
                                               use_safetensors=True).eval().cuda().to(torch.bfloat16)
        self.prompt, self.tmp = prompt, tempfile.mkdtemp()

    def run(self, path):
        from vlm_bench import to_text
        raw = self.model.infer(self.tokenizer, prompt=self.prompt, image_file=str(path), output_path=self.tmp,
                               base_size=1024, image_size=768, crop_mode=True, save_results=False, eval_mode=True)
        return to_text("deepseek", raw)


class ApiVLM(Engine):
    """vLLM 등 OpenAI 호환 서버. 08장 vlm_bench.py의 요청·파싱 함수를 그대로 씁니다."""

    def __init__(self, kind, model, base_url="http://localhost:8000/v1", max_tokens=4096, prompt="html",
                 presence_penalty=0.0, no_think=True):
        from openai import OpenAI
        from vlm_bench import PROMPTS
        PROMPTS.setdefault(prompt, prompt)  # 08장에 없는 프롬프트(GLM-OCR의 "Text Recognition:" 등)는 그대로 씀
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
        self.args = Namespace(kind=kind, model=model, max_tokens=max_tokens, prompt=prompt,
                              presence_penalty=presence_penalty, no_think=no_think, ds_prompt="free")

    def run(self, path):
        from vlm_bench import run_one, to_text
        raw = run_one(self.client, self.args, path)[0]
        return to_text(self.args.kind, raw)


class GlmSdk(Engine):
    """GLM-OCR 공식 SDK(glmocr): 레이아웃 분석(PP-DocLayout-V3) 뒤 영역마다 GLM-OCR로 인식하는 2단계 방식."""

    def __init__(self, base_url="http://localhost:8000/v1", model="zai-org/GLM-OCR"):
        from urllib.parse import urlparse
        from glmocr import GlmOcr
        url = urlparse(base_url)
        # 설정 파일의 기본값은 클라우드(MaaS) 모드라, mode를 지정하지 않으면 문서가 외부 API로 전송될 수 있음
        self.parser = GlmOcr(mode="selfhosted", ocr_api_host=url.hostname, ocr_api_port=url.port, model=model,
                             layout_device="cuda")

    def run(self, path):
        from mdtext import markdown_to_text
        return markdown_to_text(self.parser.parse(str(path)).markdown_result or "", md_tables=True)


class MinerU(Engine):
    batch = True

    def __init__(self, tier="standard"):
        self.tier = tier

    def run_batch(self, paths):
        """폴더째 넘겨야 VLM 서버를 한 번만 띄웁니다. 결과는 이미지 id → (글자, 표)."""
        import json
        import shutil
        import subprocess
        from mdtext import markdown_to_text
        from score_parsers import mineru_json_to_md
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in", Path(tmp) / "out"
            src.mkdir()
            for p in paths:
                shutil.copy(p, src / Path(p).name)
            subprocess.run(["mineru-kit", "parse", str(src), "-o", str(out), "-f", "middle_json", "--tier", self.tier],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            texts = {}
            for p in paths:
                js = out / f"{Path(p).stem}.json"
                md = mineru_json_to_md(json.loads(js.read_text(encoding="utf-8"))) if js.exists() else ""
                texts[Path(p).stem] = markdown_to_text(md, md_tables=True)
            return texts


class Docling(Engine):
    def __init__(self, threshold=0.2):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption
        ocr = EasyOcrOptions(lang=["ko", "en"], confidence_threshold=threshold)
        options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr)
        self.converter = DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)})

    def run(self, path):
        from mdtext import markdown_to_text
        md = self.converter.convert(str(path)).document.export_to_markdown()
        return markdown_to_text(md, md_tables=True)


ENGINES = {
    "tesseract": Tesseract,
    "paddle": Paddle,
    "paddlevl": PaddleVL,
    "surya": Surya,
    "deepseek": DeepSeek,
    "qwen": lambda **kw: ApiVLM("qwen", kw.pop("model", "Qwen/Qwen3.5-4B"), **{"presence_penalty": 1.5, **kw}),
    "glm": lambda **kw: ApiVLM("qwen", kw.pop("model", "zai-org/GLM-OCR"),
                               **{"prompt": "Text Recognition:", "no_think": False, **kw}),
    # VARCO는 어절마다 좌표를 붙여 출력이 이미지당 4,000토큰을 넘으므로 한도를 8,192로 (08장과 같음)
    "glmsdk": GlmSdk,
    "varco": lambda **kw: ApiVLM("varco", kw.pop("model", "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"), **{"max_tokens": 8192, **kw}),
    "mineru": MinerU,
    "docling": Docling,
}
