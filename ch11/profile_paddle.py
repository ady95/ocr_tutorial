"""11장 — PaddleOCR 파이프라인의 단계별 시간을 재고, 검출 입력 크기·인식 배치 크기에 따른 속도와 정확도를 비교합니다.

PaddleX의 벤치마크 기능(환경 변수 PADDLE_PDX_PIPELINE_BENCHMARK)으로 단계마다 걸린 시간을 모읍니다.
실행 예:
  python profile_paddle.py --dataset ko
  python profile_paddle.py --dataset aihub_page --limit-side-len 1280 --limit-type max
  python profile_paddle.py --dataset ko --device cpu
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["PADDLE_PDX_PIPELINE_BENCHMARK"] = "True"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "ch10"))
from bench import load_items  # noqa: E402

# 단계별 시간 표에서 볼 항목 (PaddleX가 붙인 이름)
STAGES = ["ReadImage", "DetResizeForTest", "NormalizeImage", "ToBatch", "RunnerInfer", "DBPostProcess",
          "OCRReisizeNormImg", "CTCLabelDecode", "TextDetRunnerPredictor.apply", "TextRecRunnerPredictor.apply"]


def print_stages(summary_csv):
    """summary.csv에서 단계별 평균 시간(장당 ms)을 꺼내 출력합니다."""
    import csv
    rows = {}
    with open(summary_csv, encoding="utf-8") as fp:
        for row in csv.reader(fp):
            if len(row) == 3 and row[1] in STAGES + ["_OCRPipeline.predict"]:
                rows[row[1]] = float(row[2])
    total = rows.get("_OCRPipeline.predict", 0)
    print(f"{'단계':32s} {'ms':>8s} {'비율':>6s}")
    for name in ["_OCRPipeline.predict"] + STAGES:
        if name in rows:
            print(f"{name:32s} {rows[name]:8.1f} {rows[name] / total * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ko", choices=["ko", "aihub_page", "aihub_word"])
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--limit-side-len", type=int, help="검출 모델 입력 크기 기준 (기본: 모델 설정)")
    ap.add_argument("--limit-type", choices=["min", "max"], help="min: 짧은 변을 키움, max: 긴 변을 줄임")
    ap.add_argument("--rec-batch", type=int, help="인식 모델에 한 번에 넣을 글자줄 수")
    ap.add_argument("--det", help="검출 모델 이름 (예: PP-OCRv5_mobile_det). 기본은 lang=korean이 고르는 PP-OCRv5_server_det")
    ap.add_argument("--hpi", action="store_true", help="고성능 추론: 환경에 맞는 추론 엔진(ONNX Runtime·TensorRT 등)을 자동 선택")
    ap.add_argument("--tensorrt", action="store_true", help="Paddle Inference의 TensorRT 하위 그래프 사용")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="TensorRT 사용 시 연산 정밀도")
    ap.add_argument("--name", help="결과 파일 이름 (기본: 설정으로 만듦)")
    args = ap.parse_args()

    from paddleocr import PaddleOCR
    from paddlex.inference.utils.benchmark import benchmark
    kwargs = dict(lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False, device=args.device)
    if args.limit_side_len:
        kwargs["text_det_limit_side_len"] = args.limit_side_len
    if args.limit_type:
        kwargs["text_det_limit_type"] = args.limit_type
    if args.rec_batch:
        kwargs["text_recognition_batch_size"] = args.rec_batch
    if args.det:  # 모델 이름을 직접 주면 lang 대신 검출·인식 모델을 모두 지정해야 함
        kwargs.pop("lang")
        kwargs.update(text_detection_model_name=args.det, text_recognition_model_name="korean_PP-OCRv5_mobile_rec")
    if args.hpi:
        kwargs["enable_hpi"] = True
    if args.tensorrt:
        kwargs.update(use_tensorrt=True, precision=args.precision)
    ocr = PaddleOCR(**kwargs)

    items = load_items(args.dataset)
    for _, path in items[:2]:  # 첫 실행(모델 적재·초기화)은 기록에서 뺌
        ocr.predict(path)
    benchmark.reset()

    saved, times = {}, []
    for sid, path in items:
        start = time.perf_counter()
        res = ocr.predict(path)[0]
        times.append(time.perf_counter() - start)
        saved[sid] = {"text": "\n".join(res["rec_texts"]), "tables": [], "time": times[-1], "error": False,
                      "lines": len(res["rec_texts"])}

    name = args.name or (f"paddle_{args.device}_side{args.limit_side_len or 'def'}{args.limit_type or ''}"
                         f"_rb{args.rec_batch or 'def'}" + ("_hpi" if args.hpi else "")
                         + (f"_trt{args.precision}" if args.tensorrt else "") + (f"_{args.det}" if args.det else ""))
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    stage_file = out / "stages" / f"{name}_{args.dataset}"
    benchmark.save_pipeline_data(str(stage_file))  # 폴더 안에 detail.csv(호출마다)·summary.csv(평균)가 생김
    print_stages(stage_file / "summary.csv")
    saved["_meta"] = {"engine": name, "dataset": args.dataset, "n": len(items), "time_median": statistics.median(times),
                      "time_mean": statistics.mean(times), "opts": kwargs}
    (out / f"{name}__{args.dataset}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name} {args.dataset}: {len(items)}장, 장당 평균 {statistics.mean(times) * 1000:.0f}ms, "
          f"중앙값 {statistics.median(times) * 1000:.0f}ms, 글자줄 평균 {statistics.mean(v['lines'] for k, v in saved.items() if k != '_meta'):.1f}개")


if __name__ == "__main__":
    main()
