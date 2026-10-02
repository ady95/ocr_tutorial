"""10장 — AI Hub의 실제 촬영·손글씨 데이터를 이 책의 평가셋 형식으로 바꿉니다.

AI Hub 데이터는 재배포할 수 없으므로 이 저장소에 들어 있지 않습니다. 각자 AI Hub에서 내려받은 뒤
이 스크립트로 변환하세요. 결과는 datasets/aihub-local/ 에 저장되고 git에는 올라가지 않습니다.

  금융업 특화 문서 OCR 데이터   https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=632  → photo_doc
  대용량 손글씨 OCR 데이터      https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=605  → handwriting
  야외 실제 촬영 한글 이미지    https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=105  → scene

실행 예:
  python aihub_prepare.py --photo "<금융 폴더>" --handwriting "<손글씨 폴더>" --scene "<야외 폴더>/2.Validation"

변환할 때 하는 일
  - 사진의 EXIF 회전 정보를 픽셀에 적용한 뒤 EXIF(촬영 기기, GPS 등)를 지우고 저장
  - 라벨에서 글자와 좌표만 남김 (작업자 나이·성별·번호 같은 메타데이터는 버림)
  - 읽을 수 없는 단어(식별 불가, xxx)는 ignore로 표시
  - 손글씨·간판은 단어 상자를 잘라 단어 평가셋(aihub-local/words/)도 만듦
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent.parent / "datasets" / "aihub-local"
SOURCES = {
    "photo_doc": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=632",
    "handwriting": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=605",
    "scene": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=105",
}


def load_json(path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    raise ValueError(f"JSON을 읽을 수 없음: {path}")


def box_poly(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def read_photo_doc(label):
    """금융: annotations[0].polygons[] = {text, type(0 식별불가, 1 인쇄, 2 손글씨), points}"""
    d = load_json(label)
    words = []
    for p in d["annotations"][0]["polygons"]:
        xs, ys = [q[0] for q in p["points"]], [q[1] for q in p["points"]]
        words.append({"text": p["text"], "poly": box_poly(min(xs), min(ys), max(xs), max(ys)),
                      "ignore": p["type"] == 0 or not p["text"], "kind": {1: "print", 2: "hand"}.get(p["type"])})
    return words


def read_handwriting(label):
    """손글씨: bbox[] = {data, x[4], y[4]} — 손글씨 칸만 라벨이 있음"""
    d = load_json(label)
    words = []
    for b in d["bbox"]:
        words.append({"text": b["data"], "poly": box_poly(min(b["x"]), min(b["y"]), max(b["x"]), max(b["y"])),
                      "ignore": b["data"].lower() == "xxx" or not b["data"], "kind": "hand"})
    return words


def read_scene(label):
    """야외: annotations[] = {text, bbox[x, y, 폭, 높이]} — 이미지당 최대 3단어, 나머지 글자는 xxx"""
    d = load_json(label)
    words = []
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        words.append({"text": a["text"], "poly": box_poly(x, y, x + w, y + h),
                      "ignore": a["text"].lower() == "xxx" or not a["text"], "kind": None})
    return words


READERS = {"photo_doc": read_photo_doc, "handwriting": read_handwriting, "scene": read_scene}


def pairs(root):
    """라벨링데이터의 json과 원천데이터의 같은 이름 이미지를 짝짓습니다."""
    images = {p.stem: p for p in (Path(root) / "원천데이터").rglob("*")
              if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    return [(lab, images[lab.stem]) for lab in sorted((Path(root) / "라벨링데이터").rglob("*.json"))
            if lab.stem in images]


def save_clean(src, dst):
    """EXIF 회전을 적용하고 메타데이터 없이 저장합니다 (라벨 좌표는 회전을 적용한 뒤 기준)."""
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if dst.suffix == ".jpg":
        im.save(dst, quality=95, subsampling=0)
    else:
        im.save(dst)
    return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", help="금융업 특화 문서 OCR 데이터 폴더 (라벨링데이터·원천데이터가 있는 곳)")
    ap.add_argument("--handwriting", help="대용량 손글씨 OCR 데이터 폴더")
    ap.add_argument("--scene", help="야외 실제 촬영 한글 이미지의 Training 또는 Validation 폴더")
    ap.add_argument("-n", type=int, nargs=3, default=[50, 50, 300], metavar=("PHOTO", "HAND", "SCENE"),
                    help="범주마다 무작위로 뽑을 장 수")
    ap.add_argument("--max-words", type=int, nargs=2, default=[1000, 1000], metavar=("HAND", "SCENE"),
                    help="단어 평가셋(words/)에 넣을 최대 단어 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "gt").mkdir(exist_ok=True)
    manifest = []
    for category, root, n in zip(READERS, (args.photo, args.handwriting, args.scene), args.n):
        if not root:
            continue
        items = pairs(root)
        random.Random(args.seed).shuffle(items)
        count = 0
        for label, image in items:
            words = READERS[category](label)
            if not any(not w["ignore"] for w in words):
                continue
            count += 1
            sid = f"{category}_{count:03d}"
            ext = ".png" if image.suffix.lower() == ".png" else ".jpg"
            width, height = save_clean(image, OUT / "images" / f"{sid}{ext}")
            gt = {"id": sid, "category": category, "source": SOURCES[category], "width": width, "height": height,
                  "text": " ".join(w["text"] for w in words if not w["ignore"]), "words": words}
            (OUT / "gt" / f"{sid}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
            manifest.append({"id": sid, "category": category, "image": f"images/{sid}{ext}", "gt": f"gt/{sid}.json",
                             "width": width, "height": height, "words": sum(not w["ignore"] for w in words)})
            if count == n:
                break
        print(f"{category}: {count}장 (후보 {len(items)}장)")
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as fp:
        for m in manifest:
            fp.write(json.dumps(m, ensure_ascii=False) + "\n")
    make_words(manifest, {"handwriting": args.max_words[0], "scene": args.max_words[1]}, args.seed)
    print(f"→ {OUT}")


def make_words(manifest, limits, seed):
    """손글씨·간판은 라벨이 없는 글자가 있어 쪽 전체를 채점할 수 없으므로, 단어 상자를 잘라 따로 평가셋을 만듭니다."""
    out = OUT / "words"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(exist_ok=True)
    rows = []
    for category, limit in limits.items():
        boxes = []
        for m in manifest:
            if m["category"] != category:
                continue
            gt = json.loads((OUT / m["gt"]).read_text(encoding="utf-8"))
            boxes += [(m, w) for w in gt["words"] if not w["ignore"]]
        random.Random(seed).shuffle(boxes)
        for i, (m, w) in enumerate(boxes[:limit], 1):
            wid = f"{category}_word_{i:04d}"
            x0, y0 = w["poly"][0]
            x1, y1 = w["poly"][2]
            pad = max(4, int((y1 - y0) * 0.15))  # 글자가 상자 경계에 붙어 있어 조금 넓혀 자름
            with Image.open(OUT / m["image"]) as im:
                crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))
            crop.save(out / "images" / f"{wid}.png")
            (out / "gt" / f"{wid}.json").write_text(json.dumps(
                {"id": wid, "category": f"{category}_word", "source": SOURCES[category], "parent": m["id"],
                 "text": w["text"]}, ensure_ascii=False), encoding="utf-8")
            rows.append({"id": wid, "category": f"{category}_word", "image": f"images/{wid}.png", "gt": f"gt/{wid}.json",
                         "width": crop.width, "height": crop.height})
        print(f"{category}_word: {min(limit, len(boxes))}단어 (후보 {len(boxes)}단어)")
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
