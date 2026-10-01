"""평가셋 렌더링에 쓰는 한글 글꼴(OFL)을 Google Fonts 저장소에서 내려받는다.

글꼴 파일은 용량이 커서 저장소에 넣지 않는다. render.py 실행 전에 한 번 실행한다.
실행: python fetch_fonts.py
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
FONTS = {
    "NotoSansKR.ttf": "notosanskr/NotoSansKR%5Bwght%5D.ttf",
    "NotoSerifKR.ttf": "notoserifkr/NotoSerifKR%5Bwght%5D.ttf",
    "NanumGothic-Regular.ttf": "nanumgothic/NanumGothic-Regular.ttf",
    "NanumGothic-Bold.ttf": "nanumgothic/NanumGothic-Bold.ttf",
    "NanumMyeongjo-Regular.ttf": "nanummyeongjo/NanumMyeongjo-Regular.ttf",
    "GowunDodum-Regular.ttf": "gowundodum/GowunDodum-Regular.ttf",
    "IBMPlexSansKR-Regular.ttf": "ibmplexsanskr/IBMPlexSansKR-Regular.ttf",
    "NanumGothicCoding-Regular.ttf": "nanumgothiccoding/NanumGothicCoding-Regular.ttf",
    "NanumPenScript-Regular.ttf": "nanumpenscript/NanumPenScript-Regular.ttf",
    "DoHyeon-Regular.ttf": "dohyeon/DoHyeon-Regular.ttf",
}


def main():
    out = Path(__file__).resolve().parent.parent / "fonts"
    out.mkdir(exist_ok=True)
    for name, path in FONTS.items():
        dst = out / name
        if dst.exists():
            print("있음:", name)
            continue
        urllib.request.urlretrieve(BASE + path, dst)
        print("받음:", name, dst.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
