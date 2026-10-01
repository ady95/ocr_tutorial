"""한국어 OCR 평가셋 — 렌더링 (원문 → 이미지 + 정답).

HTML로 문서를 그린 뒤 Chromium으로 캡처한다. 어절마다 <span>을 씌워 두고
브라우저가 줄바꿈한 결과를 읽어 오므로, 줄 단위 텍스트와 좌표가 정확하게 나온다.

실행: python render.py            (playwright, pillow 필요)
출력: ../images/<id>.png, ../gt/<id>.json
"""
import html
import json
import random
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CONTENT, FONTS = ROOT / "content", ROOT / "fonts"
IMAGES, GT, TMP = ROOT / "images", ROOT / "gt", ROOT / "build" / "_tmp"

FONT_FILES = {
    "NotoSansKR": "NotoSansKR.ttf", "NotoSerifKR": "NotoSerifKR.ttf",
    "NanumGothic": "NanumGothic-Regular.ttf", "NanumMyeongjo": "NanumMyeongjo-Regular.ttf",
    "GowunDodum": "GowunDodum-Regular.ttf", "IBMPlexSansKR": "IBMPlexSansKR-Regular.ttf",
    "NanumGothicCoding": "NanumGothicCoding-Regular.ttf", "NanumPenScript": "NanumPenScript-Regular.ttf",
    "DoHyeon": "DoHyeon-Regular.ttf",
}
DOC_FONTS = ["NotoSansKR", "NanumMyeongjo", "NanumGothic", "NotoSerifKR", "GowunDodum", "IBMPlexSansKR"]

# 브라우저 안에서 줄 단위 정답을 모으는 스크립트.
# .blk 요소마다 어절 span을 top 좌표로 묶어 줄을 만든다 (DOM 순서 = 읽기 순서).
COLLECT_JS = """
() => {
  const doc = document.querySelector('#doc').getBoundingClientRect();
  const lines = [];
  for (const blk of document.querySelectorAll('.blk')) {
    let cur = null;
    for (const w of blk.querySelectorAll('span.w')) {
      const r = w.getBoundingClientRect();
      if (cur && Math.abs(r.top - cur.top) < r.height * 0.5) {
        cur.words.push(w.textContent);
        cur.x2 = Math.max(cur.x2, r.right); cur.y2 = Math.max(cur.y2, r.bottom);
        cur.y1 = Math.min(cur.y1, r.top);
      } else {
        if (cur) lines.push(cur);
        cur = {top: r.top, words: [w.textContent], x1: r.left, y1: r.top, x2: r.right, y2: r.bottom,
               block: blk.dataset.role || 'text'};
      }
    }
    if (cur) lines.push(cur);
  }
  return {width: Math.ceil(doc.width), height: Math.ceil(doc.height),
          lines: lines.map(l => ({text: l.words.join(' '), block: l.block,
                                  box: [l.x1 - doc.left, l.y1 - doc.top, l.x2 - doc.left, l.y2 - doc.top]}))};
}
"""


def words(text):
    """어절마다 span을 씌운다. 어절 사이 공백은 일반 공백으로 둔다."""
    text = text.replace("`", "")  # 원문에 섞인 Markdown 백틱 제거
    return " ".join(f'<span class="w">{html.escape(w)}</span>' for w in text.split())


def blk(tag, text, role, cls=""):
    c = f' class="blk {cls}"' if cls else ' class="blk"'
    return f'<{tag}{c} data-role="{role}">{words(text)}</{tag}>'


def page(body, font, size=20, width=1100, extra_css=""):
    faces = "\n".join(
        f"@font-face {{ font-family: '{name}'; src: url('{(FONTS / f).as_uri()}'); }}"
        for name, f in FONT_FILES.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{faces}
body {{ margin: 0; background: #fff; }}
#doc {{ width: {width}px; padding: 56px 64px; box-sizing: border-box; background: #fff; color: #111;
        font-family: '{font}'; font-size: {size}px; line-height: 1.7; word-break: keep-all; }}
h1 {{ font-size: 1.55em; margin: 0 0 0.8em; line-height: 1.35; }}
h2 {{ font-size: 1.15em; margin: 1.1em 0 0.4em; }}
p {{ margin: 0 0 0.9em; }}
span.w {{ white-space: nowrap; }}
{extra_css}
</style></head><body><div id="doc">{body}</div></body></html>"""


def tbox_to_poly(b):
    x1, y1, x2, y2 = (round(v, 1) for v in b)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class Renderer:
    def __init__(self, pw):
        self.browser = pw.chromium.launch()
        self.page = self.browser.new_page(viewport={"width": 1400, "height": 1000})

    def render(self, sample_id, html_text):
        TMP.mkdir(parents=True, exist_ok=True)
        f = TMP / f"{sample_id}.html"
        f.write_text(html_text, encoding="utf-8")
        self.page.goto(f.as_uri())
        self.page.evaluate("document.fonts.ready")
        info = self.page.evaluate(COLLECT_JS)
        self.page.locator("#doc").screenshot(path=str(IMAGES / f"{sample_id}.png"))
        return info


def save_gt(sample_id, category, info, font, extra=None):
    lines = [{"text": l["text"], "block": l["block"], "poly": tbox_to_poly(l["box"])} for l in info["lines"]]
    gt = {
        "id": sample_id, "category": category, "source": "rendered", "font": font,
        "width": info["width"], "height": info["height"],
        "text": "\n".join(l["text"] for l in lines), "lines": lines,
    }
    gt.update(extra or {})
    (GT / f"{sample_id}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    return gt


def load(kind):
    return json.loads((CONTENT / f"{kind}.json").read_text(encoding="utf-8"))


# 문서형 ---------------------------------------------------------------------

def render_docs(r):
    out = []
    for kind, category in [("doc_ko", "print_ko"), ("doc_mixed", "print_mixed"), ("doc_numeric", "print_numeric")]:
        for i, item in enumerate(load(kind)):
            font = DOC_FONTS[i % len(DOC_FONTS)]
            body = blk("h1", item["title"], "title") + "".join(blk("p", p, "text") for p in item["paragraphs"])
            info = r.render(item["id"], page(body, font))
            out.append(save_gt(item["id"], category, info, font))
    return out


def render_small(r):
    out = []
    for i, item in enumerate(load("small")):
        font = DOC_FONTS[i % len(DOC_FONTS)]
        size = [10, 11, 12][i % 3]
        body = blk("h2", item["title"], "title") + "".join(blk("p", p, "text") for p in item["paragraphs"])
        info = r.render(item["id"], page(body, font, size=size, width=640,
                                         extra_css="#doc { padding: 24px 28px; line-height: 1.5; }"))
        out.append(save_gt(item["id"], "small", info, font, {"font_px": size}))
    return out


def render_multicol(r):
    out = []
    css = ".cols { column-count: 2; column-gap: 48px; } .cols h2 { break-after: avoid; } .cols p { text-align: justify; }"
    for i, item in enumerate(load("multicol")):
        font = DOC_FONTS[i % len(DOC_FONTS)]
        arts = "".join(blk("h2", a["heading"], "heading") + "".join(blk("p", p, "text") for p in a["paragraphs"])
                       for a in item["articles"])
        body = blk("h1", item["title"], "title") + f'<div class="cols">{arts}</div>'
        info = r.render(item["id"], page(body, font, size=18, width=1200, extra_css=css))
        out.append(save_gt(item["id"], "multicol", info, font, {"reading_order": "column"}))
    return out


def table_spans(rows):
    """첫 열에서 같은 값이 연속되면 rowspan으로 병합한다. (값, rowspan) 또는 None(병합되어 생략)."""
    spans = [[(c, 1) for c in row] for row in rows]
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        if j > i:
            spans[i][0] = (rows[i][0], j - i + 1)
            for k in range(i + 1, j + 1):
                spans[k][0] = None
        i = j + 1
    return spans


def render_tables(r):
    out = []
    css = ("table { border-collapse: collapse; width: 100%; font-size: 0.9em; } "
           "th, td { border: 1px solid #333; padding: 8px 10px; text-align: center; } th { background: #e8e8e8; }")
    for i, item in enumerate(load("table")):
        font = DOC_FONTS[i % len(DOC_FONTS)]
        merge = i % 2 == 0           # 절반은 병합 셀 포함
        spans = table_spans(item["rows"]) if merge else [[(c, 1) for c in row] for row in item["rows"]]
        head = "".join(f'<th class="blk" data-role="cell">{words(str(h))}</th>' for h in item["header"])
        gt_rows = ["<tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in item["header"]) + "</tr>"]
        body_rows = []
        for row in spans:
            cells, gcells = "", ""
            for cell in row:
                if cell is None:
                    continue
                val, rs = cell
                attr = f' rowspan="{rs}"' if rs > 1 else ""
                cells += f'<td class="blk" data-role="cell"{attr}>{words(str(val))}</td>'
                gcells += f"<td{attr}>{html.escape(str(val))}</td>"
            body_rows.append(f"<tr>{cells}</tr>")
            gt_rows.append(f"<tr>{gcells}</tr>")
        body = blk("h1", item["title"], "title") + f"<table><tr>{head}</tr>{''.join(body_rows)}</table>"
        info = r.render(item["id"], page(body, font, size=19, width=1100, extra_css=css))
        out.append(save_gt(item["id"], "table", info, font,
                           {"table_html": "<table>" + "".join(gt_rows) + "</table>", "merged_cells": merge}))
    return out


# 서식형: 가상 데이터로 생성 ------------------------------------------------------

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류홍")
GIVEN = list("민서지윤도현수아준우하은예린시온유진태호채원")
STORE_A = ["푸른", "한빛", "새솔", "다온", "하늘", "은하", "바른", "소담", "늘봄", "해들"]
STORE_B = ["마트", "식당", "카페", "베이커리", "문구", "분식", "정육점", "과일가게"]
CITY = ["새빛시 한마음구", "가온시 누리구", "다솜시 해솔구", "미르시 별빛구"]
ITEMS = [("유기농 우유 900ml", 3200), ("식빵", 2800), ("계란 10구", 4500), ("아메리카노", 4000),
         ("카페라떼", 4500), ("김밥", 3500), ("라면", 1200), ("생수 2L", 1100), ("사과 1kg", 8900),
         ("볼펜 3자루", 2400), ("공책 A4", 1800), ("돼지고기 앞다리 500g", 9800), ("두부", 1900),
         ("바나나", 3900), ("치즈 슬라이스", 5200), ("떡볶이", 4000), ("크루아상", 3300), ("샐러드", 6500)]
COMPANY = ["가상테크", "누리소프트", "한빛데이터", "새솔물산", "다온디자인", "미르건설", "은하에너지", "바른컨설팅"]
TITLES = [("개발팀", "선임연구원"), ("영업본부", "과장"), ("디자인팀", "책임"), ("경영지원팀", "대리"),
          ("품질관리팀", "팀장"), ("마케팅팀", "주임")]


def biz_number(rng):
    """체크섬이 맞는 가상의 사업자등록번호 (06-1 검증 실습용)."""
    d = [rng.randint(0, 9) for _ in range(9)]
    w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    s = sum(a * b for a, b in zip(d, w)) + (d[8] * 5) // 10
    d.append((10 - s % 10) % 10)
    t = "".join(map(str, d))
    return f"{t[:3]}-{t[3:5]}-{t[5:]}"


def person(rng):
    return rng.choice(SURNAMES) + rng.choice(GIVEN) + rng.choice(GIVEN)


def phone(rng, mobile=False):
    # 국번 000/0000은 실제로 쓰이지 않는 번호대다
    return f"010-0000-{rng.randint(1000, 9999)}" if mobile else f"02-000-{rng.randint(1000, 9999)}"


def render_receipts(r, n=10):
    out = []
    css = ("#doc { font-size: 17px; line-height: 1.55; } .c { text-align: center; } .row { display: flex; "
           "justify-content: space-between; } .hr { border-top: 1px dashed #333; margin: 8px 0; }")
    for i in range(n):
        rng = random.Random(1000 + i)
        sid = f"receipt_{i + 1:02d}"
        store = rng.choice(STORE_A) + rng.choice(STORE_B)
        picked = rng.sample(ITEMS, rng.randint(3, 6))
        qty = [rng.randint(1, 3) for _ in picked]
        total = sum(p * q for (_, p), q in zip(picked, qty))
        date = f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d} {rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}"
        fields = {"store": store, "biz_no": biz_number(rng), "tel": phone(rng), "date": date,
                  "items": [{"name": n_, "qty": q, "amount": p * q} for (n_, p), q in zip(picked, qty)],
                  "total": total, "vat": round(total / 11), "address": f"{rng.choice(CITY)} 예시로 {rng.randint(1, 300)}"}
        rows = ""
        for it in fields["items"]:
            left = words(f"{it['name']} x{it['qty']}")
            right = words(f"{it['amount']:,}")
            rows += f'<div class="row blk" data-role="item"><span>{left}</span><span>{right}</span></div>'
        vat = fields["vat"]
        body = (blk("div", store, "title", "c") + blk("div", f"사업자 {fields['biz_no']}", "text", "c")
                + blk("div", fields["address"], "text", "c") + blk("div", f"TEL {fields['tel']}", "text", "c")
                + '<div class="hr"></div>' + blk("div", f"거래일시 {date}", "text") + '<div class="hr"></div>'
                + rows + '<div class="hr"></div>'
                + f'<div class="row blk" data-role="total"><span>{words("합계")}</span><span>{words(f"{total:,}원")}</span></div>'
                + f'<div class="row blk" data-role="text"><span>{words("부가세")}</span><span>{words(f"{vat:,}원")}</span></div>'
                + '<div class="hr"></div>' + blk("div", "감사합니다", "text", "c"))
        font = ["NanumGothicCoding", "NanumGothic", "NotoSansKR"][i % 3]
        info = r.render(sid, page(body, font, width=460, extra_css=css + " #doc { padding: 28px 30px; }"))
        out.append(save_gt(sid, "receipt", info, font, {"fields": fields}))
    return out


def render_invoices(r, n=10):
    out = []
    css = ("table { border-collapse: collapse; width: 100%; font-size: 0.85em; } "
           "th, td { border: 1px solid #c33; padding: 6px 8px; } th { background: #fbeaea; color: #a22; }"
           " h1 { text-align: center; color: #a22; }")
    for i in range(n):
        rng = random.Random(2000 + i)
        sid = f"invoice_{i + 1:02d}"
        sup, buy = rng.sample(COMPANY, 2)
        picked = rng.sample(ITEMS, rng.randint(2, 4))
        lines = []
        for name, price in picked:
            q = rng.randint(10, 200)
            supply = price * q
            lines.append({"name": name, "qty": q, "price": price, "supply": supply, "tax": supply // 10})
        fields = {
            "supplier": {"biz_no": biz_number(rng), "name": sup, "ceo": person(rng)},
            "buyer": {"biz_no": biz_number(rng), "name": buy, "ceo": person(rng)},
            "date": f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
            "items": lines,
            "supply_total": sum(x["supply"] for x in lines), "tax_total": sum(x["tax"] for x in lines),
        }
        fields["total"] = fields["supply_total"] + fields["tax_total"]

        def td(v, role="cell", attr=""):
            return f'<td class="blk" data-role="{role}"{attr}>{words(str(v))}</td>'

        def th(v, attr=""):
            return f'<th class="blk" data-role="cell"{attr}>{words(str(v))}</th>'

        s, b = fields["supplier"], fields["buyer"]
        party = (f"<table><tr>{th('공급자', ' colspan=2')}{th('공급받는자', ' colspan=2')}</tr>"
                 f"<tr>{th('등록번호')}{td(s['biz_no'])}{th('등록번호')}{td(b['biz_no'])}</tr>"
                 f"<tr>{th('상호')}{td(s['name'])}{th('상호')}{td(b['name'])}</tr>"
                 f"<tr>{th('대표자')}{td(s['ceo'])}{th('대표자')}{td(b['ceo'])}</tr></table>")
        items = (f"<table style='margin-top:14px'><tr>{th('품목')}{th('수량')}{th('단가')}{th('공급가액')}{th('세액')}</tr>"
                 + "".join("<tr>" + td(x["name"]) + td(x["qty"]) + td(f"{x['price']:,}") + td(f"{x['supply']:,}")
                           + td(f"{x['tax']:,}") + "</tr>" for x in lines)
                 + "<tr>" + th("합계", " colspan=3") + td(f"{fields['supply_total']:,}")
                 + td(f"{fields['tax_total']:,}") + "</tr>"
                 + "</table>")
        body = (blk("h1", "세금계산서", "title") + blk("p", f"작성일자 {fields['date']}", "text")
                + party + items + blk("p", f"합계금액 {fields['total']:,}원", "total"))
        font = DOC_FONTS[i % len(DOC_FONTS)]
        info = r.render(sid, page(body, font, size=18, width=1000, extra_css=css))
        out.append(save_gt(sid, "invoice", info, font, {"fields": fields}))
    return out


def render_cards(r, n=10):
    out = []
    css = ("#doc { height: 500px; padding: 48px 56px; line-height: 1.5; position: relative; } "
           ".co { font-size: 1.1em; color: #2a5; } .nm { font-size: 1.9em; font-weight: 700; margin-top: 70px; }"
           " .bt { position: absolute; left: 56px; bottom: 44px; font-size: 0.85em; }")
    for i in range(n):
        rng = random.Random(3000 + i)
        sid = f"card_{i + 1:02d}"
        dept, title = rng.choice(TITLES)
        name = person(rng)
        fields = {"company": rng.choice(COMPANY), "name": name, "dept": dept, "title": title,
                  "tel": phone(rng), "mobile": phone(rng, mobile=True),
                  "email": f"user{rng.randint(10, 99)}@example.com",
                  "address": f"{rng.choice(CITY)} 예시로 {rng.randint(1, 300)}, {rng.randint(2, 15)}층"}
        body = (blk("div", fields["company"], "title", "co") + blk("div", name, "name", "nm")
                + blk("div", f"{dept} {title}", "text")
                + '<div class="bt">' + blk("div", f"T {fields['tel']}  M {fields['mobile']}", "text")
                + blk("div", f"E {fields['email']}", "text") + blk("div", fields["address"], "text") + "</div>")
        font = DOC_FONTS[i % len(DOC_FONTS)]
        info = r.render(sid, page(body, font, size=19, width=900, extra_css=css))
        out.append(save_gt(sid, "card", info, font, {"fields": fields}))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    IMAGES.mkdir(parents=True, exist_ok=True)
    GT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    steps = [("docs", render_docs), ("small", render_small), ("multicol", render_multicol),
             ("table", render_tables), ("receipt", render_receipts), ("invoice", render_invoices),
             ("card", render_cards)]
    with sync_playwright() as pw:
        r = Renderer(pw)
        for name, fn in steps:
            if only and name not in only:
                continue
            if name in ("docs", "small", "multicol", "table"):
                kinds = {"docs": ["doc_ko", "doc_mixed", "doc_numeric"]}.get(name, [name])
                if not all((CONTENT / f"{k}.json").exists() for k in kinds):
                    print("원문 없음, 건너뜀:", name)
                    continue
            res = fn(r)
            print(f"{name}: {len(res)}장")
        r.browser.close()


if __name__ == "__main__":
    main()
