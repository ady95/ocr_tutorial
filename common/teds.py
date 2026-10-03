"""OCR 따라하기 — 표 평가 지표 TEDS (Tree-Edit-Distance-based Similarity).

PubTabNet(IBM)의 TEDS 구현(Apache 2.0, PaddleOCR ppstructure/table/table_metric에 포함)을
이 책의 평가셋에 맞게 줄였습니다. 두 표를 HTML 트리로 보고 트리 편집 거리로 유사도를 잽니다.

  TEDS = 1 - (트리 편집 거리 / 두 트리 중 큰 쪽의 노드 수)   (1이면 완전히 같음)

이 책의 변경점: th와 td를 같은 칸으로 취급, 칸 내용은 공백을 지우고 비교(띄어쓰기 차이 무시),
thead·tbody 같은 묶음 태그는 무시. structure_only=True이면 내용 없이 구조만 비교(TEDS-S).

필요 패키지: pip install apted lxml rapidfuzz
"""
import re
import unicodedata

from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from rapidfuzz.distance import Levenshtein


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        result = f'"tag": {self.tag}'
        for child in self.children:
            result += child.bracket()
        return "{" + result + "}"


class CellConfig(Config):
    def rename(self, node1, node2):
        """두 노드를 바꾸는 비용: 태그·병합이 다르면 1, 칸이면 내용의 정규화 편집 거리."""
        if node1.tag != node2.tag or node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if node1.tag == "td" and (node1.content or node2.content):
            return Levenshtein.normalized_distance(node1.content, node2.content)
        return 0.0


def _cell_text(node):
    text = unicodedata.normalize("NFC", "".join(node.itertext()))
    return re.sub(r"\s+", "", text)


def _to_tree(node, structure_only):
    tag = "td" if node.tag in ("td", "th") else node.tag
    if tag == "td":
        return TableTree("td", int(node.attrib.get("colspan", "1")), int(node.attrib.get("rowspan", "1")),
                         [] if structure_only else list(_cell_text(node)))
    tree = TableTree(tag)
    for child in node:
        tree.children.append(_to_tree(child, structure_only))
    return tree


def _parse_table(text):
    if not text:
        return None
    root = html.fromstring(f"<html><body>{text}</body></html>",
                           parser=html.HTMLParser(remove_comments=True, encoding="utf-8"))
    tables = root.xpath("//table")
    if not tables:
        return None
    table = tables[0]
    etree.strip_tags(table, "thead", "tbody", "tfoot", "b", "strong", "i", "em", "span", "br")
    return table


def teds(pred_html, true_html, structure_only=False):
    """예측 표와 정답 표의 TEDS (0~1). 예측에 표가 없으면 0.

    칸이 하나도 없는 빈 표끼리는 1.0, 한쪽만 비었으면 0.0으로 셉니다 (0으로 나누기 방지).
    """
    pred, true = _parse_table(pred_html), _parse_table(true_html)
    if pred is None or true is None:
        return 0.0
    n_pred, n_true = len(pred.xpath(".//*")), len(true.xpath(".//*"))
    if n_pred == 0 or n_true == 0:
        return 1.0 if n_pred == n_true else 0.0
    n_nodes = max(n_pred, n_true)
    distance = APTED(_to_tree(pred, structure_only), _to_tree(true, structure_only),
                     CellConfig()).compute_edit_distance()
    return 1.0 - distance / n_nodes
