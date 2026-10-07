"""A line MinerU types header/footer but no other page prints is body text.

Spot check 2026-10-07: on a page of slide thumbnails MinerU typed each slide's
title bar and caption strip as header/footer, and the page's brand slogan,
product claims, product lines and product names were dropped as furniture.
Replayed over the 44 stored documents, 94 lines in 5 documents come back:
mostly section titles and whole profile pages, a few cover logos and
copyright lines.
"""
from app.pipeline.stages.normalize import NormalizeStage, is_page_furniture
from app.pipeline.stages.package import PackageStage
from test_page_furniture_excluded_from_delivery import _ir


def _parse(items):
    stage = NormalizeStage()
    blocks = [stage._parse_block(dict(item), index) for index, item in enumerate(items)]
    blocks = stage._release_one_off_furniture(blocks)
    return {block.payload["text"]: block for block in stage._tag_layout_furniture(blocks, page_count=3)}


def test_slide_text_typed_as_header_on_one_page_is_delivered():
    blocks = _parse([
        {"type": "header", "text": "• 品牌標語：示範清新，每日安心", "page_idx": 1, "bbox": [521, 206, 684, 220]},
        {"type": "footer", "text": "示範品牌個性", "page_idx": 1, "bbox": [220, 666, 369, 685]},
        {"type": "header", "text": "示範顧問 Example Consultancy", "page_idx": 0, "bbox": [92, 20, 347, 40]},
        {"type": "header", "text": "示範顧問 Example Consultancy", "page_idx": 1, "bbox": [92, 20, 347, 40]},
        {"type": "page_number", "text": "52", "page_idx": 1, "bbox": [489, 942, 509, 954]},
        {"type": "footer", "text": "10", "page_idx": 1, "bbox": [509, 521, 524, 530]},
        {"type": "text", "text": "示範內文段落", "page_idx": 2, "bbox": [100, 300, 900, 400]},
    ])

    assert not is_page_furniture(blocks["• 品牌標語：示範清新，每日安心"])
    assert not is_page_furniture(blocks["示範品牌個性"])
    assert is_page_furniture(blocks["示範顧問 Example Consultancy"]), "a running head recurs"
    assert is_page_furniture(blocks["52"]) and is_page_furniture(blocks["10"])

    rag_md, _ = PackageStage()._render_rag_md(document_ir=_ir(list(blocks.values()), pages=3),
                                              asset_map={}, enrichments={})
    assert "品牌標語：示範清新" in rag_md and "示範品牌個性" in rag_md
    assert "Example Consultancy" not in rag_md


def test_a_header_another_page_prints_as_body_stays_furniture():
    stage = NormalizeStage()
    header = stage._parse_block({"type": "header", "text": "第三章 示範市場", "page_idx": 1}, 0)
    body = stage._parse_block({"type": "text", "text": "第三章　示範市場", "page_idx": 0}, 1)

    stage._release_one_off_furniture([header, body])

    assert is_page_furniture(header), "page 0 already delivers the text"
    assert not is_page_furniture(body)
