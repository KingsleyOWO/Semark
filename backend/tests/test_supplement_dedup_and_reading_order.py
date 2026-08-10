"""The PyMuPDF text supplement must not duplicate text MinerU already has,
and merging it must not destroy MinerU's reading order.

Live evidence (2026-08-10, 100-document store): 2,617 orphan column fragments
across all 100 documents (median 26 each) and 155 duplicated paragraph heads in
49 of them. Two independent mechanisms, both reproduced below with the strings
that produced them in ``a273c9e754b4a257``:

* the paragraph that covers a fragment often sits on the *previous* page —
  MinerU merges a paragraph across the page break, but the coverage check only
  ever compared against blocks of the same page;
* markdown escaping (``1\\~10月`` in MinerU's text vs ``1~10月`` from PyMuPDF)
  costs four 4-grams, dropping a real duplicate to 0.59 against a 0.60
  threshold.

The reading-order half is separate: merging supplements re-sorted *every* block
by y alone, which interleaves the columns of a two-column layout. Observed:
「地位。然而…」 (right column, y=597) was delivered before the sentence it
continues, 「回顧2025年…」 (left column, y=598).
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import NormalizeStage

# Verbatim from the live document: MinerU's merged paragraph starts on page 1
# and runs over onto page 2, where PyMuPDF re-reads its tail as "missing".
MINERU_PARAGRAPH = (
    "國內生產方面，受惠於人工智慧、高效能運算與雲端服務需求續強，加上消費性電子新品備貨效應挹注，"
    "資訊電子產業成為支撐國內生產的核心動能。2025年1\\~10月工業生產指數較上年同期成長16.3%，"
    "其中製造業年增率達17.4%，惟其他工業部門均呈衰退，顯示產業成長動能高度集中。"
)
SPILLOVER_FRAGMENT = "產的核心動能。2025年1~10月工業生產指數"
ESCAPED_FRAGMENT = "17.4%，惟其他工業部門均呈衰退，顯示產業"


def _block(block_id: str, text: str, page_idx: int, y0: int = 400, order: int = 0) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=page_idx,
        bbox_norm=[87, y0, 480, y0 + 120],
        reading_order=order,
        payload={"text": text, "text_level": 0},
    )


# ---------------------------------------------------------------------------
# Coverage: a fragment whose paragraph lives on a neighbouring page
# ---------------------------------------------------------------------------


def test_fragment_is_covered_by_a_paragraph_on_the_previous_page():
    stage = NormalizeStage()
    blocks = [_block("b000021", MINERU_PARAGRAPH, page_idx=1)]

    assert stage._is_covered_by_blocks([0, 0, 0, 0], SPILLOVER_FRAGMENT, blocks, page_idx=2)


def test_markdown_escaping_does_not_hide_a_duplicate():
    """``1\\~10月`` vs ``1~10月`` cost four 4-grams and dropped 0.60 to 0.59."""
    stage = NormalizeStage()
    blocks = [_block("b000021", MINERU_PARAGRAPH, page_idx=2)]

    assert stage._is_covered_by_blocks([0, 0, 0, 0], ESCAPED_FRAGMENT, blocks, page_idx=2)


def test_a_bare_folio_does_not_cover_unrelated_prose():
    """A one-character block ("5") must not swallow every text containing a 5."""
    stage = NormalizeStage()
    blocks = [_block("b000007", "5", page_idx=2)]

    assert not stage._is_covered_by_blocks([0, 0, 0, 0], SPILLOVER_FRAGMENT, blocks, page_idx=2)


def test_genuinely_missing_text_is_still_supplemented():
    """Anti-over-correction: widening coverage must not suppress real gaps."""
    stage = NormalizeStage()
    blocks = [_block("b000021", MINERU_PARAGRAPH, page_idx=2)]
    missing = "在就業數據方面，整體勞動市場維持穩健，平均失業率為3.35%，較上年同期改善。"

    assert not stage._is_covered_by_blocks([0, 0, 0, 0], missing, blocks, page_idx=2)


def test_coverage_does_not_reach_across_distant_pages():
    """Scope stays at the page break; an unrelated page must not mask a gap."""
    stage = NormalizeStage()
    blocks = [_block("b000021", MINERU_PARAGRAPH, page_idx=0)]

    assert not stage._is_covered_by_blocks([0, 0, 0, 0], SPILLOVER_FRAGMENT, blocks, page_idx=7)


# ---------------------------------------------------------------------------
# Merge: MinerU's reading order is authoritative
# ---------------------------------------------------------------------------


def test_merge_preserves_mineru_order_across_columns():
    """The left-column sentence must keep preceding its right-column follow-on
    even though the right column starts one unit higher on the page."""
    left = _block("b000003", "回顧2025年，台灣經濟展現強勁韌性。", page_idx=0, y0=598, order=0)
    right = _block("b000004", "地位。然而，產業發展亦呈現「K型」分歧。", page_idx=0, y0=597, order=1)
    supplement = _block("s000075", "補充段落文字內容。", page_idx=0, y0=700, order=99)

    merged = NormalizeStage()._merge_supplements_in_order([left, right], [supplement])

    assert [block.block_id for block in merged] == ["b000003", "b000004", "s000075"]
    assert [block.reading_order for block in merged] == [0, 1, 2]


def test_supplement_lands_next_to_the_block_it_follows():
    top = _block("b000000", "頁面最上方的段落內容。", page_idx=0, y0=100, order=0)
    bottom = _block("b000001", "頁面下半部的段落內容。", page_idx=0, y0=800, order=1)
    supplement = _block("s000010", "介於兩段之間的補充。", page_idx=0, y0=400, order=99)

    merged = NormalizeStage()._merge_supplements_in_order([top, bottom], [supplement])

    assert [block.block_id for block in merged] == ["b000000", "s000010", "b000001"]


def test_supplements_are_grouped_by_page():
    page0 = _block("b000000", "第一頁的段落。", page_idx=0, y0=100, order=0)
    page1 = _block("b000001", "第二頁的段落。", page_idx=1, y0=100, order=1)
    supplement = _block("s000010", "第二頁的補充。", page_idx=1, y0=500, order=99)

    merged = NormalizeStage()._merge_supplements_in_order([page0, page1], [supplement])

    assert [block.block_id for block in merged] == ["b000000", "b000001", "s000010"]


def test_supplement_on_a_page_mineru_missed_entirely_is_kept():
    page0 = _block("b000000", "第一頁的段落。", page_idx=0, y0=100, order=0)
    supplement = _block("s000010", "MinerU 完全沒讀到的一頁。", page_idx=3, y0=200, order=99)

    merged = NormalizeStage()._merge_supplements_in_order([page0], [supplement])

    assert [block.block_id for block in merged] == ["b000000", "s000010"]


# ---------------------------------------------------------------------------
# Residue found after the first deploy (2026-08-10, re-measured over 100 docs)
# ---------------------------------------------------------------------------


def test_a_supplement_inside_an_existing_block_is_covered_even_when_the_text_differs():
    """PyMuPDF re-reads the vertical running head 「示範政經瞭望」 as
    「示政瞭範經望」 — column order, so no text comparison can match it. Its box
    sits wholly inside the block MinerU already produced."""
    stage = NormalizeStage()
    head = Block(
        block_id="b000008",
        type=BlockType.TEXT,
        page_idx=0,
        bbox_norm=[379, 63, 615, 116],
        reading_order=0,
        payload={"text": "示範政經瞭望", "text_level": 0, "origin": "page_furniture"},
    )

    assert stage._is_covered_by_blocks([400, 85, 599, 101], "示政瞭範經望", [head], page_idx=0)


def test_a_supplement_outside_every_block_is_not_covered_by_geometry():
    stage = NormalizeStage()
    head = Block(
        block_id="b000008",
        type=BlockType.TEXT,
        page_idx=0,
        bbox_norm=[379, 63, 615, 116],
        reading_order=0,
        payload={"text": "示範政經瞭望", "text_level": 0},
    )

    assert not stage._is_covered_by_blocks([95, 700, 480, 760], "完全不同位置的新段落文字。", [head], page_idx=0)


def _line(text: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"text": text, "bbox": [x0, y0, x1, y1]}


def test_contiguous_supplement_lines_merge_into_one_paragraph():
    """PyMuPDF hands back one block per printed line; delivering them unmerged
    is what produced 「望當前國際淨零碳排趨勢，各主要國家」 as its own paragraph."""
    lines = [
        _line("望當前國際淨零碳排趨勢，各主要國家", 147, 729, 476, 743),
        _line("莫不奠基各國情勢，研提資源循環經濟", 147, 754, 476, 768),
        _line("相關的政策方案。", 147, 779, 300, 793),
    ]

    merged = NormalizeStage()._merge_adjacent_pdf_lines(lines)

    assert len(merged) == 1
    assert merged[0]["text"] == "望當前國際淨零碳排趨勢，各主要國家莫不奠基各國情勢，研提資源循環經濟相關的政策方案。"
    assert merged[0]["bbox"] == [147, 729, 476, 793]


def test_lines_in_different_columns_do_not_merge():
    lines = [
        _line("左欄的第一行文字內容", 95, 400, 476, 414),
        _line("右欄的第一行文字內容", 520, 402, 900, 416),
    ]

    merged = NormalizeStage()._merge_adjacent_pdf_lines(lines)

    assert len(merged) == 2


def test_lines_separated_by_a_paragraph_gap_do_not_merge():
    lines = [
        _line("第一段的最後一行文字", 147, 400, 476, 414),
        _line("第二段的第一行文字", 147, 520, 476, 534),
    ]

    merged = NormalizeStage()._merge_adjacent_pdf_lines(lines)

    assert len(merged) == 2


def test_latin_lines_merge_with_a_separating_space():
    lines = [
        _line("Regulation (EU) 2023/1542 on batteries", 95, 400, 476, 414),
        _line("and waste batteries, Official Journal.", 95, 418, 476, 432),
    ]

    merged = NormalizeStage()._merge_adjacent_pdf_lines(lines)

    assert len(merged) == 1
    assert merged[0]["text"] == (
        "Regulation (EU) 2023/1542 on batteries and waste batteries, Official Journal."
    )


def test_lines_in_the_page_margin_never_merge_into_a_paragraph():
    """Regression: merging by geometry alone glued the running head to the
    title below it — 「示範5-6再生能源憑證制度之發展趨勢」 — which then
    could no longer be dropped as furniture, because half of it was content."""
    page_height = 1000.0
    lines = [
        _line("示範5-6", 95, 60, 180, 78),
        _line("再生能源憑證制度之發展趨勢", 95, 82, 476, 100),
        _line("本文分析各國再生能源氣體憑證制度的發展。", 95, 300, 476, 318),
        _line("並比較其與電力憑證制度的差異之處。", 95, 322, 476, 340),
    ]

    merged = NormalizeStage()._merge_adjacent_pdf_lines(lines, page_height=page_height)

    texts = [m["text"] for m in merged]
    assert "示範5-6" in texts
    assert "再生能源憑證制度之發展趨勢" in texts
    assert "本文分析各國再生能源氣體憑證制度的發展。並比較其與電力憑證制度的差異之處。" in texts


def test_body_lines_still_merge_when_the_page_height_is_known():
    lines = [
        _line("望當前國際淨零碳排趨勢，各主要國家", 147, 729, 476, 743),
        _line("莫不奠基各國情勢，研提資源循環經濟", 147, 754, 476, 768),
    ]

    merged = NormalizeStage()._merge_adjacent_pdf_lines(lines, page_height=1000.0)

    assert len(merged) == 1
