"""A table's caption and footnote must not delete a neighbour's source line.

Coverage counts a fragment as already-present when a nearby block holds 0.60 of
its 4-grams. Once a table's ``table_caption``/``table_footnote`` joined that
comparison — needed, because PyMuPDF reads them inside the table's own box — the
threshold started firing on boilerplate. Nearly every table in the corpus signs
off 「資料來源：本研究整理(20XX)。」, so a *figure's* 「…本研究繪製(20XX)。」 one
page away is a 0.64 match and vanished; 「資料來源：APEC(2024)。」 was deleted by a
table's 「資料來源：APEC(2017)。」 at 0.69. A replay over the 167-document store
found eight such deletions, all silent — the line simply is not in rag.md.

The lines that must survive top out at 0.786; a table's own note re-read by
PyMuPDF matches verbatim or at 0.95. So annotations may confirm a fragment the
table body already largely accounts for, but must effectively *be* the line to
suppress one on their own.

Wording below is fictional; the collision is the one measured in the corpus.
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import (
    NormalizeStage,
    _coverage_texts,
    _normalize_for_coverage,
)

TABLE = Block(
    block_id="b000051",
    type=BlockType.TABLE,
    page_idx=3,
    bbox_norm=[109, 244, 887, 368],
    payload={
        "table_body": (
            "<table>"
            "<tr><td>構面</td><td>說明</td><td>關鍵議題</td></tr>"
            "<tr><td>願景</td><td>希望為客戶達成的目標</td><td>我們為何存在？</td></tr>"
            "<tr><td>承諾</td><td>能為利害關係人創造的獨特價值</td><td>能帶來什麼？</td></tr>"
            "<tr><td>個性</td><td>對外展現的一致風格與語氣</td><td>我們聽起來像誰？</td></tr>"
            "<tr><td>證據</td><td>可被外部查核的具體事實</td><td>憑什麼相信？</td></tr>"
            "</table>"
        ),
        "table_caption": ["示範定位構面及關鍵議題"],
        "table_footnote": ["資料來源：本研究整理(2025)。"],
    },
)

# The figure on the facing page signs off with 繪製, not 整理. Same boilerplate
# either side of one word.
FIGURE_SOURCE_LINE = "資料來源：本研究繪製(2025)。"


def _stage() -> NormalizeStage:
    return NormalizeStage.__new__(NormalizeStage)


def _ratio(fragment: str, haystack: str) -> float:
    """Share of the fragment's 4-grams a given comparison text holds."""
    clean = _normalize_for_coverage(fragment)
    hay = _normalize_for_coverage(haystack)
    grams = [clean[i:i + 4] for i in range(len(clean) - 3)]
    return sum(1 for gram in grams if gram in hay) / len(grams)


def _combined_ratio(fragment: str, block: Block) -> float:
    _stored, stripped, annotations = _coverage_texts(block)
    return _ratio(fragment, stripped + annotations)


def test_the_collision_is_in_the_band_that_used_to_fire():
    """Pins the fixture to the failure it stands for.

    Above 0.60 the old code suppressed it; below 0.95 the new code must not.
    If the wording drifts out of this band the tests below stop proving
    anything, so fail here instead.
    """
    ratio = _combined_ratio(FIGURE_SOURCE_LINE, TABLE)

    assert 0.60 < ratio < 0.95


def test_a_neighbours_source_line_survives_the_tables_footnote():
    stage = _stage()

    covered = stage._is_covered_by_blocks(
        [120, 700, 880, 730], FIGURE_SOURCE_LINE, [TABLE], 4
    )

    assert covered is False


def test_the_tables_own_footnote_is_still_suppressed():
    """Read back off the page verbatim, it renders with the table already."""
    stage = _stage()

    covered = stage._is_covered_by_blocks(
        [120, 380, 880, 400], "資料來源：本研究整理(2025)。", [TABLE], 3
    )

    assert covered is True


def test_annotations_still_confirm_a_fragment_the_body_carries():
    """The whole-table duplicate this coverage text was widened for.

    The body accounts for most of the printed region; the caption and footnote
    only top it up. That must stay suppressed — it is the duplicate rag.md was
    emitting beside the proper record render.
    """
    printed_region = (
        "示範定位構面及關鍵議題"
        "構面說明關鍵議題"
        "願景希望為客戶達成的目標我們為何存在？"
        "承諾能為利害關係人創造的獨特價值能帶來什麼？"
        "個性對外展現的一致風格與語氣我們聽起來像誰？"
        "證據可被外部查核的具體事實憑什麼相信？"
        "資料來源：本研究整理(2025)。"
    )
    stage = _stage()

    assert len(_normalize_for_coverage(printed_region)) >= (
        stage.COVERAGE_TABLE_DUMP_MIN_CHARS
    )

    covered = stage._is_covered_by_blocks(
        [114, 246, 885, 418], printed_region, [TABLE], 3
    )

    assert covered is True


# A cross-reference is the whole point of the line, and it is printed nowhere
# else. Everything around it is table vocabulary read straight across the cell
# boundaries the stored HTML keeps apart.
FIGURE_AXIS_LINE = "構面說明關鍵議題願景希望為客戶達成的目標見圖2"


def test_a_short_cross_cell_line_is_not_deleted_by_the_stripped_body():
    """The floor's own case: only tag-stripping gets this over the threshold.

    Against the table as MinerU stores it the fragment is a minority match,
    because every 4-gram spanning two cells is cut by the markup. Strip the
    tags and the same fragment clears 0.60 on invented adjacencies — 「構面說明」
    is two headings, never a printed phrase. Long enough to be the whole re-read
    box, that is the duplicate worth removing; at this length it is a line with
    something of its own to say.
    """
    stage = _stage()
    stored, stripped, _annotations = _coverage_texts(TABLE)

    assert _ratio(FIGURE_AXIS_LINE, stored) < 0.60
    assert _ratio(FIGURE_AXIS_LINE, stripped) > 0.60
    assert len(_normalize_for_coverage(FIGURE_AXIS_LINE)) < (
        stage.COVERAGE_TABLE_DUMP_MIN_CHARS
    )

    covered = stage._is_covered_by_blocks(
        [120, 700, 880, 730], FIGURE_AXIS_LINE, [TABLE], 3
    )

    assert covered is False
