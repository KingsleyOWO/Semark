"""Repairing tables whose parsed cell boundaries collapsed.

MinerU sometimes emits a whole data row inside one ``colspan=6`` cell. The
columns are interleaved by printed line rather than by column, and CJK wraps
without a separator, so the row cannot be split back apart from the text alone
— a word is routinely cut in half around another column's content. The crop
image is kept for exactly this reason, so the repair reads the table again from
the image and the model decides whether the parse was really wrong.
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.package_utils import (
    table_may_have_collapsed_rows,
    wide_colspan_cells,
)
from app.pipeline.stages.enrich import EnrichStage

# A row that lost its boundaries: four columns flattened into one cell, with
# the column gaps surviving only as runs of spaces.
COLLAPSED_HTML = (
    "<table><tr><td>類型</td><td>說明</td><td>估值</td><td>件數</td></tr>"
    '<tr><td colspan="4">示範島衛星        示範企業            ~630            3,236</td></tr>'
    "</table>"
)


def test_wide_colspan_cell_is_flagged_for_re_reading():
    cells = wide_colspan_cells(COLLAPSED_HTML)

    assert len(cells) == 1
    assert "示範島衛星" in cells[0]
    assert table_may_have_collapsed_rows(COLLAPSED_HTML) is True


def test_ordinary_table_is_not_flagged():
    html = (
        "<table><tr><td>項目</td><td>金額</td></tr>"
        "<tr><td>設備</td><td>1,200</td></tr></table>"
    )

    assert table_may_have_collapsed_rows(html) is False


def test_short_merged_label_is_not_flagged():
    """A merged header label is a normal colspan, not a collapsed row."""
    html = (
        '<table><tr><td colspan="2">年度統計</td></tr>'
        "<tr><td>2024</td><td>2025</td></tr></table>"
    )

    assert table_may_have_collapsed_rows(html) is False


def _table_block(html: str = COLLAPSED_HTML) -> Block:
    return Block(
        block_id="tbl0",
        type=BlockType.TABLE,
        page_idx=0,
        payload={"table_body": html, "img_path": "images/table_01.jpg"},
    )


def test_reconstruction_replaces_the_table_and_keeps_the_original():
    block = _table_block()
    output = {
        "parse_was_wrong": True,
        "header": ["類型", "說明", "估值", "件數"],
        "rows": [["示範島衛星", "示範企業", "~630", "3,236"]],
    }

    assert EnrichStage._apply_table_reconstruction(block, output) is True
    assert "<td>示範島衛星</td><td>示範企業</td>" in block.payload["table_body"]
    assert block.payload["table_body_source"] == "vlm_reconstruct"
    # The original stays recoverable rather than being overwritten in place.
    assert 'colspan="4"' in block.payload["table_body_mineru"]


def test_model_saying_the_parse_was_fine_leaves_the_table_alone():
    block = _table_block()
    output = {"parse_was_wrong": False, "header": ["類型"], "rows": [["x"]]}

    assert EnrichStage._apply_table_reconstruction(block, output) is False
    assert block.payload["table_body"] == COLLAPSED_HTML
    assert "table_body_source" not in block.payload


def test_ragged_grid_is_rejected():
    """Losing the column count is the failure this repair exists to fix."""
    block = _table_block()
    output = {
        "parse_was_wrong": True,
        "header": ["類型", "說明", "估值", "件數"],
        "rows": [["示範島衛星", "示範企業", "~630"]],
    }

    assert EnrichStage._apply_table_reconstruction(block, output) is False
    assert block.payload["table_body"] == COLLAPSED_HTML


def test_transcription_that_drops_most_of_the_table_is_rejected():
    """A short read is invisible once the HTML is replaced, so refuse it."""
    block = _table_block()
    output = {
        "parse_was_wrong": True,
        "header": ["類型"],
        "rows": [["示範島衛星"]],
    }

    assert EnrichStage._apply_table_reconstruction(block, output) is False
    assert block.payload["table_body"] == COLLAPSED_HTML


def test_source_note_row_survives_the_reconstruction():
    """A table's 資料來源 row is a full-width cell, the same shape as a collapsed
    row, and the model — asked for a grid — drops it. Losing it deletes the
    attribution from the delivered output, so it is carried over here.
    """
    note = "注：網底為本研究分析對象。資料來源：示範資料庫、本研究整理(2025)。"
    html = (
        "<table><tr><td>類型</td><td>說明</td><td>估值</td><td>件數</td></tr>"
        '<tr><td colspan="4">示範島衛星        示範企業            ~630            3,236</td></tr>'
        f'<tr><td colspan="4">{note}</td></tr></table>'
    )
    block = _table_block(html)
    output = {
        "parse_was_wrong": True,
        "header": ["類型", "說明", "估值", "件數"],
        "rows": [["示範島衛星", "示範企業", "~630", "3,236"]],
    }

    assert EnrichStage._apply_table_reconstruction(block, output) is True
    assert note in block.payload["table_body"]


def test_a_collapsed_row_the_model_recovered_is_not_re_appended():
    """The collapsed row reappears as real cells, so carrying it over too
    would duplicate it. The reconstruction itself decides, not note wording."""
    block = _table_block()
    output = {
        "parse_was_wrong": True,
        "header": ["類型", "說明", "估值", "件數"],
        "rows": [["示範島衛星", "示範企業", "~630", "3,236"]],
    }

    assert EnrichStage._apply_table_reconstruction(block, output) is True
    assert block.payload["table_body"].count("示範島衛星") == 1
    assert "colspan" not in block.payload["table_body"]


def test_reconstructed_cells_are_html_escaped():
    block = _table_block()
    output = {
        "parse_was_wrong": True,
        "header": ["類型", "說明", "估值", "件數"],
        "rows": [["示範島<衛星>", "示範企業 & 夥伴", "~630", "3,236"]],
    }

    assert EnrichStage._apply_table_reconstruction(block, output) is True
    body = block.payload["table_body"]
    assert "&lt;衛星&gt;" in body
    assert "&amp;" in body
