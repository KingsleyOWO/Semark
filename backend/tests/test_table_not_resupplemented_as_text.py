"""A table PyMuPDF re-reads must not come back as a text block.

MinerU's layout pass can miss text regions, so normalize re-reads the page with
PyMuPDF and supplements whatever no existing block covers. A table is read there
as one box — including the caption above the border and the 注／資料來源 lines
below it — while MinerU splits those into ``table_caption``/``table_footnote``.
Coverage was measured against ``table_body`` alone, so the printed lines had
nothing to match and the fragment fell under the 0.60 4-gram threshold. The
whole table was then re-added as a TEXT block, and rag.md delivered it twice:
once as proper per-row records, once as an unsegmented run of every cell.

Live evidence (2026-08 corpus, doc 732f86c8 page 4): the PPDR table scored 0.39
against ``table_body``, 0.50 with the HTML tags stripped, and 0.85 once the
footnote and caption were included. The two collapsed-cell repairs on that page
made the duplicate louder — the repaired table renders cleanly, so the raw run
of cells beside it reads as a second, contradictory table.

Wording below is fictional; the structure and the field split are copied from
that table.
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import NormalizeStage

# MinerU's parse: cell boundaries partly collapsed, and the printed note and
# attribution filed away from the body.
TABLE_BLOCK = Block(
    block_id="b000036",
    type=BlockType.TABLE,
    page_idx=3,
    bbox_norm=[109, 244, 887, 368],
    payload={
        "table_body": (
            "<table>"
            "<tr><td>國家</td><td>機制</td><td>模式</td><td>主要頻段</td></tr>"
            "<tr><td>示範島</td><td>示範網</td><td>共享頻段</td>"
            "<td>700MHz頻段(758~763MHz、788~793MHz)</td></tr>"
            '<tr><td colspan="4">示範洲 示範網二 專用頻段 '
            "700MHz頻段(718~728MHz、773~783MHz)</td></tr>"
            "</table>"
        ),
        "table_caption": "表3 示範島與示範洲的頻段配置",
        "table_footnote": [
            "注：\\*為示範洲係以示範電信的LTE網路提供服務。",
            "資料來源：本研究整理(2024)。",
        ],
    },
)

# What PyMuPDF reads off the same region: printed order, full-width brackets,
# and the caption and both trailing lines inside the one box.
PRINTED_REGION = (
    "表3 示範島與示範洲的頻段配置"
    "國家機制模式主要頻段"
    "示範島示範網共享頻段700MHz頻段（758~763MHz、788~793MHz）"
    "示範洲示範網二專用頻段700MHz頻段（718~728MHz、773~783MHz）"
    "注：*為示範洲係以示範電信的LTE網路提供服務。"
    "資料來源：本研究整理(2024)。"
)


def _stage() -> NormalizeStage:
    return NormalizeStage.__new__(NormalizeStage)


def test_the_printed_table_region_counts_as_covered():
    stage = _stage()

    covered = stage._is_covered_by_blocks(
        [114, 246, 885, 418], PRINTED_REGION, [TABLE_BLOCK], 3
    )

    assert covered is True


def test_the_footnote_lines_are_what_carry_it_over_the_threshold():
    """Without them the fragment is a minority match and would be re-added."""
    stripped = TABLE_BLOCK.model_copy(deep=True)
    stripped.payload.pop("table_footnote")
    stripped.payload.pop("table_caption")
    stage = _stage()

    covered = stage._is_covered_by_blocks(
        [114, 246, 885, 418], PRINTED_REGION, [stripped], 3
    )

    assert covered is False


def test_a_genuinely_missing_paragraph_is_still_supplemented():
    """The widened coverage text must not swallow real gaps: a paragraph that
    merely discusses the table shares terms with it, not its 4-grams."""
    stage = _stage()
    missed = (
        "示範洲政府於2014年規劃採用公共安全LTE技術建置全國性寬頻網路，"
        "並針對鐵路與海事分別建立專用電信網路，其中以災難安全電信網路為核心。"
    )

    covered = stage._is_covered_by_blocks([120, 500, 880, 560], missed, [TABLE_BLOCK], 3)

    assert covered is False
