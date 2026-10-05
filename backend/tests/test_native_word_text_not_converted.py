"""Native Word text is what the author typed; only parser OCR gets zh-TW repair.

_normalize_zh_text exists to undo MinerU's OCR glyph misreads (税→稅). A DOCX
paragraph has no OCR step, so converting it can only change what the author
wrote: on the Validation report a consultant's name 「…杰」 shipped as 「…傑」 and
「客户」 as 「客戶」, and the delivered text no longer matched the source page.
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import NormalizeStage


def _block(text: str, *, native: bool) -> Block:
    payload = {"text": text}
    if native:
        payload["word_source"] = {"source": "native_docx", "block_index": 3}
    return Block(block_id="b0", type=BlockType.TEXT, page_idx=0, bbox_norm=[0, 0, 0, 0],
                 reading_order=0, payload=payload)


def test_native_word_text_keeps_the_authors_characters():
    [block] = NormalizeStage()._normalize_zh_text([_block("顧問 王小杰，服務客户", native=True)])

    assert block.payload["text"] == "顧問 王小杰，服務客户"


def test_native_word_table_cells_keep_the_authors_characters():
    table = Block(block_id="b1", type=BlockType.TABLE, page_idx=0, bbox_norm=[0, 0, 0, 0], reading_order=1,
                  payload={"table_body": "<table><tr><td>王小杰</td><td>客户</td></tr></table>",
                           "word_source": {"source": "native_docx", "block_index": 4}})

    [block] = NormalizeStage()._normalize_zh_text([table])

    assert "王小杰" in block.payload["table_body"]
    assert "客户" in block.payload["table_body"]


def test_parser_ocr_text_is_still_repaired():
    [block] = NormalizeStage()._normalize_zh_text([_block("營業税申報", native=False)])

    assert block.payload["text"] == "營業稅申報"
