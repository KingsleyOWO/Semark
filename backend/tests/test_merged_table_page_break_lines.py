"""Lines a cross-page table merge drops at a page break come back from the text layer.

Spot check 2026-10-08: a 164-page PDF of interview tables. MinerU merges a table
that runs over pages into the table on its first page and leaves an empty box
on each later page; the cell text at every page break of the merge is lost.
The supplement skipped those lines — the empty box "covered" them, and a mostly
present cell paragraph passed the fuzzy match — so 63 lines never reached RAG.
"""
import pytest
from test_normalize_fidelity import _FakeDoc, _FakePage, _patch_pdf, _pdf_text_block

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import HAS_PYMUPDF, NormalizeStage


def _line(text, top):
    return _pdf_text_block(text, (100.0, top, 600.0, top + 20.0))


def _blocks():
    table = Block(block_id="b000000", type=BlockType.TABLE, page_idx=0, bbox_norm=[50, 100, 950, 950],
                  payload={"table_body": "<table><tr><td>示範受訪者甲</td><td>我們每週帶孩子去上示範音樂課 "
                                         "，老師會推薦適合的樂器 這段合併後留在表格裡的句子</td></tr></table>"})
    remnant = Block(block_id="b000001", type=BlockType.UNKNOWN, page_idx=1, bbox_norm=[50, 100, 950, 950],
                    payload={"origin": "table_missing_body_and_image"})
    return [table, remnant]


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
async def test_lines_lost_at_the_page_break_of_a_merged_table_are_supplemented(monkeypatch, tmp_path):
    first = _FakePage(1000.0, 1000.0, [_line("我們每週帶孩子去上示範音樂課，", 200.0),
                                       _line("老師會推薦適合的樂器", 225.0),
                                       _line("後來也買了示範牌的木琴", 250.0)])
    second = _FakePage(1000.0, 1000.0, [_line("最開始其他家長會分享示範經驗", 200.0),
                                        _line("這段合併後留在表格裡的句子", 225.0)])
    _patch_pdf(monkeypatch, tmp_path, _FakeDoc([first, second]))

    blocks, count = await NormalizeStage()._supplement_missing_text(
        doc_id="doc-test", blocks=_blocks(), content_list_path=tmp_path / "content_list.json")

    supplements = {b.page_idx: b.payload["text"] for b in blocks if b.payload.get("origin") == "pymupdf_supplement"}
    # Only the lines no table carries; punctuation the cell dropped is not a difference.
    assert supplements == {0: "後來也買了示範牌的木琴", 1: "最開始其他家長會分享示範經驗"}
    assert count == 2


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
async def test_a_table_paragraph_differing_only_by_ocr_slips_is_not_repeated(monkeypatch, tmp_path):
    cell = ("示範品牌以教學現場需求為出發點，聚焦在啟蒙階段最關鍵的音準正確性、安全性與耐用性，"
            "避免家長與老師在選購過程中反覆試錯，降低整體教育成本，並以系列化組合延長單品的使用期間。")
    printed = cell.replace("降低", "降底")
    blocks = _blocks()
    blocks[0].payload["table_body"] = f"<table><tr><td>1</td><td>{cell}</td></tr></table>"
    _patch_pdf(monkeypatch, tmp_path, _FakeDoc([_FakePage(1000.0, 1000.0, [_line(printed, 200.0)]),
                                                _FakePage(1000.0, 1000.0, [])]))

    _, count = await NormalizeStage()._supplement_missing_text(
        doc_id="doc-test", blocks=blocks, content_list_path=tmp_path / "content_list.json")

    assert count == 0
