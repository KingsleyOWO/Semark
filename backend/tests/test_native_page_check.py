"""Native-Word pages are checked against the Word-rendered PDF text layer, without a model.

On the 47-page Validation report the model audit spent 32 calls / 44 minutes on
pages holding only native Word text and tables, mostly to say 「與來源逐字一致」.
The same conversion's text layer answers that by string comparison, and it is
stricter: it caught an auto-numbered 「1、」, a text box and a name the zh-TW
converter had rewritten, all of which the model audit passed. Whatever it
cannot prove is left to the model audit; it never fails a page on its own.
"""

import fitz
import pytest

from app.models.document_ir import Block, BlockType
from app.pipeline.native_page_check import NativePageText

FONT = "china-t"


def _pdf(tmp_path, pages):
    path = tmp_path / "word.pdf"
    with fitz.open() as pdf:
        for lines in pages:
            page = pdf.new_page()
            for row, line in enumerate(lines):
                page.insert_text((60, 60 + 24 * row), line, fontname=FONT, fontsize=11)
        pdf.save(path)
    return path


def _text(bid, text, order):
    return Block(block_id=bid, type=BlockType.TEXT, page_idx=0, reading_order=order, bbox_norm=[0, 0, 0, 0],
                 payload={"text": text, "word_source": {"source": "native_docx", "block_index": order}})


def _table(bid, cells, order):
    body = "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells) + "</table>"
    return Block(block_id=bid, type=BlockType.TABLE, page_idx=0, reading_order=order, bbox_norm=[0, 0, 0, 0],
                 payload={"table_body": body, "word_source": {"source": "native_docx", "block_index": order}})


def _delivered(blocks):
    documents = [{"file": "outputs/documents/main.md", "paragraph": i, "text": b.get_text(),
                  "block_ids": [b.block_id], "binding": "source_marker"} for i, b in enumerate(blocks)]
    chunks = [{"chunk_id": "c0", "content": "\n".join(b.get_text() for b in blocks),
               "block_ids": [b.block_id for b in blocks]}]
    return documents, chunks


def _check(tmp_path, pages, blocks, mapping, target=0, documents=None, chunks=None):
    """Hand the check what document_review does: the target page's blocks, neighbours apart."""
    evidence = NativePageText(_pdf(tmp_path, pages))
    on_page = [b for b in blocks if target in mapping[b.block_id]]
    nearby = [b for b in blocks if b not in on_page and set(mapping[b.block_id]) & {target - 1, target + 1}]
    delivered = _delivered(blocks)
    return evidence.check(target, on_page, mapping, documents=documents or delivered[0],
                          chunks=delivered[1] if chunks is None else chunks, chunks_required=True,
                          nearby=nearby)


def test_page_whose_every_printed_line_is_delivered_passes(tmp_path):
    blocks = [_text("b0", "一、計畫摘要", 0), _text("b1", "本計畫協助企業建立品牌識別。", 1)]

    assert _check(tmp_path, [["一、計畫摘要", "本計畫協助企業建立品牌識別。"]], blocks, {"b0": [0], "b1": [0]}) is None


def test_printed_line_missing_from_the_parse_is_left_to_the_model(tmp_path):
    blocks = [_text("b0", "結案報告書", 0)]

    reason = _check(tmp_path, [["結案報告書", "合約編號：示範001"]], blocks, {"b0": [0]})

    assert reason and "合約編號" in reason


def test_delivered_text_that_differs_from_the_source_is_left_to_the_model(tmp_path):
    blocks = [_text("b0", "服務客戶", 0)]

    assert _check(tmp_path, [["服務客户"]], blocks, {"b0": [0]})


def test_block_absent_from_chunks_is_left_to_the_model(tmp_path):
    blocks = [_text("b0", "品牌定位", 0), _text("b1", "品牌承諾", 1)]
    chunks = [{"chunk_id": "c0", "content": "品牌定位", "block_ids": ["b0"]}]

    reason = _check(tmp_path, [["品牌定位", "品牌承諾"]], blocks, {"b0": [0], "b1": [0]}, chunks=chunks)

    assert reason and "b1" in reason


def test_unattributed_delivered_line_is_left_to_the_model(tmp_path):
    blocks = [_text("b0", "品牌定位", 0)]
    documents = _delivered(blocks)[0] + [{"file": "outputs/documents/main.md", "paragraph": 1,
                                          "text": "憑空出現的一行", "block_ids": ["b0"],
                                          "binding": "unbound_neighbor_context"}]

    assert _check(tmp_path, [["品牌定位"]], blocks, {"b0": [0]}, documents=documents)


def test_page_numbers_and_running_headers_are_furniture(tmp_path):
    pages = [["品牌輔導結案報告", f"第{i}章內容段落", str(i + 1)] for i in range(3)]
    blocks = [_text("b0", "第0章內容段落", 0), _text("b1", "第1章內容段落", 1), _text("b2", "第2章內容段落", 2)]

    assert _check(tmp_path, pages, blocks, {"b0": [0], "b1": [1], "b2": [2]}, target=1) is None


def test_source_order_mismatch_is_left_to_the_model(tmp_path):
    blocks = [_text("b0", "第一段", 0), _text("b1", "第二段", 1)]

    assert _check(tmp_path, [["第二段", "第一段"]], blocks, {"b0": [0], "b1": [0]})


def test_table_cell_split_by_a_page_break_still_matches(tmp_path):
    pages = [["項目", "說明", "官網改版"], ["項目", "說明", "與社群內容規劃"]]
    blocks = [_table("t0", [["項目", "說明"], ["網站", "官網改版與社群內容規劃"]], 0)]

    reason = _check(tmp_path, pages, blocks, {"t0": [0, 1]}, target=0)

    # 「網站」 is not printed on either page, so the check must still refuse.
    assert reason and "網站" in reason
    blocks = [_table("t0", [["項目", "說明"], ["官網改版與社群內容規劃"]], 0)]
    assert _check(tmp_path, pages, blocks, {"t0": [0, 1]}, target=0) is None


def test_page_with_a_picture_block_is_not_checked_here(tmp_path):
    picture = Block(block_id="i0", type=BlockType.IMAGE, page_idx=0, reading_order=1, bbox_norm=[0, 0, 0, 0],
                    payload={"img_path": "a.png", "word_source": {"source": "native_docx"}})

    reason = _check(tmp_path, [["品牌定位"]], [_text("b0", "品牌定位", 0), picture], {"b0": [0], "i0": [0]})

    assert reason == "page_has_picture_blocks"


@pytest.mark.parametrize("payload", [{"text": "品牌定位"}, {"text": "品牌定位", "agentic_gap": {"reason": "x"},
                                                           "word_source": {"source": "native_docx"}}])
def test_only_plain_native_word_blocks_qualify(tmp_path, payload):
    block = Block(block_id="b0", type=BlockType.TEXT, page_idx=0, reading_order=0, bbox_norm=[0, 0, 0, 0],
                  payload=payload)

    assert _check(tmp_path, [["品牌定位"]], [block], {"b0": [0]})


def _unbound(text, paragraph=9):
    return {"file": "outputs/documents/main.md", "paragraph": paragraph, "text": text, "block_ids": ["b0"],
            "binding": "unbound_neighbor_context"}


def test_package_labels_around_source_text_are_not_untraceable_output(tmp_path):
    blocks = [_text("b0", "一、經費分攤簡表", 0)]
    documents = _delivered(blocks)[0] + [_unbound("表格名稱：一、經費分攤簡表"), _unbound("表格名稱：Word 表格 4", 10),
                                         _unbound("## 補充原文內容", 11)]

    assert _check(tmp_path, [["一、經費分攤簡表"]], blocks, {"b0": [0]}, documents=documents) is None


def test_package_label_around_text_the_page_never_printed_is_still_refused(tmp_path):
    blocks = [_text("b0", "一、經費分攤簡表", 0)]
    documents = _delivered(blocks)[0] + [_unbound("表格名稱：憑空的名字")]

    assert _check(tmp_path, [["一、經費分攤簡表"]], blocks, {"b0": [0]}, documents=documents)


def test_package_label_around_another_pages_source_text_is_not_untraceable(tmp_path):
    blocks = [_text("b0", "一、經費分攤簡表", 0)]
    documents = _delivered(blocks)[0] + [_unbound("表格名稱：二、總結算表")]
    evidence = NativePageText(_pdf(tmp_path, [["一、經費分攤簡表"]]))
    delivered = _delivered(blocks)

    reason = evidence.check(0, blocks, {"b0": [0]}, documents=documents, chunks=delivered[1],
                            chunks_required=True, document_text="一經費分攤簡表二總結算表")

    assert reason is None
