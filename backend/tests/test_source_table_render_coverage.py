"""Trusted cell geometry cannot be discarded by legacy semantic-table heuristics."""
from app.models.document_ir import Block, BlockType
from app.pipeline.package_utils import semantic_table_to_text
from app.pipeline.table_structure import canonical_table_html


def source_render(body):
    block = Block(block_id='t', type=BlockType.TABLE, page_idx=0, payload={'table_body': body})
    return semantic_table_to_text(canonical_table_html(block))


def test_partial_repeated_header_preserves_shared_grade_scale_and_column_span():
    text = source_render(
        '<table><tr><th rowspan="2">指標</th><th rowspan="2">指標說明</th>'
        '<th>輔導前 / 企業分數</th><th>輔導後 / 企業分數</th><th rowspan="2">企業現況說明</th></tr>'
        '<tr><td colspan="2">評價等級 / (5優,4良,3中,2可,1差)</td></tr>'
        '<tr><td>辨識度</td><td>品牌差異</td><td>3.67</td><td>4</td><td>尚待提升</td></tr></table>')
    assert text.count('評價等級') == 1
    assert '第 3–4 欄：評價等級 / (5優,4良,3中,2可,1差)' in text
    assert '第 3 欄：3.67' in text and '第 4 欄：4' in text
    assert '第 1 欄（跨 2 列）：指標' in text


def test_trusted_trailing_cell_is_not_truncated_to_shorter_header_width():
    text = source_render('<table><tr><th>項目</th><th>金額</th></tr>'
                         '<tr><td>研發</td><td>100</td><td>尚未核定</td></tr></table>')
    assert '第 3 欄：尚未核定' in text
    assert '第 1 欄：研發' in text and '第 2 欄：100' in text


def test_trusted_data_row_matching_three_headers_does_not_drop_new_information():
    text = source_render('<table><tr><th>甲</th><th>乙</th><th>丙</th><th>附註</th></tr>'
                         '<tr><td>甲</td><td>乙</td><td>丙</td><td>適用範圍不同</td></tr>'
                         '<tr><td>項目一</td><td>10</td><td>20</td><td>一般</td></tr></table>')
    assert '適用範圍不同' in text and '第 4 欄：適用範圍不同' in text
    assert '第 2 欄：10' in text and '第 3 欄：20' in text


def test_trusted_single_column_records_keep_named_fields():
    text = source_render('<table><tr><th>項目</th><th>金額</th></tr>'
                         '<tr><td>研發</td><td>100</td></tr></table>')
    assert '- 項目：研發' in text and '- 金額：100' in text
    assert '第 1 欄' not in text
