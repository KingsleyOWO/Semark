"""A data row's colspan is one value, not one value per column.

`_table_body_to_markdown` used to repeat a spanned cell's text across every
column it covers, in every row. That is right for a HEADER row — a
`<td colspan=2>獲投情形</td>` really does name both sub-columns — but wrong for
a DATA row, where a colspan means one value that happens to straddle two
columns. Repeating it there invents a value the source never gave.

Live evidence (2026-08 corpus, 128 tables): a spectrum-planning table under the
headers 屆期時間 | 頻段 | 釋出時間 ends with a row whose only cell is a
`<td rowspan=1 colspan=2>` holding a band range. Filling it sideways rendered
that band range a second time under 釋出時間, so the chunk asserted a release
date the document never states. 38 of the 128 tables carry a colspan outside
row 0, and they are all this shape: a wrapped paragraph, a 總計 row, a
straddling value, or a full-width 資料來源 note. None is a header.

The same guard already protects the rag.md render (expand_table_header_rows in
structured_rag.py fills only header rows); chunk.md and chunks.jsonl kept
fabricating, so the two products disagreed on the same table. chunk.py keeps
its own parser on purpose — it must not grow a dependency on the packaging
modules — so the rule is re-stated here: fill sideways in the header row only.

Every genuine multi-row header in the corpus (19 tables whose first row has
both a colspan >= 2 and a rowspan >= 2) keeps its second row colspan-free, so
filling physical row 0 alone costs no real header a name.

Wording in the fixtures below is fictional; only the span structure is copied
from live tables.
"""

from app.pipeline.stages.chunk import _table_body_to_markdown

# Shape copied from the live spectrum-planning table: three headers, a
# rowspan=2 key column, and a final data row whose single cell spans the last
# two columns.
DATA_ROW_COLSPAN_HTML = (
    "<table>"
    "<tr><td rowspan=1 colspan=1>屆期時間</td>"
    "<td rowspan=1 colspan=1>頻段</td>"
    "<td rowspan=1 colspan=1>釋出時間</td></tr>"
    "<tr><td rowspan=2 colspan=1>2030年</td>"
    "<td rowspan=1 colspan=1>甲頻段：703~748</td>"
    "<td rowspan=1 colspan=1>2013年標售</td></tr>"
    "<tr><td rowspan=1 colspan=1>乙頻段：1770~1785</td>"
    "<td rowspan=1 colspan=1>2017年標售</td></tr>"
    "<tr><td rowspan=1 colspan=1>2040年</td>"
    "<td rowspan=1 colspan=2>丙頻段：3300~3570 2019年標售</td></tr>"
    "</table>"
)

# Shape copied from a live investment-statistics table: a rowspan=2 key column
# beside a colspan=2 group label that splits into two sub-columns.
TWO_ROW_HEADER_HTML = (
    "<table>"
    "<tr><td rowspan=2 colspan=1>年度</td>"
    "<td rowspan=1 colspan=2>示範島新創獲投情形</td></tr>"
    "<tr><td rowspan=1 colspan=1>件數</td>"
    "<td rowspan=1 colspan=1>金額(億元)</td></tr>"
    "<tr><td rowspan=1 colspan=1>2024年</td>"
    "<td rowspan=1 colspan=1>18</td>"
    "<td rowspan=1 colspan=1>3.2</td></tr>"
    "</table>"
)


def test_data_row_colspan_stays_in_one_column():
    """The straddling cell fills its first column; the rest stay empty."""
    rendered = _table_body_to_markdown(DATA_ROW_COLSPAN_HTML)

    assert rendered == (
        "| 屆期時間 | 頻段 | 釋出時間 |\n"
        "| --- | --- | --- |\n"
        "| 2030年 | 甲頻段：703~748 | 2013年標售 |\n"
        "| 2030年 | 乙頻段：1770~1785 | 2017年標售 |\n"
        "| 2040年 | 丙頻段：3300~3570 2019年標售 |  |"
    )


def test_a_data_row_never_asserts_a_value_the_source_did_not_give():
    """The 釋出時間 column must not inherit the 頻段 column's text."""
    rendered = _table_body_to_markdown(DATA_ROW_COLSPAN_HTML)

    assert rendered.count("丙頻段：3300~3570 2019年標售") == 1


def test_header_row_colspan_still_fills_sideways():
    """A group label names every sub-column it covers — that is what a header is."""
    rendered = _table_body_to_markdown(TWO_ROW_HEADER_HTML)

    assert rendered == (
        "| 年度 | 示範島新創獲投情形 | 示範島新創獲投情形 |\n"
        "| --- | --- | --- |\n"
        "| 年度 | 件數 | 金額(億元) |\n"
        "| 2024年 | 18 | 3.2 |"
    )


def test_full_width_note_row_does_not_repeat_across_the_table():
    """A 資料來源 row folded into the table is one note, not one note per column."""
    body = (
        "<table>"
        "<tr><td rowspan=1 colspan=1>年度</td>"
        "<td rowspan=1 colspan=1>件數</td>"
        "<td rowspan=1 colspan=1>金額(億元)</td></tr>"
        "<tr><td rowspan=1 colspan=1>2024年</td>"
        "<td rowspan=1 colspan=1>18</td>"
        "<td rowspan=1 colspan=1>3.2</td></tr>"
        "<tr><td rowspan=1 colspan=3>資料來源：示範研究院整理，2026年。</td></tr>"
        "</table>"
    )

    rendered = _table_body_to_markdown(body)

    assert rendered == (
        "| 年度 | 件數 | 金額(億元) |\n"
        "| --- | --- | --- |\n"
        "| 2024年 | 18 | 3.2 |\n"
        "| 資料來源：示範研究院整理，2026年。 |  |  |"
    )


def test_rowspan_carried_from_a_data_row_colspan_stays_in_one_column():
    """The carried-down copy must inherit the blanks, not re-fill them."""
    body = (
        "<table>"
        "<tr><td rowspan=1 colspan=1>項目</td>"
        "<td rowspan=1 colspan=1>甲類</td>"
        "<td rowspan=1 colspan=1>乙類</td></tr>"
        "<tr><td rowspan=1 colspan=1>試辦說明</td>"
        "<td rowspan=2 colspan=2>兩類共用同一份說明</td></tr>"
        "<tr><td rowspan=1 colspan=1>續辦說明</td></tr>"
        "</table>"
    )

    rendered = _table_body_to_markdown(body)

    assert rendered == (
        "| 項目 | 甲類 | 乙類 |\n"
        "| --- | --- | --- |\n"
        "| 試辦說明 | 兩類共用同一份說明 |  |\n"
        "| 續辦說明 | 兩類共用同一份說明 |  |"
    )


def test_table_without_spans_renders_exactly_as_before():
    """Regression guard: the 76 corpus tables with no colspan must not move."""
    body = (
        "<table>"
        "<tr><td rowspan=1 colspan=1>指標</td>"
        "<td rowspan=1 colspan=1>現況</td>"
        "<td rowspan=1 colspan=1>目標</td></tr>"
        "<tr><td rowspan=1 colspan=1>峰值速率</td>"
        "<td rowspan=1 colspan=1>20Gbps</td>"
        "<td rowspan=1 colspan=1>1Tbps</td></tr>"
        "<tr><td rowspan=1 colspan=1>延遲</td>"
        "<td rowspan=1 colspan=1>1ms</td>"
        "<td rowspan=1 colspan=1>&lt;0.1ms</td></tr>"
        "</table>"
    )

    rendered = _table_body_to_markdown(body)

    assert rendered == (
        "| 指標 | 現況 | 目標 |\n"
        "| --- | --- | --- |\n"
        "| 峰值速率 | 20Gbps | 1Tbps |\n"
        "| 延遲 | 1ms | <0.1ms |"
    )


def test_rowspan_only_column_still_repeats_down_the_rows():
    """Regression guard: rowspan is unchanged — the key column stays labelled."""
    body = (
        "<table>"
        "<tr><td rowspan=1 colspan=1>區域</td>"
        "<td rowspan=1 colspan=1>計畫</td></tr>"
        "<tr><td rowspan=2 colspan=1>示範島北區</td>"
        "<td rowspan=1 colspan=1>青年培力</td></tr>"
        "<tr><td rowspan=1 colspan=1>社區共學</td></tr>"
        "</table>"
    )

    rendered = _table_body_to_markdown(body)

    assert rendered == (
        "| 區域 | 計畫 |\n"
        "| --- | --- |\n"
        "| 示範島北區 | 青年培力 |\n"
        "| 示範島北區 | 社區共學 |"
    )


def test_non_table_markup_still_falls_back_to_stripped_text():
    """Regression guard: bodies that do not parse into rows keep their old path."""
    assert _table_body_to_markdown("<p>示範島  年度說明</p>") == "示範島 年度說明"
    assert _table_body_to_markdown("純文字表格內容") == "純文字表格內容"
