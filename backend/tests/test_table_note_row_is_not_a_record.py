"""A table's 注／資料來源 row is attribution, not a data record.

Papers print the source line inside the table's own border, as a final merged
cell spanning every column. After grid expansion that row has the same shape as
a data row with only its first column filled, so the record renderer keyed it by
the first header and emitted

    ### 注：網底標示…資料來源：示範資料庫、本研究整理(2025)。
    - HS 4位碼：注：網底標示…資料來源：示範資料庫、本研究整理(2025)。

— a row heading, and a claim that the note is that column's value. Retrieval
then has a record whose key is a sentence and whose only field is false.

Live evidence (2026-08 corpus): the note row surfaced this way on every table
that has both a detected header and a trailing attribution cell. It is matched
on its own wording rather than its shape, because a row whose cell boundaries
collapsed is full-width too and is genuine data.
"""

from app.pipeline.package_utils import is_table_note_row, semantic_table_to_text

NOTE = "注：網底為本研究分析對象。資料來源：示範資料庫、本研究整理(2025)。"

TABLE_WITH_NOTE_ROW = (
    "<table>"
    "<tr><td>貨品碼</td><td>品名</td><td>金額</td></tr>"
    "<tr><td>8542</td><td>積體電路</td><td>444</td></tr>"
    "<tr><td>8471</td><td>資料處理機</td><td>128</td></tr>"
    f'<tr><td colspan="3">{NOTE}</td></tr>'
    "</table>"
)


def test_note_row_is_recognised_by_its_wording():
    assert is_table_note_row([NOTE, "", ""]) is True
    assert is_table_note_row(["資料來源：示範資料庫。", "", ""]) is True
    assert is_table_note_row(["Source: Demo Database (2025).", ""]) is True


def test_a_collapsed_data_row_is_not_treated_as_a_note():
    """Same shape, no note wording — it is data, and must stay data."""
    collapsed = "示範島衛星        示範企業            ~630            3,236"

    assert is_table_note_row([collapsed, "", "", ""]) is False


def test_a_row_with_real_cells_is_not_a_note():
    assert is_table_note_row(["資料來源：示範資料庫", "444", ""]) is False


def test_note_row_renders_as_a_trailing_line_not_a_record():
    text = semantic_table_to_text(TABLE_WITH_NOTE_ROW, "示範進口統計")
    lines = text.splitlines()

    # The attribution survives …
    assert any(NOTE in line for line in lines)
    # … but never as a row heading or as a column's value.
    assert f"### {NOTE}" not in lines
    assert not any(line.startswith("- 貨品碼：注：") for line in lines)
    # It trails the records rather than sitting between two of them.
    last_record = max(i for i, line in enumerate(lines) if line.startswith("### "))
    assert lines.index(next(l for l in lines if NOTE in l)) > last_record


def test_the_real_rows_are_untouched():
    text = semantic_table_to_text(TABLE_WITH_NOTE_ROW, "示範進口統計")

    assert text.count("### ") == 2
    assert "- 品名：積體電路" in text
    assert "- 金額：128" in text
