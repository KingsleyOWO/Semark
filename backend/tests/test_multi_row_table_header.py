"""Two-row table headers must survive rendering, and row keys must be unique.

MinerU hands us complete span information — every `<td>` carries `rowspan=`
and `colspan=` — and 67 of the 128 tables in the 2026-08 corpus contain a span
of 2 or more. Rendering threw it away twice over:

1. parse_html_table() writes a spanned cell's text into the FIRST grid column
   only and pads the rest with "". A header such as
   `<td colspan=2>出口金額及其占比</td>` therefore names one of its two
   sub-columns and leaves the other anonymous, so the data cell underneath is
   rendered as `- 欄位4：22.40` (926 such placeholders across 22 tables /
   17 documents).
2. semantic_table_to_text() took `rows[header_idx]` — always exactly one row.
   The second physical header row (the sub-label row 「金額 | 占比 | 排名」)
   was then rendered as the first data record, producing self-referential
   garbage records such as `- 產品名稱：產品名稱` (8 tables / 5 documents).

The fix merges the header block top-down and fills colspans sideways, guarded
by three conditions that each have a live counter-example in the corpus. Every
guard is pinned by a test below; removing any one of them corrupts real tables.
"""

import re
import unittest

from app.pipeline.package_utils import semantic_table_to_text
from app.pipeline.structured_rag import parse_html_table, parse_html_table_grid

# Structure copied verbatim from a live trade-statistics table (2 header rows:
# two rowspan=2 key columns plus two colspan=2 groups that split into 金額/占比
# and 排名/占比); the wording is fictional.
TWO_ROW_HEADER_HTML = (
    "<table>"
    "<tr><td rowspan=2 colspan=1>前30大產品HS 4位碼</td>"
    "<td rowspan=2 colspan=1>產品名稱</td>"
    "<td rowspan=1 colspan=2>示範島自甲國進口產品金額及其占總進口額比率(2024)</td>"
    "<td rowspan=1 colspan=2>甲國於示範島該項產品進口之金額排名與占比(2024)</td></tr>"
    "<tr><td rowspan=1 colspan=1>金額</td><td rowspan=1 colspan=1>占比</td>"
    "<td rowspan=1 colspan=1>排名</td><td rowspan=1 colspan=1>占比</td></tr>"
    "<tr><td>8471</td><td>電腦／伺服器</td><td>26,045</td><td>22.40</td>"
    "<td>3</td><td>18.59</td></tr>"
    "<tr><td>8542</td><td>積體電路</td><td>10,332</td><td>8.89</td>"
    "<td>5</td><td>4.11</td></tr>"
    "<tr><td>8534</td><td>印刷電路板</td><td>4,208</td><td>3.62</td>"
    "<td>2</td><td>21.07</td></tr>"
    "</table>"
)


class TwoRowHeaderMergeTests(unittest.TestCase):
    def test_spanned_subcolumns_get_a_name_instead_of_a_placeholder(self):
        text = semantic_table_to_text(TWO_ROW_HEADER_HTML, "示範島出口結構")

        # The four sub-columns under the two colspan=2 groups used to be
        # 欄位3/欄位4/欄位5/欄位6.
        self.assertIsNone(re.search(r"欄位\d", text))
        self.assertIn("示範島自甲國進口產品金額及其占總進口額比率(2024)－金額", text)
        self.assertIn("示範島自甲國進口產品金額及其占總進口額比率(2024)－占比", text)
        self.assertIn("甲國於示範島該項產品進口之金額排名與占比(2024)－排名", text)
        self.assertIn("甲國於示範島該項產品進口之金額排名與占比(2024)－占比", text)

    def test_values_stay_attached_to_the_column_they_came_from(self):
        text = semantic_table_to_text(TWO_ROW_HEADER_HTML, "示範島出口結構")

        record = text.split("### 8471")[1].split("###")[0]
        self.assertIn("- 前30大產品HS 4位碼：8471", record)
        self.assertIn("- 產品名稱：電腦／伺服器", record)
        self.assertIn("- 示範島自甲國進口產品金額及其占總進口額比率(2024)－金額：26,045", record)
        self.assertIn("- 示範島自甲國進口產品金額及其占總進口額比率(2024)－占比：22.40", record)
        self.assertIn("- 甲國於示範島該項產品進口之金額排名與占比(2024)－排名：3", record)
        self.assertIn("- 甲國於示範島該項產品進口之金額排名與占比(2024)－占比：18.59", record)

    def test_second_header_row_is_not_emitted_as_a_data_record(self):
        text = semantic_table_to_text(TWO_ROW_HEADER_HTML, "示範島出口結構")

        # The self-referential record: header row 2 rendered as data used to
        # produce 「- 前30大產品HS 4位碼：前30大產品HS 4位碼」.
        self.assertNotIn("- 前30大產品HS 4位碼：前30大產品HS 4位碼", text)
        self.assertNotIn("- 產品名稱：產品名稱", text)
        self.assertEqual(text.count("### "), 3)

    def test_blank_rows_do_not_desynchronise_the_header_index(self):
        # parse_html_table() drops all-blank rows, so a grid row index is NOT a
        # physical <tr> index. Reading spans at the wrong physical row silently
        # disables the whole fix; the corpus happens not to expose it, so it is
        # pinned here instead.
        html = (
            "<table>"
            "<tr><td></td><td></td><td></td><td></td></tr>"
            "<tr><td rowspan=2 colspan=1>項目</td>"
            "<td rowspan=1 colspan=2>示範島出口金額</td>"
            "<td rowspan=2 colspan=1>單位</td></tr>"
            "<tr><td>2024年</td><td>2025年</td></tr>"
            "<tr><td>研發支出</td><td>1,200</td><td>1,380</td><td>百萬元</td></tr>"
            "<tr><td>行銷支出</td><td>860</td><td>910</td><td>百萬元</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範島支出")

        self.assertIn("示範島出口金額－2024年", text)
        self.assertIn("示範島出口金額－2025年", text)
        self.assertIsNone(re.search(r"欄位\d", text))
        self.assertNotIn("- 項目：項目", text)
        self.assertEqual(text.count("### "), 2)

    def test_repeated_merged_column_names_are_disambiguated(self):
        # Two colspan=2 groups whose sub-labels repeat produce two identical
        # column names; without a suffix a single record renders 「- 占比：」
        # twice and the second value is unattributable.
        html = (
            "<table>"
            "<tr><td rowspan=2 colspan=1>項目</td>"
            "<td rowspan=1 colspan=2>金額</td>"
            "<td rowspan=2 colspan=1>單位</td></tr>"
            "<tr><td>占比</td><td>占比</td></tr>"
            "<tr><td>研發支出</td><td>12.5</td><td>13.1</td><td>百分比</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範島支出")

        self.assertIn("- 金額－占比：12.5", text)
        self.assertIn("- 金額－占比(2)：13.1", text)


class HeaderMergeGuardTests(unittest.TestCase):
    """Each guard has a live counter-example; dropping it corrupts real data."""

    def test_rowspan_without_colspan_keeps_the_second_row_as_data(self):
        # Guard 1. Live shape: a spectrum-allocation table whose third column
        # is rowspan=2 while every colspan is 1. Its second physical row is a
        # genuine data row; treating the table as two-row-headed swallows the
        # first record.
        html = (
            "<table>"
            "<tr><td rowspan=1 colspan=1>頻率範圍</td><td rowspan=1 colspan=1>頻寬</td>"
            "<td rowspan=2 colspan=1>供示範用途實驗網路用</td></tr>"
            "<tr><td rowspan=1 colspan=1>806~816MHz、847~857MHz</td>"
            "<td rowspan=1 colspan=1>20MHz</td></tr>"
            "<tr><td rowspan=1 colspan=1>816~821MHz、857~862MHz</td>"
            "<td rowspan=1 colspan=1>10MHz</td>"
            "<td rowspan=1 colspan=1>供示範物聯網實驗網路用</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範頻段使用現況")

        self.assertIn("- 頻率範圍：806~816MHz、847~857MHz", text)
        self.assertIn("- 頻寬：20MHz", text)
        self.assertIn("- 頻率範圍：816~821MHz、857~862MHz", text)
        self.assertEqual(text.count("### "), 2)

    def test_a_paragraph_in_the_header_row_is_not_a_two_row_header(self):
        # Guard 1, second half. Live shape: MinerU folded a 75-character
        # description into the header row and gave it rowspan=2, which made the
        # first real record (公用事業 / 能源輸出最佳化 / E.ON in the original)
        # look like a sub-label row. A column label is a label — every genuine
        # two-row header in the corpus keeps its top row under 26 characters.
        html = (
            "<table>"
            "<tr><td rowspan=1 colspan=2>案例</td><td rowspan=1 colspan=1>公司</td>"
            "<td rowspan=2 colspan=1>描述：示範顧問公司與示範電力公司合作導入數位孿生，"
            "監控資產健康狀況並收集變壓器性能數據，轉為以預防與風險為基礎的方法</td></tr>"
            "<tr><td>公用事業</td><td>能源輸出最佳化</td><td>示範電力(2023)</td></tr>"
            "<tr><td>港口</td><td>營運優化</td><td>示範港務(2024)</td>"
            "<td>示範通訊與示範港合作，透過數位孿生找出優化港口營運的方法</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範數位孿生案例")

        self.assertIn("- 案例：公用事業", text)
        self.assertIn("- 公司：示範電力(2023)", text)
        self.assertNotIn("案例－公用事業", text)
        self.assertEqual(text.count("### "), 2)

    def test_colspan_in_a_data_row_is_not_filled_sideways(self):
        # Guard 2. Live shape: a spectrum table whose last row merges the last
        # two columns (`<td colspan=2>3500MHz…2019年拍賣</td>`). Filling data
        # rows sideways invents 「釋出時間：3500MHz…」 out of nothing.
        html = (
            "<table>"
            "<tr><td>屆期時間</td><td>頻段</td><td>釋出時間</td></tr>"
            "<tr><td rowspan=2 colspan=1>2030年</td>"
            "<td>700MHz：703~748、758~803</td><td>2013年拍賣</td></tr>"
            "<tr><td>1800MHz：1770~1785、1865~1880</td><td>2017年拍賣</td></tr>"
            "<tr><td>2040年</td>"
            "<td rowspan=1 colspan=2>3500MHz：3300~3570，2019年拍賣</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範頻段屆期時程")

        self.assertIn("- 頻段：3500MHz：3300~3570，2019年拍賣", text)
        self.assertNotIn("- 釋出時間：3500MHz", text)

    def test_table_width_follows_the_header_row_not_the_widest_grid_row(self):
        # Guard 3. Rowspan carry-over makes some grid rows wider than the
        # header (live: a five-industry matrix whose second grid row is 8 wide
        # against a 6-wide header). Sizing the table by the global maximum
        # appends unnamed columns and manufactures NEW 欄位N placeholders.
        html = (
            "<table>"
            "<tr><td rowspan=2 colspan=1>項目</td>"
            "<td rowspan=1 colspan=2>示範島出口金額</td>"
            "<td rowspan=2 colspan=1>單位</td></tr>"
            "<tr><td>2024年</td><td>2025年</td></tr>"
            "<tr><td>研發支出</td><td>1,200</td><td>1,380</td><td>百萬元</td></tr>"
            "<tr><td>行銷支出</td><td>860</td><td>910</td><td>百萬元</td>"
            "<td>暫估</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範島支出")

        self.assertIsNone(re.search(r"欄位\d", text))
        self.assertNotIn("暫估", text)


class ParseHtmlTableContractTests(unittest.TestCase):
    """parse_html_table() has nine call sites, two of which (the travel
    allowance and domestic rate extractors) read the grid column by column. Its
    output must not move; the new grid API is additive."""

    def test_grid_rows_match_parse_html_table_exactly(self):
        for html in (
            TWO_ROW_HEADER_HTML,
            "<table><tr><td colspan=3>合計</td></tr><tr><td>a</td><td>b</td><td>c</td></tr></table>",
            "項目 | 金額\n研發 | 100",
            "",
        ):
            with self.subTest(html=html[:30]):
                self.assertEqual(parse_html_table(html), parse_html_table_grid(html).rows)

    def test_data_rows_still_write_a_spanned_cell_into_the_first_column_only(self):
        rows = parse_html_table(TWO_ROW_HEADER_HTML)
        self.assertEqual(
            rows[0],
            [
                "前30大產品HS 4位碼",
                "產品名稱",
                "示範島自甲國進口產品金額及其占總進口額比率(2024)",
                "",
                "甲國於示範島該項產品進口之金額排名與占比(2024)",
                "",
            ],
        )
        self.assertEqual(rows[2], ["8471", "電腦／伺服器", "26,045", "22.40", "3", "18.59"])

    def test_source_row_index_maps_grid_rows_back_to_physical_rows(self):
        grid = parse_html_table_grid(
            "<table><tr><td></td><td></td></tr><tr><td>項目</td><td>金額</td></tr></table>"
        )
        self.assertEqual(len(grid.rows), 1)
        self.assertEqual(grid.source_row_index, [1])


class PlainTableRegressionTests(unittest.TestCase):
    """Reverse gate: a span-free table must come out byte for byte as before.

    The whole rendering is pinned, so an accidental column rename, a dropped
    first data row, a stray dedupe suffix or a new placeholder shows up here.
    Its header carries a known key column (項目), so the row heading also keeps
    taking the known-key path untouched.
    """

    PLAIN_HTML = (
        "<table>"
        "<tr><th>項目</th><th>金額</th><th>單位</th></tr>"
        "<tr><td>研發支出</td><td>1,200</td><td>百萬元</td></tr>"
        "<tr><td>行銷支出</td><td>860</td><td>百萬元</td></tr>"
        "</table>"
    )

    def test_single_row_header_output_is_byte_identical(self):
        text = semantic_table_to_text(self.PLAIN_HTML, "示範島支出")

        self.assertEqual(
            text,
            "\n".join(
                [
                    "## 示範島支出",
                    "",
                    "表格名稱：示範島支出",
                    "欄位：項目、金額、單位",
                    "",
                    "## 資料列",
                    "",
                    "### 研發支出",
                    "- 項目：研發支出",
                    "- 金額：1,200",
                    "- 單位：百萬元",
                    "",
                    "### 行銷支出",
                    "- 項目：行銷支出",
                    "- 金額：860",
                    "- 單位：百萬元",
                ]
            ),
        )


class RecordHeadingUniquenessTests(unittest.TestCase):
    """`### 長` five times in a row is not a retrievable row key. The original
    HTML is fine — 22 tables / 93 headings / 14 documents lost their key purely
    in rendering, because the fallback heading was the first column alone."""

    def test_heading_uses_the_first_two_values(self):
        html = (
            "<table>"
            "<tr><td>屆期時間</td><td>頻段</td><td>釋出時間</td></tr>"
            "<tr><td rowspan=2 colspan=1>2030年</td>"
            "<td>700MHz：703~748、758~803</td><td>2013年拍賣</td></tr>"
            "<tr><td>1800MHz：1770~1785、1865~1880</td><td>2017年拍賣</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範頻段屆期時程")

        self.assertIn("### 2030年 700MHz：703~748、758~803", text)
        self.assertIn("### 2030年 1800MHz：1770~1785、1865~1880", text)

    def test_headings_that_still_collide_get_a_sequence_suffix(self):
        html = (
            "<table>"
            "<tr><td>期間</td><td>單位</td><td>保存年限</td></tr>"
            "<tr><td>長</td><td>甲部門</td><td>3年</td></tr>"
            "<tr><td>長</td><td>甲部門</td><td>5年</td></tr>"
            "<tr><td>長</td><td>乙部門</td><td>10年</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, "示範保存年限表")

        self.assertIn("### 長 甲部門\n", text)
        self.assertIn("### 長 甲部門 #2\n", text)
        self.assertIn("### 長 乙部門\n", text)
        self.assertNotIn("### 長 乙部門 #", text)


if __name__ == "__main__":
    unittest.main()
