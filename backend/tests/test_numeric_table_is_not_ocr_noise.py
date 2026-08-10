"""A statistics table's numbers are its payload, not OCR noise.

table_rows_are_low_confidence() asks whether a parsed table is mostly OCR or
layout garbage; when it says yes, semantic_table_to_text() deliberately drops
every data row and emits only the labels it still trusts. The verdict is driven
by _is_weak_table_cell(), which classified ANY purely numeric cell as weak, and
by _is_meaningful_table_cell(), which demanded letters or CJK. A wide numeric
table was therefore structurally guaranteed to be read as noise.

Measured on the 2026-08-07 corpus: 7 tables across 4 economic reports were
classified low-confidence. Every one had zero garbled-OCR markers, and the
"weak" cells were legitimate figures — 26,045 and 22.40 (US imports from Taiwan
and its share), 3,337,229 and 537,990 (revenue), 3.3 / 4.3 / -0.1 (IMF growth
forecasts). Those numbers reached none of rag.md, structured_rag.md,
chunks.jsonl or structured_chunks.jsonl, and their absence then tripped
structured_output_empty, the VLM audit and a ~100k-token reviewer pass.

The discriminator is the SHAPE of the numbers. A report's figures are
well-formed (decimals, thousands separators, multiple digits); garbled OCR
scatters lone digits and fragments like ".7" through a layout grid.
"""

import unittest

from app.pipeline.package_utils import semantic_table_to_text, table_rows_are_low_confidence

# Verbatim from doc 9a668e60d50d76e8 block b000035 (IMF outlook, run
# 01KZDNKM6DS946GSRZEZABEBNA) — well-formed MinerU HTML, 10 columns.
IMF_TABLE_ROWS = [
    ["項目", "全球", "", "", "主要經濟體", "", "", "新興市場及開發中經濟體", "", ""],
    ["項目", "2024年", "2025年(e)", "2026年(f)", "2024年", "2025年(e)", "2026年(f)", "2024年", "2025年(e)", "2026年(f)"],
    ["實質GDP成長率", "3.3", "3.2", "3.1", "1.8", "1.6", "1.6", "4.3", "4.2", "4.0"],
    ["經常帳順（逆）差／GDP", "0.4", "0.4", "0.3", "0.1", "-0.1", "0.1", "1.0", "1.0", "0.6"],
    ["消費者物價上漲率", "5.8", "4.2", "3.7", "2.6", "2.5", "2.2", "7.9", "5.3", "4.7"],
    ["失業率（主要經濟體）", "-", "-", "-", "4.6", "4.7", "4.7", "-", "-", "-"],
    ["政府負債/GDP", "92.4", "94.7", "96.8", "109.1", "110.2", "111.8", "69.0", "72.7", "75.8"],
]

# Verbatim shape from doc ba4242775667a51c block b000028 (trade statistics).
TRADE_TABLE_ROWS = [
    ["HS碼", "產品名稱", "金額", "占比", "排名", "占該項進口比重"],
    ["8471", "自動資料處理機", "26,045", "22.40", "3", "18.59"],
    ["8473", "零附件", "25,030", "21.53", "2", "17.12"],
    ["8542", "積體電路", "12,880", "11.08", "5", "9.44"],
]

# The failure this heuristic exists for: a form scanned so badly that the parser
# spread lone digits and punctuation across a grid.
GARBLED_OCR_ROWS = [
    ["圖隊填表前精樣閱本表下方之填表注意事項", "3", ".7", "驗", ""],
    ["", "1", "□", "、", "2"],
    ["範，惟仍應事先向所屬單位主管報備。", "", "4", ".", "5"],
    ["", "6", "7", "8", "9"],
]


class LowConfidenceVerdictTest(unittest.TestCase):
    def test_imf_forecast_table_is_not_ocr_noise(self):
        self.assertFalse(table_rows_are_low_confidence(IMF_TABLE_ROWS))

    def test_trade_statistics_table_is_not_ocr_noise(self):
        self.assertFalse(table_rows_are_low_confidence(TRADE_TABLE_ROWS))

    def test_genuinely_garbled_scan_is_still_ocr_noise(self):
        """The lone digits and ".7"/"□"/"、" fragments must keep tripping it."""
        self.assertTrue(table_rows_are_low_confidence(GARBLED_OCR_ROWS))


class DeliveredTableTextTest(unittest.TestCase):
    """The numbers have to survive into the text the RAG index consumes."""

    @staticmethod
    def _html(rows: list[list[str]]) -> str:
        body = "".join(
            "<tr>" + "".join(f"<td rowspan=1 colspan=1>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return f"<table>{body}</table>"

    def test_forecast_values_reach_the_rendered_text(self):
        text = semantic_table_to_text(self._html(IMF_TABLE_ROWS), "IMF 全球經濟預測")

        self.assertNotIn("低可信度表格 OCR", text)
        for value in ("3.3", "4.3", "-0.1", "92.4", "111.8"):
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_trade_values_and_their_labels_stay_together(self):
        text = semantic_table_to_text(self._html(TRADE_TABLE_ROWS), "美國自臺進口主要產品")

        self.assertNotIn("低可信度表格 OCR", text)
        for value in ("26,045", "22.40", "18.59"):
            with self.subTest(value=value):
                self.assertIn(value, text)
        self.assertIn("自動資料處理機", text)

    def test_garbled_scan_still_refuses_to_invent_rows(self):
        text = semantic_table_to_text(self._html(GARBLED_OCR_ROWS), "表單 OCR 片段")

        self.assertIn("低可信度表格 OCR", text)
