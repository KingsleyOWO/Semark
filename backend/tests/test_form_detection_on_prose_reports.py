"""A research report with a bibliography is not a fillable form.

is_form_like_document() scores the text returned by _document_table_rows(),
which includes every TEXT block — not just table cells — so a twenty-page report
is measured with thresholds calibrated for a one-page form. Two of its signals
then misfire on ordinary academic prose:

  * colon_label_score counts "Word:" as a field label, so a reference list of
    `DOI:` / `https:` entries and colon-terminated citation titles
    («The General Agreement on Trade in Services:») reads as a form's labels.
  * form_name_score matches 申請單 inside 申請單位 — "applicant *unit*", a
    routine phrase — so any document naming a department scores a form name.

Live on the 2026-08-07 corpus: 4 of 100 documents were flagged form-like on this
evidence alone. Each raised form_like_document_not_structured, and the
form_like_text signal also blocked the generic_document exemption in
_check_structured_output_presence, so each additionally raised
structured_output_empty (high) and its downstream VLM-audit issue.
"""

import unittest

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.structured_rag import is_form_like_document


def _document(paragraphs: list[str], *, path: str = "報告.pdf", ext: str = "pdf") -> DocumentIR:
    return DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path=path, ext=ext, sha256="abc", size_bytes=1000),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id=f"b{idx:03d}",
                type=BlockType.TEXT,
                page_idx=0,
                reading_order=idx,
                payload={"text": text},
            )
            for idx, text in enumerate(paragraphs)
        ],
    )


# Shape taken from the live reports: Chinese body prose, one incidental English
# form word, and a reference list. Institutions are fictional.
REPORT_WITH_BIBLIOGRAPHY = [
    "本報告回顧2025年全球經濟情勢，並展望2026年的政策風險與產業動能。",
    "由於採紀錄與宣告(Book and Claim)原則，憑證的交易與實體移轉可以分離。",
    "## 參考文獻",
    "示範研究院，2026/01，全球景氣回顧，DOI:10.29656/DEMO.202601",
    "Demo Council, Principles for 6G: A Joint Statement, https://example.org/a",
    "Demo Agency, The Framework for IMT-2030: Objectives, https://example.org/b",
    "Demo Institute, Wireless Innovation Fund: Progress Update, https://example.org/c",
    "Demo Board, Annex B: Methodology Notes, https://example.org/d",
    "Demo Group, Trade Facilitation Indicators: 2025 Update, https://example.org/e",
    "Demo Secretariat, The General Agreement on Trade in Services: Review, https://example.org/f",
]

# 申請單位 = "applicant unit"; the document is a procedural notice, not a form.
NOTICE_NAMING_A_DEPARTMENT = [
    "為推動計畫執行，申請單位應於期限內完成內部審查。",
    "計畫金額由申請單位自行編列，合計不得超過核定額度。",
    "審查結果將函知申請單位，並副知相關單位。",
]

# A real form: the form name stands alone and the field labels are dense.
GENUINE_TRAVEL_FORM = [
    "出差申請單",
    "申請人：　　　　申請單位：　　　　申請日期：",
    "出差事由：",
    "起始地點：　　　　到達地點：",
    "金額：　　　　合計：",
    "單位主管：　　　　簽章：",
]


class ProseReportIsNotAFormTest(unittest.TestCase):
    def test_report_with_reference_list_is_not_form_like(self):
        self.assertFalse(is_form_like_document(_document(REPORT_WITH_BIBLIOGRAPHY)))

    def test_applicant_unit_in_prose_is_not_a_form_name(self):
        self.assertFalse(is_form_like_document(_document(NOTICE_NAMING_A_DEPARTMENT)))


class GenuineFormStillDetectedTest(unittest.TestCase):
    """The corrections must not cost us the detection the signal exists for."""

    def test_travel_application_form_is_still_form_like(self):
        self.assertTrue(is_form_like_document(_document(GENUINE_TRAVEL_FORM, path="出差申請單.pdf")))

    def test_english_form_with_dense_field_labels_is_still_form_like(self):
        english_form = [
            "Request for Transcript of Tax Return",
            "Name shown on return:",
            "Address:",
            "Phone:",
            "Signature:",
            "Date signed:",
        ]

        self.assertTrue(is_form_like_document(_document(english_form, path="form-4506.pdf")))
