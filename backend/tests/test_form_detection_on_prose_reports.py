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
    "示範研究院，2026/01，全球景氣回顧，DOI:10.12345/DEMO.202601",
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


def _word(blocks: list[tuple[str, str]], *, path: str) -> DocumentIR:
    """A native Word document: no page images or boxes, so every block is on page 0."""
    return DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path=path, ext="docx", sha256="abc", size_bytes=1000),
        engine=EngineInfo(backend="native_docx", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id=f"b{idx:03d}",
                type=BlockType.TABLE if kind == "table" else BlockType.TEXT,
                page_idx=0,
                reading_order=idx,
                payload={"table_body": body} if kind == "table" else {"text": body},
            )
            for idx, (kind, body) in enumerate(blocks)
        ],
    )


# Shape of the 2026-10 brand coaching reports: tens of thousands of characters,
# a budget table with 金額/合計, and one passing mention of a 檢核表.
LONG_WORD_REPORT = [
    ("text", "示範企業品牌輔導結案報告書"),
    *[("text", f"第{n}節 品牌定位與市場策略說明，輔導團隊與企業共同盤點品牌現況並提出改善建議。" * 12) for n in range(30)],
    ("text", "輔導過程依品牌健檢檢核表逐項確認，並增加單位內部溝通頻率。"),
    ("table", "<table><tr><td>項目</td><td>金額</td><td>合計</td></tr>"
              "<tr><td>人事費</td><td>120,000</td><td>120,000</td></tr>"
              "<tr><td>負責人姓名</td><td>王示範</td><td></td></tr></table>"),
]

SHORT_WORD_FORM = [
    ("text", "出差申請單"),
    ("table", "<table><tr><td>申請人</td><td></td><td>申請日期</td><td></td></tr>"
              "<tr><td>出差事由</td><td></td><td>金額</td><td></td></tr>"
              "<tr><td>單位主管</td><td></td><td>簽章</td><td></td></tr></table>"),
]


class NativeWordReportIsNotAFormTest(unittest.TestCase):
    def test_long_word_report_is_not_delivered_as_a_form(self):
        from app.pipeline.structured_rag import build_form_documents_rag

        report = _word(LONG_WORD_REPORT, path="示範企業結案報告書.docx")

        self.assertFalse(is_form_like_document(report))
        output = build_form_documents_rag(report, {}, semantic_output_language="zh-TW")
        self.assertEqual(output.plan.document_type, "generic_document")
        self.assertEqual(output.records, [])

    def test_short_word_form_is_still_a_form(self):
        from app.pipeline.structured_rag import build_form_documents_rag

        form = _word(SHORT_WORD_FORM, path="出差申請單.docx")

        self.assertTrue(is_form_like_document(form))
        output = build_form_documents_rag(form, {}, semantic_output_language="zh-TW")
        self.assertIn(output.plan.document_type, {"form_document", "form_collection"})
        self.assertTrue(output.records)

    def test_unit_names_ending_in_dan_wei_are_not_form_names(self):
        notice = [
            "各申報單位應於期限內增加單位內部稽核，異動單位須另行函報。",
            "金額與合計由各單位自行填列。",
        ]

        self.assertFalse(is_form_like_document(_document(notice)))


def _form_enrichment(block_id: str) -> dict:
    return {block_id: {
        "kind": "form_asset",
        "input": {"page_idx": 0},
        "output": {"title": "企業發展重要記事", "document_type": "form",
                   "structured_content": "表單語意：企業發展重要記事"},
    }}


class LongWordReportKeepsItsTextTest(unittest.TestCase):
    """A form read from one table of a long Word report must not stand in for the report."""

    def test_rag_text_keeps_every_paragraph_of_a_long_word_report(self):
        from app.pipeline.stages.package import PackageStage

        report = _word(LONG_WORD_REPORT, path="示範企業結案報告書.docx")
        table_id = report.blocks[-1].block_id

        rag_md, _ = PackageStage()._render_rag_md(
            document_ir=report, asset_map={}, enrichments=_form_enrichment(table_id))

        self.assertIn("第29節 品牌定位與市場策略說明", rag_md)
        self.assertIn("輔導過程依品牌健檢檢核表逐項確認", rag_md)
        self.assertIn("120,000", rag_md)

    def test_a_short_word_form_still_reads_as_its_form(self):
        from app.pipeline.stages.package import PackageStage

        form = _word(SHORT_WORD_FORM, path="出差申請單.docx")

        rag_md, _ = PackageStage()._render_rag_md(
            document_ir=form, asset_map={}, enrichments=_form_enrichment(form.blocks[-1].block_id))

        self.assertIn("表單語意：企業發展重要記事", rag_md)
