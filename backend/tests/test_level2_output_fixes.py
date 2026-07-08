"""Level 2 output-quality fixes surfaced by the first clean golden E2E:
B1 raw-OCR-dump leak on zh-TW forms, B2 missing heading_path on the dominant
(structured-repair/fallback) chunk path, B3 over-severe empty-structured gate."""

import unittest

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.quality_gate import _check_structured_output_presence
from app.pipeline.stages.package import PackageStage
from app.pipeline.structured_rag import _form_supplementary_fact_lines


def _doc(blocks, filename="x.pdf"):
    return DocumentIR(
        doc_id="d",
        run_id="r",
        source=SourceInfo(path=filename, ext=".pdf", sha256="x", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=blocks,
    )


# --- B1: zh-TW forms must not dump raw OCR when structured content exists -----


class SourceTextDumpTest(unittest.TestCase):
    def _lines(self, all_text, guide="", fields=("申請人",)):
        return _form_supplementary_fact_lines(
            all_text,
            fields=[{"name": n} for n in fields],
            output={"all_text": all_text, "filling_guide": guide},
        )

    def test_redundant_lines_are_dropped(self):
        # Every value already in the guide → nothing worth keeping (no residue).
        kept = self._lines(
            ["申請日期 114.12.11", "金額 500"],
            guide="版本日期 114.12.11，核銷金額 500 元由申請人填寫。",
        )
        self.assertEqual(kept, [])

    def test_uncovered_amount_line_is_kept(self):
        # NT$100,000 is only in the OCR text — its line must survive.
        kept = self._lines(["財力證明 NT$100,000"], guide="由申請人填寫基本資料。")
        self.assertEqual(kept, ["財力證明 NT$100,000"])

    def test_uncovered_legal_reference_line_is_kept(self):
        # 第八條 the field extractor missed must survive (the real 2-12 case).
        kept = self._lines(["依本院人員訓練辦法第八條規定辦理"], guide="由申請人填寫。")
        self.assertEqual(kept, ["依本院人員訓練辦法第八條規定辦理"])

    def test_blank_field_labels_are_dropped_but_facts_kept(self):
        kept = self._lines(
            ["申請單位：", "申請日期： 年 月 日", "姓名(員工編號)： ( )", "依第八條辦理", "版本 114.12.11"],
            guide="本表單版本日期 114.12.11 由申請單位填寫。",
        )
        self.assertEqual(kept, ["依第八條辦理"])  # blank labels + covered date dropped

    def test_no_all_text_returns_empty(self):
        self.assertEqual(self._lines([], guide="x"), [])


# --- B2: heading_path on the structured-repair/fallback chunk path -----------


class HeadingSectionsTest(unittest.TestCase):
    def test_sections_carry_full_heading_path(self):
        md = (
            "# 示範研究院人員請假辦法\n\n"
            "## 第一條\n" + "本辦法依本院人員管理辦法訂定之。" * 24 + "\n\n"
            "## 第八條\n" + "請假應依規定辦理。" * 40 + "\n"
        )
        sections = PackageStage._heading_sections(md)
        paths = [p for _, p in sections]
        self.assertIn(["示範研究院人員請假辦法", "第一條"], paths)
        self.assertIn(["示範研究院人員請假辦法", "第八條"], paths)

    def test_subsplit_chunks_inherit_section_path(self):
        body = "條文內容。" * 600  # 3000 chars, forces a max_chars sub-split
        md = f"# 規則\n\n## 長條\n{body}\n"
        sections = PackageStage._heading_sections(md)
        self.assertGreater(len(sections), 1)
        for _, path in sections:
            self.assertEqual(path, ["規則", "長條"])

    def test_split_into_chunks_still_returns_plain_text(self):
        md = "# A\n\n## B\n" + "x" * 300
        chunks = PackageStage()._split_repair_markdown_into_chunks(md)
        self.assertTrue(all(isinstance(c, str) for c in chunks))


# --- B3: empty-structured gate severity tracks signal strength ---------------


class StructuredEmptySeverityTest(unittest.TestCase):
    def _issue(self, blocks, enrichments, filename="x.pdf"):
        issues = _check_structured_output_presence(
            _doc(blocks, filename),
            structured_output=None,
            structured_text="",
            source_md="來源文字內容" * 20,
            enrichments=enrichments,
        )
        return issues[0] if issues else None

    def test_table_document_empty_structured_is_high(self):
        issue = self._issue(
            [Block(block_id="t", type=BlockType.TABLE, page_idx=0, bbox_norm=[0, 0, 1000, 500], payload={})],
            enrichments={},
        )
        self.assertEqual(issue.severity, "high")

    def test_form_page_document_empty_structured_still_flags(self):
        # A form page with empty structured output is a real extraction miss
        # and stays high severity — the naive "downgrade prose false alarms"
        # fix would mask true form-extraction failures, so it is deferred (L3).
        issue = self._issue(
            [Block(block_id="tx", type=BlockType.TEXT, page_idx=0, bbox_norm=[0, 0, 1000, 100], payload={"text": "第一條"})],
            enrichments={"form_page_0011": {"kind": "form_asset", "output": {"title": "請假單"}}},
            filename="請假辦法.pdf",
        )
        self.assertIsNotNone(issue)
        self.assertEqual(issue.severity, "high")


if __name__ == "__main__":
    unittest.main()
