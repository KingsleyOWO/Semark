"""Level 2 output-quality fixes surfaced by the first clean golden E2E:
B1 raw-OCR-dump leak on zh-TW forms, B2 missing heading_path on the dominant
(structured-repair/fallback) chunk path, B3 over-severe empty-structured gate."""

import unittest

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.quality_gate import _check_structured_output_presence
from app.pipeline.stages.package import PackageStage
from app.pipeline.structured_rag import _should_emit_form_source_text_record


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
    def test_redundant_dump_is_skipped(self):
        # All the dump's facts already live in the guide → it is pure noise
        # ("來源抽取文字" residue) and must be dropped from the shipped form.
        emit = _should_emit_form_source_text_record(
            language="zh-TW",
            output={
                "all_text": ["申請日期 114.12.11", "金額 500"],
                "filling_guide": "版本日期 114.12.11，核銷金額 500 元由申請人填寫。",
            },
            fields=[{"name": "申請人"}],
        )
        self.assertFalse(emit)

    def test_dump_with_uncovered_fact_is_kept(self):
        # NT$100,000 appears only in the OCR text, not in fields/guide —
        # dropping the dump would lose a real fact, so it must be kept.
        emit = _should_emit_form_source_text_record(
            language="zh-TW",
            output={
                "all_text": ["財力證明 NT$100,000"],
                "filling_guide": "由申請人填寫基本資料。",
            },
            fields=[{"name": "申請人"}],
        )
        self.assertTrue(emit)

    def test_no_all_text_is_not_emitted(self):
        self.assertFalse(
            _should_emit_form_source_text_record(
                language="zh-TW", output={"all_text": [], "filling_guide": "x"}, fields=[{"name": "a"}]
            )
        )

    def test_blank_field_labels_do_not_keep_dump(self):
        # A blank form's dump is just empty field labels (申請日期：年 月 日) —
        # structure, not values. With dates covered by the guide it must be
        # skipped, not retained as "來源抽取文字" noise.
        emit = _should_emit_form_source_text_record(
            language="zh-TW",
            output={
                "all_text": ["申請單位：", "申請日期： 年 月 日", "姓名(員工編號)： ( )", "版本 114.12.11"],
                "filling_guide": "本表單版本日期 114.12.11，由申請單位與申請人填寫。",
            },
            fields=[{"name": "申請人"}, {"name": "申請單位"}],
        )
        self.assertFalse(emit)


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
