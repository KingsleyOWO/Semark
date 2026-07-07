"""Safety tests for the semantic-repair reviewer path.

Covers Level 1 output-quality fixes:
1. The reviewer is grounded with the rendered page image (not blind).
2. Repairs that drop fact tokens from the pre-repair markdown are rejected.
3. After an applied repair the rule-based quality gate is re-run instead of
   rubber-stamping the gate state.
"""

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

if importlib.util.find_spec("pydantic_settings") is None:
    raise unittest.SkipTest("pydantic_settings is required to import package stage")

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.stages.package import PackageStage


class RecordingRepairAdapter:
    """Fake reviewer adapter that records enrich() kwargs."""

    def __init__(self, repaired_markdown: str, summary: str = "rewrote document"):
        self.calls: list[dict] = []
        self._repaired_markdown = repaired_markdown
        self._summary = summary

    async def enrich(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            output={
                "status": "repaired",
                "repaired_markdown": self._repaired_markdown,
                "summary": self._summary,
                "applied_repairs": ["rewrite_semantic_markdown"],
                "confidence": 0.9,
            },
            tokens_used=42,
            duration_seconds=0.1,
        )


def _make_document_ir(pages: list[PageInfo], text: str = "Authorization form fields and signature.") -> DocumentIR:
    return DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="authorization.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=pages,
        blocks=[
            Block(
                block_id="b0",
                type=BlockType.TEXT,
                page_idx=pages[0].page_idx if pages else 0,
                payload={"text": text},
            )
        ],
    )


def _write_form_fixture(outputs: Path, *, form_markdown: str, page_indices: list[int]) -> Path:
    forms = outputs / "forms"
    docs = outputs / "documents"
    forms.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    form_md = forms / "form_0000.md"
    form_md.write_text(form_markdown, encoding="utf-8")
    (docs / "form_0000.md").write_text(form_markdown, encoding="utf-8")
    (outputs / "forms_index.json").write_text(
        json.dumps(
            [
                {
                    "form_id": "form_0000",
                    "subdoc_id": "doc:form:0000",
                    "title": "Authorization Form",
                    "page_indices": page_indices,
                    "page_label": "Page 1",
                    "field_count": 2,
                    "files": {"markdown": str(form_md)},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "form_0000",
                    "kind": "form",
                    "title": "Authorization Form",
                    "source_filename": "authorization.pdf",
                    "file": str(docs / "form_0000.md"),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (outputs / "structured_chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "old",
                "doc_id": "doc:form:0000",
                "content": "old parser fallback",
                "metadata": {"subdoc_id": "doc:form:0000"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return form_md


FORM_QUALITY_GATE = SimpleNamespace(
    issues=[
        SimpleNamespace(
            code="merged_field_detected",
            severity="warning",
            message="Field labels are merged.",
            page_idx=None,
            evidence={},
        )
    ],
    stats={"semantic_quality": {"rag_readiness_score": 0.8, "recommended_repairs": ["split_merged_fields"]}},
)

GOOD_FORM_REWRITE = (
    "# Authorization Form\n\n"
    "## Purpose\n"
    "Use this form to authorize disclosure of records for a specific request.\n\n"
    "## Fields\n"
    "- Name of the requester\n"
    "- Signature and the date signed\n"
)


class SemanticRepairPageImageTest(unittest.TestCase):
    def test_form_repair_reviewer_receives_first_evidence_page_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            outputs = run_path / "outputs"
            pages_dir = run_path / "assets" / "pages"
            pages_dir.mkdir(parents=True)
            (pages_dir / "page_0000.png").write_bytes(b"png0")
            (pages_dir / "page_0001.png").write_bytes(b"png1")
            document_ir = _make_document_ir(
                [
                    PageInfo(page_idx=0, page_image_path="assets/pages/page_0000.png"),
                    PageInfo(page_idx=1, page_image_path="assets/pages/page_0001.png"),
                ]
            )
            _write_form_fixture(outputs, form_markdown="# Authorization Form\n\nOld body", page_indices=[1])

            adapter = RecordingRepairAdapter(GOOD_FORM_REWRITE)
            asyncio.run(
                PackageStage()._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Authorization Form",
                    structured_output=SimpleNamespace(plan=SimpleNamespace(document_type="form_collection")),
                    quality_gate=FORM_QUALITY_GATE,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=adapter,
                    run_path=run_path,
                )
            )

            self.assertEqual(len(adapter.calls), 1)
            self.assertEqual(adapter.calls[0]["image_path"], pages_dir / "page_0001.png")

    def test_structured_repair_reviewer_receives_first_document_page_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            outputs = run_path / "outputs"
            outputs.mkdir(parents=True)
            pages_dir = run_path / "assets" / "pages"
            pages_dir.mkdir(parents=True)
            (pages_dir / "page_0000.png").write_bytes(b"png0")
            document_ir = _make_document_ir(
                [PageInfo(page_idx=0, page_image_path="assets/pages/page_0000.png")]
            )

            adapter = RecordingRepairAdapter(GOOD_FORM_REWRITE)
            asyncio.run(
                PackageStage()._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Authorization Form",
                    structured_output=SimpleNamespace(
                        plan=SimpleNamespace(document_type="generic_document", title="Authorization Form"),
                        rag_markdown="# Authorization Form\n\nOld structured body",
                    ),
                    quality_gate=FORM_QUALITY_GATE,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=adapter,
                    run_path=run_path,
                )
            )

            self.assertEqual(len(adapter.calls), 1)
            self.assertEqual(adapter.calls[0]["image_path"], pages_dir / "page_0000.png")

    def test_repair_without_rendered_page_image_stays_text_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            outputs = run_path / "outputs"
            outputs.mkdir(parents=True)
            document_ir = _make_document_ir([PageInfo(page_idx=0)])

            adapter = RecordingRepairAdapter(GOOD_FORM_REWRITE)
            asyncio.run(
                PackageStage()._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Authorization Form",
                    structured_output=SimpleNamespace(
                        plan=SimpleNamespace(document_type="generic_document", title="Authorization Form"),
                        rag_markdown="# Authorization Form\n\nOld structured body",
                    ),
                    quality_gate=FORM_QUALITY_GATE,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=adapter,
                    run_path=run_path,
                )
            )

            self.assertEqual(len(adapter.calls), 1)
            self.assertIsNone(adapter.calls[0]["image_path"])


ZH_FACT_MARKDOWN = (
    "# 差旅費申請單\n\n"
    "## 表單用途\n"
    "出差人依第15條規定，於 114.12.11 前提出申請。\n\n"
    "## 欄位說明\n"
    "- 申請人：出差本人簽名\n"
    "- 補助金額：上限 NT$1,200（８０％）\n"
)

ZH_FACT_DROPPING_REWRITE = (
    "# 差旅費申請單\n\n"
    "## 表單用途\n"
    "本表單用於出差申請流程說明，適用全體員工與外聘顧問。\n\n"
    "## 填寫重點\n"
    "- 請完整填寫個人資料與行程資訊，並附上相關證明文件。\n"
    "- 送交單位主管審核後由人資歸檔保存。\n"
)

EN_FEE_MARKDOWN = (
    "# Fee Schedule\n\n"
    "## Fees\n"
    "- Application fee is NT$3,500 (12%)\n"
    "- Deadline: 2026.01.15\n"
)

EN_FEE_DROPPING_REWRITE = (
    "# Fee Schedule\n\n"
    "## Overview\n"
    "- The schedule lists application fees for the visa program.\n"
    "- Contact the service desk for the current amounts and dates.\n"
)

EN_FEE_PRESERVING_REWRITE = (
    "# Fee Schedule\n\n"
    "## Fees\n"
    "- Application fee: NT$3500 (12 %)\n"
    "- Deadline: 2026.01.15\n"
    "- Submit the application early to avoid processing delays.\n"
)


class RepairGuardFunctionTest(unittest.TestCase):
    def test_repair_guard_passes_when_fact_tokens_survive(self):
        from app.pipeline.repair_guard import repair_preserves_facts

        repaired = (
            "# 差旅費申請單\n\n"
            "## 欄位\n"
            "- 申請人：需由出差本人填寫\n"
            "- 補助金額：上限 1200 元（80%）\n"
            "- 於 114.12.11 前送出\n\n"
            "## 依據\n"
            "- 依第 15 條規定辦理\n"
        )

        ok, details = repair_preserves_facts(ZH_FACT_MARKDOWN, "", repaired)

        self.assertTrue(ok)
        self.assertEqual(details["missing_tokens"], [])
        self.assertGreater(details["checked_token_count"], 0)

    def test_repair_guard_rejects_rewrite_that_drops_fact_tokens(self):
        from app.pipeline.repair_guard import repair_preserves_facts

        ok, details = repair_preserves_facts(ZH_FACT_MARKDOWN, "", ZH_FACT_DROPPING_REWRITE)

        self.assertFalse(ok)
        self.assertIn("114.12.11", details["missing_tokens"])
        self.assertIn("第15條", details["missing_tokens"])
        self.assertIn("申請人", details["missing_tokens"])
        self.assertLess(details["survival_ratio"], 0.9)

    def test_repair_guard_ignores_evidence_only_tokens(self):
        from app.pipeline.repair_guard import repair_preserves_facts

        original = "# 通知\n\n- 生效日：114.12.11\n"
        evidence = "第99條 費率 3.75% 承辦人：李四"
        repaired = "# 通知\n\n- 生效日：114.12.11（自公告日起適用）\n"

        ok, details = repair_preserves_facts(original, evidence, repaired)

        self.assertTrue(ok)
        self.assertEqual(details["missing_tokens"], [])

    def test_repair_guard_passes_documents_without_fact_tokens(self):
        from app.pipeline.repair_guard import repair_preserves_facts

        ok, details = repair_preserves_facts(
            "# Notes\n\nGeneral guidance without figures.",
            "",
            "# Notes\n\nRewritten guidance, still without figures.",
        )

        self.assertTrue(ok)
        self.assertEqual(details["checked_token_count"], 0)
        self.assertEqual(details["survival_ratio"], 1.0)

    def test_repair_guard_threshold_allows_ten_percent_loss(self):
        from app.pipeline.repair_guard import repair_preserves_facts

        numbers = ["121", "232", "343", "454", "565", "676", "787", "898", "919", "242"]
        original = "# 資料\n\n" + "\n".join(f"- 值 {value}" for value in numbers)
        keep_nine = "# 資料\n\n" + "\n".join(f"- 值 {value}" for value in numbers[:9])
        keep_eight = "# 資料\n\n" + "\n".join(f"- 值 {value}" for value in numbers[:8])

        ok_nine, _ = repair_preserves_facts(original, "", keep_nine)
        ok_eight, details_eight = repair_preserves_facts(original, "", keep_eight)

        self.assertTrue(ok_nine)
        self.assertFalse(ok_eight)
        self.assertIn("242", details_eight["missing_tokens"])
        self.assertIn("919", details_eight["missing_tokens"])


class RepairGuardWiringTest(unittest.TestCase):
    def test_fact_losing_form_repair_is_rejected_and_marked_needs_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            document_ir = _make_document_ir([PageInfo(page_idx=0)], text="差旅費申請單 申請人 補助金額")
            form_md = _write_form_fixture(outputs, form_markdown=ZH_FACT_MARKDOWN, page_indices=[0])

            adapter = RecordingRepairAdapter(ZH_FACT_DROPPING_REWRITE)
            stats = asyncio.run(
                PackageStage()._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# 差旅費申請單",
                    structured_output=SimpleNamespace(plan=SimpleNamespace(document_type="form_collection")),
                    quality_gate=FORM_QUALITY_GATE,
                    enrichments={},
                    semantic_output_language="zh-TW",
                    review_adapter=adapter,
                )
            )

            self.assertEqual(stats["applied_count"], 0)
            self.assertEqual(stats["fallback_count"], 1)
            item = stats["items"][0]
            self.assertEqual(item["status"], "fallback_retained")
            self.assertEqual(item["reason"], "fact_loss")
            self.assertEqual(item["repair_rejected_reason"], "fact_loss")
            self.assertIn("114.12.11", item["missing_fact_tokens"])
            form_text = form_md.read_text(encoding="utf-8")
            self.assertIn("114.12.11", form_text)
            self.assertNotIn("填寫重點", form_text)
            chunks = [
                json.loads(line)
                for line in (outputs / "structured_chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(chunks)
            for chunk in chunks:
                metadata = chunk["metadata"]
                self.assertTrue(metadata["needs_review"])
                self.assertFalse(metadata["auto_rag_ready"])
                self.assertEqual(metadata["semantic_repair_reason"], "fact_loss")

    def test_fact_losing_structured_repair_keeps_pre_repair_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            document_ir = _make_document_ir([PageInfo(page_idx=0)], text="Fee schedule for applications.")

            adapter = RecordingRepairAdapter(EN_FEE_DROPPING_REWRITE)
            stats = asyncio.run(
                PackageStage()._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Fee Schedule",
                    structured_output=SimpleNamespace(
                        plan=SimpleNamespace(document_type="generic_document", title="Fee Schedule"),
                        rag_markdown=EN_FEE_MARKDOWN,
                    ),
                    quality_gate=FORM_QUALITY_GATE,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=adapter,
                )
            )

            self.assertEqual(stats["applied_count"], 0)
            self.assertEqual(stats["fallback_count"], 1)
            item = stats["items"][0]
            self.assertEqual(item["reason"], "fact_loss")
            self.assertEqual(item["repair_rejected_reason"], "fact_loss")
            self.assertIn("2026.01.15", item["missing_fact_tokens"])
            retained = (outputs / "structured_rag.md").read_text(encoding="utf-8")
            self.assertIn("NT$3,500", retained)
            self.assertNotIn("service desk", retained)
            chunks_text = (outputs / "structured_chunks.jsonl").read_text(encoding="utf-8")
            self.assertIn("fact_loss", chunks_text)

    def test_fact_preserving_structured_repair_still_applies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            document_ir = _make_document_ir([PageInfo(page_idx=0)], text="Fee schedule for applications.")

            adapter = RecordingRepairAdapter(EN_FEE_PRESERVING_REWRITE)
            stats = asyncio.run(
                PackageStage()._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Fee Schedule",
                    structured_output=SimpleNamespace(
                        plan=SimpleNamespace(document_type="generic_document", title="Fee Schedule"),
                        rag_markdown=EN_FEE_MARKDOWN,
                    ),
                    quality_gate=FORM_QUALITY_GATE,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=adapter,
                )
            )

            self.assertEqual(stats["applied_count"], 1)
            self.assertEqual(stats["items"][0]["status"], "applied")
            self.assertIn("NT$3500", (outputs / "structured_rag.md").read_text(encoding="utf-8"))


CLEAN_REPAIRED_MARKDOWN = (
    "# Policy\n\n"
    "## Scope\n"
    "- Applies to all staff of the institute.\n\n"
    "## Rules\n"
    "- Submit requests five days in advance.\n"
)


def _make_issue(code: str, severity: str) -> SimpleNamespace:
    return SimpleNamespace(code=code, severity=severity, message="", page_idx=None, evidence={})


def _make_gate(issues: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        status="needs_review",
        score=0.25,
        issues=issues,
        vlm_audit_candidates=[{"page_idx": 0, "reasons": ["structured_output_empty"]}],
        vlm_audits=[{"success": False}],
        stats={
            "issue_count": len(issues),
            "semantic_quality": {"rag_readiness_score": 0.5, "recommended_repairs": ["compress_summary_without_ellipsis"]},
        },
    )


def _make_settle_fixture(run_path: Path, *, write_repaired: bool = True) -> tuple[Path, DocumentIR, SimpleNamespace]:
    outputs = run_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    if write_repaired:
        (outputs / "structured_rag.md").write_text(CLEAN_REPAIRED_MARKDOWN, encoding="utf-8")
    document_ir = DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="policy.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/page_0000.png")],
        blocks=[
            Block(
                block_id="b0",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Policy overview describing scope and submission rules."},
            )
        ],
    )
    structured_output = SimpleNamespace(
        plan=SimpleNamespace(document_type="generic_document", title="Policy"),
        records=[],
        chunks=[],
        rag_markdown="# Policy\n\nOld body with trailing ellipsis ...",
    )
    return outputs, document_ir, structured_output


def _run_settle(quality_gate, semantic_repair_stats, *, run_path, outputs, document_ir, structured_output, enrichments):
    asyncio.run(
        PackageStage()._settle_quality_gate_after_semantic_repair(
            quality_gate,
            semantic_repair_stats,
            outputs_dir=outputs,
            document_ir=document_ir,
            source_md="# Policy\n\nRaw parser text ...",
            assets=[],
            structured_output=structured_output,
            enrichments=enrichments,
            run_path=run_path,
            semantic_output_language="en",
        )
    )


class PostRepairRecheckTest(unittest.TestCase):
    def test_post_repair_recheck_keeps_unresolved_issues_and_blocks_auto_rag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            outputs, document_ir, structured_output = _make_settle_fixture(run_path)
            enrichments = {
                "b0": {
                    "kind": "form_asset",
                    "output": {"_error": "JSON_PARSE_FAILED: truncated reply"},
                    "evidence": {"page_idx": 0},
                    "input": {},
                    "quality": {},
                }
            }
            quality_gate = _make_gate(
                [
                    _make_issue("vlm_enrichment_parse_failed", "high"),
                    _make_issue("summary_contains_ellipsis", "warning"),
                ]
            )

            _run_settle(
                quality_gate,
                {"applied_count": 1, "fallback_count": 0, "blocked_count": 0},
                run_path=run_path,
                outputs=outputs,
                document_ir=document_ir,
                structured_output=structured_output,
                enrichments=enrichments,
            )

            self.assertEqual(quality_gate.status, "needs_review")
            self.assertEqual([issue.code for issue in quality_gate.issues], ["vlm_enrichment_parse_failed"])
            self.assertFalse(quality_gate.stats["auto_rag_ready"])
            recheck = quality_gate.stats["post_repair_recheck"]
            self.assertEqual(recheck["issue_count"], 1)
            self.assertIn("summary_contains_ellipsis", recheck["cleared"])
            self.assertIn("vlm_enrichment_parse_failed", recheck["remaining"])
            self.assertLess(quality_gate.score, 1.0)

    def test_post_repair_recheck_clears_issues_only_when_rules_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            outputs, document_ir, structured_output = _make_settle_fixture(run_path)
            quality_gate = _make_gate([_make_issue("summary_contains_ellipsis", "warning")])

            _run_settle(
                quality_gate,
                {"applied_count": 1, "fallback_count": 0, "blocked_count": 0},
                run_path=run_path,
                outputs=outputs,
                document_ir=document_ir,
                structured_output=structured_output,
                enrichments={},
            )

            self.assertEqual(quality_gate.status, "pass")
            self.assertEqual(quality_gate.issues, [])
            self.assertEqual(quality_gate.score, 1.0)
            self.assertTrue(quality_gate.stats["auto_rag_ready"])
            self.assertEqual(quality_gate.vlm_audit_candidates, [])
            self.assertEqual(quality_gate.vlm_audits, [])
            recheck = quality_gate.stats["post_repair_recheck"]
            self.assertEqual(recheck["issue_count"], 0)
            self.assertIn("summary_contains_ellipsis", recheck["cleared"])
            self.assertEqual(recheck["remaining"], [])

    def test_post_repair_recheck_skipped_without_applied_repair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            outputs, document_ir, structured_output = _make_settle_fixture(run_path)
            quality_gate = _make_gate([_make_issue("summary_contains_ellipsis", "warning")])

            _run_settle(
                quality_gate,
                {"applied_count": 0, "fallback_count": 0, "blocked_count": 0},
                run_path=run_path,
                outputs=outputs,
                document_ir=document_ir,
                structured_output=structured_output,
                enrichments={},
            )

            self.assertEqual(quality_gate.status, "needs_review")
            self.assertEqual(len(quality_gate.issues), 1)
            self.assertNotIn("post_repair_recheck", quality_gate.stats)

    def test_post_repair_recheck_missing_repaired_markdown_keeps_gate_conservative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            outputs, document_ir, structured_output = _make_settle_fixture(run_path, write_repaired=False)
            quality_gate = _make_gate([_make_issue("summary_contains_ellipsis", "warning")])

            _run_settle(
                quality_gate,
                {"applied_count": 1, "fallback_count": 0, "blocked_count": 0},
                run_path=run_path,
                outputs=outputs,
                document_ir=document_ir,
                structured_output=structured_output,
                enrichments={},
            )

            self.assertEqual(len(quality_gate.issues), 1)
            self.assertEqual(quality_gate.status, "needs_review")
            self.assertEqual(
                quality_gate.stats["post_repair_recheck"]["skipped_reason"],
                "repaired_markdown_missing",
            )
            self.assertIsNot(quality_gate.stats.get("auto_rag_ready"), True)


if __name__ == "__main__":
    unittest.main()
