"""Delivery-boundary regression tests.

Four delivered surfaces bypassed the privacy scrub (dataset.md, the
structured-repair main.md re-export, chunks.jsonl metadata, quality_gate.json
/ llm_vlm_outputs.md), and no delivered artifact was written atomically. See
app/pipeline/delivery.py for the atomic-write primitives shared by every
pipeline write, and app/pipeline/privacy.py for the scrub itself.
"""

import json

from app.config import PackageConfig, PipelineConfig
from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.pipeline.privacy import scrub_transcribed_privacy, set_privacy_scrub_enabled

# Proven privacy-scrub sample (masked by _DOMAIN_ACCOUNT_RE); reused from
# test_privacy_scrub.py / test_chunk_stage.py instead of inventing a new one.
_SENSITIVE_TEXT = "描述：DEMO\\d32755分享給您。狀態：仍在等待中。"


def test_sensitive_sample_is_genuinely_masked_by_the_scrub():
    # Guard the fixture itself: every test below assumes this string is
    # something scrub_transcribed_privacy actually changes.
    masked = scrub_transcribed_privacy(_SENSITIVE_TEXT)
    assert "d32755" not in masked
    assert "DEMO\\d*****" in masked


def _make_document_ir(blocks: list[Block], path: str = "guide.pdf") -> DocumentIR:
    return DocumentIR(
        doc_id="doc-delivery",
        run_id="run-delivery",
        source=SourceInfo(path=path, ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# 1. dataset.md was never scrubbed (package.py "compatibility alias" write)
# ---------------------------------------------------------------------------


def test_finalize_dataset_markdown_scrubs_sensitive_text():
    from app.pipeline.stages.package import PackageStage

    out = PackageStage._finalize_dataset_markdown(_SENSITIVE_TEXT)

    assert "d32755" not in out
    assert "DEMO\\d*****" in out


def test_finalize_dataset_markdown_passthrough_when_toggle_disabled():
    from app.pipeline.stages.package import PackageStage

    try:
        set_privacy_scrub_enabled(False)
        out = PackageStage._finalize_dataset_markdown(_SENSITIVE_TEXT)
        assert out == _SENSITIVE_TEXT
    finally:
        set_privacy_scrub_enabled(True)


# ---------------------------------------------------------------------------
# 2. Structured-repair re-export wrote unscrubbed documents/main.md
# ---------------------------------------------------------------------------


def test_write_document_exports_scrubs_raw_repair_source_md(tmp_path):
    # Simulates the structured-repair call site, which passes raw,
    # pre-scrub source_md straight into _write_document_exports.
    from app.pipeline.stages.package import PackageStage

    outputs_dir = tmp_path / "outputs"
    document_ir = _make_document_ir(blocks=[])

    paths = PackageStage()._write_document_exports(
        outputs_dir=outputs_dir,
        source_md=f"# 主文件\n\n{_SENSITIVE_TEXT}",
        assets=[],
        structured_paths={},
        document_ir=document_ir,
    )

    main_text = (outputs_dir / "documents" / "main.md").read_text(encoding="utf-8")
    assert "d32755" not in main_text
    assert "DEMO\\d*****" in main_text
    assert paths["main_document"].endswith("main.md")


def test_write_document_exports_keeps_stale_docs_if_new_write_fails(tmp_path, monkeypatch):
    # Old behaviour unlinked every documents/*.md before writing the new
    # ones; a crash mid-export left deleted files still referenced by the
    # (unchanged) old documents_index.json. New writes must land first.
    from app.pipeline.stages import package as package_module
    from app.pipeline.stages.package import PackageStage

    outputs_dir = tmp_path / "outputs"
    documents_dir = outputs_dir / "documents"
    documents_dir.mkdir(parents=True)
    stale = documents_dir / "stale_asset.md"
    stale.write_text("# stale\n\nmust survive a crash mid-export", encoding="utf-8")

    document_ir = _make_document_ir(blocks=[])

    def _boom(path, text, encoding="utf-8"):
        raise RuntimeError("simulated crash while writing a new document")

    monkeypatch.setattr(package_module, "atomic_write_text", _boom)

    raised = False
    try:
        PackageStage()._write_document_exports(
            outputs_dir=outputs_dir,
            source_md="# 主文件\n\n內容",
            assets=[],
            structured_paths={},
            document_ir=document_ir,
        )
    except RuntimeError:
        raised = True

    assert raised, "expected the simulated crash to propagate"
    assert stale.exists(), "stale doc must not be deleted before new docs are confirmed written"


def test_write_document_exports_removes_stale_docs_not_in_new_set(tmp_path):
    from app.pipeline.stages.package import PackageStage

    outputs_dir = tmp_path / "outputs"
    documents_dir = outputs_dir / "documents"
    documents_dir.mkdir(parents=True)
    stale = documents_dir / "stale_asset.md"
    stale.write_text("# stale\n\nshould be removed once new docs land", encoding="utf-8")

    document_ir = _make_document_ir(blocks=[])

    PackageStage()._write_document_exports(
        outputs_dir=outputs_dir,
        source_md="# 主文件\n\n內容",
        assets=[],
        structured_paths={},
        document_ir=document_ir,
    )

    assert not stale.exists()
    assert (documents_dir / "main.md").is_file()


# ---------------------------------------------------------------------------
# 3. chunks.jsonl metadata (table_html / heading / heading_path) leaked raw
# ---------------------------------------------------------------------------


def test_chunks_jsonl_metadata_table_html_and_heading_are_scrubbed(tmp_path):
    import asyncio

    from app.pipeline.stages.chunk import ChunkStage

    document_ir = _make_document_ir(
        blocks=[
            Block(
                block_id="h1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": _SENSITIVE_TEXT, "text_level": 1},
            ),
            Block(
                block_id="tbl0",
                type=BlockType.TABLE,
                page_idx=0,
                payload={"table_body": f"<table><tr><td>{_SENSITIVE_TEXT}</td></tr></table>"},
            ),
        ]
    )
    run_path = tmp_path / "run"
    (run_path / "outputs").mkdir(parents=True)

    result = asyncio.run(ChunkStage().run("doc", "run", document_ir, run_path))

    assert result.success
    content = (run_path / "outputs" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "d32755" not in content
    # jsonl is a json.dumps'd line, so the mask's single backslash is doubled
    # in the raw file text (DEMO\d***** -> DEMO\\d*****).
    assert "DEMO\\\\d*****" in content
    chunk_rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    masked = scrub_transcribed_privacy(_SENSITIVE_TEXT)
    assert any(masked in row["metadata"].get("heading", "") for row in chunk_rows)
    assert any(
        masked in html for row in chunk_rows for html in row["metadata"].get("table_html", [])
    )
    assert any(masked in part for row in chunk_rows for part in row["metadata"].get("heading_path", []))


def test_chunks_jsonl_metadata_passthrough_when_toggle_disabled(tmp_path):
    import asyncio

    from app.pipeline.stages.chunk import ChunkStage

    document_ir = _make_document_ir(
        blocks=[
            Block(
                block_id="tbl0",
                type=BlockType.TABLE,
                page_idx=0,
                payload={"table_body": f"<table><tr><td>{_SENSITIVE_TEXT}</td></tr></table>"},
            ),
        ]
    )
    run_path = tmp_path / "run"
    (run_path / "outputs").mkdir(parents=True)
    config = PipelineConfig(package=PackageConfig(scrub_private_info=False))

    try:
        result = asyncio.run(ChunkStage(config=config).run("doc", "run", document_ir, run_path))
        assert result.success
        content = (run_path / "outputs" / "chunks.jsonl").read_text(encoding="utf-8")
        assert "d32755" in content
    finally:
        # ChunkStage.run() flips the module-level toggle; reset so later
        # tests in the same process are not left with scrubbing disabled.
        set_privacy_scrub_enabled(True)


# ---------------------------------------------------------------------------
# 4. quality_gate.json / llm_vlm_outputs.md leaked raw VLM audit text
# ---------------------------------------------------------------------------


def test_quality_gate_issue_to_dict_scrubs_missing_notes_and_samples():
    from app.pipeline.quality_gate import QualityGateIssue

    issue = QualityGateIssue(
        code="table_notes_missing",
        severity="high",
        message="表格後方有備註/註解，但最終語意文件沒有完整包含。",
        evidence={"missing_notes": [_SENSITIVE_TEXT], "missing_count": 1},
    )

    data = issue.to_dict()

    assert "d32755" not in json.dumps(data, ensure_ascii=False)
    assert data["evidence"]["missing_count"] == 1  # structural field untouched


def test_quality_gate_result_to_dict_scrubs_vlm_audit_output():
    from app.pipeline.quality_gate import QualityGateResult

    result = QualityGateResult(
        status="needs_review",
        score=0.5,
        vlm_audits=[
            {
                "success": True,
                "page_idx": 0,
                "reasons": ["structured_output_empty"],
                "output": {"status": "needs_fix", "missing_items": [_SENSITIVE_TEXT]},
                "error": None,
                "tokens_used": 10,
                "duration_seconds": 1.0,
                "needs_review": True,
            }
        ],
    )

    data = result.to_dict()
    masked_items = data["vlm_audits"][0]["output"]["missing_items"]

    assert masked_items == [scrub_transcribed_privacy(_SENSITIVE_TEXT)]
    assert "d32755" not in json.dumps(data, ensure_ascii=False)
    # The in-memory record itself is untouched; llm_vlm_outputs.md scrubs
    # independently at its own write site rather than depending on to_dict().
    assert result.vlm_audits[0]["output"]["missing_items"] == [_SENSITIVE_TEXT]


def test_write_quality_gate_is_atomic_and_scrubbed(tmp_path):
    from app.pipeline.quality_gate import QualityGateResult, write_quality_gate

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)
    result = QualityGateResult(
        status="pass",
        score=1.0,
        vlm_audits=[{"success": True, "page_idx": 0, "output": {"summary": _SENSITIVE_TEXT}}],
    )

    path = write_quality_gate(result, outputs_dir)

    content = path.read_text(encoding="utf-8")
    assert "d32755" not in content
    saved = json.loads(content)
    assert saved["vlm_audits"][0]["output"]["summary"] == scrub_transcribed_privacy(_SENSITIVE_TEXT)
    assert not list(outputs_dir.glob("*.tmp"))


def test_llm_vlm_outputs_markdown_scrubs_audit_output(tmp_path):
    from types import SimpleNamespace

    from app.pipeline.stages.package import PackageStage

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)
    quality_gate = SimpleNamespace(
        vlm_audits=[
            {
                "success": True,
                "page_idx": 0,
                "reasons": ["structured_output_empty"],
                "tokens_used": 12,
                "error": None,
                "output": {"status": "needs_fix", "missing_items": [_SENSITIVE_TEXT]},
            }
        ],
    )

    PackageStage()._write_llm_vlm_outputs(
        outputs_dir=outputs_dir,
        enrichments={},
        quality_gate=quality_gate,
        semantic_repair_stats={},
    )

    content = (outputs_dir / "llm_vlm_outputs.md").read_text(encoding="utf-8")
    assert "d32755" not in content
    # The masked text is embedded as a json.dumps'd code block, so its single
    # backslash (DEMO\d*****) is doubled in the markdown text (DEMO\\d*****).
    assert "DEMO\\\\d*****" in content
    assert not list(outputs_dir.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 5. Atomic-write primitives (backend/app/pipeline/delivery.py)
# ---------------------------------------------------------------------------


def test_atomic_write_text_leaves_no_tmp_file_on_success(tmp_path):
    from app.pipeline.delivery import atomic_write_text

    target = tmp_path / "out.md"

    atomic_write_text(target, "hello world", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "hello world"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_text_preserves_previous_content_when_replace_fails(tmp_path, monkeypatch):
    from app.pipeline import delivery

    target = tmp_path / "out.md"
    target.write_text("previous content", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise OSError("simulated os.replace failure")

    monkeypatch.setattr(delivery.os, "replace", _boom)

    raised = False
    try:
        delivery.atomic_write_text(target, "new content", encoding="utf-8")
    except OSError:
        raised = True

    assert raised, "expected the simulated os.replace failure to propagate"
    assert target.read_text(encoding="utf-8") == "previous content"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_round_trips(tmp_path):
    from app.pipeline.delivery import atomic_write_json

    target = tmp_path / "out.json"

    atomic_write_json(target, {"a": 1, "b": "中文"}, ensure_ascii=False, indent=2)

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": "中文"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_jsonl_round_trips(tmp_path):
    from app.pipeline.delivery import atomic_write_jsonl

    target = tmp_path / "out.jsonl"

    atomic_write_jsonl(target, [{"id": 1}, {"id": 2}])

    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"id": 1}, {"id": 2}]
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_structured_rag_outputs_is_atomic(tmp_path):
    from types import SimpleNamespace

    from app.pipeline.structured_rag import write_structured_rag_outputs

    plan = SimpleNamespace(
        document_type="generic_document",
        to_dict=lambda: {"document_type": "generic_document"},
    )
    output = SimpleNamespace(
        plan=plan,
        records=[],
        rag_markdown="# Doc\n\nBody",
        chunks=[],
        stats={},
    )

    write_structured_rag_outputs(output, tmp_path)

    assert not list(tmp_path.rglob("*.tmp"))
    assert (tmp_path / "structured_rag.md").read_text(encoding="utf-8") == "# Doc\n\nBody"


# ---------------------------------------------------------------------------
# 6. Repair-rejection retention paths are delivery surfaces too
# ---------------------------------------------------------------------------


def test_retain_structured_candidate_writes_scrubbed_delivered_surfaces(tmp_path):
    # When reviewer repair is rejected, the pre-review candidate overwrites
    # source.md / rag.md / structured_rag.md — the same files the run APIs
    # serve. The retention write must never deliver unscrubbed OCR content.
    from app.pipeline.stages.package import PackageStage

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "source.md").write_text("# scrubbed\n", encoding="utf-8")
    (outputs_dir / "rag.md").write_text("# scrubbed\n", encoding="utf-8")

    PackageStage()._retain_structured_candidate_for_review(
        outputs_dir=outputs_dir,
        document_ir=_make_document_ir(blocks=[]),
        current_markdown=f"# 標題\n\n{_SENSITIVE_TEXT}",
        title="標題",
        reason="review_fallback",
        summary="kept pre-review candidate",
        semantic_output_language="zh-TW",
    )

    for filename in ("structured_rag.md", "source.md", "rag.md"):
        text = (outputs_dir / filename).read_text(encoding="utf-8")
        assert "d32755" not in text, filename
    assert not list(outputs_dir.rglob("*.tmp"))


def test_retain_form_candidate_scrubs_markdown_file(tmp_path):
    # The form-retention path writes current_markdown verbatim to the form's
    # markdown file; OCR'd personal content must be masked on that surface too.
    from app.pipeline.stages.package import PackageStage

    outputs_dir = tmp_path / "outputs"
    documents_dir = outputs_dir / "documents"
    documents_dir.mkdir(parents=True)
    md_path = documents_dir / "form_0.md"

    form_item = {
        "form_id": "form_0",
        "title": "申請表",
        "files": {"markdown": str(md_path)},
        "page_indices": [0],
    }

    PackageStage()._retain_form_candidate_for_review(
        outputs_dir=outputs_dir,
        form_item=form_item,
        item_stats={},
        current_markdown=f"# 申請表\n\n{_SENSITIVE_TEXT}",
        document_ir=_make_document_ir(blocks=[]),
        semantic_output_language="zh-TW",
    )

    text = md_path.read_text(encoding="utf-8")
    assert "d32755" not in text
    assert "DEMO\\d*****" in text
