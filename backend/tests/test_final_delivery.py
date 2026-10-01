"""Final-output contracts and crash-safe publication."""
import json

import pytest

from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.delivery import atomic_write_json
from app.pipeline.final_delivery import finalize_delivery, preserve_document_content
from app.pipeline.revisions import begin_revision, publish_revision


def source():
    return DocumentIR(doc_id="doc", run_id="run",
        source=SourceInfo(path="a.pdf", ext="pdf", sha256="source", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"), pages=[PageInfo(page_idx=0)],
        blocks=[Block(block_id="budget", type=BlockType.TEXT, page_idx=0,
                      payload={"text": "數位行銷預算 1,100,000 元。"})])


def write_outputs(root, main, chunks):
    out = root / "outputs"
    (out / "documents").mkdir(parents=True, exist_ok=True)
    (out / "documents/main.md").write_text(main)
    atomic_write_json(out / "documents_index.json", [{"document_id": "main", "file": str(out / "documents/main.md")}])
    (out / "chunks.jsonl").write_text("\n".join(json.dumps({
        "chunk_id": f"c{i}", "content": text, "block_ids": ["budget"], "page_indices": [0],
    }, ensure_ascii=False) for i, text in enumerate(chunks)))
    atomic_write_json(out / "quality_gate.json", {"status": "pass", "issues": [], "stats": {}})
    # A complete upstream surface must not hide an incomplete delivery.
    (out / "source.md").write_text("數位行銷預算 1,100,000 元。")


def test_complete_source_cannot_mask_final_chunk_loss(tmp_path):
    write_outputs(tmp_path, "數位行銷預算 1,100,000 元。", ["預算已整理"])
    result = finalize_delivery(source(), tmp_path, PipelineConfig())
    assert result["quality_status"] == "needs_review"
    assert result["coverage"]["missing_chunks"] == 1
    assert not result["auto_rag_ready"]


def test_main_content_is_restored_but_source_review_is_still_required(tmp_path):
    write_outputs(tmp_path, "# 計畫", ["數位行銷預算 1,100,000 元。"])
    assert preserve_document_content(source(), tmp_path) == 1
    assert preserve_document_content(source(), tmp_path) == 0
    result = finalize_delivery(source(), tmp_path, PipelineConfig())
    assert not result["final_validation_passed"]
    assert not result["auto_rag_ready"]
    assert result["review"]["reason"] == "source_review_missing"
    ledger = json.loads((tmp_path / "outputs/coverage.json").read_text())
    assert ledger["blocks"][0]["documents"] and ledger["blocks"][0]["chunks"]


def test_publish_rejects_changed_file_and_keeps_previous_revision(tmp_path):
    root = tmp_path / "run"
    first, _ = begin_revision(root, "fingerprint-1")
    write_outputs(first, "數位行銷預算 1,100,000 元。", ["數位行銷預算 1,100,000 元。"])
    finalize_delivery(source(), first, PipelineConfig())
    publish_revision(root, first)
    next_revision, _ = begin_revision(root, "fingerprint-2")
    write_outputs(next_revision, "數位行銷預算 1,100,000 元。", ["數位行銷預算 1,100,000 元。"])
    finalize_delivery(source(), next_revision, PipelineConfig())
    (next_revision / "outputs/chunks.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="changed after validation"):
        publish_revision(root, next_revision)
    assert (root / "current").resolve() == first
    assert "1,100,000" in (root / "outputs/documents/main.md").read_text()


def test_missing_index_member_cannot_publish_as_complete(tmp_path):
    write_outputs(tmp_path, "content", ["content"])
    (tmp_path / "outputs/documents/main.md").unlink()
    with pytest.raises(ValueError, match="Missing delivered document"):
        finalize_delivery(source(), tmp_path, PipelineConfig())



def test_changed_visual_evidence_prevents_publication(tmp_path):
    root = tmp_path / "run"
    revision, _ = begin_revision(root, "fingerprint")
    write_outputs(revision, "數位行銷預算 1,100,000 元。", ["數位行銷預算 1,100,000 元。"])
    image = revision / "assets/pages/p0000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"original visual evidence")
    finalize_delivery(source(), revision, PipelineConfig())
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after validation"):
        publish_revision(root, revision)
    assert not (root / "current").exists()


def test_changed_word_agent_evidence_prevents_publication(tmp_path):
    root = tmp_path / "run"
    revision, _ = begin_revision(root, "fingerprint")
    write_outputs(revision, "數位行銷預算 1,100,000 元。", ["數位行銷預算 1,100,000 元。"])
    evidence = revision / "agentic/evidence/word/page.png"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"verified Word page")
    finalize_delivery(source(), revision, PipelineConfig())
    evidence.write_bytes(b"different page")
    with pytest.raises(ValueError, match="changed after validation"):
        publish_revision(root, revision)
