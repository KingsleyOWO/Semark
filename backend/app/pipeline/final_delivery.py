"""Validate the files actually delivered, after all rewriting and chunking."""
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import PipelineConfig
from app.models.document_ir import DocumentIR
from app.pipeline.content_coverage import TextCoverage
from app.pipeline.delivery import atomic_write_json, atomic_write_text
from app.pipeline.privacy import scrub_transcribed_privacy, set_privacy_scrub_enabled
from app.pipeline.stages.normalize import is_non_content

# Section heading for blocks the documents missed; not source text itself.
SUPPLEMENT_HEADINGS = {"en": "Additional Source Content", "zh-TW": "補充原文內容"}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def delivered_documents(outputs: Path) -> list[Path]:
    """Resolve index members inside this output, including relocated legacy runs."""
    paths = []
    for entry in read_json(outputs / "documents_index.json", []):
        filename = str(entry.get("file") or "")
        candidate = outputs / "documents" / Path(filename).name
        if candidate.suffix != ".md" or not candidate.is_file():
            raise ValueError(f"Missing delivered document: {entry.get('document_id', '')}")
        if not candidate.resolve().is_relative_to(outputs.resolve()):
            raise ValueError("Document index escapes the delivery directory")
        paths.append(candidate)
    if not paths and (outputs / "documents/main.md").is_file():
        paths.append(outputs / "documents/main.md")
    return paths


def expected_blocks(ir: DocumentIR, outputs: Path, config: PipelineConfig | None = None):
    from app.pipeline.agentic_content import replacement_text
    from app.pipeline.stages.chunk import ChunkStage

    renderer = ChunkStage(config, ir)
    renderer._asset_text = renderer._load_asset_retrieval_text(outputs)
    for block in ir.blocks:
        if is_non_content(block):
            yield block, "", str(block.payload.get("origin"))
        elif block.block_id in renderer._decorative_blocks:
            yield block, "", "decorative"
        else:
            text = scrub_transcribed_privacy("\n\n".join(part for part in (
                renderer._block_to_text(block), "" if replacement_text(block) else str(block.payload.get("verified_semantics") or "")
            ) if part))
            yield block, text, "" if text.strip() else "no_transcribed_content"


def missing_part(text: str, delivered: list[str]) -> str:
    """The lines of a block the delivery lacks, or the whole block when those lines alone leave it short.

    Demo run 2026-10-21: a flowchart's main document lacked one opening sentence of its picture
    text, and the whole flow (about 40 lines) was appended a second time. A line counts as
    delivered by the same rule as the block.
    """
    coverage = TextCoverage(delivered)
    part = "\n".join(line for line in text.split("\n") if not coverage.covers(line)).strip()
    return part if part and TextCoverage([*delivered, part]).covers(text) else text


def preserve_document_content(ir: DocumentIR, run_path: Path, config: PipelineConfig | None = None) -> int:
    """Retain source blocks not represented in any delivered child/main file."""
    outputs = run_path / "outputs"
    files = delivered_documents(outputs)
    if not files:
        return 0
    coverage = TextCoverage(path.read_text(encoding="utf-8") for path in files)
    missing = [(block, text) for block, text, reason in expected_blocks(ir, outputs, config)
               if not reason and not coverage.covers(text)]
    if not missing:
        return 0
    main = outputs / "documents/main.md"
    if main not in files:
        raise ValueError("A delivery with missing content needs a main document")
    from app.pipeline.stages.chunk import ChunkStage

    heading = SUPPLEMENT_HEADINGS[ChunkStage(config, ir).table_language]
    additions = [f"\n\n## {heading}\n"]
    delivered = [path.read_text(encoding="utf-8") for path in files]
    for block, text in missing:
        additions.append(f"<!-- source-block:{block.block_id} -->\n{missing_part(text, delivered)}\n")
        delivered.append(additions[-1])
    atomic_write_text(main, main.read_text(encoding="utf-8") + "\n".join(additions))
    return len(missing)


def finalize_delivery(ir: DocumentIR, run_path: Path, config: PipelineConfig) -> dict:
    """No source.md/IR text is allowed to masquerade as delivered evidence."""
    set_privacy_scrub_enabled(config.package.scrub_private_info)
    outputs = run_path / "outputs"
    docs = delivered_documents(outputs)
    chunks = read_jsonl(outputs / "chunks.jsonl")
    doc_coverage = TextCoverage(path.read_text(encoding="utf-8") for path in docs)
    chunk_coverage = TextCoverage(str(c.get("content") or "") for c in chunks)
    ledger = []
    for block, text, reason in expected_blocks(ir, outputs, config):
        ledger.append({
            "block_id": block.block_id, "page_idx": block.page_idx,
            "type": block.type.value, "excluded_reason": reason or None,
            "documents": bool(reason) or doc_coverage.covers(text),
            "chunks": None if not config.package.generate_chunks else bool(reason) or chunk_coverage.covers(text),
            "privacy_policy": config.package.scrub_private_info,
        })
    missing_documents = [r["block_id"] for r in ledger if not r["documents"]]
    missing_chunks = [r["block_id"] for r in ledger if r["chunks"] is False]
    ids = [c.get("chunk_id") for c in chunks]
    issues = []
    if missing_documents:
        issues.append({"code": "final_documents_missing_content", "severity": "high",
                       "message": "Delivered documents omit source blocks.", "block_ids": missing_documents})
    if missing_chunks:
        issues.append({"code": "final_chunks_missing_content", "severity": "high",
                       "message": "Final chunks omit source blocks.", "block_ids": missing_chunks})
    if config.package.generate_chunks and (not chunks or len(set(ids)) != len(ids)):
        issues.append({"code": "final_chunks_invalid", "severity": "high",
                       "message": "Final chunks are empty or contain duplicate IDs."})
    if not docs:
        issues.append({"code": "final_documents_missing", "severity": "high",
                       "message": "No delivered document was indexed."})
    from app.pipeline.review_sources import validate_review_coverage

    review = validate_review_coverage(ir, run_path, config)
    if not review["complete"]:
        issues.append({"code": "agentic_review_incomplete", "severity": "high",
                       "message": "Source review is incomplete or no longer matches the delivery.",
                       "review": review})
    gate = read_json(outputs / "quality_gate.json", {})
    old_issues = [item for item in gate.get("issues", [])
                  if not str(item.get("code", "")).startswith(("final_", "agentic_"))]
    gate["issues"] = old_issues + issues
    if issues:
        gate["status"] = "needs_review"
    elif not gate.get("status"):
        gate["status"] = "pass"
    stats = gate.setdefault("stats", {})
    upstream_veto = stats.get("upstream_auto_rag_veto", bool(
        not stats.get("final_delivery_checked") and stats.get("auto_rag_ready") is False))
    stats["upstream_auto_rag_veto"] = upstream_veto
    ready = not issues and gate["status"] == "pass" and config.package.generate_chunks
    # Preserve an explicit upstream veto (e.g. an unverified conversion).
    ready = ready and stats.get("auto_rag_ready", True) is not False
    stats["auto_rag_ready"] = bool(ready)
    stats["final_delivery_checked"] = True
    stats["agentic_review"] = review
    from app.pipeline.autonomous_delivery import build_automatic_delivery
    automatic = build_automatic_delivery(ir, run_path, config, review, ledger, upstream_veto, complete_allowed=not issues and gate["status"] == "pass")
    stats["automatic_delivery"] = automatic
    stats["partial_rag_ready"] = automatic["rag_ready"]
    recovery = read_json(outputs / "recovery_report.json", {})
    if recovery:
        issues_by_id = recovery.get("issues", {}).values()
        statuses = [i.get("status") for i in issues_by_id]
        stats["agentic_recovery"] = {
            "applied": sum(s.get("status") == "applied" for s in recovery.get("steps", [])),
            "rejected": sum(s.get("status") == "rejected" for s in recovery.get("steps", [])),
            "resolved": statuses.count("resolved"),
            "unresolved": sum(s != "resolved" for s in statuses),
            # Never tried for lack of time, as against tried and stopped by it (2026-10-16: the
            # report counted 57 issues out of time where 55 had had one to three attempts).
            "budget_exhausted": statuses.count("budget_exhausted"),
            "stopped_by_budget": sum(bool(i.get("stopped_by_budget")) for i in issues_by_id),
            "attempts": sum(i.get("attempts", 0) for i in issues_by_id),
            "deferred": statuses.count("deferred"),
        }
    stats["final_coverage"] = {
        "expected_blocks": len([r for r in ledger if not r["excluded_reason"]]),
        "missing_documents": len(missing_documents), "missing_chunks": len(missing_chunks),
    }
    gate.setdefault("score", 1.0 if not issues else 0.0)
    atomic_write_json(outputs / "coverage.json", {"schema_version": 1, "blocks": ledger}, ensure_ascii=False, indent=2)
    atomic_write_json(outputs / "quality_gate.json", gate, ensure_ascii=False, indent=2)
    revision = run_path.name
    result = {
        "schema_version": 1, "revision": revision, "execution_status": "completed",
        "quality_status": gate["status"], "auto_rag_ready": bool(ready),
        "final_validation_passed": not issues, "coverage": stats["final_coverage"],
        "issue_count": len(gate["issues"]), "review": review,
        "processing_status": automatic["status"], "requires_human_review": False,
        "partial_rag_ready": automatic["rag_ready"], "automatic_delivery": automatic,
    }
    atomic_write_json(outputs / "delivery_status.json", result, ensure_ascii=False, indent=2)
    quality = read_json(outputs / "quality.json", {})
    quality["delivery"] = result
    atomic_write_json(outputs / "quality.json", quality, ensure_ascii=False, indent=2)
    files = {}
    for directory in (outputs, run_path / "assets", run_path / "agentic/evidence", run_path / "agentic/recovery"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "delivery_manifest.json":
                files[str(path.relative_to(run_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("document_ir.json", "source_map.json", "manifest.json"):
        path = run_path / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_write_json(outputs / "delivery_manifest.json", {
        "schema_version": 1, "revision": revision, "files": files, "status": result,
    }, ensure_ascii=False, indent=2)
    return result
