"""Collect baseline metrics from existing pipeline run artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.eval.validation import (
    validate_asset_entry,
    validate_chunk_entry,
    validate_source_map,
)

REQUIRED_RUN_FILES = (
    "document_ir.json",
    "source_map.json",
    "manifest.json",
    "outputs/dataset.md",
    "outputs/rag.md",
    "outputs/assets_index.jsonl",
    "outputs/quality.json",
    "outputs/chunks.jsonl",
)

ASSET_TOKEN_RE = re.compile(r"\[\[asset:([^\]]+)\]\]")


@dataclass
class RunArtifactStatus:
    """Presence and size of a required run artifact."""

    path: str
    exists: bool
    size_bytes: int = 0


@dataclass
class RunBaselineMetrics:
    """Metrics for one completed or partially completed run."""

    doc_id: str
    run_id: str
    run_path: str
    success: bool
    artifacts: list[RunArtifactStatus] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    parse_backend: str | None = None
    parse_method: str | None = None
    vlm_model: str | None = None
    vlm_api_mode: str | None = None
    block_counts: dict[str, int] = field(default_factory=dict)
    page_count: int = 0
    dataset_chars: int = 0
    rag_chars: int = 0
    asset_count: int = 0
    asset_count_by_type: dict[str, int] = field(default_factory=dict)
    asset_needs_review_count: int = 0
    field_schema_count: int = 0
    asset_token_count_in_rag: int = 0
    chunk_count: int = 0
    chunk_attachment_count: int = 0
    enrichment_count: int = 0
    enrichment_count_by_kind: dict[str, int] = field(default_factory=dict)
    enrichment_needs_review_count: int = 0
    source_map_anchor_count: int = 0
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_read_error": f"line {line_no}: {exc}"})
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def iter_run_paths(workspace_path: Path) -> list[Path]:
    """Return run directories under workspace/store/docs."""
    docs_root = workspace_path / "store" / "docs"
    if not docs_root.exists():
        return []

    run_paths: list[Path] = []
    for doc_dir in sorted(p for p in docs_root.iterdir() if p.is_dir()):
        runs_dir = doc_dir / "runs"
        if not runs_dir.exists():
            continue
        run_paths.extend(sorted(p for p in runs_dir.iterdir() if p.is_dir()))
    return run_paths


def collect_run_metrics(run_path: Path) -> RunBaselineMetrics:
    """Collect artifact, packaging, and validation metrics for one run."""
    doc_id = run_path.parent.parent.name
    run_id = run_path.name

    artifacts: list[RunArtifactStatus] = []
    missing: list[str] = []
    for rel in REQUIRED_RUN_FILES:
        artifact_path = run_path / rel
        exists = artifact_path.exists()
        size = artifact_path.stat().st_size if exists else 0
        artifacts.append(RunArtifactStatus(path=rel, exists=exists, size_bytes=size))
        if not exists:
            missing.append(rel)

    manifest = read_json(run_path / "manifest.json")
    quality = read_json(run_path / "outputs" / "quality.json")
    source_map = read_json(run_path / "source_map.json")
    assets = read_jsonl(run_path / "outputs" / "assets_index.jsonl")
    chunks = read_jsonl(run_path / "outputs" / "chunks.jsonl")
    enrichments = read_jsonl(run_path / "outputs" / "enrichments.jsonl")

    dataset_md = _read_text(run_path / "outputs" / "dataset.md")
    rag_md = _read_text(run_path / "outputs" / "rag.md")

    asset_type_counts = Counter(str(a.get("type", "unknown")) for a in assets)
    enrichment_kind_counts = Counter(str(e.get("kind", "unknown")) for e in enrichments)

    validation_errors: list[str] = []
    for idx, asset in enumerate(assets):
        for error in validate_asset_entry(asset):
            validation_errors.append(f"assets_index line {idx + 1}: {error}")

    for idx, chunk in enumerate(chunks):
        for error in validate_chunk_entry(chunk):
            validation_errors.append(f"chunks line {idx + 1}: {error}")

    for error in validate_source_map(source_map):
        validation_errors.append(f"source_map: {error}")

    vlm_engine = manifest.get("engines", {}).get("vlm", {})
    mineru_engine = manifest.get("engines", {}).get("mineru", {})

    metrics = RunBaselineMetrics(
        doc_id=doc_id,
        run_id=run_id,
        run_path=str(run_path),
        success=not missing,
        artifacts=artifacts,
        missing_artifacts=missing,
        parse_backend=mineru_engine.get("backend"),
        parse_method=mineru_engine.get("method"),
        vlm_model=vlm_engine.get("model"),
        vlm_api_mode=vlm_engine.get("api_mode"),
        block_counts=quality.get("block_counts", {}) if isinstance(quality, dict) else {},
        page_count=int(quality.get("page_count", 0) or 0) if isinstance(quality, dict) else 0,
        dataset_chars=len(dataset_md),
        rag_chars=len(rag_md),
        asset_count=len(assets),
        asset_count_by_type=dict(asset_type_counts),
        asset_needs_review_count=sum(1 for a in assets if bool(a.get("needs_review"))),
        field_schema_count=sum(
            len(a.get("field_schema", []))
            for a in assets
            if isinstance(a.get("field_schema", []), list)
        ),
        asset_token_count_in_rag=len(ASSET_TOKEN_RE.findall(rag_md)),
        chunk_count=len(chunks),
        chunk_attachment_count=sum(
            len(c.get("attachments", []))
            for c in chunks
            if isinstance(c.get("attachments", []), list)
        ),
        enrichment_count=len(enrichments),
        enrichment_count_by_kind=dict(enrichment_kind_counts),
        enrichment_needs_review_count=sum(
            1 for e in enrichments if bool(e.get("quality", {}).get("needs_review"))
        ),
        source_map_anchor_count=len(source_map.get("md_anchors", []))
        if isinstance(source_map.get("md_anchors"), list)
        else 0,
        validation_errors=validation_errors,
    )
    return metrics


def collect_workspace_baseline(workspace_path: Path) -> dict[str, Any]:
    """Collect metrics for every run in a workspace."""
    runs = [collect_run_metrics(path) for path in iter_run_paths(workspace_path)]
    aggregate_asset_counts = Counter()
    aggregate_block_counts = Counter()
    validation_error_count = 0

    for run in runs:
        aggregate_asset_counts.update(run.asset_count_by_type)
        aggregate_block_counts.update(run.block_counts)
        validation_error_count += len(run.validation_errors)

    return {
        "workspace_path": str(workspace_path),
        "run_count": len(runs),
        "successful_run_count": sum(1 for run in runs if run.success),
        "asset_count_by_type": dict(aggregate_asset_counts),
        "block_count_by_type": dict(aggregate_block_counts),
        "validation_error_count": validation_error_count,
        "runs": [run.to_dict() for run in runs],
    }


def write_baseline_reports(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write JSON and Markdown reports for baseline metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_metrics.json"
    markdown_path = output_dir / "baseline_metrics.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_baseline_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_baseline_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable baseline report."""
    lines = [
        "# Baseline Metrics",
        "",
        f"- Workspace: `{report.get('workspace_path', '')}`",
        f"- Runs discovered: {report.get('run_count', 0)}",
        f"- Successful runs: {report.get('successful_run_count', 0)}",
        f"- Validation errors: {report.get('validation_error_count', 0)}",
        "",
        "## Aggregate Counts",
        "",
        f"- Blocks: `{json.dumps(report.get('block_count_by_type', {}), ensure_ascii=False)}`",
        f"- Assets: `{json.dumps(report.get('asset_count_by_type', {}), ensure_ascii=False)}`",
        "",
        "## Runs",
        "",
        "| doc_id | run_id | pages | blocks | assets | chunks | rag chars | validation errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for run in report.get("runs", []):
        block_total = sum(int(v) for v in run.get("block_counts", {}).values())
        lines.append(
            "| {doc_id} | {run_id} | {pages} | {blocks} | {assets} | {chunks} | {rag_chars} | {errors} |".format(
                doc_id=run.get("doc_id", ""),
                run_id=run.get("run_id", ""),
                pages=run.get("page_count", 0),
                blocks=block_total,
                assets=run.get("asset_count", 0),
                chunks=run.get("chunk_count", 0),
                rag_chars=run.get("rag_chars", 0),
                errors=len(run.get("validation_errors", [])),
            )
        )

    lines.append("")
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

