"""Golden-set evaluation: score run outputs against per-document manifests.

Every quality-affecting pipeline change should be measured against a small
set of golden documents instead of eyeballed. A golden manifest declares
what a good run of one document must contain (fact tokens, required and
forbidden strings, chunk-shape bounds); ``score_run`` grades an existing
run outputs directory (``<run>/outputs/``) against it and ``compare_runs``
reports per-metric deltas between two runs of the same document.

Matching is whitespace- and fullwidth-insensitive, reusing the
normalization helpers from :mod:`app.pipeline.repair_guard` so the eval
harness and the in-pipeline fact guard agree on what "the same token"
means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.pipeline.repair_guard import _compact, _normalize, _token_survives
from app.pipeline.stages.chunk import estimate_tokens

__all__ = [
    "GoldenDoc",
    "GoldenExpectations",
    "GoldenManifest",
    "ChunkStats",
    "QualityGateSummary",
    "GoldenReport",
    "load_manifest",
    "score_run",
    "compare_runs",
    "render_markdown",
    "render_comparison_markdown",
]

# Joined between corpus files before normalization. The private-use
# character is not whitespace (``_compact`` strips whitespace and commas)
# and never occurs in real outputs, so a token can never be stitched
# together from the tail of one file and the head of the next.
_FILE_BOUNDARY = "\n\ue000\n"

# Final chunk artifact first: the chunk stage overwrites ``chunks.jsonl``
# with structured chunks when those are authoritative.
_CHUNK_FILES = ("chunks.jsonl", "structured_chunks.jsonl")


class GoldenDoc(BaseModel):
    """Identity of a golden document."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_pdf: str  # absolute path; informational only, never read at score time
    notes: str | None = None


class GoldenExpectations(BaseModel):
    """What a good run of this document must (not) contain."""

    model_config = ConfigDict(extra="forbid")

    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    fact_tokens: list[str] = Field(default_factory=list)
    min_chunk_count: int = 0
    max_chunk_tokens: int | None = None


class GoldenManifest(BaseModel):
    """One golden manifest file: a document plus its expectations."""

    model_config = ConfigDict(extra="forbid")

    doc: GoldenDoc
    expectations: GoldenExpectations


class ChunkStats(BaseModel):
    """Shape statistics for the run's final chunk artifact."""

    chunks_file: str | None = None
    count: int = 0
    max_tokens: int = 0
    avg_tokens: float = 0.0
    heading_path_coverage: float = 0.0
    min_chunk_count_ok: bool = True
    max_chunk_tokens_ok: bool = True
    oversized_chunk_ids: list[str] = Field(default_factory=list)


class QualityGateSummary(BaseModel):
    """Passthrough of the run's own quality gate verdict."""

    present: bool = False
    status: str | None = None
    score: float | None = None


class GoldenReport(BaseModel):
    """Scored comparison of one run against one golden manifest."""

    doc_name: str
    outputs_dir: str
    fact_token_count: int
    fact_tokens_found: int
    fact_recall: float
    missing_fact_tokens: list[str]
    must_include_count: int
    must_include_found: int
    must_include_rate: float
    missing_must_include: list[str]
    must_not_include_violations: list[str]
    chunks: ChunkStats
    quality_gate: QualityGateSummary

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def load_manifest(path: Path | str) -> GoldenManifest:
    """Load and validate a golden manifest JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldenManifest.model_validate(data)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _semantic_corpus(outputs_dir: Path) -> str:
    """Concatenate structured_rag.md and every documents/*.md for matching."""

    parts: list[str] = []
    structured_rag = outputs_dir / "structured_rag.md"
    if structured_rag.is_file():
        parts.append(_read_text(structured_rag))
    documents_dir = outputs_dir / "documents"
    if documents_dir.is_dir():
        for doc_path in sorted(documents_dir.glob("*.md")):
            parts.append(_read_text(doc_path))
    return _FILE_BOUNDARY.join(parts)


def _corpus_contains(needle: str, corpus_normalized: str, corpus_compact: str) -> bool:
    return _token_survives(_normalize(needle), corpus_normalized, corpus_compact)


def _load_chunk_records(outputs_dir: Path) -> tuple[str | None, list[dict[str, Any]]]:
    for name in _CHUNK_FILES:
        path = outputs_dir / name
        if not path.is_file():
            continue
        records: list[dict[str, Any]] = []
        for line in _read_text(path).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        if records:
            return name, records
    return None, []


def _chunk_stats(outputs_dir: Path, expectations: GoldenExpectations) -> ChunkStats:
    chunks_file, records = _load_chunk_records(outputs_dir)
    token_sizes = [estimate_tokens(str(record.get("content") or "")) for record in records]
    count = len(records)
    with_heading = sum(
        1 for record in records if (record.get("metadata") or {}).get("heading_path")
    )
    limit = expectations.max_chunk_tokens
    oversized = [
        str(record.get("chunk_id") or f"#{idx}")
        for idx, (record, tokens) in enumerate(zip(records, token_sizes))
        if limit is not None and tokens > limit
    ]
    return ChunkStats(
        chunks_file=chunks_file,
        count=count,
        max_tokens=max(token_sizes, default=0),
        avg_tokens=round(sum(token_sizes) / count, 1) if count else 0.0,
        heading_path_coverage=with_heading / count if count else 0.0,
        min_chunk_count_ok=count >= expectations.min_chunk_count,
        max_chunk_tokens_ok=not oversized,
        oversized_chunk_ids=oversized,
    )


def _quality_gate_summary(outputs_dir: Path) -> QualityGateSummary:
    path = outputs_dir / "quality_gate.json"
    if not path.is_file():
        return QualityGateSummary()
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError:
        return QualityGateSummary()
    if not isinstance(data, dict):
        return QualityGateSummary()
    status = data.get("status")
    score = data.get("score")
    return QualityGateSummary(
        present=True,
        status=str(status) if status is not None else None,
        score=float(score) if isinstance(score, (int, float)) else None,
    )


def score_run(manifest: GoldenManifest, outputs_dir: Path | str) -> GoldenReport:
    """Score one run outputs directory against a golden manifest."""

    outputs = Path(outputs_dir)
    if not outputs.is_dir():
        raise FileNotFoundError(f"run outputs directory not found: {outputs}")

    expectations = manifest.expectations
    corpus_normalized = _normalize(_semantic_corpus(outputs))
    corpus_compact = _compact(corpus_normalized)

    missing_facts = [
        token
        for token in expectations.fact_tokens
        if not _corpus_contains(token, corpus_normalized, corpus_compact)
    ]
    fact_token_count = len(expectations.fact_tokens)
    fact_tokens_found = fact_token_count - len(missing_facts)

    missing_must_include = [
        text
        for text in expectations.must_include
        if not _corpus_contains(text, corpus_normalized, corpus_compact)
    ]
    must_include_count = len(expectations.must_include)
    must_include_found = must_include_count - len(missing_must_include)

    violations = [
        text
        for text in expectations.must_not_include
        if _corpus_contains(text, corpus_normalized, corpus_compact)
    ]

    return GoldenReport(
        doc_name=manifest.doc.name,
        outputs_dir=str(outputs),
        fact_token_count=fact_token_count,
        fact_tokens_found=fact_tokens_found,
        fact_recall=fact_tokens_found / fact_token_count if fact_token_count else 1.0,
        missing_fact_tokens=missing_facts,
        must_include_count=must_include_count,
        must_include_found=must_include_found,
        must_include_rate=(
            must_include_found / must_include_count if must_include_count else 1.0
        ),
        missing_must_include=missing_must_include,
        must_not_include_violations=violations,
        chunks=_chunk_stats(outputs, expectations),
        quality_gate=_quality_gate_summary(outputs),
    )


def compare_runs(
    manifest: GoldenManifest,
    outputs_dir_a: Path | str,
    outputs_dir_b: Path | str,
) -> dict[str, Any]:
    """Score two runs of the same document and report per-metric deltas (b - a)."""

    report_a = score_run(manifest, outputs_dir_a)
    report_b = score_run(manifest, outputs_dir_b)

    score_a = report_a.quality_gate.score
    score_b = report_b.quality_gate.score
    delta: dict[str, Any] = {
        "fact_recall": round(report_b.fact_recall - report_a.fact_recall, 4),
        "fact_tokens_found": report_b.fact_tokens_found - report_a.fact_tokens_found,
        "must_include_rate": round(report_b.must_include_rate - report_a.must_include_rate, 4),
        "must_include_found": report_b.must_include_found - report_a.must_include_found,
        "must_not_include_violations": (
            len(report_b.must_not_include_violations)
            - len(report_a.must_not_include_violations)
        ),
        "chunk_count": report_b.chunks.count - report_a.chunks.count,
        "max_chunk_tokens": report_b.chunks.max_tokens - report_a.chunks.max_tokens,
        "avg_chunk_tokens": round(report_b.chunks.avg_tokens - report_a.chunks.avg_tokens, 1),
        "heading_path_coverage": round(
            report_b.chunks.heading_path_coverage - report_a.chunks.heading_path_coverage, 4
        ),
        "quality_gate_score": (
            round(score_b - score_a, 4) if score_a is not None and score_b is not None else None
        ),
    }
    return {
        "doc_name": manifest.doc.name,
        "a": report_a.to_dict(),
        "b": report_b.to_dict(),
        "delta": delta,
    }


def _format_ratio(found: int, total: int, rate: float) -> str:
    return f"{found}/{total} ({rate:.1%})" if total else "n/a (none declared)"


def render_markdown(report: GoldenReport) -> str:
    """Render a GoldenReport as a compact markdown table."""

    chunks = report.chunks
    quality_gate = report.quality_gate
    if quality_gate.present:
        score_text = "-" if quality_gate.score is None else f"{quality_gate.score:.2f}"
        gate_value = f"{quality_gate.status or '-'} (score {score_text})"
    else:
        gate_value = "missing"
    max_tokens_value = str(chunks.max_tokens)
    if not chunks.max_chunk_tokens_ok:
        max_tokens_value += f" (over limit: {', '.join(chunks.oversized_chunk_ids)})"

    lines = [
        f"# Golden Eval — {report.doc_name}",
        "",
        f"outputs: `{report.outputs_dir}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| fact recall | {_format_ratio(report.fact_tokens_found, report.fact_token_count, report.fact_recall)} |",
        f"| must_include | {_format_ratio(report.must_include_found, report.must_include_count, report.must_include_rate)} |",
        f"| must_not_include violations | {len(report.must_not_include_violations)} |",
        f"| chunks | {chunks.count} ({chunks.chunks_file or 'no chunk file'}) |",
        f"| min chunk count ok | {'yes' if chunks.min_chunk_count_ok else 'no'} |",
        f"| max chunk tokens | {max_tokens_value} |",
        f"| avg chunk tokens | {chunks.avg_tokens} |",
        f"| heading_path coverage | {chunks.heading_path_coverage:.1%} |",
        f"| quality gate | {gate_value} |",
    ]
    if report.missing_fact_tokens:
        lines += ["", "missing fact tokens: " + "、".join(report.missing_fact_tokens)]
    if report.missing_must_include:
        lines += ["", "missing must_include: " + "、".join(report.missing_must_include)]
    if report.must_not_include_violations:
        lines += ["", "violations: " + "、".join(report.must_not_include_violations)]
    lines.append("")
    return "\n".join(lines)


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a compare_runs() result as a compact markdown delta table."""

    report_a = comparison["a"]
    report_b = comparison["b"]
    delta = comparison["delta"]

    def gate(report: dict[str, Any]) -> str:
        quality_gate = report["quality_gate"]
        if not quality_gate["present"]:
            return "missing"
        score = quality_gate["score"]
        score_text = "-" if score is None else f"{score:.2f}"
        return f"{quality_gate['status'] or '-'} ({score_text})"

    rows = [
        ("fact recall", report_a["fact_recall"], report_b["fact_recall"], delta["fact_recall"]),
        (
            "must_include rate",
            report_a["must_include_rate"],
            report_b["must_include_rate"],
            delta["must_include_rate"],
        ),
        (
            "must_not_include violations",
            len(report_a["must_not_include_violations"]),
            len(report_b["must_not_include_violations"]),
            delta["must_not_include_violations"],
        ),
        ("chunk count", report_a["chunks"]["count"], report_b["chunks"]["count"], delta["chunk_count"]),
        (
            "max chunk tokens",
            report_a["chunks"]["max_tokens"],
            report_b["chunks"]["max_tokens"],
            delta["max_chunk_tokens"],
        ),
        (
            "avg chunk tokens",
            report_a["chunks"]["avg_tokens"],
            report_b["chunks"]["avg_tokens"],
            delta["avg_chunk_tokens"],
        ),
        (
            "heading_path coverage",
            report_a["chunks"]["heading_path_coverage"],
            report_b["chunks"]["heading_path_coverage"],
            delta["heading_path_coverage"],
        ),
        ("quality gate", gate(report_a), gate(report_b), delta["quality_gate_score"]),
    ]
    lines = [
        f"# Golden Compare — {comparison['doc_name']}",
        "",
        f"baseline (a): `{report_a['outputs_dir']}`",
        f"candidate (b): `{report_b['outputs_dir']}`",
        "",
        "| metric | a | b | delta |",
        "| --- | --- | --- | --- |",
    ]
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    for name, value_a, value_b, value_delta in rows:
        if value_delta is None:
            delta_text = "-"
        elif isinstance(value_delta, (int, float)):
            delta_text = f"{value_delta:+.4g}"
        else:
            delta_text = str(value_delta)
        lines.append(f"| {name} | {cell(value_a)} | {cell(value_b)} | {delta_text} |")
    lines.append("")
    return "\n".join(lines)
