"""Lightweight retrieval smoke evaluation over generated chunks and assets."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.eval.baseline import read_jsonl

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "組織圖": ("組織架構", "架構圖", "系統圖", "流程圖", "flowchart", "hierarchical", "hierarchy", "chart", "diagram", "structure"),
    "架構": ("結構", "層級", "階層", "hierarchical", "hierarchy", "structure"),
    "流程圖": ("flowchart", "chart", "diagram"),
    "系統圖": ("架構圖", "流程圖", "flowchart", "diagram", "structure"),
}

ASSET_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "figure_asset": ("figure", "image", "visual_asset", "圖", "圖表"),
    "form_asset": ("form", "sheet", "表單", "申請單", "簽核", "欄位"),
    "table_asset": ("table", "spreadsheet", "表格", "欄位", "列", "欄"),
}


@dataclass
class RetrievalCandidate:
    candidate_id: str
    candidate_type: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_queries(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_corpus(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def local_sample_doc_ids(corpus: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sample in corpus.get("local_samples", []):
        sample_id = sample.get("sample_id")
        doc_id = sample.get("doc_id")
        if sample_id and doc_id:
            mapping[sample_id] = doc_id
    return mapping


def latest_run_path(workspace_path: Path, doc_id: str) -> Path | None:
    runs_dir = workspace_path / "store" / "docs" / doc_id / "runs"
    if not runs_dir.exists():
        return None
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def build_retrieval_candidates(run_path: Path) -> list[RetrievalCandidate]:
    """Build lexical retrieval candidates from chunks and assets."""
    candidates: list[RetrievalCandidate] = []

    chunks_path = run_path / "outputs" / "chunks.jsonl"
    for chunk in read_jsonl(chunks_path):
        candidates.append(
            RetrievalCandidate(
                candidate_id=str(chunk.get("chunk_id", "")),
                candidate_type="chunk",
                text=str(chunk.get("content", "")),
                source_path=str(chunks_path),
                metadata={
                    "block_ids": chunk.get("block_ids", []),
                    "attachments": chunk.get("attachments", []),
                    "page_indices": chunk.get("page_indices", []),
                },
            )
        )

    assets_path = run_path / "outputs" / "assets_index.jsonl"
    for asset in read_jsonl(assets_path):
        candidate_type = str(asset.get("type", "asset"))
        text_parts = [
            " ".join(ASSET_TYPE_ALIASES.get(candidate_type, ())),
            str(asset.get("title", "")),
            " ".join(str(t) for t in asset.get("triggers", []) if t),
            str(asset.get("retrieval_text", "")),
            str(asset.get("semantic_caption", "")),
            str(asset.get("filling_guide", "")),
        ]
        candidates.append(
            RetrievalCandidate(
                candidate_id=str(asset.get("asset_id", "")),
                candidate_type=candidate_type,
                text="\n".join(part for part in text_parts if part),
                source_path=str(assets_path),
                metadata={
                    "asset_path": asset.get("asset_path"),
                    "block_id": asset.get("block_id"),
                    "page_idx": asset.get("page_idx"),
                    "needs_review": asset.get("needs_review"),
                },
            )
        )

    return candidates


def score_candidate(query: str, candidate_text: str) -> int:
    """Simple token-overlap score for smoke tests."""
    query_terms = query_terms_for_scoring(query)
    if not query_terms:
        return 0

    candidate_lower = candidate_text.lower()
    return sum(1 for term in query_terms if term in candidate_lower)


def query_terms_for_scoring(query: str) -> set[str]:
    """Extract ASCII tokens and whitespace-delimited CJK terms."""
    terms = {term.lower() for term in TOKEN_RE.findall(query)}
    for part in re.split(r"\s+", query.strip()):
        cleaned = part.strip().lower()
        if len(cleaned) < 2:
            continue
        if TOKEN_RE.fullmatch(cleaned):
            continue
        terms.add(cleaned)
    expanded_terms = set(terms)
    for term in terms:
        for expansion in QUERY_EXPANSIONS.get(term, ()):
            expanded_terms.add(expansion.lower())
    return expanded_terms


def rank_candidates(query: str, candidates: list[RetrievalCandidate], top_k: int = 5) -> list[RetrievalCandidate]:
    ranked: list[RetrievalCandidate] = []
    for candidate in candidates:
        ranked.append(
            RetrievalCandidate(
                candidate_id=candidate.candidate_id,
                candidate_type=candidate.candidate_type,
                text=candidate.text,
                source_path=candidate.source_path,
                metadata=candidate.metadata,
                score=score_candidate(query, candidate.text),
            )
        )
    ranked.sort(key=lambda item: (item.score, item.candidate_type, item.candidate_id), reverse=True)
    return ranked[:top_k]


def run_retrieval_smoke_eval(
    workspace_path: Path,
    corpus_path: Path,
    queries_path: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run starter retrieval checks using lexical ranking."""
    corpus = load_corpus(corpus_path)
    doc_ids = local_sample_doc_ids(corpus)
    queries = load_queries(queries_path)

    results: list[dict[str, Any]] = []
    for query in queries:
        sample_id = query.get("sample_id")
        doc_id = doc_ids.get(str(sample_id))
        run_path = latest_run_path(workspace_path, doc_id) if doc_id else None
        if not run_path:
            results.append(
                {
                    "query_id": query.get("query_id"),
                    "sample_id": sample_id,
                    "status": "missing_run",
                    "passed": False,
                    "top": [],
                }
            )
            continue

        candidates = build_retrieval_candidates(run_path)
        ranked = rank_candidates(str(query.get("query", "")), candidates, top_k=top_k)
        expected_asset_type = query.get("expected_asset_type")
        expected_artifact = query.get("expected_artifact")

        passed = True
        if expected_asset_type:
            passed = any(item.candidate_type == expected_asset_type for item in ranked)
        if expected_artifact:
            passed = (run_path / "outputs" / str(expected_artifact)).exists()

        results.append(
            {
                "query_id": query.get("query_id"),
                "sample_id": sample_id,
                "doc_id": doc_id,
                "run_id": run_path.name,
                "status": "ok",
                "passed": passed,
                "expected_asset_type": expected_asset_type,
                "expected_artifact": expected_artifact,
                "top": [item.to_dict() for item in ranked],
            }
        )

    return {
        "workspace_path": str(workspace_path),
        "query_count": len(results),
        "passed_count": sum(1 for item in results if item.get("passed")),
        "top_k": top_k,
        "results": results,
    }


def write_retrieval_reports(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_smoke.json"
    markdown_path = output_dir / "retrieval_smoke.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_retrieval_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_retrieval_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Smoke Eval",
        "",
        f"- Workspace: `{report.get('workspace_path', '')}`",
        f"- Queries: {report.get('query_count', 0)}",
        f"- Passed: {report.get('passed_count', 0)}",
        f"- Top K: {report.get('top_k', 0)}",
        "",
        "| query_id | sample_id | status | passed | top candidate | score |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for result in report.get("results", []):
        top = result.get("top", [{}])
        best = top[0] if top else {}
        lines.append(
            "| {query_id} | {sample_id} | {status} | {passed} | {candidate} | {score} |".format(
                query_id=result.get("query_id", ""),
                sample_id=result.get("sample_id", ""),
                status=result.get("status", ""),
                passed="yes" if result.get("passed") else "no",
                candidate=best.get("candidate_id", ""),
                score=best.get("score", 0),
            )
        )
    lines.append("")
    return "\n".join(lines)
