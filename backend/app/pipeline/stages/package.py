"""
Package stage - Generate source/extracted markdown, assets, and reports.

Output files:
- outputs/source.md: Original-document markdown (preserves source structure)
- outputs/extracted.md: Extracted form/table/figure semantic markdown
- outputs/documents/: Split markdown files for systems that use file boundaries
- outputs/assets_index.jsonl: Asset index for retrieval
- source_map.json: MD anchor to block mapping (for viewer)
- outputs/quality.json: Quality report
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.adapters.vlm import VLMAdapter
from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR
from app.models.org_chart import (
    EdgeType,
    OrgCategory,
    OrgChartGraph,
    OrgEdge,
    OrgGroup,
    OrgNode,
)
from app.pipeline.package_utils import (
    clean_html_table,
    clean_latex_symbols,
    html_table_to_text,
    infer_table_asset_title,
    semantic_table_to_text,
)
from app.pipeline.quality_gate import run_quality_gate, write_quality_gate
from app.pipeline.semantic.language import resolve_semantic_output_language
from app.pipeline.structured_rag import (
    build_form_documents_rag,
    build_structured_rag,
    build_structured_rag_with_vlm_fallback,
    looks_like_reference_table,
    parse_html_table,
    plan_document,
    write_structured_rag_outputs,
)

__all__ = [
    "PackageStage",
    "html_table_to_text",
    "infer_table_asset_title",
    "semantic_table_to_text",
]


def render_vlm_text(value: Any) -> str:
    """Normalize VLM text fields that may come back as str/list/dict."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            text = render_vlm_text(item)
            if text:
                lines.append(text)
        return "\n".join(lines).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()




def _decode_json_string_fragment(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value.replace('\\n', '\n').replace('\\"', '"').strip()


def _salvage_visual_jsonish_text(text: str) -> dict[str, Any]:
    """Recover useful figure fields from malformed JSON-like VLM output."""
    result: dict[str, Any] = {}
    if not text or not text.lstrip().startswith("{"):
        return result

    for key in ("semantic_caption", "image_type"):
        match = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
        if match:
            result[key] = _decode_json_string_fragment(match.group(1)).strip()

    array_match = re.search(r'"structured_content"\s*:\s*\[(.*?)(?:\]\s*,|\]\s*}|$)', text, re.DOTALL)
    if array_match:
        items = []
        for match in re.finditer(r'"((?:\\.|[^"\\])*)"', array_match.group(1), re.DOTALL):
            item = _decode_json_string_fragment(match.group(1)).strip()
            if item:
                items.append(item)
        if items:
            result["structured_content"] = items

    return result


def coerce_visual_vlm_output(output: dict[str, Any]) -> dict[str, Any]:
    """Merge JSON-encoded or JSON-like figure output back into VLM fields."""
    result = dict(output or {})
    candidates = [result.get("structured_content"), result.get("semantic_caption")]
    parsed: dict[str, Any] = {}
    for raw in candidates:
        text = render_vlm_text(raw)
        if not text.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(text)
        except Exception:
            value = _salvage_visual_jsonish_text(text)
        if isinstance(value, dict) and value:
            parsed = value
            break
    if not parsed:
        return result

    for key in ("semantic_caption", "image_type", "structured_content", "all_text", "facts", "keywords"):
        value = parsed.get(key)
        if value:
            result[key] = value
    return result

def split_vlm_lines(value: Any) -> list[str]:
    """Return clean, unique lines from VLM text/list fields."""
    text = render_vlm_text(value)
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip().lstrip("-•0123456789.、 "))
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


@dataclass
class Manifest:
    """Run manifest with engine versions and configuration."""

    doc_id: str
    run_id: str
    config_hash: str
    created_at: str
    engines: dict[str, Any] = field(default_factory=dict)
    pipeline_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "engines": self.engines,
            "pipeline_config": self.pipeline_config,
        }

    @staticmethod
    def from_config(
        doc_id: str,
        run_id: str,
        config_hash: str,
        config: PipelineConfig,
    ) -> "Manifest":
        """Create manifest from pipeline config."""
        from datetime import datetime

        # Extract MinerU engine info
        mineru_engine = {
            "name": "mineru",
            "backend": config.mineru.backend.value,
            "method": config.mineru.method.value,
            "lang": config.mineru.lang,
            "table": config.mineru.table,
            "formula": config.mineru.formula,
            "api_url": config.mineru.api_url,
            "vlm_url": config.mineru.vlm_url,
            "vlm_model_name": config.mineru.vlm_model_name,
            "model_source": config.mineru.model_source,
        }

        # Extract VLM engine info
        vlm_engine = {
            "name": "vlm",
            "base_url": config.vlm.base_url,
            "model": config.vlm.model,
            "api_mode": config.vlm.api_mode.value,
            "chat_template": config.vlm.chat_template,
            "decode_params": {
                "temperature": config.vlm.decode_params.temperature,
                "top_p": config.vlm.decode_params.top_p,
                "max_tokens": config.vlm.decode_params.max_tokens,
            },
            "image_mode": config.vlm.image_mode.value,
        }

        review_vlm_engine = {
            "name": "review_vlm",
            "base_url": config.review_vlm.base_url,
            "model": config.review_vlm.model,
            "api_mode": config.review_vlm.api_mode.value,
            "decode_params": {
                "temperature": config.review_vlm.decode_params.temperature,
                "top_p": config.review_vlm.decode_params.top_p,
                "max_tokens": config.review_vlm.decode_params.max_tokens,
            },
            "image_mode": config.review_vlm.image_mode.value,
        }

        # Pipeline configuration summary
        pipeline_summary = {
            "enrich": {
                "enable_vlm": config.enrich.enable_vlm,
                "vlm_enrich_forms": config.enrich.vlm_enrich_forms,
                "vlm_enrich_figures": config.enrich.vlm_enrich_figures,
                "vlm_enrich_tables": config.enrich.vlm_enrich_tables,
                "table_vlm_budget": config.enrich.table_vlm_budget,
            },
            "package": {
                "generate_dataset_md": config.package.generate_dataset_md,
                "generate_rag_md": config.package.generate_rag_md,
                "generate_chunks": config.package.generate_chunks,
                "semantic_output_language": config.package.semantic_output_language.value,
                "enable_semantic_repair": config.package.enable_semantic_repair,
                "chunk_max_tokens": config.package.chunk_max_tokens,
            },
        }

        return Manifest(
            doc_id=doc_id,
            run_id=run_id,
            config_hash=config_hash,
            created_at=datetime.now().isoformat(),
            engines={
                "mineru": mineru_engine,
                "vlm": vlm_engine,
                "review_vlm": review_vlm_engine,
            },
            pipeline_config=pipeline_summary,
        )


@dataclass
class AssetEntry:
    """Entry in assets_index.jsonl."""

    type: str  # form_asset, figure_asset, table_asset
    asset_id: str
    doc_id: str
    run_id: str
    title: str
    page_idx: int
    asset_path: str
    block_id: str
    retrieval_text: str
    triggers: list[str] = field(default_factory=list)
    guide_ref: str | None = None
    # Additional fields for forms
    filling_guide: str | None = None
    field_schema: list[dict[str, Any]] = field(default_factory=list)
    # Figure-specific fields
    semantic_caption: str | None = None
    structured_content: str | None = None
    facts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # Quality flag
    needs_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = {
            "type": self.type,
            "asset_id": self.asset_id,
            "doc_id": self.doc_id,
            "run_id": self.run_id,
            "title": self.title,
            "triggers": self.triggers,
            "page_idx": self.page_idx,
            "asset_path": self.asset_path,
            "block_id": self.block_id,
            "guide_ref": self.guide_ref,
            "retrieval_text": self.retrieval_text,
            "needs_review": self.needs_review,
        }
        # Add optional fields if present
        if self.filling_guide:
            result["filling_guide"] = self.filling_guide
        if self.field_schema:
            result["field_schema"] = self.field_schema
        if self.semantic_caption:
            result["semantic_caption"] = self.semantic_caption
        if self.structured_content:
            result["structured_content"] = self.structured_content
        if self.facts:
            result["facts"] = self.facts
        if self.keywords:
            result["keywords"] = self.keywords
        return result


@dataclass
class MdAnchor:
    """Markdown anchor mapping to blocks."""

    anchor_id: str
    md_range: list[int]  # [start, end] character positions
    block_ids: list[str]


@dataclass
class SourceMap:
    """Source map for viewer."""

    md_anchors: list[MdAnchor] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "md_anchors": [
                {
                    "anchor_id": a.anchor_id,
                    "md_range": a.md_range,
                    "block_ids": a.block_ids,
                }
                for a in self.md_anchors
            ],
        }


@dataclass
class QualityReport:
    """Quality report for a run."""

    doc_id: str
    run_id: str
    block_counts: dict[str, int] = field(default_factory=dict)
    page_count: int = 0
    total_text_length: int = 0
    asset_count: int = 0
    coverage: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "run_id": self.run_id,
            "block_counts": self.block_counts,
            "page_count": self.page_count,
            "total_text_length": self.total_text_length,
            "asset_count": self.asset_count,
            "coverage": self.coverage,
        }


@dataclass
class PackageStageResult:
    """Result from package stage."""

    success: bool
    dataset_md_path: Path | None = None
    rag_md_path: Path | None = None
    assets_index_path: Path | None = None
    source_map_path: Path | None = None
    quality_path: Path | None = None
    manifest_path: Path | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class PackageStage:
    """
    Package stage - generates final output files.

    Input: DocumentIR
    Output: source.md, extracted.md, documents/, assets/, assets_index.jsonl,
    source_map.json, quality.json
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.package_config = self.config.package

    async def run(
        self,
        doc_id: str,
        run_id: str,
        document_ir: DocumentIR,
        run_path: Path,
        parse_cache_path: Path | None = None,
        config_hash: str = "",
    ) -> PackageStageResult:
        """
        Run package stage.

        Args:
            doc_id: Document ID
            run_id: Run ID
            document_ir: Normalized document IR
            run_path: Path to run output directory
            parse_cache_path: Path to parse cache (for asset extraction)
            config_hash: Configuration hash for manifest

        Returns:
            PackageStageResult with output paths
        """
        try:
            outputs_dir = run_path / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            assets_dir = run_path / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            semantic_output_language = resolve_semantic_output_language(
                self.config.package.semantic_output_language.value,
                document_ir,
            )

            # Load enrichments from enrich stage (if exists)
            enrichments = self._load_enrichments(outputs_dir)

            # 1. Export assets and build index (with enrichments integration)
            assets, asset_map = await self._export_assets(
                document_ir=document_ir,
                assets_dir=assets_dir,
                parse_cache_path=parse_cache_path,
                enrichments=enrichments,
                semantic_output_language=semantic_output_language,
            )

            # 2. Generate source markdown (high-fidelity original document)
            dataset_md, dataset_source_map = self._render_dataset_md(
                document_ir=document_ir,
            )

            # 3. Generate retrieval markdown used by the internal chunker.
            rag_md, rag_source_map = self._render_rag_md(
                document_ir=document_ir,
                asset_map=asset_map,
                enrichments=enrichments,
                semantic_output_language=semantic_output_language,
            )

            # 4. Generate quality report
            quality = self._generate_quality_report(
                document_ir=document_ir,
                assets=assets,
            )

            # 5. Generate planner/extractor row-level RAG artifacts.
            if self.config.enrich.enable_vlm and self.config.enrich.vlm_enrich_tables:
                structured_output = await build_structured_rag_with_vlm_fallback(
                    document_ir=document_ir,
                    run_path=run_path,
                    vlm_adapter=VLMAdapter(self.config.vlm),
                    max_pages=min(self.config.enrich.table_vlm_budget, 3),
                    semantic_output_language=semantic_output_language,
                )
            else:
                structured_output = build_structured_rag(
                    document_ir,
                    semantic_output_language=semantic_output_language,
                )
            if not structured_output.records:
                structured_output = build_form_documents_rag(
                    document_ir=document_ir,
                    enrichments=enrichments,
                    semantic_output_language=semantic_output_language,
                )

            if structured_output.records:
                if structured_output.plan.document_type == "form_collection":
                    form_page_indices = self._collect_structured_form_page_indices(structured_output.records)
                    rag_md, rag_source_map = self._render_rag_md(
                        document_ir=document_ir,
                        asset_map=asset_map,
                        enrichments=enrichments,
                        suppress_form_enrichment=True,
                        excluded_page_indices=form_page_indices,
                        semantic_output_language=semantic_output_language,
                    )
                else:
                    rag_md = structured_output.rag_markdown

            source_md = rag_md

            # 6. Write all outputs
            source_md_path = outputs_dir / "source.md"
            source_md_path.write_text(source_md, encoding="utf-8")

            # Compatibility aliases for older endpoints/evaluators.
            dataset_md_path = outputs_dir / "dataset.md"
            dataset_md_path.write_text(dataset_md, encoding="utf-8")

            rag_md_path = outputs_dir / "rag.md"
            rag_md_path.write_text(source_md, encoding="utf-8")

            assets_index_path = outputs_dir / "assets_index.jsonl"
            with open(assets_index_path, "w", encoding="utf-8") as f:
                for asset in assets:
                    f.write(json.dumps(asset.to_dict(), ensure_ascii=False) + "\n")

            source_map_path = run_path / "source_map.json"
            # Combine source maps (use rag view as primary)
            combined_source_map = SourceMap(
                md_anchors=rag_source_map.md_anchors,
            )
            with open(source_map_path, "w", encoding="utf-8") as f:
                json.dump(combined_source_map.to_dict(), f, ensure_ascii=False, indent=2)

            quality_path = outputs_dir / "quality.json"
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump(quality.to_dict(), f, ensure_ascii=False, indent=2)

            structured_paths = write_structured_rag_outputs(
                output=structured_output,
                outputs_dir=outputs_dir,
            )
            document_export_paths = self._write_document_exports(
                outputs_dir=outputs_dir,
                source_md=source_md,
                assets=assets,
                structured_paths=structured_paths,
                document_ir=document_ir,
                semantic_output_language=semantic_output_language,
            )

            quality_gate = await run_quality_gate(
                document_ir=document_ir,
                source_md=source_md,
                assets=assets,
                structured_output=structured_output,
                enrichments=enrichments,
                run_path=run_path,
                vlm_adapter=VLMAdapter(self.config.review_vlm) if self.config.enrich.enable_vlm else None,
                max_vlm_audits=2 if self.config.enrich.enable_vlm else 0,
                semantic_output_language=semantic_output_language,
            )

            if self._quality_gate_needs_structured_repair(quality_gate):
                repaired_output = build_form_documents_rag(
                    document_ir=document_ir,
                    enrichments=enrichments,
                    semantic_output_language=semantic_output_language,
                )
                if repaired_output.records:
                    structured_output = repaired_output
                    structured_paths = write_structured_rag_outputs(
                        output=structured_output,
                        outputs_dir=outputs_dir,
                    )
                    document_export_paths = self._write_document_exports(
                        outputs_dir=outputs_dir,
                        source_md=source_md,
                        assets=assets,
                        structured_paths=structured_paths,
                        document_ir=document_ir,
                        semantic_output_language=semantic_output_language,
                    )
                    quality_gate = await run_quality_gate(
                        document_ir=document_ir,
                        source_md=source_md,
                        assets=assets,
                        structured_output=structured_output,
                        enrichments=enrichments,
                        run_path=run_path,
                        vlm_adapter=VLMAdapter(self.config.review_vlm) if self.config.enrich.enable_vlm else None,
                        max_vlm_audits=1 if self.config.enrich.enable_vlm else 0,
                        semantic_output_language=semantic_output_language,
                    )
                    quality_gate.stats["structured_repair_applied"] = True

            semantic_repair_enabled = bool(
                self.config.enrich.enable_vlm and self.config.package.enable_semantic_repair
            )
            semantic_repair_stats = {"enabled": semantic_repair_enabled, "applied_count": 0}
            if semantic_repair_enabled and self._quality_gate_needs_semantic_repair(quality_gate):
                semantic_repair_stats = await self._apply_semantic_repair(
                    outputs_dir=outputs_dir,
                    document_ir=document_ir,
                    source_md=source_md,
                    structured_output=structured_output,
                    quality_gate=quality_gate,
                    enrichments=enrichments,
                    semantic_output_language=semantic_output_language,
                )
                quality_gate.stats["semantic_repair"] = semantic_repair_stats
                if semantic_repair_stats.get("applied_count", 0) > 0:
                    quality_gate.stats["semantic_repair_applied"] = True
                    quality_gate.stats["post_repair_note"] = (
                        "Reviewer model rewrote semantic markdown/chunks after the rule-based quality gate."
                    )
                    self._settle_quality_gate_after_semantic_repair(quality_gate, semantic_repair_stats)
                if semantic_repair_stats.get("blocked_count", 0) > 0:
                    quality_gate.stats["semantic_repair_blocked"] = True
                    quality_gate.stats["auto_rag_ready"] = False
                    quality_gate.stats["post_repair_note"] = (
                        "Reviewer model blocked unsafe parser fallback output from automatic RAG ingestion."
                    )
                if semantic_repair_stats.get("fallback_count", 0) > 0:
                    quality_gate.stats["semantic_repair_fallback_retained"] = True
                    quality_gate.stats["auto_rag_ready"] = False
                    quality_gate.stats["post_repair_note"] = (
                        "Reviewer model could not produce a valid final rewrite; retained the pre-review structured output/chunks with needs_review metadata."
                    )
            else:
                if not self.config.package.enable_semantic_repair:
                    semantic_repair_stats["skipped_reason"] = "semantic_repair_disabled"
                else:
                    semantic_repair_stats["skipped_reason"] = "quality_gate_passed_or_vlm_disabled"

            write_quality_gate(quality_gate, outputs_dir)
            self._write_llm_vlm_outputs(
                outputs_dir=outputs_dir,
                enrichments=enrichments,
                quality_gate=quality_gate,
                semantic_repair_stats=semantic_repair_stats,
            )

            # 7. Generate manifest with engine versions
            manifest = Manifest.from_config(
                doc_id=doc_id,
                run_id=run_id,
                config_hash=config_hash,
                config=self.config,
            )
            manifest_path = run_path / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)

            stats = {
                "source_md_chars": len(source_md),
                "asset_count": len(assets),
                "anchor_count": len(combined_source_map.md_anchors),
                "structured_rag": structured_output.stats,
                "structured_outputs": structured_paths,
                "document_exports": document_export_paths,
                "quality_gate": quality_gate.to_dict(),
                "semantic_repair": semantic_repair_stats,
            }

            return PackageStageResult(
                success=True,
                dataset_md_path=dataset_md_path,
                rag_md_path=rag_md_path,
                assets_index_path=assets_index_path,
                source_map_path=source_map_path,
                quality_path=quality_path,
                manifest_path=manifest_path,
                stats=stats,
            )

        except Exception as e:
            import logging
            import traceback
            logging.error(f"Package stage failed: {type(e).__name__}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return PackageStageResult(
                success=False,
                error=str(e),
            )

    @staticmethod
    def _quality_gate_needs_structured_repair(quality_gate: Any) -> bool:
        repair_codes = {
            "html_table_without_semantic_text",
            "semantic_template_incomplete",
            "semantic_summary_too_dense",
            "form_signature_fields_missing",
            "table_notes_missing",
            "form_like_document_not_structured",
            "possible_over_split_form",
        }
        for issue in getattr(quality_gate, "issues", []) or []:
            if getattr(issue, "code", None) in repair_codes:
                return True
        return False

    @staticmethod
    def _quality_gate_needs_semantic_repair(quality_gate: Any) -> bool:
        repair_codes = {
            "html_table_without_semantic_text",
            "semantic_output_too_short",
            "semantic_template_incomplete",
            "semantic_summary_too_dense",
            "form_signature_fields_missing",
            "table_notes_missing",
            "form_like_document_not_structured",
            "possible_over_split_form",
            "field_name_too_long",
            "merged_field_detected",
            "too_many_generic_fields",
            "version_misclassified_as_note",
            "summary_contains_ellipsis",
            "raw_parser_residue",
            "ocr_title_noise",
            "english_noise_high",
            "structured_output_empty",
            "vlm_enrichment_parse_failed",
            "target_language_mismatch",
            "vlm_audit_missing_items",
        }
        for issue in getattr(quality_gate, "issues", []) or []:
            if getattr(issue, "code", None) in repair_codes:
                return True

        stats = dict(getattr(quality_gate, "stats", {}) or {})
        if stats.get("structured_document_type") == "form_document" and int(stats.get("structured_record_count") or 0) > 0:
            return True

        semantic_quality = dict(stats.get("semantic_quality") or {})
        recommended_repairs = semantic_quality.get("recommended_repairs") or []
        readiness = float(semantic_quality.get("rag_readiness_score", 1.0) or 1.0)
        correctness = float(semantic_quality.get("correctness_score", 1.0) or 1.0)
        return bool(recommended_repairs and (readiness < 0.92 or correctness < 0.92))

    async def _apply_semantic_repair(
        self,
        *,
        outputs_dir: Path,
        document_ir: DocumentIR,
        source_md: str,
        structured_output: Any,
        quality_gate: Any,
        enrichments: dict[str, dict[str, Any]],
        semantic_output_language: str,
        review_adapter: Any | None = None,
    ) -> dict[str, Any]:
        """Use the reviewer model to rewrite low-quality semantic Markdown.

        The deterministic records remain as evidence; the repaired Markdown is
        written back to form/structured outputs and appended as RAG chunks.
        """

        review_adapter = review_adapter or VLMAdapter(self.config.review_vlm)
        plan = getattr(structured_output, "plan", None)
        document_type = str(getattr(plan, "document_type", "") or "")
        stats: dict[str, Any] = {
            "enabled": True,
            "document_type": document_type,
            "applied_count": 0,
            "attempted_count": 0,
            "items": [],
            "trigger_codes": self._semantic_repair_trigger_codes(quality_gate),
        }

        if document_type == "form_collection":
            stats.update(
                await self._repair_form_markdown_outputs(
                    outputs_dir=outputs_dir,
                    document_ir=document_ir,
                    quality_gate=quality_gate,
                    enrichments=enrichments,
                    semantic_output_language=semantic_output_language,
                    review_adapter=review_adapter,
                )
            )
        else:
            stats.update(
                await self._repair_structured_markdown_output(
                    outputs_dir=outputs_dir,
                    document_ir=document_ir,
                    source_md=source_md,
                    structured_output=structured_output,
                    quality_gate=quality_gate,
                    enrichments=enrichments,
                    semantic_output_language=semantic_output_language,
                    review_adapter=review_adapter,
                )
            )

        (outputs_dir / "semantic_repair.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return stats


    def _write_llm_vlm_outputs(
        self,
        *,
        outputs_dir: Path,
        enrichments: dict[str, dict[str, Any]],
        quality_gate: Any,
        semantic_repair_stats: dict[str, Any],
    ) -> None:
        """Write a human-readable trace of model outputs used by packaging."""

        lines = ["# LLM/VLM Outputs", ""]
        lines.append("## Enrichment Outputs")
        if enrichments:
            for block_id, enrichment in enrichments.items():
                output = enrichment.get("output") or {}
                quality = enrichment.get("quality") or {}
                input_info = enrichment.get("input") or {}
                evidence = enrichment.get("evidence") or {}
                lines.extend([
                    f"### {block_id}",
                    f"- kind: {enrichment.get('kind') or ''}",
                    f"- page: {input_info.get('page_idx', evidence.get('page_idx', 'unknown'))}",
                    f"- needs_review: {quality.get('needs_review', output.get('needs_review', ''))}",
                    f"- tokens_used: {quality.get('tokens_used', '')}",
                    f"- error: {output.get('_error', enrichment.get('error', '')) or ''}",
                    "",
                    "```json",
                    self._truncate_text(json.dumps(output, ensure_ascii=False, indent=2), 4000),
                    "```",
                    "",
                ])
        else:
            lines.extend(["No enrichment outputs were recorded.", ""])

        lines.append("## VLM Quality Audits")
        audits = getattr(quality_gate, "vlm_audits", []) or []
        if audits:
            for idx, audit in enumerate(audits, start=1):
                lines.extend([
                    f"### Audit {idx}",
                    f"- success: {audit.get('success')}",
                    f"- page: {audit.get('page_idx')}",
                    f"- reasons: {', '.join(str(item) for item in audit.get('reasons', []) or [])}",
                    f"- tokens_used: {audit.get('tokens_used', '')}",
                    f"- error: {audit.get('error') or ''}",
                    "",
                    "```json",
                    self._truncate_text(json.dumps(audit.get("output") or {}, ensure_ascii=False, indent=2), 3000),
                    "```",
                    "",
                ])
        else:
            lines.extend(["No VLM quality audits were run.", ""])

        lines.append("## Semantic Repair")
        lines.extend([
            "```json",
            self._truncate_text(json.dumps(semantic_repair_stats or {}, ensure_ascii=False, indent=2), 5000),
            "```",
            "",
        ])
        (outputs_dir / "llm_vlm_outputs.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    async def _repair_form_markdown_outputs(
        self,
        *,
        outputs_dir: Path,
        document_ir: DocumentIR,
        quality_gate: Any,
        enrichments: dict[str, dict[str, Any]],
        semantic_output_language: str,
        review_adapter: Any,
        max_forms: int = 3,
    ) -> dict[str, Any]:
        forms_index_path = outputs_dir / "forms_index.json"
        if not forms_index_path.exists():
            return {"skipped_reason": "forms_index_missing"}

        try:
            forms_index = json.loads(forms_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"skipped_reason": "forms_index_unreadable", "error": str(exc)}
        if not isinstance(forms_index, list) or not forms_index:
            return {"skipped_reason": "no_form_entries"}

        targets = self._semantic_repair_form_targets(forms_index, quality_gate, max_forms=max_forms)
        repaired_items: list[dict[str, Any]] = []
        blocked_items: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []

        for item in targets:
            form_id = str(item.get("form_id") or "")
            md_path = Path(str((item.get("files") or {}).get("markdown") or ""))
            item_stats: dict[str, Any] = {
                "form_id": form_id,
                "subdoc_id": item.get("subdoc_id"),
                "status": "skipped",
            }
            if not md_path.exists():
                item_stats["reason"] = "markdown_missing"
                items.append(item_stats)
                continue

            current_markdown = md_path.read_text(encoding="utf-8")
            page_indices = self._safe_page_indices(item.get("page_indices"))
            issues = self._quality_issues_as_dicts(quality_gate, page_indices)
            must_rewrite = bool(issues)
            source_evidence = self._semantic_repair_source_evidence(
                document_ir=document_ir,
                page_indices=page_indices,
                enrichments=enrichments,
                fallback_text=current_markdown,
            )
            result = await review_adapter.enrich(
                kind="semantic_repair",
                image_path=None,
                context=source_evidence,
                doc_id=document_ir.doc_id,
                run_id=document_ir.run_id,
                page_idx=page_indices[0] if page_indices else None,
                bbox=None,
                extra_vars={
                    "semantic_output_language": semantic_output_language,
                    "quality_issues_json": json.dumps(issues, ensure_ascii=False, indent=2),
                    "source_evidence": source_evidence,
                    "current_markdown": self._truncate_text(current_markdown, 12000),
                },
            )
            item_stats["status"] = "attempted"
            item_stats["model_success"] = bool(getattr(result, "success", False))
            item_stats["tokens_used"] = int(getattr(result, "tokens_used", 0) or 0)
            item_stats["duration_seconds"] = float(getattr(result, "duration_seconds", 0) or 0)
            if not getattr(result, "success", False):
                item_stats["status"] = "fallback_retained"
                item_stats["reason"] = "semantic_repair_failed"
                item_stats["error"] = str(getattr(result, "error", "") or "semantic_repair_failed")
                self._retain_form_candidate_for_review(
                    outputs_dir=outputs_dir,
                    form_item=item,
                    item_stats=item_stats,
                    current_markdown=current_markdown,
                    document_ir=document_ir,
                    semantic_output_language=semantic_output_language,
                )
                blocked_items.append(item)
                items.append(item_stats)
                continue

            output = dict(getattr(result, "output", {}) or {})
            status = str(output.get("status") or "").lower()
            repaired_markdown = render_vlm_text(output.get("repaired_markdown"))
            title = self._clean_export_title(str(item.get("title") or form_id or "Form"))
            if status == "pass" and not repaired_markdown:
                item_stats["summary"] = str(output.get("summary") or "")
                if must_rewrite:
                    item_stats["status"] = "fallback_retained"
                    item_stats["reason"] = "reviewer_passed_despite_quality_issues"
                    self._retain_form_candidate_for_review(
                        outputs_dir=outputs_dir,
                        form_item=item,
                        item_stats=item_stats,
                        current_markdown=current_markdown,
                        document_ir=document_ir,
                        semantic_output_language=semantic_output_language,
                    )
                    blocked_items.append(item)
                else:
                    item_stats["status"] = "passed_by_reviewer"
                    item["semantic_repair"] = {
                        "status": "passed_by_reviewer",
                        "summary": item_stats["summary"],
                        "auto_rag_ready": True,
                    }
                items.append(item_stats)
                continue

            repaired_markdown = self._normalize_repaired_markdown(repaired_markdown, title)
            if not self._semantic_repair_markdown_is_usable(
                repaired_markdown,
                current_markdown,
                semantic_output_language,
            ):
                item_stats["status"] = "fallback_retained"
                item_stats["reason"] = "repaired_markdown_not_usable"
                item_stats["summary"] = str(output.get("summary") or "")
                self._retain_form_candidate_for_review(
                    outputs_dir=outputs_dir,
                    form_item=item,
                    item_stats=item_stats,
                    current_markdown=current_markdown,
                    document_ir=document_ir,
                    semantic_output_language=semantic_output_language,
                )
                blocked_items.append(item)
                items.append(item_stats)
                continue

            md_path.write_text(repaired_markdown, encoding="utf-8")
            self._write_repaired_split_form_document(
                outputs_dir=outputs_dir,
                form_item=item,
                repaired_markdown=repaired_markdown,
                document_ir=document_ir,
                semantic_output_language=semantic_output_language,
            )
            repair_record = {
                "form_id": form_id,
                "subdoc_id": item.get("subdoc_id"),
                "logical_doc_id": item.get("logical_doc_id"),
                "files": item.get("files") or {},
                "title": title,
                "page_indices": page_indices,
                "markdown": repaired_markdown,
                "summary": str(output.get("summary") or ""),
                "applied_repairs": [str(value) for value in output.get("applied_repairs", []) if str(value).strip()],
                "confidence": float(output.get("confidence", 0) or 0),
            }
            repaired_items.append(repair_record)
            item["semantic_repair"] = {
                "status": "applied",
                "summary": repair_record["summary"],
                "applied_repairs": repair_record["applied_repairs"],
                "confidence": repair_record["confidence"],
                "auto_rag_ready": True,
            }
            item_stats.update(item["semantic_repair"])
            item_stats["status"] = "applied"
            items.append(item_stats)

        if repaired_items or blocked_items:
            forms_index_path.write_text(json.dumps(forms_index, ensure_ascii=False, indent=2), encoding="utf-8")
            self._rewrite_authoritative_form_chunks(
                outputs_dir=outputs_dir,
                document_ir=document_ir,
                repaired_items=repaired_items,
                blocked_items=[],
            )
            self._rebuild_structured_rag_from_form_files(outputs_dir, document_ir)

        return {
            "attempted_count": len(targets),
            "applied_count": len(repaired_items),
            "blocked_count": 0,
            "fallback_count": len(blocked_items),
            "items": items,
        }

    async def _repair_structured_markdown_output(
        self,
        *,
        outputs_dir: Path,
        document_ir: DocumentIR,
        source_md: str,
        structured_output: Any,
        quality_gate: Any,
        enrichments: dict[str, dict[str, Any]],
        semantic_output_language: str,
        review_adapter: Any,
    ) -> dict[str, Any]:
        current_markdown = str(getattr(structured_output, "rag_markdown", "") or source_md or "")
        if not current_markdown.strip():
            return {"skipped_reason": "current_markdown_empty"}

        issues = self._quality_issues_as_dicts(quality_gate, [])
        source_evidence = self._semantic_repair_source_evidence(
            document_ir=document_ir,
            page_indices=[],
            enrichments=enrichments,
            fallback_text=current_markdown,
        )
        result = await review_adapter.enrich(
            kind="semantic_repair",
            image_path=None,
            context=source_evidence,
            doc_id=document_ir.doc_id,
            run_id=document_ir.run_id,
            page_idx=None,
            bbox=None,
            extra_vars={
                "semantic_output_language": semantic_output_language,
                "quality_issues_json": json.dumps(issues, ensure_ascii=False, indent=2),
                "source_evidence": source_evidence,
                "current_markdown": self._truncate_text(current_markdown, 14000),
            },
        )
        item_stats = {
            "target": "structured_rag",
            "model_success": bool(getattr(result, "success", False)),
            "tokens_used": int(getattr(result, "tokens_used", 0) or 0),
            "duration_seconds": float(getattr(result, "duration_seconds", 0) or 0),
        }
        if not getattr(result, "success", False):
            item_stats["status"] = "fallback_retained"
            item_stats["reason"] = "semantic_repair_failed"
            item_stats["error"] = str(getattr(result, "error", "") or "semantic_repair_failed")
            self._retain_structured_candidate_for_review(
                outputs_dir=outputs_dir,
                document_ir=document_ir,
                current_markdown=current_markdown,
                title=str(getattr(getattr(structured_output, "plan", None), "title", "") or Path(document_ir.source.path).stem),
                reason=item_stats["reason"],
                summary=item_stats["error"],
                semantic_output_language=semantic_output_language,
            )
            return {"attempted_count": 1, "applied_count": 0, "blocked_count": 0, "fallback_count": 1, "items": [item_stats]}

        output = dict(getattr(result, "output", {}) or {})
        repaired_markdown = render_vlm_text(output.get("repaired_markdown"))
        if str(output.get("status") or "").lower() == "pass" and not repaired_markdown:
            item_stats["summary"] = str(output.get("summary") or "")
            if issues:
                item_stats["status"] = "fallback_retained"
                item_stats["reason"] = "reviewer_passed_despite_quality_issues"
                self._retain_structured_candidate_for_review(
                    outputs_dir=outputs_dir,
                    document_ir=document_ir,
                    current_markdown=current_markdown,
                    title=str(getattr(getattr(structured_output, "plan", None), "title", "") or Path(document_ir.source.path).stem),
                    reason=item_stats["reason"],
                    summary=item_stats["summary"],
                    semantic_output_language=semantic_output_language,
                )
                return {"attempted_count": 1, "applied_count": 0, "blocked_count": 0, "fallback_count": 1, "items": [item_stats]}
            item_stats["status"] = "passed_by_reviewer"
            return {"attempted_count": 1, "applied_count": 0, "blocked_count": 0, "fallback_count": 0, "items": [item_stats]}

        title = self._semantic_repair_title(
            document_ir=document_ir,
            source_md=source_md,
            structured_output=structured_output,
        )
        repaired_markdown = self._normalize_repaired_markdown(repaired_markdown, title)
        if not self._semantic_repair_markdown_is_usable(repaired_markdown, current_markdown, semantic_output_language):
            item_stats["status"] = "fallback_retained"
            item_stats["reason"] = "repaired_markdown_not_usable"
            item_stats["summary"] = str(output.get("summary") or "")
            self._retain_structured_candidate_for_review(
                outputs_dir=outputs_dir,
                document_ir=document_ir,
                current_markdown=current_markdown,
                title=title,
                reason=item_stats["reason"],
                summary=item_stats["summary"],
                semantic_output_language=semantic_output_language,
            )
            return {"attempted_count": 1, "applied_count": 0, "blocked_count": 0, "fallback_count": 1, "items": [item_stats]}

        for filename in ("structured_rag.md", "source.md", "rag.md"):
            path = outputs_dir / filename
            if path.exists() or filename == "structured_rag.md":
                path.write_text(repaired_markdown, encoding="utf-8")
        self._write_repaired_main_document_export(outputs_dir, repaired_markdown)
        repair_record = {
            "form_id": "structured_rag",
            "subdoc_id": "structured_rag",
            "title": title,
            "page_indices": self._document_page_indices(document_ir),
            "markdown": repaired_markdown,
            "summary": str(output.get("summary") or ""),
            "applied_repairs": [str(value) for value in output.get("applied_repairs", []) if str(value).strip()],
            "confidence": float(output.get("confidence", 0) or 0),
        }
        self._write_semantic_repair_chunks(outputs_dir, document_ir, [repair_record], replace=True)
        item_stats.update({
            "status": "applied",
            "summary": repair_record["summary"],
            "applied_repairs": repair_record["applied_repairs"],
            "confidence": repair_record["confidence"],
        })
        return {"attempted_count": 1, "applied_count": 1, "blocked_count": 0, "fallback_count": 0, "items": [item_stats]}

    @staticmethod
    def _write_repaired_main_document_export(outputs_dir: Path, markdown: str) -> None:
        index_path = outputs_dir / "documents_index.json"
        target: Path | None = None
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                index = []
            if isinstance(index, list):
                for entry in index:
                    if not isinstance(entry, dict) or str(entry.get("document_id") or "") != "main":
                        continue
                    file_value = str(entry.get("file") or "")
                    if file_value:
                        target = Path(file_value)
                    break
        fallback = outputs_dir / "documents" / "main.md"
        if target is None and fallback.exists():
            target = fallback
        if target is None or not target.parent.exists():
            return
        target.write_text(markdown, encoding="utf-8")


    def _retain_structured_candidate_for_review(
        self,
        *,
        outputs_dir: Path,
        document_ir: DocumentIR,
        current_markdown: str,
        title: str,
        reason: str,
        summary: str,
        semantic_output_language: str,
    ) -> None:
        """Keep the best pre-review semantic output when reviewer repair is unusable."""

        markdown = self._normalize_repaired_markdown(current_markdown, title)
        for filename in ("structured_rag.md", "source.md", "rag.md"):
            path = outputs_dir / filename
            if path.exists() or filename == "structured_rag.md":
                path.write_text(markdown, encoding="utf-8")
        self._write_repaired_main_document_export(outputs_dir, markdown)

        chunks_path = outputs_dir / "structured_chunks.jsonl"
        if not self._annotate_jsonl_chunks_review_fallback(
            chunks_path,
            reason=reason,
            summary=summary,
        ):
            self._write_markdown_fallback_chunks(
                target_path=chunks_path,
                document_ir=document_ir,
                markdown=markdown,
                title=title,
                page_indices=self._document_page_indices(document_ir),
                reason=reason,
                summary=summary,
                replace=True,
            )

    def _retain_form_candidate_for_review(
        self,
        *,
        outputs_dir: Path,
        form_item: dict[str, Any],
        item_stats: dict[str, Any],
        current_markdown: str,
        document_ir: DocumentIR,
        semantic_output_language: str,
    ) -> None:
        """Keep form candidate output while marking it as reviewer-fallback content."""

        reason = str(item_stats.get("reason") or "semantic_repair_not_usable")
        summary = str(item_stats.get("summary") or item_stats.get("error") or "")
        item_stats["auto_rag_ready"] = False
        item_stats["needs_review"] = True
        item_stats["fallback_source"] = "pre_reviewer_structured_output"

        md_path = Path(str((form_item.get("files") or {}).get("markdown") or ""))
        if md_path.parent.exists() and not md_path.exists():
            md_path.write_text(current_markdown, encoding="utf-8")
        self._write_repaired_split_form_document(
            outputs_dir=outputs_dir,
            form_item=form_item,
            repaired_markdown=current_markdown,
            document_ir=document_ir,
            semantic_output_language=semantic_output_language,
        )

        changed = self._annotate_jsonl_chunks_review_fallback(
            outputs_dir / "structured_chunks.jsonl",
            reason=reason,
            summary=summary,
            form_item=form_item,
        )
        chunks_value = str((form_item.get("files") or {}).get("chunks") or "")
        form_chunks_path = Path(chunks_value) if chunks_value else None
        form_chunks_changed = False
        if form_chunks_path is not None:
            form_chunks_changed = self._annotate_jsonl_chunks_review_fallback(
                form_chunks_path,
                reason=reason,
                summary=summary,
            )
        page_indices = self._safe_page_indices(form_item.get("page_indices"))
        title = self._clean_export_title(str(form_item.get("title") or form_item.get("form_id") or "Form"))
        form_id = str(form_item.get("form_id") or "form")
        if not changed:
            self._write_markdown_fallback_chunks(
                target_path=outputs_dir / "structured_chunks.jsonl",
                document_ir=document_ir,
                markdown=current_markdown,
                title=title,
                page_indices=page_indices,
                reason=reason,
                summary=summary,
                form_id=form_id,
                subdoc_id=form_item.get("subdoc_id"),
                logical_doc_id=form_item.get("logical_doc_id"),
                append=True,
            )
        if form_chunks_path is not None and not form_chunks_changed:
            self._write_markdown_fallback_chunks(
                target_path=form_chunks_path,
                document_ir=document_ir,
                markdown=current_markdown,
                title=title,
                page_indices=page_indices,
                reason=reason,
                summary=summary,
                form_id=form_id,
                subdoc_id=form_item.get("subdoc_id"),
                logical_doc_id=form_item.get("logical_doc_id"),
                replace=True,
            )

        form_item["semantic_repair"] = {
            "status": "fallback_retained",
            "reason": reason,
            "summary": summary,
            "auto_rag_ready": False,
            "needs_review": True,
            "fallback_source": "pre_reviewer_structured_output",
        }

    def _annotate_jsonl_chunks_review_fallback(
        self,
        path: Path,
        *,
        reason: str,
        summary: str,
        form_item: dict[str, Any] | None = None,
    ) -> bool:
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            return False
        lines: list[str] = []
        changed = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                lines.append(raw_line)
                continue
            if form_item is not None and not self._chunk_matches_form_item(chunk, form_item):
                lines.append(json.dumps(chunk, ensure_ascii=False))
                continue
            metadata = dict(chunk.get("metadata") or {})
            metadata.update({
                "auto_rag_ready": False,
                "needs_review": True,
                "semantic_repair_status": "fallback_retained",
                "semantic_repair_reason": reason,
                "semantic_repair_summary": self._truncate_text(summary, 800),
                "fallback_source": "pre_reviewer_structured_output",
            })
            chunk["metadata"] = metadata
            lines.append(json.dumps(chunk, ensure_ascii=False))
            changed = True
        if changed:
            path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return changed

    def _write_markdown_fallback_chunks(
        self,
        *,
        target_path: Path,
        document_ir: DocumentIR,
        markdown: str,
        title: str,
        page_indices: list[int],
        reason: str,
        summary: str,
        form_id: str = "structured_rag",
        subdoc_id: Any | None = None,
        logical_doc_id: Any | None = None,
        replace: bool = False,
        append: bool = False,
    ) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        sections = self._split_repair_markdown_into_chunks(markdown)
        if not sections:
            return
        safe_id = re.sub(r"[^0-9A-Za-z_:-]+", "_", str(subdoc_id or form_id or "structured_rag")).strip("_") or "structured_rag"
        mode = "w" if replace or not append else "a"
        needs_leading_newline = False
        if mode == "a" and target_path.exists():
            existing = target_path.read_text(encoding="utf-8")
            needs_leading_newline = bool(existing and not existing.endswith("\n"))
        with open(target_path, mode, encoding="utf-8") as f:
            if needs_leading_newline:
                f.write("\n")
            for section_idx, section in enumerate(sections):
                chunk = {
                    "chunk_id": f"sr_fallback_{safe_id}_{section_idx:04d}",
                    "doc_id": str(subdoc_id or form_id or document_ir.doc_id),
                    "run_id": document_ir.run_id,
                    "view": "structured_fallback",
                    "content": section,
                    "block_ids": [f"structured_fallback:{form_id}"],
                    "page_indices": page_indices,
                    "attachments": [],
                    "metadata": {
                        "document_type": "structured_fallback",
                        "content_type": "structured_fallback",
                        "form_name": title,
                        "form_id": form_id,
                        "subdoc_id": subdoc_id,
                        "logical_doc_id": logical_doc_id,
                        "parent_doc_id": document_ir.doc_id,
                        "auto_rag_ready": False,
                        "needs_review": True,
                        "semantic_repair_status": "fallback_retained",
                        "semantic_repair_reason": reason,
                        "semantic_repair_summary": self._truncate_text(summary, 800),
                        "fallback_source": "pre_reviewer_structured_output",
                    },
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def _block_structured_output_from_auto_rag(
        self,
        *,
        outputs_dir: Path,
        document_ir: DocumentIR,
        title: str,
        reason: str,
        summary: str,
        semantic_output_language: str,
    ) -> None:
        blocked_markdown = self._render_blocked_structured_markdown(
            title=title,
            reason=reason,
            summary=summary,
            semantic_output_language=semantic_output_language,
        )
        for filename in ("structured_rag.md", "source.md", "rag.md"):
            path = outputs_dir / filename
            if path.exists() or filename == "structured_rag.md":
                path.write_text(blocked_markdown, encoding="utf-8")
        self._write_repaired_main_document_export(outputs_dir, blocked_markdown)
        for filename in ("structured_chunks.jsonl", "structured_records.jsonl"):
            path = outputs_dir / filename
            if path.exists():
                path.write_text("", encoding="utf-8")

    def _render_blocked_structured_markdown(
        self,
        *,
        title: str,
        reason: str,
        summary: str,
        semantic_output_language: str,
    ) -> str:
        clean_title = self._clean_export_title(title or "Document") or "Document"
        reason_text = reason.replace("_", " ").strip()
        if semantic_output_language == "en":
            lines = [
                f"# {clean_title}",
                "",
                "> LLM/VLM final review did not produce a trustworthy semantic rewrite. This output is blocked from automatic RAG ingestion.",
                "",
                "## Review Status",
                "- Auto RAG ready: no",
                f"- Reason: {reason_text}",
            ]
            if summary:
                lines.extend(["", "## Reviewer Summary", summary.strip()])
        else:
            lines = [
                f"# {clean_title}",
                "",
                "> LLM/VLM 最終審核沒有產出可信的語意重寫。本輸出已阻擋自動進入 RAG。",
                "",
                "## 審核狀態",
                "- 可自動進 RAG：否",
                f"- 原因：{reason_text}",
            ]
            if summary:
                lines.extend(["", "## Reviewer 摘要", summary.strip()])
        return "\n".join(lines).strip() + "\n"

    def _semantic_repair_form_targets(
        self,
        forms_index: list[dict[str, Any]],
        quality_gate: Any,
        *,
        max_forms: int,
    ) -> list[dict[str, Any]]:
        issue_pages = {
            int(getattr(issue, "page_idx"))
            for issue in getattr(quality_gate, "issues", []) or []
            if getattr(issue, "page_idx", None) is not None
        }
        if issue_pages:
            targeted = [
                item for item in forms_index
                if issue_pages.intersection(self._safe_page_indices(item.get("page_indices")))
            ]
            if targeted:
                return targeted[:max_forms]
        return forms_index[:max_forms]

    @staticmethod
    def _semantic_repair_trigger_codes(quality_gate: Any) -> list[str]:
        codes = []
        for issue in getattr(quality_gate, "issues", []) or []:
            code = str(getattr(issue, "code", "") or "")
            if code and code not in codes:
                codes.append(code)
        return codes

    @classmethod
    def _settle_quality_gate_after_semantic_repair(cls, quality_gate: Any, semantic_repair_stats: dict[str, Any]) -> None:
        if int(semantic_repair_stats.get("applied_count", 0) or 0) <= 0:
            return
        if int(semantic_repair_stats.get("fallback_count", 0) or 0) > 0:
            return
        if int(semantic_repair_stats.get("blocked_count", 0) or 0) > 0:
            return

        repairable_codes = {
            "structured_output_empty",
            "vlm_enrichment_parse_failed",
            "semantic_output_too_short",
            "semantic_template_incomplete",
            "semantic_summary_too_dense",
            "form_signature_fields_missing",
            "table_notes_missing",
            "form_like_document_not_structured",
            "possible_over_split_form",
            "field_name_too_long",
            "merged_field_detected",
            "too_many_generic_fields",
            "version_misclassified_as_note",
            "summary_contains_ellipsis",
            "raw_parser_residue",
            "ocr_title_noise",
            "english_noise_high",
            "target_language_mismatch",
            "vlm_audit_missing_items",
        }
        before_issues = list(getattr(quality_gate, "issues", []) or [])
        remaining_issues = [
            issue for issue in before_issues if str(getattr(issue, "code", "") or "") not in repairable_codes
        ]
        cleared_codes: list[str] = []
        for issue in before_issues:
            code = str(getattr(issue, "code", "") or "")
            if code in repairable_codes and code not in cleared_codes:
                cleared_codes.append(code)

        quality_gate.issues = remaining_issues
        stats = dict(getattr(quality_gate, "stats", {}) or {})
        stats["pre_semantic_repair_issue_count"] = len(before_issues)
        stats["post_semantic_repair_issue_count"] = len(remaining_issues)
        stats["pre_semantic_repair_vlm_audit_candidate_count"] = len(getattr(quality_gate, "vlm_audit_candidates", []) or [])
        stats["pre_semantic_repair_vlm_audit_count"] = len(getattr(quality_gate, "vlm_audits", []) or [])
        stats["semantic_repair_cleared_issue_codes"] = cleared_codes
        stats["semantic_repair_quality_settled"] = True
        stats["auto_rag_ready"] = not remaining_issues
        stats["issue_count"] = len(remaining_issues)
        stats["issues_by_code"] = cls._count_values(str(getattr(issue, "code", "") or "") for issue in remaining_issues)
        stats["issues_by_severity"] = cls._count_values(
            str(getattr(issue, "severity", "") or "") for issue in remaining_issues
        )
        semantic_quality = dict(stats.get("semantic_quality", {}) or {})
        semantic_quality["post_repair_issue_count"] = len(remaining_issues)
        semantic_quality["post_repair_status"] = cls._quality_status_from_issues(remaining_issues)
        if not remaining_issues:
            semantic_quality["rag_readiness_score"] = max(float(semantic_quality.get("rag_readiness_score", 0.0) or 0.0), 0.95)
            semantic_quality["recommended_repairs"] = []
        stats["semantic_quality"] = semantic_quality
        quality_gate.stats = stats
        quality_gate.status = cls._quality_status_from_issues(remaining_issues)
        quality_gate.score = cls._quality_score_from_issues(remaining_issues)
        if not remaining_issues:
            quality_gate.vlm_audit_candidates = []
            quality_gate.vlm_audits = []

    @staticmethod
    def _quality_status_from_issues(issues: list[Any]) -> str:
        if any(str(getattr(issue, "severity", "") or "") == "high" for issue in issues):
            return "needs_review"
        if any(str(getattr(issue, "severity", "") or "") == "medium" for issue in issues):
            return "warning"
        return "pass"

    @staticmethod
    def _quality_score_from_issues(issues: list[Any]) -> float:
        penalty = 0.0
        for issue in issues:
            severity = str(getattr(issue, "severity", "") or "")
            if severity == "high":
                penalty += 0.25
            elif severity == "medium":
                penalty += 0.12
            elif severity == "warning":
                penalty += 0.05
        return max(0.0, round(1.0 - penalty, 3))

    @staticmethod
    def _count_values(values: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            if not value:
                continue
            counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

    def _quality_issues_as_dicts(self, quality_gate: Any, page_indices: list[int]) -> list[dict[str, Any]]:
        pages = set(page_indices)
        result: list[dict[str, Any]] = []
        for issue in getattr(quality_gate, "issues", []) or []:
            page_idx = getattr(issue, "page_idx", None)
            if pages and page_idx is not None and int(page_idx) not in pages:
                continue
            result.append({
                "code": getattr(issue, "code", ""),
                "severity": getattr(issue, "severity", ""),
                "message": getattr(issue, "message", ""),
                "page_idx": page_idx,
                "evidence": getattr(issue, "evidence", {}) or {},
            })
            if len(result) >= 16:
                break
        return result

    def _semantic_repair_source_evidence(
        self,
        *,
        document_ir: DocumentIR,
        page_indices: list[int],
        enrichments: dict[str, dict[str, Any]],
        fallback_text: str,
    ) -> str:
        selected_pages = page_indices or self._document_page_indices(document_ir)[:4]
        lines = [
            f"Source file: {Path(document_ir.source.path).name}",
            f"Source path: {document_ir.source.path}",
            f"Pages: {', '.join(str(page + 1) for page in selected_pages) if selected_pages else 'unknown'}",
        ]
        for page_idx in selected_pages[:6]:
            lines.append(f"\n[Page {page_idx + 1} MinerU evidence]")
            blocks = document_ir.get_blocks_by_page(page_idx)
            for block in blocks[:80]:
                block_text = self._repair_block_text(block)
                if not block_text:
                    continue
                block_type = getattr(block.type, "value", str(block.type))
                lines.append(f"- {block_type}: {self._truncate_text(block_text, 900)}")

            enrichment_text = self._repair_enrichment_text_for_page(enrichments, page_idx)
            if enrichment_text:
                lines.append(f"\n[Page {page_idx + 1} enrich/VLM evidence]")
                lines.append(enrichment_text)

        if len("\n".join(lines)) < 600 and fallback_text:
            lines.append("\n[Current semantic text fallback evidence]")
            lines.append(self._truncate_text(fallback_text, 5000))
        return self._truncate_text("\n".join(lines), 16000)

    def _repair_enrichment_text_for_page(self, enrichments: dict[str, dict[str, Any]], page_idx: int) -> str:
        parts: list[str] = []
        for enrichment in enrichments.values():
            input_page = (enrichment.get("input") or {}).get("page_idx")
            evidence_page = (enrichment.get("evidence") or {}).get("page_idx")
            if input_page != page_idx and evidence_page != page_idx:
                continue
            output = dict(enrichment.get("output") or {})
            title = str(output.get("title") or "").strip()
            document_type = str(output.get("document_type") or enrichment.get("kind") or "").strip()
            if title or document_type:
                parts.append(f"- {document_type}: {title}".strip())
            error = str(output.get("_error") or "").strip()
            if error:
                parts.append("  model_error: " + self._truncate_text(error, 500))
            all_text = [str(item) for item in output.get("all_text", []) if str(item).strip()]
            if all_text:
                parts.append("  visible_text: " + self._truncate_text(" | ".join(all_text), 1800))
            caption = render_vlm_text(output.get("semantic_caption") or "")
            if caption:
                parts.append("  model_caption_or_raw_output: " + self._truncate_text(caption, 2600))
            structured_content = render_vlm_text(output.get("structured_content") or "")
            if structured_content and structured_content != caption:
                parts.append("  structured_content: " + self._truncate_text(structured_content, 2200))
            facts = [str(item) for item in output.get("facts", []) or [] if str(item).strip()]
            if facts:
                parts.append("  facts: " + self._truncate_text(" | ".join(facts), 1600))
            fields = []
            for field_item in output.get("field_schema", []) or []:
                if isinstance(field_item, dict) and field_item.get("name"):
                    fields.append(str(field_item.get("name")))
            if fields:
                parts.append("  fields: " + ", ".join(fields[:40]))
            guide = render_vlm_text(output.get("filling_guide") or "")
            if guide:
                parts.append("  semantic_output: " + self._truncate_text(guide, 1800))
        return "\n".join(parts)

    @staticmethod
    def _repair_block_text(block: Block) -> str:
        if block.type == BlockType.TEXT:
            return re.sub(r"\s+", " ", str(block.payload.get("text") or "")).strip()
        if block.type == BlockType.TABLE:
            text = str(block.payload.get("table_body") or "")
            text = re.sub(r"</tr>|<tr[^>]*>", "\n", text, flags=re.IGNORECASE)
            text = re.sub(r"</t[dh]>", " | ", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", text).strip()
        if block.type == BlockType.IMAGE:
            return re.sub(
                r"\s+",
                " ",
                " ".join(str(block.payload.get(key) or "") for key in ("caption", "footnote", "img_path")),
            ).strip()
        return re.sub(r"\s+", " ", block.get_text() or "").strip()

    def _write_repaired_split_form_document(
        self,
        *,
        outputs_dir: Path,
        form_item: dict[str, Any],
        repaired_markdown: str,
        document_ir: DocumentIR,
        semantic_output_language: str,
    ) -> None:
        index_path = outputs_dir / "documents_index.json"
        if not index_path.exists():
            return
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(index, list):
            return
        main_entry = index[0] if index else {}
        source_title = str(main_entry.get("title") or Path(document_ir.source.path).stem)
        source_filename = str(main_entry.get("source_filename") or Path(document_ir.source.path).name)
        form_id = str(form_item.get("form_id") or "")
        for entry in index:
            if str(entry.get("document_id") or "") != form_id:
                continue
            file_path = Path(str(entry.get("file") or ""))
            if not file_path.parent.exists():
                continue
            render_item = dict(form_item)
            render_item["title"] = entry.get("title") or form_item.get("title")
            file_path.write_text(
                self._render_split_form_document(
                    raw_markdown=repaired_markdown,
                    item=render_item,
                    source_title=source_title,
                    source_filename=source_filename,
                    semantic_output_language=semantic_output_language,
                ),
                encoding="utf-8",
            )
            return

    def _block_form_from_auto_rag(
        self,
        *,
        outputs_dir: Path,
        form_item: dict[str, Any],
        item_stats: dict[str, Any],
        current_markdown: str,
        document_ir: DocumentIR,
        semantic_output_language: str,
    ) -> None:
        """Replace unsafe parser fallback with a non-ingestable review marker."""

        reason = str(item_stats.get("reason") or "semantic_repair_not_usable")
        summary = str(item_stats.get("summary") or item_stats.get("error") or "")
        blocked_markdown = self._render_blocked_form_markdown(
            form_item=form_item,
            reason=reason,
            summary=summary,
            current_markdown=current_markdown,
            semantic_output_language=semantic_output_language,
        )
        md_path = Path(str((form_item.get("files") or {}).get("markdown") or ""))
        if md_path.parent.exists():
            md_path.write_text(blocked_markdown, encoding="utf-8")
        chunks_value = str((form_item.get("files") or {}).get("chunks") or "")
        if chunks_value:
            chunks_path = Path(chunks_value)
            if chunks_path.parent.exists():
                chunks_path.write_text("", encoding="utf-8")
        self._write_repaired_split_form_document(
            outputs_dir=outputs_dir,
            form_item=form_item,
            repaired_markdown=blocked_markdown,
            document_ir=document_ir,
            semantic_output_language=semantic_output_language,
        )
        form_item["semantic_repair"] = {
            "status": "blocked",
            "reason": reason,
            "summary": summary,
            "auto_rag_ready": False,
            "blocked_by": "review_vlm",
        }

    def _render_blocked_form_markdown(
        self,
        *,
        form_item: dict[str, Any],
        reason: str,
        summary: str,
        current_markdown: str,
        semantic_output_language: str,
    ) -> str:
        title = self._clean_export_title(str(form_item.get("title") or form_item.get("form_id") or "Form"))
        page_label = str(form_item.get("page_label") or "").strip()
        language = "en" if semantic_output_language == "en" else "zh-TW"
        reason_text = reason.replace("_", " ").strip()
        if language == "en":
            lines = [
                f"# {title}",
                "",
                "> LLM/VLM final review did not produce a trustworthy semantic rewrite. This page is blocked from automatic RAG ingestion.",
                "",
                "## Review Status",
                "- Auto RAG ready: no",
                f"- Reason: {reason_text}",
            ]
            if page_label:
                lines.append(f"- Source page: {page_label}")
            if summary:
                lines.extend(["", "## Reviewer Summary", summary.strip()])
            lines.extend([
                "",
                "## Next Step",
                "Re-run with a stronger review VLM/LLM or inspect the original page before allowing this content into a vector index.",
            ])
        else:
            lines = [
                f"# {title}",
                "",
                "> LLM/VLM 最終審核沒有產出可信的語意重寫。本頁已阻擋自動進入 RAG。",
                "",
                "## 審核狀態",
                "- 可自動進 RAG：否",
                f"- 原因：{reason_text}",
            ]
            if page_label:
                lines.append(f"- 來源頁面：{page_label}")
            if summary:
                lines.extend(["", "## Reviewer 摘要", summary.strip()])
            lines.extend([
                "",
                "## 後續處理",
                "請改用更強的 review VLM/LLM 重新處理，或人工對照原頁後再允許進入向量索引。",
            ])
        if len(current_markdown.strip()) < 120:
            lines.extend(["", "## 原始候選輸出", current_markdown.strip()])
        return "\n".join(line for line in lines).strip() + "\n"

    def _rewrite_authoritative_form_chunks(
        self,
        *,
        outputs_dir: Path,
        document_ir: DocumentIR,
        repaired_items: list[dict[str, Any]],
        blocked_items: list[dict[str, Any]],
    ) -> None:
        """Remove unsafe fallback chunks for repaired/blocked forms, then write reviewer chunks."""

        chunks_path = outputs_dir / "structured_chunks.jsonl"
        target_items = [*repaired_items, *blocked_items]
        existing_chunks: list[dict[str, Any]] = []
        if chunks_path.exists():
            for line in chunks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    existing_chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        kept_chunks = [
            chunk for chunk in existing_chunks
            if not any(self._chunk_matches_form_item(chunk, item) for item in target_items)
        ]
        repair_chunks = self._semantic_repair_chunks_for_items(document_ir, repaired_items)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in [*kept_chunks, *repair_chunks]:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        for item in repaired_items:
            form_chunks = [chunk for chunk in repair_chunks if self._chunk_matches_form_item(chunk, item)]
            chunks_value = str((item.get("files") or {}).get("chunks") or "")
            if chunks_value:
                chunks_file = Path(chunks_value)
                if chunks_file.parent.exists():
                    with open(chunks_file, "w", encoding="utf-8") as f:
                        for chunk in form_chunks:
                            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        for item in blocked_items:
            chunks_value = str((item.get("files") or {}).get("chunks") or "")
            if chunks_value:
                chunks_file = Path(chunks_value)
                if chunks_file.parent.exists():
                    chunks_file.write_text("", encoding="utf-8")

    @staticmethod
    def _chunk_matches_form_item(chunk: dict[str, Any], item: dict[str, Any]) -> bool:
        metadata = dict(chunk.get("metadata") or {})
        item_ids = {
            str(item.get("form_id") or ""),
            str(item.get("subdoc_id") or ""),
            str(item.get("logical_doc_id") or ""),
        }
        item_ids.discard("")
        chunk_ids = {
            str(chunk.get("doc_id") or ""),
            str(metadata.get("form_id") or ""),
            str(metadata.get("subdoc_id") or ""),
            str(metadata.get("logical_doc_id") or ""),
        }
        chunk_ids.discard("")
        if item_ids.intersection(chunk_ids):
            return True
        form_id = str(item.get("form_id") or "")
        subdoc_id = str(item.get("subdoc_id") or "")
        for block_id in chunk.get("block_ids") or []:
            value = str(block_id or "")
            if (form_id and form_id in value) or (subdoc_id and subdoc_id in value):
                return True
        return False

    def _append_semantic_repair_chunks(
        self,
        outputs_dir: Path,
        document_ir: DocumentIR,
        repaired_items: list[dict[str, Any]],
    ) -> None:
        self._write_semantic_repair_chunks(outputs_dir, document_ir, repaired_items, replace=False)

    def _write_semantic_repair_chunks(
        self,
        outputs_dir: Path,
        document_ir: DocumentIR,
        repaired_items: list[dict[str, Any]],
        *,
        replace: bool,
    ) -> None:
        chunks_path = outputs_dir / "structured_chunks.jsonl"
        mode = "w" if replace else "a"
        needs_leading_newline = False
        if not replace and chunks_path.exists():
            existing = chunks_path.read_text(encoding="utf-8")
            needs_leading_newline = bool(existing and not existing.endswith("\n"))
        with open(chunks_path, mode, encoding="utf-8") as f:
            if needs_leading_newline:
                f.write("\n")
            for chunk in self._semantic_repair_chunks_for_items(document_ir, repaired_items):
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def _semantic_repair_chunks_for_items(
        self,
        document_ir: DocumentIR,
        repaired_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for item in repaired_items:
            form_id = str(item.get("form_id") or "semantic_repair")
            sections = self._split_repair_markdown_into_chunks(str(item.get("markdown") or ""))
            page_indices = self._safe_page_indices(item.get("page_indices"))
            for section_idx, section in enumerate(sections):
                chunks.append(
                    {
                        "chunk_id": f"sr_repair_{form_id}_{section_idx:04d}",
                        "doc_id": str(item.get("subdoc_id") or form_id or document_ir.doc_id),
                        "run_id": document_ir.run_id,
                        "view": "semantic_repair",
                        "content": section,
                        "block_ids": [f"semantic_repair:{form_id}"],
                        "page_indices": page_indices,
                        "attachments": [],
                        "metadata": {
                            "document_type": "semantic_repair",
                            "content_type": "semantic_repair",
                            "form_name": item.get("title"),
                            "form_id": form_id,
                            "subdoc_id": item.get("subdoc_id"),
                            "logical_doc_id": item.get("logical_doc_id"),
                            "parent_doc_id": document_ir.doc_id,
                            "repaired_by": "review_vlm",
                            "applied_repairs": item.get("applied_repairs", []),
                            "confidence": item.get("confidence"),
                            "auto_rag_ready": True,
                        },
                    }
                )
        return chunks

    def _rebuild_structured_rag_from_form_files(self, outputs_dir: Path, document_ir: DocumentIR) -> None:
        forms_index_path = outputs_dir / "forms_index.json"
        if not forms_index_path.exists():
            return
        try:
            forms_index = json.loads(forms_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(forms_index, list):
            return
        title = self._clean_export_title(Path(document_ir.source.path).stem) or Path(document_ir.source.path).name
        parts = [f"# {title}", ""]
        for item in forms_index:
            md_path = Path(str((item.get("files") or {}).get("markdown") or ""))
            if md_path.exists():
                parts.append(md_path.read_text(encoding="utf-8").strip())
                parts.append("")
        (outputs_dir / "structured_rag.md").write_text("\n".join(parts).strip() + "\n", encoding="utf-8")

    def _normalize_repaired_markdown(self, markdown: str, title: str) -> str:
        text = render_vlm_text(markdown)
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        clean_title = self._clean_export_title(title or "")
        if clean_title and not re.match(r"^#\s+", text):
            text = f"# {clean_title}\n\n{text}".strip()
        elif clean_title:
            match = re.match(r"^#\s+([^\n]+)(\n|$)", text)
            current_title = self._clean_export_title(match.group(1) if match else "")
            if match and self._is_weak_repair_heading(current_title, clean_title):
                text = re.sub(r"^#\s+[^\n]+", f"# {clean_title}", text, count=1)
        text = self._remove_duplicate_repair_heading(text, clean_title)
        return text.strip() + "\n"

    @classmethod
    def _remove_duplicate_repair_heading(cls, markdown: str, title: str) -> str:
        lines = markdown.strip().splitlines()
        if not lines:
            return markdown
        first_idx = next((idx for idx, line in enumerate(lines) if line.strip()), None)
        if first_idx is None:
            return markdown
        first_match = re.match(r"^#\s+(.+?)\s*$", lines[first_idx].strip())
        if not first_match:
            return markdown
        first_title = cls._clean_export_title(first_match.group(1))
        expected_title = cls._clean_export_title(title or first_title)
        probe_idx = first_idx + 1
        while probe_idx < len(lines) and not lines[probe_idx].strip():
            probe_idx += 1
        if probe_idx >= len(lines):
            return markdown
        duplicate_match = re.match(r"^#{1,3}\s+(.+?)\s*$", lines[probe_idx].strip())
        if not duplicate_match:
            return markdown
        duplicate_title = cls._clean_export_title(duplicate_match.group(1))
        if duplicate_title not in {first_title, expected_title}:
            return markdown
        del lines[probe_idx]
        return "\n".join(lines).strip()

    def _semantic_repair_title(self, *, document_ir: DocumentIR, source_md: str, structured_output: Any) -> str:
        plan = getattr(structured_output, "plan", None)
        plan_title = self._clean_export_title(str(getattr(plan, "title", "") or ""))
        document_type = str(getattr(plan, "document_type", "") or "")
        inferred_title = self._clean_export_title(self._infer_source_title(source_md, document_ir.source.path))
        if document_type == "generic_document" and inferred_title:
            return inferred_title
        if self._is_weak_repair_heading(plan_title, inferred_title or plan_title):
            return inferred_title or self._clean_export_title(Path(document_ir.source.path).stem)
        return plan_title or inferred_title or self._clean_export_title(Path(document_ir.source.path).stem)

    @classmethod
    def _is_weak_repair_heading(cls, heading: str, replacement: str = "") -> bool:
        value = cls._clean_export_title(heading)
        if not value:
            return True
        if replacement and value == cls._clean_export_title(replacement):
            return False
        if re.search(r"\.(?:pdf|docx?|xlsx?|ods|png|jpe?g)$", value, flags=re.IGNORECASE):
            return True
        compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value)
        if re.fullmatch(r"\d{10,}|[0-9a-f]{12,}", compact, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"(?:source|document|untitled|file|page)\s*\d*", value, flags=re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _semantic_repair_markdown_is_usable(markdown: str, current_markdown: str, semantic_output_language: str) -> bool:
        text = markdown.strip()
        if len(text) < 80:
            return False
        if text.startswith("{") or text.startswith("["):
            return False
        body_without_title = re.sub(r"^#\s+[^\n]+\n+", "", text, count=1).lstrip()
        jsonish_prefix = body_without_title[:1200]
        if body_without_title.startswith(("{", "[")):
            return False
        if "\"repaired_markdown\"" in jsonish_prefix or ("\"status\"" in jsonish_prefix and "\"confidence\"" in jsonish_prefix):
            return False
        if "JSON_PARSE_FAILED" in jsonish_prefix or "raw_response_preview" in jsonish_prefix:
            return False
        if "QUALITY ISSUES JSON" in text or "SOURCE EVIDENCE" in text:
            return False
        if re.search(r"\.\.\.|…", text):
            return False
        if not re.search(r"^#{1,3}\s+", text, re.MULTILINE) and "- " not in text:
            return False
        if len(text) < max(80, min(len(current_markdown.strip()) * 0.2, 500)):
            return False
        language = "en" if semantic_output_language == "en" else "zh-TW"
        if language == "en" and re.search(r"^#{1,3}\s*(表單用途|適用場景|填寫重點|注意事項|RAG 查詢摘要)", text, re.MULTILINE):
            return False
        return True

    def _split_repair_markdown_into_chunks(self, markdown: str, max_chars: int = 2400) -> list[str]:
        lines = markdown.strip().splitlines()
        sections: list[str] = []
        current: list[str] = []
        for line in lines:
            if re.match(r"^#{1,3}\s+", line) and current and len("\n".join(current)) >= 240:
                sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current).strip())

        chunks: list[str] = []
        for section in sections:
            if len(section) <= max_chars:
                chunks.append(section)
                continue
            paragraphs = re.split(r"\n\s*\n", section)
            buffer: list[str] = []
            for paragraph in paragraphs:
                candidate = "\n\n".join(buffer + [paragraph]).strip()
                if buffer and len(candidate) > max_chars:
                    chunks.append("\n\n".join(buffer).strip())
                    buffer = [paragraph]
                else:
                    buffer.append(paragraph)
            if buffer:
                chunks.append("\n\n".join(buffer).strip())
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _safe_page_indices(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        result: list[int] = []
        if isinstance(value, list):
            for item in value:
                try:
                    result.append(int(item))
                except (TypeError, ValueError):
                    continue
        return sorted(set(result))

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 1].rstrip() + "…"

    def _write_document_exports(
        self,
        outputs_dir: Path,
        source_md: str,
        assets: list[AssetEntry],
        structured_paths: dict[str, str],
        document_ir: DocumentIR,
        semantic_output_language: str = "zh-TW",
    ) -> dict[str, str]:
        """
        Write split markdown documents for tools that treat each uploaded file as a document.

        This keeps OpenWebUI-style ingestion from mixing the original source body with
        extracted forms/tables/figures inside one uploaded markdown file.
        """

        documents_dir = outputs_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        for old_markdown in documents_dir.glob("*.md"):
            old_markdown.unlink()

        paths: dict[str, Path] = {
            "documents_dir": documents_dir,
            "main_document": documents_dir / "main.md",
            "documents_index": outputs_dir / "documents_index.json",
        }

        source_filename = Path(document_ir.source.path).name
        if Path(document_ir.source.path).suffix.lower() in {".xls", ".xlsx", ".ods"}:
            source_title = Path(document_ir.source.path).stem[:120] or source_filename
        else:
            source_title = self._infer_source_title(source_md, document_ir.source.path)
        source_title = self._clean_export_title(source_title)

        index: list[dict[str, Any]] = [
            {
                "document_id": "main",
                "kind": "main",
                "title": source_title,
                "source_filename": source_filename,
                "page_indices": self._document_page_indices(document_ir),
                "page_image_path": self._document_page_image_path(document_ir, 0),
                "file": str(paths["main_document"]),
            },
        ]

        form_title_by_page = self._form_asset_titles_by_page(assets, source_title)
        form_entries: list[dict[str, Any]] = []
        forms_index_value = structured_paths.get("forms_index")
        forms_index_path = Path(forms_index_value) if forms_index_value else None
        table_collection_groups = self._table_collection_groups(assets)
        table_collection_asset_ids = {
            asset.asset_id
            for group in table_collection_groups
            for asset in group["assets"]
        }
        structured_document_type = self._structured_document_type(structured_paths)
        source_is_single_visual_document = self._source_is_single_visual_document(document_ir, assets, source_md)
        if forms_index_path and forms_index_path.is_file():
            forms_index = json.loads(forms_index_path.read_text(encoding="utf-8"))
            for item in forms_index:
                form_file = Path(item.get("files", {}).get("markdown") or "")
                if not form_file.exists():
                    continue
                dst = documents_dir / f"{item['form_id']}.md"
                form_md = form_file.read_text(encoding="utf-8")
                if self._is_low_value_form_markdown(item, form_md, source_title, source_filename):
                    continue
                display_title = self._best_form_export_title(item, source_title, form_title_by_page)
                render_item = dict(item)
                render_item["title"] = display_title
                dst.write_text(
                    self._render_split_form_document(
                        raw_markdown=form_md,
                        item=render_item,
                        source_title=source_title,
                        source_filename=source_filename,
                        semantic_output_language=semantic_output_language,
                    ),
                    encoding="utf-8",
                )
                form_entry = {
                    "document_id": item["form_id"],
                    "kind": "form",
                    "title": display_title,
                    "source_title": source_title,
                    "source_filename": source_filename,
                    "page_indices": item.get("page_indices", []),
                    "page_label": item.get("page_label"),
                    "page_image_path": self._document_page_image_path(document_ir, self._first_page_index(item.get("page_indices", []))),
                    "logical_doc_id": item.get("logical_doc_id"),
                    "parent_doc_id": item.get("parent_doc_id"),
                    "file": str(dst),
                }
                form_entries.append(form_entry)
                index.append(
                    form_entry
                )
        else:
            for collection_idx, group in enumerate(table_collection_groups):
                collection_id = f"table_collection_{collection_idx:04d}"
                dst = documents_dir / f"{collection_id}.md"
                dst.write_text(
                    self._render_table_collection_document(
                        group=group,
                        source_title=source_title,
                        semantic_output_language=semantic_output_language,
                    ),
                    encoding="utf-8",
                )
                page_indices = sorted({asset.page_idx for asset in group["assets"] if asset.page_idx is not None})
                index.append(
                    {
                        "document_id": collection_id,
                        "kind": "table_collection",
                        "title": group["title"],
                        "source_title": source_title,
                        "source_filename": source_filename,
                        "page_indices": page_indices,
                        "page_image_path": self._document_page_image_path(
                            document_ir,
                            page_indices[0] if page_indices else None,
                        ),
                        "asset_ids": [asset.asset_id for asset in group["assets"]],
                        "file": str(dst),
                    }
                )

            for asset in assets:
                if asset.asset_id in table_collection_asset_ids:
                    continue
                if structured_document_type == "form_document" and asset.type == "form_asset":
                    continue
                if source_is_single_visual_document and asset.type == "figure_asset":
                    continue
                if not self._should_export_asset_document(asset, source_md, assets):
                    continue
                dst = documents_dir / f"{asset.asset_id}.md"
                dst.write_text(
                    self._render_split_asset_document(
                        asset=asset,
                        source_title=source_title,
                        source_filename=source_filename,
                        semantic_output_language=semantic_output_language,
                    ),
                    encoding="utf-8",
                )
                index.append(
                    {
                        "document_id": asset.asset_id,
                        "kind": asset.type,
                        "title": self._clean_export_title(asset.title),
                        "source_title": source_title,
                        "source_filename": source_filename,
                        "page_indices": [asset.page_idx],
                        "page_image_path": self._document_page_image_path(document_ir, asset.page_idx),
                        "asset_path": asset.asset_path,
                        "file": str(dst),
                    }
                )

        if form_entries:
            promoted_source_title = self._promoted_source_title_from_forms(
                source_title=source_title,
                source_filename=source_filename,
                form_entries=form_entries,
            )
            if promoted_source_title != source_title:
                source_title = promoted_source_title
                index[0]["title"] = source_title
                for entry in form_entries:
                    entry["source_title"] = source_title
                for entry in index[1:]:
                    if "source_title" in entry:
                        entry["source_title"] = source_title

        paths["main_document"].write_text(
            self._render_split_main_document(
                source_md=source_md,
                source_title=source_title,
                source_filename=source_filename,
                form_entries=form_entries,
                asset_entries=assets,
                semantic_output_language=semantic_output_language,
            ),
            encoding="utf-8",
        )
        paths["documents_index"].write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {key: str(path) for key, path in paths.items()}

    def _source_is_single_visual_document(
        self,
        document_ir: DocumentIR,
        assets: list[AssetEntry],
        source_md: str,
    ) -> bool:
        figure_assets = [asset for asset in assets if asset.type == "figure_asset"]
        if not figure_assets:
            return False
        page_indices = self._document_page_indices(document_ir)
        if len(page_indices) > max(1, len({asset.page_idx for asset in figure_assets})):
            return False
        if len(page_indices) > 2:
            return False
        combined = " ".join(
            render_vlm_text(part)
            for asset in figure_assets
            for part in [asset.title, asset.semantic_caption, asset.structured_content, asset.retrieval_text]
            if part
        )
        source_title = self._infer_source_title(source_md, document_ir.source.path)
        haystack = f"{source_title} {Path(document_ir.source.path).stem} {combined}"
        if not re.search(r"流程圖|流程|作業流程|flowchart|workflow|decision tree|diagram", haystack, re.IGNORECASE):
            return False
        text_without_asset_refs = re.sub(r"\[\[asset:[^\]]+\]\]", " ", source_md or "")
        compact = re.sub(r"\s+", "", self._clean_export_title(text_without_asset_refs))
        return len(compact) <= 1800


    @staticmethod
    def _structured_document_type(structured_paths: dict[str, str]) -> str:
        plan_value = structured_paths.get("document_plan") or ""
        if not plan_value:
            return ""
        plan_path = Path(plan_value)
        if not plan_path.exists():
            return ""
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        return str(plan.get("document_type") or "")


    @classmethod
    def _remove_duplicate_title_lines(cls, body: str, title: str, source_title: str) -> str:
        lines = body.splitlines()
        title_keys = {
            re.sub(r"\s+", "", cls._clean_export_title(title)),
            re.sub(r"\s+", "", cls._clean_export_title(source_title)),
        }
        cleaned: list[str] = []
        skipped = False
        for idx, line in enumerate(lines):
            compact = re.sub(r"\s+", "", cls._clean_export_title(line.strip().strip("#")))
            if idx <= 2 and compact in title_keys and compact:
                skipped = True
                continue
            if skipped and not line.strip() and not cleaned:
                continue
            cleaned.append(line)
        return cls._remove_empty_display_sections("\n".join(cleaned)).strip()

    def _form_asset_titles_by_page(self, assets: list[AssetEntry], source_title: str) -> dict[int, str]:
        titles: dict[int, str] = {}
        source_key = re.sub(r"\s+", "", self._clean_export_title(source_title)).lower()
        for asset in assets:
            if asset.type != "form_asset":
                continue
            title = self._clean_export_title(asset.title)
            title_key = re.sub(r"\s+", "", title).lower()
            if not title or title_key == source_key or self._is_unreliable_export_title(title):
                continue
            titles.setdefault(asset.page_idx, title)
        return titles

    def _best_form_export_title(
        self,
        item: dict[str, Any],
        source_title: str,
        form_title_by_page: dict[int, str],
    ) -> str:
        page_idx = self._first_page_index(item.get("page_indices", []))
        asset_title = form_title_by_page.get(page_idx) if page_idx is not None else None
        if asset_title and not self._is_unreliable_export_title(asset_title):
            return asset_title
        fallback_title = self._clean_export_title(str(item.get("title") or ""))
        source_key = re.sub(r"\s+", "", self._clean_export_title(source_title)).lower()
        fallback_key = re.sub(r"\s+", "", fallback_title).lower()
        if not fallback_title or fallback_key == source_key or self._is_unreliable_export_title(fallback_title):
            return source_title if not self._is_unreliable_export_title(source_title) else "表單"
        return fallback_title

    def _promoted_source_title_from_forms(
        self,
        source_title: str,
        source_filename: str,
        form_entries: list[dict[str, Any]],
    ) -> str:
        if not form_entries:
            return source_title
        first_title = self._clean_export_title(str(form_entries[0].get("title") or ""))
        if not first_title or self._is_unreliable_export_title(first_title):
            return source_title
        source_clean = self._clean_export_title(source_title)
        source_key = re.sub(r"\s+", "", source_clean).lower()
        stem_key = re.sub(r"\s+", "", self._clean_export_title(Path(source_filename).stem)).lower()
        first_key = re.sub(r"\s+", "", first_title).lower()
        if first_key == source_key:
            return source_title
        source_lower = source_clean.lower()
        source_is_file_stub = bool(stem_key and (source_key == stem_key or source_key in stem_key or stem_key in source_key))
        source_is_instruction_or_chart = bool(
            re.search(
                r"\b(explanation of form|chart for individual transcripts|instructions? for form)\b",
                source_lower,
                re.IGNORECASE,
            )
        )
        if self._is_unreliable_export_title(source_clean) or source_is_file_stub or source_is_instruction_or_chart:
            return first_title
        if len(form_entries) == 1 and self._looks_like_meaningful_english_title(first_title):
            return first_title
        return source_title


    def _is_low_value_form_markdown(
        self,
        item: dict[str, Any],
        markdown: str,
        source_title: str,
        source_filename: str,
    ) -> bool:
        field_count = self._safe_int(item.get("field_count"), 0)
        title = self._clean_export_title(str(item.get("title") or ""))
        title_key = re.sub(r"\s+", "", title).lower()
        source_keys = {
            re.sub(r"\s+", "", self._clean_export_title(source_title)).lower(),
            re.sub(r"\s+", "", self._clean_export_title(Path(source_filename).stem)).lower(),
        }
        text = markdown or ""
        text_lower = text.lower()
        if re.search(r"\bblank page\b|no visible text|no visible fields|empty page", text_lower):
            return True
        if title_key in {"tableofcontents", "contents"} or re.search(r"^#\s*(table of contents|contents)\b", text, re.IGNORECASE | re.MULTILINE):
            return True
        if field_count > 0:
            return False
        has_specific_sections = bool(
            re.search(
                r"^#{2,3}\s*(Form Fields|Filling Guidance|Approval Flow|Notes|表單欄位|填寫重點|簽核流程|注意事項)",
                text,
                re.IGNORECASE | re.MULTILINE,
            )
        )
        generic_source_stub = bool(
            "source file" in text_lower
            and "use cases" in text_lower
            and "form structure" in text_lower
            and not has_specific_sections
        )
        if generic_source_stub:
            return True
        if not has_specific_sections and title_key in source_keys:
            return True
        compact = re.sub(r"\s+", "", self._clean_export_title(text))
        return len(compact) < 180 and not has_specific_sections

    @staticmethod
    def _safe_int(value: Any, fallback: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _table_collection_groups(self, assets: list[AssetEntry]) -> list[dict[str, Any]]:
        candidates = [
            asset
            for asset in assets
            if self._is_table_collection_candidate(asset)
        ]
        if len(candidates) < 30:
            return []

        grouped: dict[str, list[AssetEntry]] = {}
        for asset in candidates:
            title = self._table_collection_title(asset.title)
            grouped.setdefault(title, []).append(asset)

        groups = [
            {
                "title": title,
                "assets": sorted(items, key=lambda item: (item.page_idx, item.asset_id)),
            }
            for title, items in grouped.items()
        ]
        groups.sort(key=lambda group: (group["assets"][0].page_idx, group["title"]))
        return groups

    def _is_table_collection_candidate(self, asset: AssetEntry) -> bool:
        if asset.type != "table_asset":
            return False
        if not render_vlm_text(asset.structured_content):
            return False
        if self._is_fragment_table_asset(asset) or self._is_low_confidence_table_asset(asset):
            return False
        return True

    @staticmethod
    def _is_low_confidence_table_asset(asset: AssetEntry) -> bool:
        text = "\n".join(
            render_vlm_text(part)
            for part in [asset.structured_content, asset.retrieval_text]
            if part
        ).lower()
        return "低可信度表格 ocr" in text or "low-confidence table ocr" in text

    def _is_low_value_form_asset(self, asset: AssetEntry) -> bool:
        if asset.type != "form_asset" or asset.field_schema:
            return False
        title_key = re.sub(r"\s+", "", self._clean_export_title(asset.title)).lower()
        body = "\n".join(
            render_vlm_text(part)
            for part in [asset.filling_guide, asset.structured_content, asset.semantic_caption, asset.retrieval_text]
            if part
        ).strip()
        body_lower = body.lower()
        body_key = re.sub(r"\s+", "", self._clean_export_title(body)).lower()
        if title_key in {"tableofcontents", "contents"}:
            return True
        if re.search(r"\bblank page\b|no visible text|no visible fields|empty page", body_lower):
            return True
        if re.fullmatch(r"formpage\d+|form\d*", title_key) and body_key in {title_key, ""}:
            return True
        has_specific_sections = bool(
            re.search(
                r"^#{2,3}\s*(Form Fields|Filling Guidance|Approval Flow|Notes|表單欄位|填寫重點|簽核流程|注意事項)",
                body,
                re.IGNORECASE | re.MULTILINE,
            )
        )
        return len(body_key) < 180 and not has_specific_sections

    def _table_collection_title(self, value: Any) -> str:
        title = self._clean_export_title(value)
        title = re.sub(r"\s*(?:第\s*\d+\s*頁|Page\s*\d+)\s*(?:表格|Table)\s*\d+\s*$", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s*(?:表格|Table)\s*\d+\s*$", "", title, flags=re.IGNORECASE)
        for marker in ("檔案分類及保存年限區分表", "保存年限區分表"):
            idx = title.find(marker)
            if idx >= 0:
                title = title[: idx + len(marker)]
                break
        if self._is_unreliable_export_title(title):
            return "Table Collection"
        return title[:100]

    def _render_table_collection_document(
        self,
        group: dict[str, Any],
        source_title: str,
        semantic_output_language: str = "zh-TW",
    ) -> str:
        language = "en" if semantic_output_language == "en" else "zh-TW"
        title = str(group["title"])
        assets = list(group["assets"])
        parts = [f"# {title}", ""]
        if language == "en":
            parts.append(f"This table collection groups {len(assets)} related table blocks from the source document.")
            parts.extend(["", "## Included Tables"])
            for asset in assets[:80]:
                parts.append(f"- {self._clean_export_title(asset.title)} ({self._page_label(asset.page_idx, language)})")
            if len(assets) > 80:
                parts.append(f"- ... {len(assets) - 80} additional table blocks")
            parts.extend(["", "## Table Content"])
        else:
            parts.append(f"本表格集合整理來源文件中的 {len(assets)} 個相關表格區塊。")
            parts.extend(["", "## 包含表格"])
            for asset in assets[:80]:
                parts.append(f"- {self._clean_export_title(asset.title)}（{self._page_label(asset.page_idx, language)}）")
            if len(assets) > 80:
                parts.append(f"- 另有 {len(assets) - 80} 個表格區塊")
            parts.extend(["", "## 表格內容"])

        for asset in assets:
            asset_title = self._clean_export_title(asset.title) or asset.asset_id
            body = render_vlm_text(asset.structured_content or asset.retrieval_text)
            body = self._strip_table_collection_repeated_heading(body, asset_title, title)
            page = self._page_label(asset.page_idx, language)
            parts.extend(["", f"### {asset_title}", f"來源頁面：{page}" if language != "en" else f"Source page: {page}"])
            if body:
                parts.append(body)

        return "\n".join(parts).strip() + "\n"

    def _strip_table_collection_repeated_heading(self, body: str, asset_title: str, collection_title: str) -> str:
        lines = body.strip().splitlines()
        cleaned: list[str] = []
        title_keys = {
            re.sub(r"\s+", "", self._clean_export_title(asset_title)).lower(),
            re.sub(r"\s+", "", self._clean_export_title(collection_title)).lower(),
        }
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if idx <= 2 and stripped.startswith("## "):
                heading_key = re.sub(r"\s+", "", self._clean_export_title(stripped.lstrip("#"))).lower()
                if heading_key in title_keys:
                    continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    @staticmethod
    def _page_label(page_idx: int | None, language: str) -> str:
        if page_idx is None:
            return "unknown page" if language == "en" else "未知頁面"
        return f"Page {page_idx + 1}" if language == "en" else f"第 {page_idx + 1} 頁"

    def _collapse_table_collection_sections(self, source_md: str, groups: list[dict[str, Any]]) -> str:
        ranges: list[tuple[int, int]] = []
        for group in groups:
            for asset in group["assets"]:
                token = f"[[asset:{asset.asset_id}]]"
                token_idx = source_md.find(token)
                if token_idx < 0:
                    continue
                start = self._find_table_section_start(source_md, asset, token_idx)
                end = source_md.find("\n\n", token_idx)
                end = len(source_md) if end < 0 else end + 2
                if start < end:
                    ranges.append((start, end))
        if not ranges:
            return source_md

        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        parts: list[str] = []
        cursor = 0
        for start, end in merged:
            parts.append(source_md[cursor:start])
            cursor = end
        parts.append(source_md[cursor:])
        return "".join(parts)

    def _find_table_section_start(self, source_md: str, asset: AssetEntry, token_idx: int) -> int:
        title = self._clean_export_title(asset.title)
        heading = f"## {title}"
        if source_md.startswith(heading):
            return 0
        heading_idx = source_md.rfind("\n" + heading, 0, token_idx)
        if heading_idx >= 0:
            return heading_idx + 1

        meta_candidates = [
            source_md.rfind("\n表格名稱：", 0, token_idx),
            source_md.rfind("\nTable name:", 0, token_idx),
        ]
        meta_idx = max(meta_candidates)
        if meta_idx >= 0:
            prior_heading = source_md.rfind("\n## ", 0, meta_idx)
            if prior_heading >= 0:
                return prior_heading + 1
            return meta_idx + 1

        boundary = source_md.rfind("\n\n", 0, token_idx)
        return 0 if boundary < 0 else boundary + 2

    def _clean_split_main_body(
        self,
        source_md: str,
        source_title: str,
        assets: list[AssetEntry],
        semantic_output_language: str = "zh-TW",
    ) -> str:
        body = source_md.strip()
        table_collection_groups = self._table_collection_groups(assets)
        if table_collection_groups:
            body = self._collapse_table_collection_sections(body, table_collection_groups).strip()
        if body.startswith(f"# {source_title}"):
            body = "\n".join(body.splitlines()[1:]).strip()
        cleaned: list[str] = []
        skip_next_blank = False
        source_title_key = re.sub(r"\s+", "", self._clean_export_title(source_title)).lower()
        last_heading_key = ""
        for line in body.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"\[\[asset:[^\]]+\]\]", stripped):
                continue
            line_title_key = re.sub(r"\s+", "", self._clean_export_title(stripped)).lower()
            if source_title_key and line_title_key == source_title_key:
                continue
            if stripped.lower() == "content type: form/table fragment" or stripped == "內容類型：表格片段或續接資料":
                continue
            if self._is_toc_like_display_line(stripped):
                skip_next_blank = True
                continue
            heading_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
            if heading_match:
                heading_text = self._clean_export_title(heading_match.group(1))
                if self._is_toc_like_display_line(heading_text):
                    skip_next_blank = True
                    continue
                if self._is_inline_option_group_heading(heading_text):
                    line = heading_text
                    stripped = line.strip()
                    last_heading_key = ""
                else:
                    if self._is_generic_display_heading(heading_text):
                        skip_next_blank = True
                        continue
                    if re.match(r"^Step\s+\d+\s*:", heading_text, flags=re.IGNORECASE):
                        heading_text = self._clean_step_heading(heading_text)
                        line = f"## {heading_text}"
                        stripped = line.strip()
                    heading_key = re.sub(r"\s+", "", self._clean_export_title(heading_text)).lower()
                    if heading_key and last_heading_key and (heading_key == last_heading_key or heading_key.startswith(last_heading_key)):
                        continue
                    last_heading_key = heading_key
            elif stripped:
                last_heading_key = ""
            image_asset_match = re.search(r"asset://assets/figures/(fig\d{4})", stripped)
            if image_asset_match:
                skip_next_blank = True
                continue
            if skip_next_blank and not stripped:
                skip_next_blank = False
                continue
            skip_next_blank = False
            if self._is_low_value_visual_ocr_line(stripped):
                continue
            line = self._clean_split_main_display_line(line)
            if semantic_output_language == "en":
                line = self._clean_english_ocr_cjk_noise(line)
            if not line.strip():
                continue
            cleaned.append(line)
        return self._remove_empty_display_sections("\n".join(cleaned)).strip()

    def _clean_split_form_body(self, raw_markdown: str, title: str, source_title: str, language: str) -> str:
        title_keys = {
            re.sub(r"\s+", "", self._clean_export_title(title)).lower(),
            re.sub(r"\s+", "", self._clean_export_title(source_title)).lower(),
        }
        cleaned: list[str] = []
        seen_section = False
        skip_section = False
        for line in raw_markdown.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                if seen_section and cleaned and cleaned[-1].strip():
                    cleaned.append("")
                continue
            if stripped.startswith("# ") or stripped.startswith("頁碼：") or stripped.startswith("Page:") or stripped.startswith("Page："):
                continue
            if stripped.startswith(("## ", "### ")):
                section_title = stripped.lstrip("#").strip()
                section_key = re.sub(r"\s+", "", self._clean_export_title(section_title)).lower()
                if section_key in title_keys:
                    continue
                skip_section = self._is_rag_summary_heading(section_title, language)
                seen_section = True
                if not skip_section:
                    cleaned.append(f"### {section_title}")
                continue
            if not seen_section or skip_section:
                continue
            content = self._strip_form_record_prefix(stripped, language)
            if content:
                cleaned.append(content)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return "\n".join(cleaned).strip()

    @staticmethod
    def _is_rag_summary_heading(section_title: str, language: str) -> bool:
        compact = re.sub(r"\s+", "", section_title).lower()
        if language == "en":
            return compact in {"ragquerysummary", "retrievalsummary"}
        return compact in {"rag查詢摘要", "檢索摘要"}

    @staticmethod
    def _strip_form_record_prefix(line: str, language: str) -> str:
        text = line.strip()
        if language == "en":
            text = re.sub(r"^Form:\s*[^.]{1,180}\.\s*Section:\s*[^.]{1,100}\.\s*", "", text)
            text = re.sub(r"^Form:\s*[^.]{1,180}\.\s*", "", text)
            return text.strip()
        text = re.sub(r"^表單：[^。]{1,180}。區塊：[^。]{1,100}。", "", text)
        text = re.sub(r"^表單：[^。]{1,180}。", "", text)
        return text.strip()

    def _is_decorative_figure_asset(self, asset: AssetEntry) -> bool:
        text = "\n".join(
            render_vlm_text(part)
            for part in [asset.title, asset.structured_content, asset.semantic_caption, asset.retrieval_text]
            if part
        ).lower()
        if not text:
            return True
        has_flow = " > " in text or any(keyword for keyword in asset.keywords if "flow" in str(keyword).lower() or "流程" in str(keyword))
        if has_flow:
            return False
        has_no_text_signal = any(token in text for token in ("no text", "there is no text", "無文字", "沒有文字", "未包含文字"))
        has_shape_signal = any(token in text for token in ("arrow", "downward", "upward", "triangle", "rectangular shaft", "箭頭", "三角", "線條"))
        if has_shape_signal and has_no_text_signal:
            return True
        title_key = re.sub(r"\s+", "", self._clean_export_title(asset.title)).lower()
        generic_figure_title = bool(re.fullmatch(r"figure\d*", title_key))
        decorative_keywords = {
            "arrow",
            "up",
            "down",
            "left",
            "right",
            "direction",
            "symbol",
            "icon",
            "black",
            "white",
            "solid",
            "shape",
        }
        keyword_values = {str(keyword).strip().lower() for keyword in asset.keywords if str(keyword).strip()}
        shape_keywords_only = bool(keyword_values) and keyword_values.issubset(decorative_keywords)
        simple_shape_signal = any(
            token in text
            for token in (
                "single graphical element",
                "single, solid",
                "simple black arrow",
                "simple, solid black arrow",
                "white background",
                "oriented vertically",
                "pointing downwards",
                "pointing upwards",
                "pointing towards",
            )
        )
        content_signal = bool(
            re.search(r"\b(charts?|graphs?|diagrams?|workflows?|process(?:es)?|tables?|labels?|nodes?)\b", text)
            or any(token in text for token in ("流程", "圖表", "節點", "標籤"))
        )
        warning_icon_signal = any(
            token in text
            for token in (
                "caution symbol",
                "warning sign",
                "warning icon",
                "exclamation mark",
                "exclamation point",
            )
        )
        if warning_icon_signal and not content_signal:
            return True
        if has_shape_signal and not content_signal and (shape_keywords_only or (generic_figure_title and simple_shape_signal)):
            return True
        meaningful_facts = [
            fact
            for fact in asset.facts
            if len(re.sub(r"\s+", "", str(fact))) >= 16
            and not any(token in str(fact).lower() for token in ("arrow", "no text", "箭頭", "無文字", "沒有文字"))
        ]
        if meaningful_facts:
            return False
        return has_shape_signal and len(re.sub(r"\s+", "", text)) < 160

    @staticmethod
    def _is_low_value_visual_ocr_line(line: str) -> bool:
        compact = re.sub(r"\s+", "", line or "")
        if not compact:
            return False
        lower_compact = compact.lower().strip(".")
        code_compact = compact.strip(".")
        if re.fullmatch(r"\d{1,3}", compact):
            return True
        if lower_compact == "printreset":
            return True
        if re.fullmatch(r"st[-a-z]{1,8}", lower_compact):
            return True
        if re.fullmatch(r"(?:[A-Z]{2}){3,8}", code_compact):
            return True
        if re.fullmatch(r"St[-a-zA-Z]{0,4}", compact):
            return True
        if len(compact) <= 5 and re.search(r"[-_]", compact):
            return True
        if (
            re.fullmatch(r"\*?[A-Z0-9]{6,14}\*?", compact)
            and re.search(r"[A-Z]", compact)
            and re.search(r"\d", compact)
        ):
            return True
        return False

    @staticmethod
    def _is_generic_display_heading(value: str) -> bool:
        compact = re.sub(r"\s+", "", str(value or "")).lower().strip("#:： ")
        return compact in {"table", "tablecontent", "表格", "表格內容"}

    @staticmethod
    def _is_toc_like_display_line(value: str) -> bool:
        text = str(value or "").strip().strip("# ")
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        if re.sub(r"[^a-z]", "", text.lower()) in {"tableofcontents", "contents", "toc"}:
            return True
        if text.startswith(("表格名稱：", "Table name:")):
            text = re.sub(r"^(表格名稱：|Table name:)\s*", "", text, flags=re.IGNORECASE)
            compact = re.sub(r"\s+", "", text)
        section_hits = sum(1 for marker in ("壹", "貳", "參", "一、", "二、", "三、") if marker in text)
        return bool(
            ("背景說明" in text and "使用說明" in text and section_hits >= 2)
            or (section_hits >= 3 and re.search(r"\.{2,}|…{1,}|……", text))
            or (len(compact) > 80 and re.search(r"\.{2,}|…{1,}|……", text))
            or (
                re.match(r"^[壹貳參肆伍陸柒捌玖拾一二三四五六七八九十附]", text)
                and re.search(r"[.．…]{2,}.*\d+$", text)
            )
        )

    @classmethod
    def _remove_empty_display_sections(cls, markdown: str) -> str:
        lines = markdown.splitlines()
        cleaned: list[str] = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            heading_match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
            if heading_match:
                heading_title = cls._clean_export_title(heading_match.group(1))
                heading_key = cls._display_heading_key(heading_title)
                if heading_key in {"內容", "資料列", "content", "rows", "tablecontent"}:
                    idx += 1
                    continue
                if heading_key in {"常見查詢主題", "commonquerytopics"}:
                    next_idx = idx + 1
                    while next_idx < len(lines) and not lines[next_idx].strip():
                        next_idx += 1
                    if next_idx >= len(lines) or re.match(r"^#{1,6}\s+", lines[next_idx].strip()):
                        idx += 1
                        continue
            else:
                next_idx = idx + 1
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1
                if next_idx < len(lines):
                    next_heading = re.match(r"^#{1,6}\s+(.+)$", lines[next_idx].strip())
                    if next_heading and cls._display_heading_key(line) == cls._display_heading_key(next_heading.group(1)):
                        idx += 1
                        continue
            cleaned.append(line)
            idx += 1
        return "\n".join(cleaned)

    @classmethod
    def _display_heading_key(cls, value: str) -> str:
        text = cls._clean_export_title(value)
        text = re.sub(r"[、，,。．.：:()（）\[\]【】\-–—_\s]+", "", text)
        return text.lower()

    @staticmethod
    def _is_inline_option_group_heading(value: str) -> bool:
        words = [word.lower() for word in re.split(r"\s+", str(value or "").strip()) if word]
        option_words = {"self", "you", "spouse", "dependent", "dependent(s)", "dependents", "checking", "savings"}
        return 2 <= len(words) <= 6 and all(word.strip("/:：") in option_words for word in words)

    @staticmethod
    def _clean_split_main_display_line(line: str) -> str:
        text = re.sub(r"^(\s*[-*]\s+)\.\s+", r"\1", line)
        text = re.sub(r"^(\s*[-*]\s+)(\d{1,3})([A-Z])", r"\1\2 \3", text)
        text = re.sub(r"^(\s*)(\d{1,3})([A-Z])", r"\1\2 \3", text)
        text = re.sub(r"^(\s*[a-z])(?=(Check|direct deposit|paper check)\b)", r"\1 ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bNR(?=Part-year\b)", "NR ", text)
        text = re.sub(r"(IL-\d{4}-V)(Staple)", r"\1 \2", text)
        text = re.sub(r"(?<=[a-z])(?=Firm[’']s phone)", " ", text)
        text = re.sub(r"\s+DR\.?\s+AP\.?\s+RR\s+DC\s+IR\s+ID\s*$", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.rstrip()

    @staticmethod
    def _clean_english_ocr_cjk_noise(line: str) -> str:
        if not re.search(r"[\u4e00-\u9fff]", line or ""):
            return line
        cjk_count = sum(1 for ch in line if "\u4e00" <= ch <= "\u9fff")
        ascii_letters = sum(1 for ch in line if ch.isascii() and ch.isalpha())
        if cjk_count > 3 or ascii_letters < max(4, cjk_count * 3):
            return line
        replacements = {"口": "", "日": "", "出": "", "文": "", "圖": "", "图": "", "一": "-"}
        chars = list(line)
        cleaned: list[str] = []
        for idx, ch in enumerate(chars):
            if ch not in replacements:
                cleaned.append(ch)
                continue
            prev_is_cjk = idx > 0 and "\u4e00" <= chars[idx - 1] <= "\u9fff"
            next_is_cjk = idx + 1 < len(chars) and "\u4e00" <= chars[idx + 1] <= "\u9fff"
            if prev_is_cjk or next_is_cjk:
                cleaned.append(ch)
                continue
            replacement = replacements[ch]
            if replacement:
                cleaned.append(f" {replacement} ")
        text = "".join(cleaned)
        text = re.sub(r"^(\s*[-*]\s+)[,;:|\-\s]+", r"\1", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"([|,;:])\s*$", "", text).rstrip()
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+-\s+", " - ", text)
        return text.strip() if line == line.lstrip() else text.rstrip()

    def _improve_table_asset_title_from_body(
        self,
        table_body: str,
        current_title: str,
        semantic_output_language: str,
    ) -> str:
        if semantic_output_language != "en" and not self._is_mechanical_table_title(current_title):
            return current_title
        rows = parse_html_table(table_body)
        for row in rows[:12]:
            row_text = " ".join(
                cell_text
                for cell in row
                for cell_text in [re.sub(r"\s+", " ", str(cell or "")).strip()]
                if cell_text
            )
            match = re.search(r"\b(Step\s+\d+\s*:\s*[^|.]+)", row_text, flags=re.IGNORECASE)
            if match:
                return self._clean_step_heading(match.group(1))[:100]
        if self._is_mechanical_table_title(current_title):
            return "Table"
        return current_title

    def _clean_step_heading(self, value: str) -> str:
        title = self._clean_export_title(value)
        match = re.match(r"^(Step\s+\d+\s*:\s*)(.+)$", title, flags=re.IGNORECASE)
        if match:
            body = match.group(2).strip()
            body = re.sub(
                r"\s+(Enter|Check|Complete|Attach|Use|See|Write)\b.+$",
                "",
                body,
                flags=re.IGNORECASE,
            ).strip()
            body = re.sub(r"\s+\d{1,3}[a-z]?\b.*$", "", body).strip()
            body = re.sub(r"\s*[-–—:]\s*$", "", body).strip()
            if body:
                return self._clean_export_title(f"{match.group(1)}{body}")
        shortened = re.sub(
            r"\s+(Enter|Check|Complete|Attach|Use|See|Write)\b.+$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()
        shortened = re.sub(r"\s*[-–—:]\s*$", "", shortened).strip()
        if re.match(r"^Step\s+\d+\s*:\s*\S.+", shortened, flags=re.IGNORECASE):
            return self._clean_export_title(shortened)
        return title

    @staticmethod
    def _is_mechanical_table_title(title: str) -> bool:
        compact = re.sub(r"\s+", "", str(title or "")).lower().strip("*#")
        return bool(
            not compact
            or re.fullmatch(r"table\d*", compact)
            or re.fullmatch(r"[*a-z0-9_-]+page\d+table\d+", compact)
            or ("page" in compact and "table" in compact and len(re.findall(r"\d", compact)) >= 2)
        )

    @staticmethod
    def _is_fragment_table_asset(asset: AssetEntry) -> bool:
        text = "\n".join(
            render_vlm_text(part)
            for part in [asset.structured_content, asset.retrieval_text]
            if part
        ).lower()
        return "table fragment" in text or "form/table fragment" in text or "表格片段" in text

    @staticmethod
    def _clean_export_title(value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip().strip("#:： ")
        text = re.sub(r"(表[一二三四五六七八九十0-9]+)[〇○昇鑑箇]+", r"\1", text)
        text = re.sub(r"(表[一二三四五六七八九十0-9]+)\s*[〇○昇鑑箇]+", r"\1", text)
        text = re.sub(r"[昇鑑](?=台灣|臺灣|國內|國外|大台北|大臺北)", "", text)
        text = re.sub(r"\bForm and receipts must be submitted\b.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bDo not sign this form\b.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bRequest may be rejected\b.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(\bRequest for Transcript of Tax Return)\s+Form\s*$", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip("#:： -")

    def _document_page_indices(self, document_ir: DocumentIR) -> list[int]:
        if document_ir.pages:
            return [page.page_idx for page in document_ir.pages]
        return sorted({block.page_idx for block in document_ir.blocks if block.page_idx is not None})

    def _first_page_index(self, page_indices: Any) -> int | None:
        if not isinstance(page_indices, list) or not page_indices:
            return None
        try:
            return int(page_indices[0])
        except (TypeError, ValueError):
            return None

    def _document_page_image_path(self, document_ir: DocumentIR, page_idx: int | None) -> str | None:
        if page_idx is None:
            return None
        for page in document_ir.pages:
            if page.page_idx == page_idx:
                return page.page_image_path
        return None

    def _infer_source_title(self, source_md: str, source_path: str) -> str:
        """Pick a stable human title for split-document metadata."""

        source_ext = Path(source_path).suffix.lower()
        body_title = self._infer_source_title_from_body(source_md)
        if body_title:
            return body_title[:120]
        for line in source_md.splitlines():
            stripped_line = line.strip()
            if not stripped_line.startswith("#") and not stripped_line.startswith("TABLE:"):
                continue
            raw_title = stripped_line.lstrip("#").strip()
            if raw_title.startswith("TABLE:"):
                raw_title = raw_title.removeprefix("TABLE:").strip()
            title = self._clean_export_title(raw_title)
            if not title:
                continue
            if title.startswith(("標 號", "修改歷程", "頁碼", "ROW:", "COLUMNS:", "...(", "[[asset:")):
                continue
            if self._is_unreliable_export_title(title) or self._is_toc_like_display_line(title):
                continue
            if source_ext in {".xls", ".xlsx", ".ods"} and self._is_weak_spreadsheet_source_title(title):
                continue
            if body_title and re.match(r"^Step\s+\d+\s*:", title, flags=re.IGNORECASE):
                continue
            return title[:120]
        fallback = self._clean_export_title(Path(source_path).stem)
        return fallback[:120] or "來源文件"

    def _infer_source_title_from_body(self, source_md: str) -> str:
        text = re.sub(r"\s+", " ", source_md or "").strip()
        if not text:
            return ""
        transcript_match = re.search(
            r"\b(?:Form\s+)?(4506-T\s+Request for Transcript of Tax Return)(?:\s+Form)?\b",
            text,
            re.IGNORECASE,
        )
        if transcript_match:
            return self._clean_export_title(f"Form {transcript_match.group(1)}")
        english_match = re.search(
            r"([A-Z][A-Za-z]+ Department of Revenue\s+20\d{2}\s+Form\s+[A-Z0-9-]+\s+[A-Za-z ]{8,80}?Return)",
            text,
        )
        if english_match:
            return self._clean_export_title(english_match.group(1))
        title_candidate = self._best_source_title_candidate(source_md)
        if title_candidate:
            return title_candidate
        match = re.search(
            r"(?P<org>(?:財團法人)?(?:台灣|臺灣)[\u4e00-\u9fff]{2,30}?)(?:（以下簡稱本院）|\(以下簡稱本院\))(?P<subject>[\u4e00-\u9fff]{2,20})[，,].{0,120}?特訂定本(?P<kind>制度|辦法|規程|要點|準則)",
            text,
        )
        if match:
            org = match.group("org")
            subject = match.group("subject").replace("内部", "內部")
            kind = match.group("kind")
            return self._clean_export_title(f"{org}{subject}{kind}")
        return ""

    def _best_source_title_candidate(self, source_md: str) -> str:
        candidates: list[tuple[int, int, str]] = []
        for order, raw_line in enumerate((source_md or "").splitlines()[:180]):
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^#{1,6}\s+", "", line)
            line = re.sub(r"^[-*]\s+", "", line)
            if line.startswith("TABLE:"):
                line = line.removeprefix("TABLE:").strip()
            title = self._clean_export_title(line)
            compact = re.sub(r"\s+", "", title)
            if not compact or len(compact) < 6 or len(compact) > 80:
                continue
            if self._is_unreliable_export_title(title) or self._is_toc_like_display_line(title):
                continue
            score = self._source_title_candidate_score(title)
            if score >= 20:
                candidates.append((score, -order, title))
        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][2][:120]

    @classmethod
    def _source_title_candidate_score(cls, title: str) -> int:
        compact = re.sub(r"\s+", "", cls._clean_export_title(title))
        score = 0
        title_terms = (
            "辦法",
            "要點",
            "準則",
            "規程",
            "規定",
            "指引",
            "手冊",
            "範本",
            "懶人包",
            "作業流程",
            "流程圖",
            "區分表",
            "申請單",
            "報告書",
            "核銷",
        )
        for term in title_terms:
            if term in compact:
                score += 18
        if re.search(r"(檔案分類|保存年限|性騷擾|申訴處理|著作權|補助案)", compact):
            score += 12
        if re.search(r"(行政院|政府|文化局|智慧財產局|台灣|臺灣|花蓮縣|臺中市|台中市)", compact):
            score += 6
        lowered = cls._clean_export_title(title).lower()
        english_terms = (
            "application",
            "authorization",
            "authorisation",
            "certificate",
            "consent",
            "declaration",
            "form",
            "guide",
            "handbook",
            "instructions",
            "manual",
            "notice",
            "permission",
            "permit",
            "report",
            "request",
            "release",
            "return",
            "statement",
            "training",
            "worksheet",
        )
        english_hits = sum(1 for term in english_terms if re.search(rf"\b{re.escape(term)}s?\b", lowered))
        if english_hits:
            score += min(english_hits, 3) * 9
        if re.search(r"\btraining\s+guide\b|\bapplication\s+form\b|\bauthori[sz]ation\s+for\s+release\b|\brelease\s+of\s+information\b", lowered):
            score += 14
        if re.search(r"\bplease\s+return\b|\breturn\s+completed\s+forms?\b|\bmail\s+completed\s+forms?\b|\bwhere\s+to\s+(?:send|return)\b", lowered):
            score -= 35
        if re.search(r"\bchart\s+for\s+individual\s+transcripts\b", lowered):
            score -= 28
        if 10 <= len(compact) <= 45:
            score += 8
        if cls._looks_like_org_only_title(compact):
            score -= 30
        if re.match(r"^(附件|附表)[一二三四五六七八九十0-9-]*$", compact):
            score -= 20
        return score

    @staticmethod
    def _looks_like_org_only_title(compact_title: str) -> bool:
        if re.search(r"(辦法|要點|準則|規程|規定|指引|手冊|範本|懶人包|流程|區分表|申請單|報告書|核銷)", compact_title):
            return False
        return bool(
            re.search(r"(?:各機關(?:（構）|\(構\))?|政府|文化局|智慧財產局|委員會|辦公室|中心|處|部|署|院|局)$", compact_title)
            or compact_title in {"行政院所屬中央及地方各機關（構）", "行政院所屬中央及地方各機關(構)"}
        )

    @classmethod
    def _looks_like_meaningful_english_title(cls, title: str) -> bool:
        text = cls._clean_export_title(title)
        if re.search(r"[\u4e00-\u9fff]", text):
            return False
        words = re.findall(r"[A-Za-z]{3,}", text)
        if len(words) < 3:
            return False
        return bool(
            re.search(
                r"\b(application|authori[sz]ation|certificate|consent|declaration|form|guide|handbook|instructions?|manual|notice|permission|permit|release|report|request|return|statement|training|worksheet)s?\b",
                text,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _is_unreliable_export_title(cls, title: str) -> bool:
        compact = re.sub(r"\s+", "", cls._clean_export_title(title))
        if not compact:
            return True
        generic_compact_titles = {
            "內容",
            "資料列",
            "content",
            "rows",
            "table",
            "tablename",
            "tablecontent",
            "contenttype",
            "sourcedocument",
            "sourcefile",
            "sourcepage",
            "documenttype",
            "relatedsource",
            "tableofcontents",
            "contents",
            "toc",
        }
        if compact.lower() in generic_compact_titles:
            return True
        if re.fullmatch(r"\d+", compact):
            return True
        if re.fullmatch(r"[A-Za-z]*Figure\d*|Table\d*|Form(?:Page)?\d*|Form\d+", compact, re.IGNORECASE):
            return True
        if (
            not re.search(r"[\u4e00-\u9fff]", compact)
            and len(re.findall(r"\d", compact)) >= 3
            and not cls._looks_like_meaningful_english_title(title)
        ):
            return True
        if len(compact) <= 2:
            return True
        if re.search(r"(?:核定|修正|訂定).{0,16}(?:施行|生效)|(?:施行|生效)[）)]?$", compact):
            return True
        if re.search(r"第\d+頁表格\d+", compact):
            return True
        if re.fullmatch(r"[（(]?[壹貳參肆伍陸柒捌玖拾一二三四五六七八九十]+[）)]?[、.．]?[\u4e00-\u9fff]{1,10}[:：]?", compact):
            return True
        if re.search(r"[。；;]", compact):
            return True
        if len(compact) > 45 and re.search(r"[，,]", compact):
            return True
        if re.search(r"事件編號|表單編號|申請日期|填表日期|xxx|xx\(|序號", compact, re.IGNORECASE):
            return True
        return False

    def _is_weak_spreadsheet_source_title(self, title: str) -> bool:
        compact = re.sub(r"\s+", "", title)
        if len(compact) < 8:
            return True
        weak_terms = {"備註單據編號", "會計核銷差旅", "元參考匯價為"}
        return compact in weak_terms

    def _infer_visual_asset_title(
        self,
        structured_content: str,
        semantic_caption: str,
        fallback: str,
        all_text: Any | None = None,
    ) -> str:
        """Use document-visible text instead of generic Figure N labels."""

        parsed = coerce_visual_vlm_output(
            {
                "structured_content": structured_content,
                "semantic_caption": semantic_caption,
                "all_text": all_text or [],
            }
        )
        for line in split_vlm_lines(parsed.get("structured_content", structured_content)):
            title = line.strip().lstrip("#- ").strip()
            if not title or " > " in title:
                continue
            if any("一" <= ch <= "鿿" for ch in title):
                return title[:80]
        caption = render_vlm_text(parsed.get("semantic_caption") or semantic_caption)
        if caption:
            zh_count = sum(1 for ch in caption if "一" <= ch <= "鿿")
            ascii_count = sum(1 for ch in caption if ch.isascii() and ch.isalpha())
            if zh_count >= 4 and zh_count >= ascii_count:
                return caption[:80]
        for line in split_vlm_lines(parsed.get("all_text", [])):
            if any("一" <= ch <= "鿿" for ch in line):
                return line[:80]
        return fallback

    def _augment_visual_output_from_page_text(
        self,
        output: dict[str, Any],
        document_ir: DocumentIR,
        image_block: Block,
    ) -> dict[str, Any]:
        """Backfill flowchart text from same-page OCR when VLM output is partial."""

        result = dict(output or {})
        page_lines = self._same_page_visual_text_lines(document_ir, image_block.page_idx)
        if len(page_lines) < 4:
            return result

        structured_lines = split_vlm_lines(result.get("structured_content", ""))
        existing_all_text = split_vlm_lines(result.get("all_text", []))
        caption = render_vlm_text(result.get("semantic_caption", ""))
        image_type = str(result.get("image_type") or "").lower()
        looks_like_flowchart = (
            image_type == "flowchart"
            or any(" > " in line for line in structured_lines)
            or "flowchart" in caption.lower()
            or any("流程圖" in line or "作業流程" in line for line in page_lines[:5])
        )
        if not looks_like_flowchart:
            return result

        result["image_type"] = "flowchart"
        merged = self._dedupe_visual_lines([*existing_all_text, *page_lines])
        structured_len = sum(len(line) for line in structured_lines)
        if len(existing_all_text) < len(page_lines) or structured_len < 180:
            result["all_text"] = merged
        return result

    @classmethod
    def _same_page_visual_text_lines(cls, document_ir: DocumentIR, page_idx: int | None) -> list[str]:
        lines: list[str] = []
        for block in sorted(document_ir.blocks, key=lambda item: (item.page_idx, item.reading_order)):
            if block.page_idx != page_idx or block.type != BlockType.TEXT:
                continue
            text = cls._clean_export_title(str(block.payload.get("text") or ""))
            if not text or cls._is_noisy_visual_page_text(text):
                continue
            lines.append(text)
        return cls._dedupe_visual_lines(lines)

    @staticmethod
    def _dedupe_visual_lines(lines: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for line in lines:
            clean = re.sub(r"\s+", " ", str(line or "")).strip()
            key = re.sub(r"\s+", "", clean).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(clean)
        return result

    @staticmethod
    def _is_noisy_visual_page_text(text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if len(compact) <= 1:
            return True
        if re.fullmatch(r"(?:page)?\d+(?:of\d+)?", compact, flags=re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _normalize_zh_visual_label(value: str) -> str:
        text = str(value or "")
        replacements = [
            (r"Government\s+Agency\s+A\s*\(甲\)", "政府機關甲"),
            (r"政府機關\s*甲\s*\(Government\s+Agency\s+A\)", "政府機關甲"),
            (r"Vendor\s+B\s*\(乙\)", "廠商乙"),
            (r"廠商\s*乙\s*\(Vendor\s+B\)", "廠商乙"),
            (r"Lecturer\s+B\s*\(乙\)", "講座乙"),
            (r"講座\s*乙\s*\(Lecturer\s+B\)", "講座乙"),
            (r"Authorize\s*\(授權\)", "授權"),
            (r"Entrusted(?:/Contracted)?\s*\(委辦\)", "委辦"),
            (r"No\s+need\s+to\s+deliver\s*\(不須交付\)", "不須交付"),
            (r"不須交付\s*\(No\s+need\s+to\s+deliver\)", "不須交付"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = re.sub(r"\((?:yes|y)\)", "（是）", text, flags=re.IGNORECASE)
        text = re.sub(r"\((?:no|n)\)", "（否）", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(政府機關|廠商|講座)\s+([甲乙丙丁])", r"\1\2", text)
        return text

    def _render_visual_semantic_content(
        self,
        title: str,
        output: dict[str, Any],
        semantic_output_language: str = "zh-TW",
    ) -> str:
        """Convert VLM figure output into retrieval-friendly semantic markdown."""

        language = "en" if semantic_output_language == "en" else "zh-TW"
        output = coerce_visual_vlm_output(output)
        structured_lines = split_vlm_lines(output.get("structured_content", ""))
        all_text_lines = split_vlm_lines(output.get("all_text", ""))
        facts = []
        for line in split_vlm_lines(output.get("facts", [])):
            if language == "en":
                if line.strip():
                    facts.append(line)
                continue
            zh_count = sum(1 for ch in line if "一" <= ch <= "鿿")
            ascii_count = sum(1 for ch in line if ch.isascii() and ch.isalpha())
            if zh_count >= 8 and zh_count >= ascii_count:
                facts.append(line)
        keywords = split_vlm_lines(output.get("keywords", []))
        image_type = str(output.get("image_type", "")).lower()

        is_flow = image_type == "flowchart" or any(" > " in line for line in structured_lines)
        if not is_flow:
            parts = []
            if facts:
                parts.extend(facts)
            if structured_lines:
                parts.extend(structured_lines)
            return "\n".join(parts or all_text_lines).strip()

        path_lines = [line for line in structured_lines if " > " in line]
        if language != "en":
            path_lines = [self._normalize_zh_visual_label(line) for line in path_lines]
            all_text_lines = [self._normalize_zh_visual_label(line) for line in all_text_lines]
            keywords = [self._normalize_zh_visual_label(keyword) for keyword in keywords]
        nodes: list[str] = []
        seen_nodes: set[str] = set()
        terminal_nodes: list[str] = []
        seen_terminal_nodes: set[str] = set()
        for path_line in path_lines:
            segments: list[str] = []
            for node in [item.strip() for item in path_line.split(">")]:
                node = re.sub(r"\s+", " ", node).strip()
                if not node:
                    continue
                segments.append(node)
                if node not in seen_nodes:
                    seen_nodes.add(node)
                    nodes.append(node)
            if segments:
                terminal = segments[-1]
                if terminal not in seen_terminal_nodes:
                    seen_terminal_nodes.add(terminal)
                    terminal_nodes.append(terminal)

        fallback_title = "Flowchart" if language == "en" else "流程圖"
        title_text = title.strip() or fallback_title
        if language != "en":
            title_text = self._normalize_zh_visual_label(title_text)
        generic_title = title_text.lower().startswith("figure ")
        if generic_title:
            title_text = fallback_title
        keywords_text = (", " if language == "en" else "、").join(keywords[:10])
        start_node = nodes[0] if nodes else (all_text_lines[0] if all_text_lines else "")
        if language == "en":
            end_text = ", ".join(terminal_nodes[:3]) if terminal_nodes else ""
        else:
            end_text = "、".join(terminal_nodes[:3]) if terminal_nodes else ""

        roles: list[str] = []
        if language != "en":
            role_candidates = (
                "受理單位(人事)",
                "被害人",
                "行為人之雇主",
                "院長",
                "調查小組",
                "處理單位",
                "申訴人",
                "被申訴人",
                "地方主管機關",
            )
            role_source = " ".join(nodes + all_text_lines)
            for role in role_candidates:
                if role in role_source and role not in roles:
                    roles.append(role)

        deadline_items: list[str] = []
        if language != "en":
            deadline_re = re.compile(r"((?:受理翌日起)?[一二三四五六七八九十兩0-9]+(?:個)?(?:工作日|週|月)內)(?:\s*[\[〔]([^\]〕]+)[\]〕])?")
            for line in (path_lines or all_text_lines):
                for match in deadline_re.finditer(line):
                    item = match.group(1)
                    if match.group(2):
                        item += f"（{match.group(2)}）"
                    context = re.sub(r"\s+", " ", line).strip()
                    sentence = f"{item}：{context}"
                    if sentence not in deadline_items:
                        deadline_items.append(sentence)

        if language == "en":
            branches = [line for line in path_lines if re.search(r"\((yes|no)\)", line, re.IGNORECASE)]
        else:
            branches = [line for line in path_lines if any(marker in line for marker in ("(是)", "(否)", "（是）", "（否）"))]

        parts: list[str] = []
        if language == "en":
            parts.append("## Semantic Summary")
            summary = (
                "This document organizes the workflow shown in the flowchart."
                if generic_title
                else f'This document is the semantic workflow content for \"{title_text}\".'
            )
            if start_node and end_text:
                if len(terminal_nodes) > 1:
                    summary += f' The flow starts with \"{start_node}\" and may end at one of: {end_text}.'
                else:
                    summary += f' The flow starts with \"{start_node}\" and may end at \"{end_text}\".'
            if keywords_text:
                summary += f" Useful query topics: {keywords_text}."
            parts.append(summary)
        else:
            parts.append("## 語意摘要")
            summary = "本文件整理流程圖中的作業流程。" if generic_title else f"本文件是「{title_text}」的流程圖語意化內容。"
            if start_node and end_text:
                if len(terminal_nodes) > 1:
                    summary += f"流程從「{start_node}」開始，可能結束於「{end_text}」。"
                else:
                    summary += f"流程從「{start_node}」開始，最後可能連到「{end_text}」。"
            elif start_node:
                summary += f"流程從「{start_node}」開始。"
            if keywords_text:
                summary += f"可用於查詢：{keywords_text}。"
            parts.append(summary)

        if roles:
            parts.append("\n## 主要角色與單位")
            parts.extend(f"- {role}" for role in roles[:12])

        if branches:
            parts.append("\n## " + ("Decisions and Branches" if language == "en" else "判斷與分支"))
            for line in branches[:8]:
                parts.append(f"- {line}")

        if deadline_items:
            parts.append("\n## 時限與依據")
            parts.extend(f"- {item}" for item in deadline_items[:12])

        if keywords:
            parts.append("\n## " + ("Common Query Topics" if language == "en" else "常見查詢主題"))
            parts.extend(f"- {keyword}" for keyword in keywords[:10])

        if facts:
            parts.append("\n## " + ("Key Facts" if language == "en" else "重要事實"))
            parts.extend(f"- {fact}" for fact in facts[:10])

        if path_lines:
            parts.append("\n## " + ("Detailed Flow Paths" if language == "en" else "詳細流程路徑"))
            parts.extend(f"- {line}" for line in path_lines)
            path_text_key = re.sub(r"\s+", "", " ".join(path_lines)).lower()
            missing_text_lines = [
                line
                for line in all_text_lines
                if len(re.sub(r"\s+", "", line)) >= 2
                and re.sub(r"\s+", "", line).lower() not in path_text_key
            ]
            if missing_text_lines:
                parts.append("\n## " + ("Text in Image" if language == "en" else "圖中文字"))
                parts.extend(f"- {line}" for line in missing_text_lines[:40])
        elif all_text_lines:
            parts.append("\n## " + ("Text in Image" if language == "en" else "圖中文字"))
            parts.extend(f"- {line}" for line in all_text_lines)

        return "\n".join(parts).strip()

    def _should_export_asset_document(self, asset: AssetEntry, source_md: str, assets: list[AssetEntry]) -> bool:
        """Avoid creating duplicate or empty child docs."""

        if asset.type == "form_asset":
            if self._is_low_value_form_asset(asset):
                return False
            title_key = re.sub(r"\s+", "", self._clean_export_title(asset.title)).lower()
            body = "\n".join(
                render_vlm_text(part)
                for part in [asset.filling_guide, asset.structured_content, asset.semantic_caption, asset.retrieval_text]
                if part
            ).strip()
            body_key = re.sub(r"\s+", "", self._clean_export_title(body)).lower()
            has_schema = bool(asset.field_schema)
            is_generic_page_stub = bool(re.fullmatch(r"formpage\d+|form\d*", title_key))
            if is_generic_page_stub and (not body or body_key in {title_key, ""}) and not has_schema:
                return False
            return True
        if asset.type == "figure_asset":
            body = "\n".join(
                render_vlm_text(part)
                for part in [asset.structured_content, asset.semantic_caption, asset.retrieval_text]
                if part
            ).strip()
            title_only = re.sub(r"\s+", "", self._clean_export_title(asset.title))
            body_key = re.sub(r"\s+", "", self._clean_export_title(body))
            if not body or body_key in {title_only, ""}:
                return False
            if self._is_decorative_figure_asset(asset):
                return False
        if asset.type == "table_asset" and (self._is_fragment_table_asset(asset) or self._is_low_confidence_table_asset(asset)):
            return False
        if asset.type not in {"figure_asset", "table_asset"}:
            return False
        same_page_assets = [item for item in assets if item.page_idx == asset.page_idx]
        if len(assets) == 1 and len(same_page_assets) == 1:
            text = source_md.strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            has_structured_flow = " > " in text or "來源頁碼" in text or "Source page" in text
            if asset.type == "figure_asset" and len(lines) <= 80 and has_structured_flow:
                return False
        return True

    def _extract_visual_text_lines_from_retrieval_text(self, retrieval_text: str, language: str) -> list[str]:
        heading_pattern = r"^##\s*(?:Text in Image|圖中文字)\s*$" if language == "en" else r"^##\s*(?:圖中文字|Text in Image)\s*$"
        lines: list[str] = []
        collecting = False
        for raw_line in str(retrieval_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.match(heading_pattern, line, re.I):
                collecting = True
                continue
            if collecting and line.startswith("## "):
                break
            if not collecting:
                continue
            line = re.sub(r"^[-*•]\s*", "", line).strip()
            if line:
                lines.append(line)
        return self._dedupe_visual_lines(lines)


    def _render_split_asset_document(
        self,
        asset: AssetEntry,
        source_title: str,
        source_filename: str,
        semantic_output_language: str = "zh-TW",
    ) -> str:
        del source_filename
        language = "en" if semantic_output_language == "en" else "zh-TW"
        title = self._clean_export_title(asset.title.strip() if asset.title else asset.asset_id)
        visual_output = coerce_visual_vlm_output(
            {
                "image_type": "flowchart" if " > " in (asset.structured_content or "") else "figure",
                "structured_content": asset.structured_content,
                "semantic_caption": asset.semantic_caption or "",
                "facts": asset.facts,
                "keywords": asset.keywords,
            }
        )
        retrieval_all_text = self._extract_visual_text_lines_from_retrieval_text(asset.retrieval_text, language)
        if retrieval_all_text:
            existing_all_text = split_vlm_lines(visual_output.get("all_text", ""))
            visual_output["all_text"] = self._dedupe_visual_lines([*existing_all_text, *retrieval_all_text])
        visual_structured_content = render_vlm_text(visual_output.get("structured_content", ""))
        if title.startswith("Figure ") and (visual_structured_content or asset.semantic_caption):
            title = self._infer_visual_asset_title(
                visual_structured_content,
                render_vlm_text(visual_output.get("semantic_caption", "")),
                title,
                all_text=visual_output.get("all_text"),
            )
            if title.startswith("Figure ") and str(visual_output.get("image_type", "")).lower() == "flowchart":
                title = source_title or ("Flowchart" if language == "en" else "流程圖")
        parts = [f"# {title}", ""]
        body_parts = []
        if asset.type == "figure_asset" and (visual_structured_content or render_vlm_text(visual_output.get("semantic_caption", ""))):
            body_parts.append(
                self._render_visual_semantic_content(
                    title,
                    visual_output,
                    semantic_output_language=semantic_output_language,
                )
            )
        elif asset.structured_content:
            body_parts.append(render_vlm_text(asset.structured_content))
        if asset.filling_guide:
            body_parts.append(render_vlm_text(asset.filling_guide))
        if not body_parts and asset.retrieval_text:
            body_parts.append(render_vlm_text(asset.retrieval_text))
        body = "\n\n".join(body_parts).strip()
        visual_caption = render_vlm_text(visual_output.get("semantic_caption", "")) if asset.type == "figure_asset" else render_vlm_text(asset.semantic_caption)
        if visual_caption and not visual_caption.lstrip().startswith("{"):
            zh_count = sum(1 for ch in visual_caption if "一" <= ch <= "鿿")
            ascii_count = sum(1 for ch in visual_caption if ch.isascii() and ch.isalpha())
            should_append_caption = language == "en" or (zh_count >= 4 and zh_count >= ascii_count)
            if should_append_caption and visual_caption not in body:
                body_parts.append(visual_caption)
                body = "\n\n".join(body_parts).strip()
        body = self._remove_duplicate_title_lines(body, title, source_title)
        return "\n".join(part for part in parts if part).strip() + "\n\n" + body + "\n"

    def _render_split_main_document(
        self,
        source_md: str,
        source_title: str,
        source_filename: str,
        form_entries: list[dict[str, Any]],
        asset_entries: list[AssetEntry] | None = None,
        semantic_output_language: str = "zh-TW",
    ) -> str:
        del source_filename
        language = "en" if semantic_output_language == "en" else "zh-TW"
        header = [f"# {source_title}"]
        if form_entries:
            header.extend(
                [
                    "",
                    "## Related Forms and Attachments" if language == "en" else "## 關聯表單與附件",
                ]
            )
            for entry in form_entries:
                title = str(entry.get("title") or entry["document_id"]).strip()
                entry_page_label = str(entry.get("page_label") or "").strip()
                if language == "en":
                    suffix = f", source page: {entry_page_label}" if entry_page_label else ""
                    header.append(f"- {title}{suffix}.")
                else:
                    suffix = f"，來源頁面：{entry_page_label}" if entry_page_label else ""
                    header.append(f"- {title}{suffix}。")
        table_collection_groups = self._table_collection_groups(asset_entries or [])
        if table_collection_groups:
            header.extend(
                [
                    "",
                    "## Related Table Collections" if language == "en" else "## 關聯表格集合",
                ]
            )
            for group in table_collection_groups:
                pages = sorted({asset.page_idx for asset in group["assets"] if asset.page_idx is not None})
                if language == "en":
                    page_text = f", pages: {', '.join(self._page_label(page, language) for page in pages[:8])}" if pages else ""
                    header.append(f"- {group['title']}: {len(group['assets'])} table blocks{page_text}.")
                else:
                    page_text = f"，頁面：{'、'.join(self._page_label(page, language) for page in pages[:8])}" if pages else ""
                    header.append(f"- {group['title']}：{len(group['assets'])} 個表格區塊{page_text}。")
        body = self._clean_split_main_body(
            source_md,
            source_title,
            asset_entries or [],
            semantic_output_language=language,
        )
        return "\n".join(header).strip() + "\n\n" + body + "\n"

    def _render_split_form_document(
        self,
        raw_markdown: str,
        item: dict[str, Any],
        source_title: str,
        source_filename: str,
        semantic_output_language: str = "zh-TW",
    ) -> str:
        del source_filename
        language = "en" if semantic_output_language == "en" else "zh-TW"
        fallback_title = "Form" if language == "en" else "表單"
        title = self._clean_export_title(str(item.get("title") or item.get("form_id") or fallback_title).strip())
        if self._is_unreliable_export_title(title):
            title = source_title if not self._is_unreliable_export_title(source_title) else fallback_title

        body = self._clean_split_form_body(raw_markdown, title, source_title, language)
        return f"# {title}\n\n{body}\n"

    def _collect_structured_form_page_indices(self, records: list[dict[str, Any]]) -> set[int]:
        """Return source pages that are exported as standalone form documents."""

        page_indices: set[int] = set()
        for record in records:
            if record.get("document_type") != "form_document":
                continue
            page_idx = record.get("page_idx")
            if isinstance(page_idx, int):
                page_indices.add(page_idx)
            page_indices_value = record.get("page_indices")
            if isinstance(page_indices_value, list):
                page_indices.update(idx for idx in page_indices_value if isinstance(idx, int))
        return page_indices

    def _load_enrichments(self, outputs_dir: Path) -> dict[str, dict[str, Any]]:
        """
        Load enrichments from enrichments.jsonl.

        Returns:
            Dict mapping block_id to enrichment data
        """
        enrichments: dict[str, dict[str, Any]] = {}
        enrichments_path = outputs_dir / "enrichments.jsonl"

        if not enrichments_path.exists():
            return enrichments

        try:
            with open(enrichments_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    block_id = entry.get("block_id", "")
                    if block_id:
                        enrichments[block_id] = entry
        except Exception:
            pass

        return enrichments

    async def _export_assets(
        self,
        document_ir: DocumentIR,
        assets_dir: Path,
        parse_cache_path: Path | None = None,
        enrichments: dict[str, dict[str, Any]] | None = None,
        semantic_output_language: str = "zh-TW",
    ) -> tuple[list[AssetEntry], dict[str, AssetEntry]]:
        """
        Export assets (figures, forms, tables) and build index.

        Integrates VLM enrichments for enhanced asset metadata.

        Returns:
            Tuple of (assets list, block_id -> asset mapping)
        """
        assets: list[AssetEntry] = []
        asset_map: dict[str, AssetEntry] = {}
        enrichments = enrichments or {}

        figures_dir = assets_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        forms_dir = assets_dir / "forms"
        forms_dir.mkdir(parents=True, exist_ok=True)

        tables_dir = assets_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        figure_idx = 0
        table_idx = 0
        form_idx = 0
        source_plan = plan_document(document_ir)
        suppress_all_form_assets = source_plan.document_type in {
            "travel_daily_allowance_table",
            "travel_domestic_expense_rate_table",
        }
        reference_table_block_ids = {
            block.block_id
            for block in document_ir.blocks
            if block.type == BlockType.TABLE
            and looks_like_reference_table(str(block.payload.get("table_body") or ""))
        }
        suppressed_form_block_ids = set(reference_table_block_ids)
        if suppress_all_form_assets:
            suppressed_form_block_ids.update(
                block.block_id for block in document_ir.blocks if block.type == BlockType.TABLE
            )

        structured_table_text_by_block: dict[str, str] = {}
        if suppressed_form_block_ids:
            structured_output = build_structured_rag(
                document_ir,
                semantic_output_language=semantic_output_language,
            )
            grouped_lines: dict[str, list[str]] = {}
            for chunk in structured_output.chunks:
                for block_id in chunk.get("block_ids", []):
                    content = str(chunk.get("content") or "").strip()
                    if content:
                        grouped_lines.setdefault(str(block_id), []).append(content)
            structured_table_text_by_block = {
                block_id: "\n\n".join(lines) for block_id, lines in grouped_lines.items()
            }

        # First, process block-level assets
        for block in document_ir.blocks:
            enrichment = enrichments.get(block.block_id, {})
            enrichment_output = enrichment.get("output", {})
            enrichment_kind = enrichment.get("kind", "")

            if block.type == BlockType.IMAGE:
                # Export figure
                img_path = block.payload.get("img_path", "")
                if img_path and parse_cache_path:
                    src_path = self._find_image(img_path, parse_cache_path)
                    if src_path and src_path.exists():
                        asset_id = f"fig{figure_idx:04d}"
                        dst_path = figures_dir / f"{asset_id}{src_path.suffix}"
                        shutil.copy2(src_path, dst_path)

                        caption = block.payload.get("caption", "")
                        title = caption[:100] if caption else f"Figure {figure_idx + 1}"

                        # Integrate VLM enrichment for figures. Coerce first so malformed
                        # JSON-like captions do not leak into split documents.
                        enrichment_output = coerce_visual_vlm_output(enrichment_output)
                        enrichment_output = self._augment_visual_output_from_page_text(
                            enrichment_output,
                            document_ir,
                            block,
                        )
                        semantic_caption = render_vlm_text(enrichment_output.get("semantic_caption", ""))
                        structured_content = render_vlm_text(enrichment_output.get("structured_content", ""))
                        facts = enrichment_output.get("facts", [])
                        keywords = enrichment_output.get("keywords", [])
                        needs_review = enrichment.get("quality", {}).get("needs_review", False)
                        if title.startswith("Figure ") and (structured_content or enrichment_output.get("all_text")):
                            title = self._infer_visual_asset_title(
                                structured_content,
                                semantic_caption,
                                title,
                                all_text=enrichment_output.get("all_text"),
                            )

                        # Build enhanced retrieval text without leaking raw English captions.
                        semantic_retrieval = self._render_visual_semantic_content(
                            title,
                            enrichment_output,
                            semantic_output_language=semantic_output_language,
                        )
                        retrieval_parts = [title, semantic_retrieval]
                        if keywords:
                            retrieval_parts.extend(keywords)
                        retrieval_text = "\n".join(part for part in retrieval_parts if part).strip()

                        asset = AssetEntry(
                            type="figure_asset",
                            asset_id=asset_id,
                            doc_id=document_ir.doc_id,
                            run_id=document_ir.run_id,
                            title=title,
                            page_idx=block.page_idx,
                            asset_path=f"assets/figures/{dst_path.name}",
                            block_id=block.block_id,
                            retrieval_text=retrieval_text,
                            semantic_caption=semantic_caption,
                            structured_content=structured_content,
                            facts=facts,
                            keywords=keywords,
                            needs_review=needs_review,
                        )
                        assets.append(asset)
                        asset_map[block.block_id] = asset
                        figure_idx += 1

            elif block.type == BlockType.TABLE:
                # Export table assets, or form assets when VLM classified a native
                # spreadsheet/layout table as a fillable form.
                table_body = block.payload.get("table_body", "")
                caption = block.payload.get("table_caption", "")

                # Ensure caption is a string (could be list from MinerU)
                if isinstance(caption, list):
                    caption = " ".join(str(x) for x in caption if x)

                if table_body:
                    needs_review = enrichment.get("quality", {}).get("needs_review", False)
                    if enrichment_kind in ("form_asset", "form_guide") and block.block_id not in suppressed_form_block_ids:
                        asset_id = f"form{form_idx:04d}"
                        title = (
                            enrichment_output.get("title")
                            or str(caption)[:100]
                            or f"Form {form_idx + 1}"
                        )
                        triggers = enrichment_output.get("triggers", [])
                        filling_guide = enrichment_output.get("filling_guide", "")
                        field_schema = enrichment_output.get("field_schema", [])
                        retrieval_text = enrichment_output.get("retrieval_text") or " ".join(
                            part for part in [title, render_vlm_text(filling_guide)] if part
                        )

                        asset = AssetEntry(
                            type="form_asset",
                            asset_id=asset_id,
                            doc_id=document_ir.doc_id,
                            run_id=document_ir.run_id,
                            title=self._clean_export_title(str(title)),
                            triggers=triggers,
                            page_idx=block.page_idx,
                            asset_path="",
                            block_id=block.block_id,
                            retrieval_text=retrieval_text,
                            filling_guide=filling_guide,
                            field_schema=field_schema,
                            needs_review=needs_review,
                        )
                        assets.append(asset)
                        asset_map[block.block_id] = asset
                        form_idx += 1
                        continue

                    asset_id = f"tbl{table_idx:04d}"
                    title = infer_table_asset_title(
                        caption=caption,
                        source_title=source_plan.title,
                        page_idx=block.page_idx,
                        table_idx=table_idx,
                        semantic_output_language=semantic_output_language,
                    )
                    title = self._improve_table_asset_title_from_body(
                        table_body,
                        title,
                        semantic_output_language,
                    )

                    # Extract first row as preview
                    preview = self._extract_table_preview(table_body)

                    # Integrate VLM enrichment for tables
                    table_summary = enrichment_output.get("table_summary", "")
                    key_columns = enrichment_output.get("key_columns", [])

                    # Ensure table_summary is a string (could be list from VLM)
                    if isinstance(table_summary, list):
                        table_summary = " ".join(str(x) for x in table_summary if x)

                    # Build enhanced retrieval text
                    structured_table_text = structured_table_text_by_block.get(block.block_id, "") or semantic_table_to_text(
                        table_body,
                        title,
                        semantic_output_language=semantic_output_language,
                    )
                    if structured_table_text:
                        retrieval_text = "\n\n".join(part for part in [title, structured_table_text] if part)
                    else:
                        retrieval_parts = [title, preview]
                        if table_summary:
                            retrieval_parts.append(str(table_summary))
                        if key_columns:
                            # Ensure all key_columns items are strings
                            key_columns_str = " ".join(str(x) for x in key_columns if x)
                            if key_columns_str:
                                retrieval_parts.append(key_columns_str)
                        retrieval_text = " ".join(filter(None, retrieval_parts))

                    asset = AssetEntry(
                        type="table_asset",
                        asset_id=asset_id,
                        doc_id=document_ir.doc_id,
                        run_id=document_ir.run_id,
                        title=title,
                        page_idx=block.page_idx,
                        asset_path="",  # Tables are inline
                        block_id=block.block_id,
                        retrieval_text=retrieval_text,
                        structured_content=structured_table_text,
                        needs_review=needs_review,
                    )
                    assets.append(asset)
                    asset_map[block.block_id] = asset
                    table_idx += 1

        # Now process form page assets (from enrichments with kind=form_asset or form_guide)
        form_enrichments = {
            k: v for k, v in enrichments.items()
            if v.get("kind") in ("form_asset", "form_guide")
            and k not in suppressed_form_block_ids
            and k not in asset_map
        }

        for block_id, enrichment in form_enrichments.items():
            enrichment_output = enrichment.get("output", {})
            input_data = enrichment.get("input", {})
            evidence_data = enrichment.get("evidence", {})

            # Get page_idx from evidence first, then input (evidence is authoritative)
            page_idx = evidence_data.get("page_idx") or input_data.get("page_idx", 0)

            # Get asset_path from evidence or input (don't guess with glob)
            stored_asset_path = evidence_data.get("asset_path") or input_data.get("asset_path")

            # Resolve to actual file path
            form_image_path = None
            if stored_asset_path:
                # stored_asset_path might be absolute or relative
                stored_path = Path(stored_asset_path)
                if stored_path.exists():
                    form_image_path = stored_path
                else:
                    # Try relative to forms_dir
                    candidate = forms_dir / stored_path.name
                    if candidate.exists():
                        form_image_path = candidate

            # Only fallback to glob if no stored path (legacy entries)
            if not form_image_path:
                for form_file in forms_dir.glob(f"form_p{page_idx:04d}.*"):
                    form_image_path = form_file
                    break

            # Create form asset entry
            asset_id = f"form{form_idx:04d}"
            title = enrichment_output.get("title", f"Form {form_idx + 1}")
            triggers = enrichment_output.get("triggers", [])
            filling_guide = enrichment_output.get("filling_guide", "")
            field_schema = enrichment_output.get("field_schema", [])
            retrieval_text = enrichment_output.get("retrieval_text", title)
            needs_review = enrichment.get("quality", {}).get("needs_review", False)

            asset = AssetEntry(
                type="form_asset",
                asset_id=asset_id,
                doc_id=document_ir.doc_id,
                run_id=document_ir.run_id,
                title=self._clean_export_title(title),
                triggers=triggers,
                page_idx=page_idx,
                asset_path=f"assets/forms/{form_image_path.name}" if form_image_path else "",
                block_id=block_id,
                retrieval_text=retrieval_text,
                filling_guide=filling_guide,
                field_schema=field_schema,
                needs_review=needs_review,
            )
            assets.append(asset)
            asset_map[block_id] = asset
            form_idx += 1

        # Also check for form images without enrichments (fallback)
        existing_form_block_ids = set(form_enrichments)
        existing_form_pages = {asset.page_idx for asset in assets if asset.type == "form_asset"}
        for form_file in forms_dir.glob("form_p*.png"):
            # Parse page_idx from filename
            stem = form_file.stem  # form_p0001
            page_str = stem.replace("form_p", "")
            try:
                page_idx = int(page_str)
            except ValueError:
                continue

            # Generate a block_id for this form page
            form_block_id = f"form_page_{page_idx:04d}"
            if form_block_id in existing_form_block_ids or page_idx in existing_form_pages:
                continue

            # Create basic form asset entry (without VLM enrichment)
            asset_id = f"form{form_idx:04d}"
            asset = AssetEntry(
                type="form_asset",
                asset_id=asset_id,
                doc_id=document_ir.doc_id,
                run_id=document_ir.run_id,
                title=f"Form Page {page_idx + 1}",
                triggers=[],
                page_idx=page_idx,
                asset_path=f"assets/forms/{form_file.name}",
                block_id=form_block_id,
                retrieval_text=f"Form Page {page_idx + 1}",
                needs_review=True,  # Mark for review since no VLM enrichment
            )
            assets.append(asset)
            form_idx += 1

        return assets, asset_map

    def _find_image(self, img_path: str, parse_cache_path: Path) -> Path | None:
        """Find image file in parse cache."""
        # Try relative path from cache
        candidate = parse_cache_path / img_path
        if candidate.exists():
            return candidate

        # Try searching in cache directory
        for p in parse_cache_path.rglob(Path(img_path).name):
            return p

        return None

    def _extract_table_preview(self, table_body: str, max_chars: int = 200) -> str:
        """Extract preview text from table body."""
        # Simple extraction: strip HTML/MD and take first N chars
        import re

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", table_body)
        # Remove markdown formatting
        text = re.sub(r"[|*_#`]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars]

    def _render_dataset_md(
        self,
        document_ir: DocumentIR,
    ) -> tuple[str, SourceMap]:
        """
        Render dataset.md - high-fidelity output.

        Preserves original structure without VLM enrichments.
        """
        lines: list[str] = []
        anchors: list[MdAnchor] = []
        char_pos = 0

        for block in document_ir.blocks:
            start_pos = char_pos
            block_lines = self._render_block_dataset(block)

            if block_lines:
                content = "\n".join(block_lines) + "\n\n"
                lines.append(content)
                char_pos += len(content)

                # Create anchor
                anchor = MdAnchor(
                    anchor_id=block.block_id,
                    md_range=[start_pos, char_pos],
                    block_ids=[block.block_id],
                )
                anchors.append(anchor)

        source_map = SourceMap(md_anchors=anchors)
        return "".join(lines), source_map

    def _render_block_dataset(self, block: Block) -> list[str]:
        """Render a single block for dataset.md."""
        lines: list[str] = []

        if block.type == BlockType.TEXT:
            text = clean_latex_symbols(block.payload.get("text", ""))
            level = block.payload.get("text_level", 0)

            if level > 0:
                # Heading
                prefix = "#" * min(level, 6)
                lines.append(f"{prefix} {text}")
            else:
                # Body text
                lines.append(text)

        elif block.type == BlockType.TABLE:
            table_body = block.payload.get("table_body", "")
            caption = block.payload.get("table_caption")

            # Ensure caption is a string (MinerU may return list)
            if isinstance(caption, list):
                caption = " ".join(str(x) for x in caption if x)

            if caption:
                lines.append(f"**{clean_latex_symbols(caption)}**")
                lines.append("")

            # Keep HTML for display, but clean empty rows and apply LaTeX cleanup
            cleaned_table = clean_html_table(table_body)
            lines.append(clean_latex_symbols(cleaned_table))

        elif block.type == BlockType.IMAGE:
            img_path = block.payload.get("img_path", "")
            caption = block.payload.get("caption", "")

            if img_path:
                alt_text = caption or "Image"
                lines.append(f"![{alt_text}]({img_path})")

            if caption:
                lines.append(f"*{caption}*")

        elif block.type == BlockType.EQUATION:
            latex = block.payload.get("latex", "")
            eq_type = block.payload.get("equation_type", "display")

            if eq_type == "inline":
                lines.append(f"${latex}$")
            else:
                lines.append(f"$$\n{latex}\n$$")

        elif block.type == BlockType.CODE:
            code = block.payload.get("code", "")
            lang = block.payload.get("language", "")

            lines.append(f"```{lang}")
            lines.append(code)
            lines.append("```")

        elif block.type == BlockType.LIST:
            items = block.payload.get("items", [])
            list_type = block.payload.get("list_type", "unordered")

            for i, item in enumerate(items):
                # 確保 item 是字符串（可能是嵌套列表）
                if isinstance(item, list):
                    item_text = " ".join(str(x) for x in item)
                else:
                    item_text = str(item) if item is not None else ""

                if list_type == "ordered":
                    lines.append(f"{i + 1}. {item_text}")
                else:
                    lines.append(f"- {item_text}")

        return lines

    def _render_rag_md(
        self,
        document_ir: DocumentIR,
        asset_map: dict[str, AssetEntry],
        enrichments: dict[str, dict[str, Any]] | None = None,
        suppress_form_enrichment: bool = False,
        excluded_page_indices: set[int] | None = None,
        semantic_output_language: str = "zh-TW",
    ) -> tuple[str, SourceMap]:
        """
        Render rag.md - retrieval-friendly output.

        Includes asset references for figures/tables.
        Integrates VLM enrichments (structured_content) when available.
        """
        lines: list[str] = []
        anchors: list[MdAnchor] = []
        char_pos = 0
        enrichments = enrichments or {}
        excluded_page_indices = excluded_page_indices or set()

        # Collect all triggers and watermarks across pages (for unified output at end)
        all_triggers: set[str] = set()
        all_watermarks: list[str] = []

        # Group blocks by page for form enrichment integration
        pages_with_form_enrichment: set[int] = set()
        form_enrichments_by_page: dict[int, dict[str, Any]] = {}

        # Find form enrichments and their page indices
        if not suppress_form_enrichment:
            for block_id, enrichment in enrichments.items():
                if enrichment.get("kind") in ("form_asset", "form_guide"):
                    page_idx = enrichment.get("input", {}).get("page_idx")
                    if page_idx is not None:
                        pages_with_form_enrichment.add(page_idx)
                        form_enrichments_by_page[page_idx] = enrichment

        visual_structured_pages = {
            block.page_idx
            for block in document_ir.blocks
            if block.type == BlockType.IMAGE
            and render_vlm_text(
                enrichments.get(block.block_id, {}).get("output", {}).get("structured_content", "")
            )
        }

        # Track which pages we've already added form enrichment content for
        pages_enriched: set[int] = set()
        first_page_with_title = True  # Only add title/date for first page

        for block in document_ir.blocks:
            start_pos = char_pos
            page_idx = block.page_idx

            if page_idx in excluded_page_indices:
                continue

            if page_idx in visual_structured_pages and block.type != BlockType.IMAGE:
                text = str(block.payload.get("text", "")).strip() if block.type == BlockType.TEXT else ""
                if not (block.reading_order == 0 and len(text) >= 10):
                    continue

            # Check if this block's page has a form enrichment
            if page_idx in pages_with_form_enrichment:
                # Only render form enrichment content once per page
                if page_idx not in pages_enriched:
                    form_enrichment = form_enrichments_by_page[page_idx]
                    form_output = form_enrichment.get("output", {})

                    # Collect triggers from this page
                    page_triggers = form_output.get("triggers", [])
                    if page_triggers:
                        all_triggers.update(page_triggers)

                    # Check document_type to determine rendering strategy
                    document_type = form_output.get("document_type", "form")

                    # For org_chart, use org_chart_graph.to_structured_content() if available
                    org_chart_data = form_output.get("org_chart_graph", {})
                    if org_chart_data and org_chart_data.get("nodes"):
                        structured_content = self._render_org_chart_graph(org_chart_data)
                    elif document_type == "org_chart":
                        # Use structured_content from VLM (PATH notation)
                        structured_content = form_output.get("structured_content", "")
                        # Do NOT fall back to form fields for org_chart
                    else:
                        # Regular form: Use filling_guide if structured_content not available
                        structured_content = form_output.get("structured_content", "")
                        if not structured_content:
                            # Build RAG content from filling_guide + field_schema
                            # Returns (content, watermarks)
                            structured_content, page_watermarks = self._build_form_rag_content(
                                form_output,
                                semantic_output_language=semantic_output_language,
                            )
                            all_watermarks.extend(page_watermarks)

                    if structured_content:
                        # Build content with title, date, and structured content
                        content_parts = []

                        # Add title and date only for first page
                        if first_page_with_title:
                            title = form_output.get("title", "")
                            date = form_output.get("date", "")
                            if title:
                                content_parts.append(f"# {title}")
                                if date:
                                    if semantic_output_language == "en":
                                        content_parts.append(f"**Date: {date}**")
                                    else:
                                        content_parts.append(f"**日期：{date}**")
                                content_parts.append("")  # Empty line after header
                            first_page_with_title = False

                        content_parts.append(render_vlm_text(structured_content))
                        content = "\n".join(content_parts) + "\n\n"

                        lines.append(content)
                        char_pos += len(content)

                        # Create anchor for form enrichment
                        form_block_id = f"form_page_{page_idx:04d}"
                        anchor = MdAnchor(
                            anchor_id=form_block_id,
                            md_range=[start_pos, char_pos],
                            block_ids=[form_block_id],
                        )
                        anchors.append(anchor)

                    pages_enriched.add(page_idx)

                # Skip ALL original blocks for form pages since VLM content is more complete
                continue

            # Check if this block has a figure enrichment with structured_content
            block_enrichment = enrichments.get(block.block_id, {})
            if block_enrichment.get("kind") in ("figure_caption", "figure_description"):
                structured_content = block_enrichment.get("output", {}).get("structured_content", "")
                if structured_content:
                    asset = asset_map.get(block.block_id)
                    if asset and self._is_decorative_figure_asset(asset):
                        continue
                    fallback_title = "Flowchart" if semantic_output_language == "en" else "流程圖"
                    asset_title = asset.title if asset else fallback_title
                    content = (
                        self._render_visual_semantic_content(
                            asset_title,
                            block_enrichment.get("output", {}),
                            semantic_output_language=semantic_output_language,
                        )
                        + "\n\n"
                    )
                    lines.append(content)
                    char_pos += len(content)

                    anchor = MdAnchor(
                        anchor_id=block.block_id,
                        md_range=[start_pos, char_pos],
                        block_ids=[block.block_id],
                    )
                    anchors.append(anchor)
                    continue

            # Default: render block normally
            block_lines = self._render_block_rag(
                block,
                asset_map,
                semantic_output_language=semantic_output_language,
            )

            if block_lines:
                content = "\n".join(block_lines) + "\n\n"
                lines.append(content)
                char_pos += len(content)

                # Create anchor
                anchor = MdAnchor(
                    anchor_id=block.block_id,
                    md_range=[start_pos, char_pos],
                    block_ids=[block.block_id],
                )
                anchors.append(anchor)

        # Add unified triggers at the end (once for entire document)
        # Note: watermarks are filtered out and not displayed
        if all_triggers:
            keyword_label = "Keywords" if semantic_output_language == "en" else "關鍵字"
            footer_content = f"**{keyword_label}**: {', '.join(sorted(all_triggers))}\n"
            lines.append(footer_content)
            char_pos += len(footer_content)

        source_map = SourceMap(md_anchors=anchors)
        return "".join(lines), source_map

    def _render_org_chart_graph(self, org_chart_data: dict[str, Any]) -> str:
        """
        Render org chart graph data to structured markdown.

        Reconstructs OrgChartGraph from dict and calls to_structured_content().
        """
        try:
            # Reconstruct nodes
            nodes = []
            for n in org_chart_data.get("nodes", []):
                # Parse category enum
                cat_value = n.get("category", "未分類")
                try:
                    category = OrgCategory(cat_value)
                except ValueError:
                    category = OrgCategory.UNKNOWN

                cat_hint_value = n.get("category_hint", "未分類")
                try:
                    category_hint = OrgCategory(cat_hint_value)
                except ValueError:
                    category_hint = OrgCategory.UNKNOWN

                node = OrgNode(
                    id=n.get("id", ""),
                    label=n.get("label", ""),
                    bbox=n.get("bbox", [0, 0, 0, 0]),
                    page_idx=n.get("page_idx", 0),
                    category=category,
                    category_hint=category_hint,
                    level=n.get("level", -1),
                )
                nodes.append(node)

            # Reconstruct edges
            edges = []
            for e in org_chart_data.get("edges", []):
                edge_type_value = e.get("type", "unknown")
                try:
                    edge_type = EdgeType(edge_type_value)
                except ValueError:
                    edge_type = EdgeType.UNKNOWN

                edge = OrgEdge(
                    from_id=e.get("from", ""),
                    to_id=e.get("to", ""),
                    edge_type=edge_type,
                    confidence=e.get("confidence", 0.5),
                )
                edges.append(edge)

            # Reconstruct groups
            groups = []
            for g in org_chart_data.get("groups", []):
                group = OrgGroup(
                    name=g.get("name", ""),
                    members=g.get("members", []),
                )
                groups.append(group)

            # Create graph and generate content
            graph = OrgChartGraph(
                nodes=nodes,
                edges=edges,
                groups=groups,
                title=org_chart_data.get("title", ""),
                date=org_chart_data.get("date", ""),
                page_idx=org_chart_data.get("page_idx", 0),
            )

            return graph.to_structured_content()

        except Exception as e:
            # Fallback: return empty string, caller will use VLM content
            import logging
            logging.warning(f"Failed to render org_chart_graph: {e}")
            return ""

    def _build_form_rag_content(
        self,
        form_output: dict[str, Any],
        semantic_output_language: str = "zh-TW",
    ) -> tuple[str, list[str]]:
        """
        Build RAG content from form_asset output.

        Priority:
        1. For non-form documents (document_type: "other"), use all_text
        2. For forms, use filling_guide + field_schema
        3. Fallback to all_text if form fields are empty

        Returns:
            tuple of (content, watermarks_list)
            - Triggers are NOT included here (collected separately by caller)
        """
        parts: list[str] = []
        watermarks: list[str] = []
        document_type = form_output.get("document_type", "form")
        form_output.get("title", "")

        # For non-form documents, prioritize all_text (contains complete document content)
        all_text = form_output.get("all_text", [])
        filling_guide = form_output.get("filling_guide", "")
        field_schema = form_output.get("field_schema", [])

        # Use all_text for non-form documents, or when form fields are empty
        if document_type == "other" or (not filling_guide and not field_schema):
            if all_text:
                # Filter and clean all_text items, collect watermarks separately
                # Note: We keep both VLM title and full table_caption from all_text
                filtered_texts = []

                for t in all_text:
                    text = str(t).strip() if t else ""
                    if not text:
                        continue

                    # Collect watermark-like content separately (filtered out)
                    if self._is_watermark(text):
                        watermarks.append(text)
                        continue

                    filtered_texts.append(text)

                if filtered_texts:
                    parts.append("\n\n".join(filtered_texts))
        else:
            # Standard form: use filling_guide + field_schema
            if filling_guide:
                parts.append(render_vlm_text(filling_guide))

            if field_schema:
                heading = "Form Fields" if semantic_output_language == "en" else "表單欄位"
                parts.append(f"\n## {heading}")
                for field in field_schema:
                    if isinstance(field, dict):
                        name = field.get("name", "")
                        field_type = field.get("type", "text")
                        if semantic_output_language == "en":
                            required = "required" if field.get("required") else "optional"
                        else:
                            required = "必填" if field.get("required") else "選填"
                        if name:
                            parts.append(f"- {name} ({field_type}, {required})")

        # Note: triggers are NOT added here - they are collected by caller and output once at end
        return "\n".join(parts), watermarks

    def _is_title_duplicate(self, text: str, title: str) -> bool:
        """Check if text is essentially the same as title (with possible suffix like date)."""
        # Normalize for comparison
        text_norm = text.replace(" ", "").replace("　", "").lower()
        title_norm = title.replace(" ", "").replace("　", "").lower()

        # Check if title is contained in text or vice versa
        if title_norm in text_norm or text_norm in title_norm:
            return True

        # Check if they share significant overlap (first 20 chars)
        if len(title_norm) >= 10 and len(text_norm) >= 10:
            if title_norm[:20] == text_norm[:20]:
                return True

        return False

    def _is_watermark(self, text: str) -> bool:
        """Check if text looks like a watermark (page number, date stamp, doc ID)."""
        import re

        text = text.strip()

        # Pure page number (single digit or small number)
        if re.match(r'^\d{1,3}$', text):
            return True

        # Date-like patterns (yyyy/mm/dd, yyyy-mm-dd, yyyy.mm.dd)
        if re.match(r'^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$', text):
            return True

        # Short alphanumeric codes (likely doc IDs or watermarks)
        if re.match(r'^[a-zA-Z]?\d{4,8}$', text) and len(text) <= 10:
            return True

        return False

    def _render_block_rag(
        self,
        block: Block,
        asset_map: dict[str, AssetEntry],
        semantic_output_language: str = "zh-TW",
    ) -> list[str]:
        """Render a single block for rag.md."""
        lines: list[str] = []
        asset = asset_map.get(block.block_id)

        if block.type == BlockType.TEXT:
            text = clean_latex_symbols(block.payload.get("text", ""))
            level = block.payload.get("text_level", 0)

            if level > 0:
                prefix = "#" * min(level, 6)
                lines.append(f"{prefix} {text}")
            else:
                lines.append(text)

        elif block.type == BlockType.TABLE:
            table_body = block.payload.get("table_body", "")
            caption = block.payload.get("table_caption")

            # Ensure caption is a string (MinerU may return list)
            if isinstance(caption, list):
                caption = " ".join(str(x) for x in caption if x)

            # Convert HTML table to semantic retrieval text; avoid raw HTML or legacy TABLE/ROW format.
            title = asset.title if asset else infer_table_asset_title(
                caption=caption,
                source_title="",
                page_idx=block.page_idx,
                table_idx=0,
                semantic_output_language=semantic_output_language,
            )
            table_text = semantic_table_to_text(
                table_body,
                title,
                semantic_output_language=semantic_output_language,
            )
            if table_text:
                lines.append(table_text)
            else:
                preview = self._extract_table_preview(table_body, max_chars=500)
                if caption:
                    lines.append(f"## {clean_latex_symbols(caption)}")
                if preview:
                    summary_label = "Table content summary: " if semantic_output_language == "en" else "表格內容摘要："
                    lines.append(summary_label + clean_latex_symbols(preview))

            # Add asset reference as explicit token
            if asset:
                lines.append("")
                lines.append(f"[[asset:{asset.asset_id}]]")

        elif block.type == BlockType.IMAGE:
            caption = block.payload.get("caption", "")

            if asset and self._is_decorative_figure_asset(asset):
                return []

            if asset:
                # Use exported asset path
                alt_text = caption or asset.title
                lines.append(f"![{alt_text}](asset://{asset.asset_path})")

                if caption:
                    lines.append(f"*{caption}*")

                # Add asset reference as explicit token
                lines.append(f"[[asset:{asset.asset_id}]]")
            else:
                # Fallback to original path
                img_path = block.payload.get("img_path", "")
                if img_path:
                    alt_text = caption or "Image"
                    lines.append(f"![{alt_text}]({img_path})")

        elif block.type == BlockType.EQUATION:
            latex = block.payload.get("latex", "")
            eq_type = block.payload.get("equation_type", "display")

            if eq_type == "inline":
                lines.append(f"${latex}$")
            else:
                lines.append(f"$$\n{latex}\n$$")

        elif block.type == BlockType.CODE:
            code = block.payload.get("code", "")
            lang = block.payload.get("language", "")

            lines.append(f"```{lang}")
            lines.append(code)
            lines.append("```")

        elif block.type == BlockType.LIST:
            items = block.payload.get("items", [])
            list_type = block.payload.get("list_type", "unordered")

            for i, item in enumerate(items):
                # 確保 item 是字符串（可能是嵌套列表）
                if isinstance(item, list):
                    item_text = " ".join(str(x) for x in item)
                else:
                    item_text = str(item) if item is not None else ""

                if list_type == "ordered":
                    lines.append(f"{i + 1}. {item_text}")
                else:
                    lines.append(f"- {item_text}")

        return lines

    def _generate_quality_report(
        self,
        document_ir: DocumentIR,
        assets: list[AssetEntry],
    ) -> QualityReport:
        """Generate quality report."""
        block_counts = document_ir.count_by_type()

        total_text = 0
        for block in document_ir.blocks:
            total_text += len(block.get_text() or "")

        # Calculate coverage per page
        coverage: dict[str, float] = {}
        page_blocks: dict[int, list[Block]] = {}

        for block in document_ir.blocks:
            if block.page_idx not in page_blocks:
                page_blocks[block.page_idx] = []
            page_blocks[block.page_idx].append(block)

        for page_idx, blocks in page_blocks.items():
            text_blocks = sum(1 for b in blocks if b.type == BlockType.TEXT)
            total_blocks = len(blocks)
            if total_blocks > 0:
                coverage[f"page_{page_idx}"] = text_blocks / total_blocks

        return QualityReport(
            doc_id=document_ir.doc_id,
            run_id=document_ir.run_id,
            block_counts=block_counts,
            page_count=len(document_ir.pages),
            total_text_length=total_text,
            asset_count=len(assets),
            coverage=coverage,
        )
