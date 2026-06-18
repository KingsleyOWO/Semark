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
from app.pipeline.structured_rag import (
    build_form_documents_rag,
    build_structured_rag,
    build_structured_rag_with_vlm_fallback,
    looks_like_reference_table,
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

            # Load enrichments from enrich stage (if exists)
            enrichments = self._load_enrichments(outputs_dir)

            # 1. Export assets and build index (with enrichments integration)
            assets, asset_map = await self._export_assets(
                document_ir=document_ir,
                assets_dir=assets_dir,
                parse_cache_path=parse_cache_path,
                enrichments=enrichments,
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
                )
            else:
                structured_output = build_structured_rag(document_ir)
            if not structured_output.records:
                structured_output = build_form_documents_rag(
                    document_ir=document_ir,
                    enrichments=enrichments,
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
            )

            if self._quality_gate_needs_structured_repair(quality_gate):
                repaired_output = build_form_documents_rag(
                    document_ir=document_ir,
                    enrichments=enrichments,
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
                    )
                    quality_gate.stats["structured_repair_applied"] = True

            write_quality_gate(quality_gate, outputs_dir)

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

    def _write_document_exports(
        self,
        outputs_dir: Path,
        source_md: str,
        assets: list[AssetEntry],
        structured_paths: dict[str, str],
        document_ir: DocumentIR,
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

        form_entries: list[dict[str, Any]] = []
        forms_index_value = structured_paths.get("forms_index")
        forms_index_path = Path(forms_index_value) if forms_index_value else None
        if forms_index_path and forms_index_path.is_file():
            forms_index = json.loads(forms_index_path.read_text(encoding="utf-8"))
            for item in forms_index:
                form_file = Path(item.get("files", {}).get("markdown") or "")
                if not form_file.exists():
                    continue
                dst = documents_dir / f"{item['form_id']}.md"
                form_md = form_file.read_text(encoding="utf-8")
                dst.write_text(
                    self._render_split_form_document(
                        raw_markdown=form_md,
                        item=item,
                        source_title=source_title,
                        source_filename=source_filename,
                    ),
                    encoding="utf-8",
                )
                form_entry = {
                    "document_id": item["form_id"],
                    "kind": "form",
                    "title": self._clean_export_title(str(item.get("title") or "")),
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
            for asset in assets:
                if not self._should_export_asset_document(asset, source_md, assets):
                    continue
                dst = documents_dir / f"{asset.asset_id}.md"
                dst.write_text(
                    self._render_split_asset_document(
                        asset=asset,
                        source_title=source_title,
                        source_filename=source_filename,
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

        paths["main_document"].write_text(
            self._render_split_main_document(
                source_md=source_md,
                source_title=source_title,
                source_filename=source_filename,
                form_entries=form_entries,
            ),
            encoding="utf-8",
        )
        paths["documents_index"].write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {key: str(path) for key, path in paths.items()}

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
        return "\n".join(cleaned).strip()

    @staticmethod
    def _clean_export_title(value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip().strip("#:： ")
        text = re.sub(r"(表[一二三四五六七八九十0-9]+)[〇○昇鑑箇]+", r"\1", text)
        text = re.sub(r"(表[一二三四五六七八九十0-9]+)\s*[〇○昇鑑箇]+", r"\1", text)
        text = re.sub(r"[昇鑑](?=台灣|臺灣|國內|國外|大台北|大臺北)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip("#:： ")

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
            if self._is_unreliable_export_title(title):
                continue
            if source_ext in {".xls", ".xlsx", ".ods"} and self._is_weak_spreadsheet_source_title(title):
                continue
            return title[:120]
        body_title = self._infer_source_title_from_body(source_md)
        if body_title:
            return body_title[:120]
        fallback = self._clean_export_title(Path(source_path).stem)
        return fallback[:120] or "來源文件"

    def _infer_source_title_from_body(self, source_md: str) -> str:
        text = re.sub(r"\s+", " ", source_md or "").strip()
        if not text:
            return ""
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

    @classmethod
    def _is_unreliable_export_title(cls, title: str) -> bool:
        compact = re.sub(r"\s+", "", cls._clean_export_title(title))
        if not compact:
            return True
        if re.fullmatch(r"\d+", compact):
            return True
        if re.fullmatch(r"[A-Za-z]*Figure\d*|Table\d*", compact, re.IGNORECASE):
            return True
        if not re.search(r"[\u4e00-\u9fff]", compact) and len(re.findall(r"\d", compact)) >= 3:
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

    def _infer_visual_asset_title(self, structured_content: str, semantic_caption: str, fallback: str) -> str:
        """Use document-visible Chinese text instead of generic Figure N labels."""

        parsed = coerce_visual_vlm_output({"structured_content": structured_content, "semantic_caption": semantic_caption})
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

    def _render_visual_semantic_content(self, title: str, output: dict[str, Any]) -> str:
        """Convert VLM figure output into retrieval-friendly semantic markdown."""

        output = coerce_visual_vlm_output(output)
        structured_lines = split_vlm_lines(output.get("structured_content", ""))
        all_text_lines = split_vlm_lines(output.get("all_text", ""))
        facts = []
        for line in split_vlm_lines(output.get("facts", [])):
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
        nodes: list[str] = []
        seen_nodes: set[str] = set()
        for path_line in path_lines:
            for node in [item.strip() for item in path_line.split(">")]:
                node = re.sub(r"\s+", " ", node).strip()
                if node and node not in seen_nodes:
                    seen_nodes.add(node)
                    nodes.append(node)

        title_text = title.strip() or "流程圖"
        generic_title = title_text.lower().startswith("figure ")
        if generic_title:
            title_text = "流程圖"
        keywords_text = "、".join(keywords[:10])
        start_node = nodes[0] if nodes else (all_text_lines[0] if all_text_lines else "")
        end_nodes = [node for node in nodes if any(term in node for term in ("結案", "通知", "追蹤", "監督", "申訴"))]
        end_text = "、".join(end_nodes[-3:]) if end_nodes else (nodes[-1] if nodes else "")

        roles: list[str] = []
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

        branches = [line for line in path_lines if "(是)" in line or "(否)" in line]

        parts: list[str] = []
        parts.append("## 語意摘要")
        summary = "本文件整理流程圖中的作業流程。" if generic_title else f"本文件是「{title_text}」的流程圖語意化內容。"
        if start_node and end_text:
            summary += f"流程從「{start_node}」開始，涵蓋受理、補正、調查、審議、通知、結案與後續追蹤等節點，最後可能連到「{end_text}」。"
        if keywords_text:
            summary += f"可用於查詢：{keywords_text}。"
        parts.append(summary)

        if roles:
            parts.append("\n## 主要角色與單位")
            parts.extend(f"- {role}" for role in roles[:12])

        if branches:
            parts.append("\n## 判斷與分支")
            for line in branches[:8]:
                parts.append(f"- {line}")

        if deadline_items:
            parts.append("\n## 時限與依據")
            parts.extend(f"- {item}" for item in deadline_items[:12])

        if keywords:
            parts.append("\n## 常見查詢主題")
            parts.extend(f"- {keyword}" for keyword in keywords[:10])

        if facts:
            parts.append("\n## 重要事實")
            parts.extend(f"- {fact}" for fact in facts[:10])

        if path_lines:
            parts.append("\n## 詳細流程路徑")
            parts.extend(f"- {line}" for line in path_lines)
        elif all_text_lines:
            parts.append("\n## 圖中文字")
            parts.extend(f"- {line}" for line in all_text_lines)

        return "\n".join(parts).strip()

    def _should_export_asset_document(self, asset: AssetEntry, source_md: str, assets: list[AssetEntry]) -> bool:
        """Avoid creating duplicate or empty child docs."""

        if asset.type == "form_asset":
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
        if asset.type not in {"figure_asset", "table_asset"}:
            return False
        same_page_assets = [item for item in assets if item.page_idx == asset.page_idx]
        if len(assets) == 1 and len(same_page_assets) == 1:
            text = source_md.strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            has_structured_flow = " > " in text or "來源頁碼" in text
            if asset.type == "figure_asset" and len(lines) <= 80 and has_structured_flow:
                return False
        return True

    def _render_split_asset_document(
        self,
        asset: AssetEntry,
        source_title: str,
        source_filename: str,
    ) -> str:
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
        visual_structured_content = render_vlm_text(visual_output.get("structured_content", ""))
        if title.startswith("Figure ") and (visual_structured_content or asset.semantic_caption):
            title = self._infer_visual_asset_title(visual_structured_content, render_vlm_text(visual_output.get("semantic_caption", "")), title)
            if title.startswith("Figure ") and str(visual_output.get("image_type", "")).lower() == "flowchart":
                title = source_title or "流程圖"
        kind_label = {
            "figure_asset": "圖示/流程圖",
            "table_asset": "表格",
            "form_asset": "表單",
        }.get(asset.type, asset.type)
        parts = [
            f"# {title}",
            "",
            f"來源文件：{source_title}",
            f"來源檔案：{source_filename}",
            f"來源頁碼：第 {asset.page_idx + 1} 頁",
            f"文件類型：{kind_label}",
            f"關聯來源：本文件來自「{source_title}」。",
            "",
        ]
        body_parts = []
        if asset.type == "figure_asset" and (visual_structured_content or render_vlm_text(visual_output.get("semantic_caption", ""))):
            body_parts.append(self._render_visual_semantic_content(title, visual_output))
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
            if zh_count >= 4 and zh_count >= ascii_count and visual_caption not in body:
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
    ) -> str:
        header = [
            f"# {source_title}",
            "",
            f"來源檔案：{source_filename}",
            "文件類型：主文/規章",
        ]
        if form_entries:
            header.extend(
                [
                    "",
                    "## 關聯表單與附件",
                ]
            )
            for entry in form_entries:
                title = str(entry.get("title") or entry["document_id"]).strip()
                page_label = str(entry.get("page_label") or "").strip()
                suffix = f"，來源頁面：{page_label}" if page_label else ""
                header.append(f"- {title}{suffix}。")
        body = source_md.strip()
        if body.startswith(f"# {source_title}"):
            body = "\n".join(body.splitlines()[1:]).strip()
        return "\n".join(header).strip() + "\n\n" + body + "\n"

    def _render_split_form_document(
        self,
        raw_markdown: str,
        item: dict[str, Any],
        source_title: str,
        source_filename: str,
    ) -> str:
        title = self._clean_export_title(str(item.get("title") or item.get("form_id") or "表單").strip())
        if self._is_unreliable_export_title(title):
            title = source_title if not self._is_unreliable_export_title(source_title) else "表單"
        page_label = str(item.get("page_label") or "").strip()
        header = [
            f"# {title}",
            "",
            f"來源文件：{source_title}",
            f"來源檔案：{source_filename}",
            f"來源頁面：{page_label}" if page_label else "",
            "文件類型：表單",
            f"關聯來源：本表單來自「{source_title}」。",
            "",
        ]

        body_lines = raw_markdown.strip().splitlines()
        cleaned: list[str] = []
        skipped_outer_title = False
        for line in body_lines:
            stripped = line.strip()
            if not skipped_outer_title and stripped.startswith("# "):
                skipped_outer_title = True
                continue
            if stripped == f"## {title}":
                continue
            cleaned.append(line)

        body = "\n".join(cleaned).strip()
        return "\n".join(part for part in header if part).strip() + "\n\n" + body + "\n"

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
            structured_output = build_structured_rag(document_ir)
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
                        semantic_caption = render_vlm_text(enrichment_output.get("semantic_caption", ""))
                        structured_content = render_vlm_text(enrichment_output.get("structured_content", ""))
                        facts = enrichment_output.get("facts", [])
                        keywords = enrichment_output.get("keywords", [])
                        needs_review = enrichment.get("quality", {}).get("needs_review", False)
                        if title.startswith("Figure ") and structured_content:
                            title = self._infer_visual_asset_title(structured_content, semantic_caption, title)

                        # Build enhanced retrieval text without leaking raw English captions.
                        semantic_retrieval = self._render_visual_semantic_content(title, enrichment_output)
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
                    structured_table_text = structured_table_text_by_block.get(block.block_id, "") or semantic_table_to_text(table_body, title)
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
                            structured_content, page_watermarks = self._build_form_rag_content(form_output)
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
                    asset_title = asset_map.get(block.block_id).title if block.block_id in asset_map else "流程圖"
                    content = self._render_visual_semantic_content(asset_title, block_enrichment.get("output", {})) + "\n\n"
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
            block_lines = self._render_block_rag(block, asset_map)

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
            footer_content = f"**關鍵字**: {', '.join(sorted(all_triggers))}\n"
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

    def _build_form_rag_content(self, form_output: dict[str, Any]) -> tuple[str, list[str]]:
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
                parts.append("\n## 表單欄位")
                for field in field_schema:
                    if isinstance(field, dict):
                        name = field.get("name", "")
                        field_type = field.get("type", "text")
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
            )
            table_text = semantic_table_to_text(table_body, title)
            if table_text:
                lines.append(table_text)
            else:
                preview = self._extract_table_preview(table_body, max_chars=500)
                if caption:
                    lines.append(f"## {clean_latex_symbols(caption)}")
                if preview:
                    lines.append("表格內容摘要：" + clean_latex_symbols(preview))

            # Add asset reference as explicit token
            if asset:
                lines.append("")
                lines.append(f"[[asset:{asset.asset_id}]]")

        elif block.type == BlockType.IMAGE:
            caption = block.payload.get("caption", "")

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
