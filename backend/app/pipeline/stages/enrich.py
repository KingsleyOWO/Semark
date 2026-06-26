"""
Enrich stage - VLM enrichment for forms, figures, tables, and visual pages.

Gating logic determines which blocks/pages need VLM enrichment.
"""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from app.adapters.vlm import EnrichmentOutput, VLMAdapter
from app.config import PipelineConfig, settings
from app.core.cache import CacheManager
from app.db.database import Database
from app.models.document_ir import Block, BlockType, DocumentIR
from app.pipeline.org_chart_parser import OrgChartParser
from app.pipeline.semantic.language import resolve_semantic_output_language
from app.pipeline.structured_rag import looks_like_reference_table

# Patterns indicating visual/diagram documents
DIAGRAM_PATTERNS = [
    r"架構圖|組織圖|流程圖|示意圖|概念圖",
    r"diagram|chart|flowchart|structure|org.chart",
]

# MinerU YOLO category IDs
class MinerUCategoryId:
    Title = 0
    Text = 1
    Abandon = 2
    ImageBody = 3  # Figures, diagrams, charts
    ImageCaption = 4
    TableBody = 5
    TableCaption = 6
    TableFootnote = 7

# Minimum score threshold for YOLO detection
YOLO_MIN_SCORE = 0.5

# Standard image sizes for VLM (max dimension)
VLM_IMAGE_MAX_SIZE = 1024

# Form page render DPI (higher for forms to preserve detail)
FORM_PAGE_DPI = 200


@dataclass
class TableStats:
    """
    Table statistics for gating decisions and debugging.

    Used to track why a table was skipped/processed.
    """

    rows: int = 0
    cols: int = 0
    cells: int = 0
    non_empty_cells: int = 0
    text_chars: int = 0
    non_empty_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cells": self.cells,
            "non_empty_cells": self.non_empty_cells,
            "text_chars": self.text_chars,
            "non_empty_ratio": round(self.non_empty_ratio, 3),
        }


@dataclass
class TableGatingResult:
    """
    Result from table gating evaluation.

    Tracks whether table should be processed and why.
    """

    should_process: bool = False
    skip_reason: str | None = None  # "budget" | "too_small" | "too_large" | "layout_table" | "no_table_body"
    stats: TableStats = field(default_factory=TableStats)
    truncated: bool = False
    truncate_policy: str | None = None  # "head10+tail10" | "non_empty_sample"
    truncated_body: str | None = None  # Truncated table body for VLM input

    def to_dict(self) -> dict[str, Any]:
        result = {
            "should_process": self.should_process,
            "stats": self.stats.to_dict(),
        }
        if self.skip_reason:
            result["skip_reason"] = self.skip_reason
        if self.truncated:
            result["truncated"] = True
            result["truncate_policy"] = self.truncate_policy
        return result


@dataclass
class EnrichmentEntry:
    """
    Entry in enrichments.jsonl.

    Stores VLM enrichment results with full traceability.
    """

    block_id: str
    kind: str  # form_asset, figure_caption, table_summary
    prompt_version: str
    model: str
    decode: dict[str, Any]
    input: dict[str, Any]  # Contains asset_path, page_idx, context_preview
    output: dict[str, Any]  # VLM-generated content (validated by Pydantic)
    quality: dict[str, Any] = field(default_factory=dict)  # needs_review, tokens_used, duration, table_stats
    # Evidence for UI highlighting and traceability:
    # - page_idx: 0-based page index
    # - bbox: MinerU 0-1000 normalized coordinates [x0, y0, x1, y1], None for full page
    # - asset_path: path to the image crop used for VLM input
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "decode": self.decode,
            "input": self.input,
            "output": self.output,
            "quality": self.quality,
            "evidence": self.evidence,
        }


@dataclass
class EnrichStageResult:
    """Result from enrich stage."""

    success: bool
    enrichments: list[EnrichmentEntry] = field(default_factory=list)
    enrichments_path: Path | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrgChartDebugBundle:
    """
    D4: Debug bundle for org chart processing.

    收集各階段資料用於排錯：
    - org_input.json: 策略分支決策
    - org_nodes_candidates.json: MinerU blocks → nodes
    - org_edge_candidates.json: heuristics 候選
    - org_vlm2_edges.raw.json: VLM#2 原始輸出
    - org_graph.canonical.json: 最終 graph + warnings
    - org_render.md: 最終輸出
    """

    page_idx: int
    doc_id: str
    run_id: str

    # 策略分支決策
    decision_trace: dict[str, Any] = field(default_factory=dict)

    # 各階段資料
    mineru_blocks: list[dict[str, Any]] = field(default_factory=list)
    nodes_candidates: list[dict[str, Any]] = field(default_factory=list)
    edge_candidates: list[dict[str, Any]] = field(default_factory=list)

    # VLM 輸出
    vlm1_raw: dict[str, Any] = field(default_factory=dict)
    vlm1_validated: dict[str, Any] = field(default_factory=dict)
    vlm2_raw: dict[str, Any] = field(default_factory=dict)
    vlm2_validated: list[dict[str, Any]] = field(default_factory=list)

    # 最終結果
    canonical_graph: dict[str, Any] = field(default_factory=dict)
    render_md: str = ""
    warnings: list[str] = field(default_factory=list)

    def save(self, run_path: Path) -> Path:
        """儲存 debug bundle 到 org_debug/ 資料夾。"""
        debug_dir = run_path / "outputs" / "org_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # org_input.json - 策略分支決策
        self._write_json(debug_dir / "org_input.json", {
            "page_idx": self.page_idx,
            "doc_id": self.doc_id,
            "run_id": self.run_id,
            "decision_trace": self.decision_trace,
        })

        # org_nodes_candidates.json - MinerU → nodes
        self._write_json(debug_dir / "org_nodes_candidates.json", {
            "mineru_blocks_count": len(self.mineru_blocks),
            "nodes_count": len(self.nodes_candidates),
            "mineru_blocks": self.mineru_blocks,
            "nodes": self.nodes_candidates,
        })

        # org_edge_candidates.json - heuristics 候選
        self._write_json(debug_dir / "org_edge_candidates.json", {
            "candidates_count": len(self.edge_candidates),
            "candidates": self.edge_candidates,
        })

        # org_vlm1_units.raw.json
        if self.vlm1_raw:
            self._write_json(debug_dir / "org_vlm1_units.raw.json", self.vlm1_raw)

        # org_vlm1_units.validated.json
        if self.vlm1_validated:
            self._write_json(debug_dir / "org_vlm1_units.validated.json", self.vlm1_validated)

        # org_vlm2_edges.raw.json
        if self.vlm2_raw:
            self._write_json(debug_dir / "org_vlm2_edges.raw.json", self.vlm2_raw)

        # org_vlm2_edges.validated.json
        if self.vlm2_validated:
            self._write_json(debug_dir / "org_vlm2_edges.validated.json", {
                "edges_count": len(self.vlm2_validated),
                "edges": self.vlm2_validated,
            })

        # org_graph.canonical.json - 最終 graph + warnings
        self._write_json(debug_dir / "org_graph.canonical.json", {
            "graph": self.canonical_graph,
            "warnings": self.warnings,
            "decision_trace": self.decision_trace,
        })

        # org_render.md - 最終輸出
        (debug_dir / "org_render.md").write_text(self.render_md, encoding="utf-8")

        return debug_dir

    def _write_json(self, path: Path, data: Any) -> None:
        """寫入 JSON 檔案（確保繁體中文正確顯示）。"""
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


class EnrichStage:
    """
    Enrich stage - VLM enrichment for visual blocks.

    Gating heuristics:
    - Forms: Detected by filename/page patterns, low text ratio
    - Figures: Image blocks without caption or with low-quality caption
    - Tables: Large tables or tables with many empty cells

    Input: DocumentIR + parse cache (for images)
    Output: enrichments.jsonl
    """

    # Form detection patterns
    FORM_PATTERNS = [
        r"申請|表單|報支|請假|加班|進修|附件|出差單|差旅|application|form",
        r"authorization|authorisation|consent|request|claim|transcript|tax|irs|ssa",
        r"簽核|審核|核准|approval",
        r"填寫|fill|complete",
    ]

    def __init__(
        self,
        db: Database,
        config: PipelineConfig | None = None,
    ):
        self.db = db
        self.config = config or PipelineConfig()
        self.enrich_config = self.config.enrich
        self.vlm_config = self.config.vlm
        self.cache_manager = CacheManager(db)
        self.vlm_adapter = VLMAdapter(self.vlm_config)
        self.org_chart_parser = OrgChartParser()
        self._semantic_output_language = "zh-TW"

    async def run(
        self,
        doc_id: str,
        run_id: str,
        document_ir: DocumentIR,
        run_path: Path,
        parse_cache_path: Path | None = None,
        use_cache: bool = True,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> EnrichStageResult:
        """
        Run enrich stage.

        Args:
            doc_id: Document ID
            run_id: Run ID
            document_ir: Document IR with blocks
            run_path: Path to run output directory
            parse_cache_path: Path to parse cache (for image files)
            use_cache: Whether to use cached enrichments

        Returns:
            EnrichStageResult with enrichments
        """
        try:
            semantic_output_language = resolve_semantic_output_language(
                self.config.package.semantic_output_language.value,
                document_ir,
            )
            self._semantic_output_language = semantic_output_language
            # Check if VLM is available
            if self.enrich_config.enable_vlm:
                vlm_ok, vlm_msg = await self.vlm_adapter.check_available()
                if not vlm_ok:
                    # VLM not available, skip enrichment
                    return EnrichStageResult(
                        success=True,
                        stats={
                            "skipped": True,
                            "reason": f"VLM not available: {vlm_msg}",
                        },
                    )

            # Compute VLM config hash for cache
            vlm_config_hash = VLMAdapter.compute_config_hash(self.vlm_config)

            # Apply gating logic to find blocks that need enrichment
            blocks_to_enrich, gating_stats = self._apply_gating(document_ir)

            enrichments: list[EnrichmentEntry] = []
            form_pages: list[int] = []
            if self.enrich_config.vlm_enrich_forms:
                form_pages = self._detect_form_pages(document_ir)

            figure_detections: list[tuple[int, dict[str, Any]]] = []
            if self.enrich_config.enable_vlm and self.enrich_config.vlm_enrich_figures:
                yolo_results = self._load_yolo_detections(parse_cache_path)
                figure_detections = self._get_figure_detections(yolo_results)

            vlm_total_items = len(blocks_to_enrich) + len(form_pages) + len(figure_detections)

            stats = {
                "total_blocks": len(document_ir.blocks),
                "gated_blocks": len(blocks_to_enrich),
                "cache_hits": 0,
                "enriched": 0,
                "errors": 0,
                # VLM detailed stats
                "vlm_calls_by_kind": {},  # {"form_asset": 3, "figure_caption": 5, ...}
                "vlm_total_duration_seconds": 0.0,
                "vlm_total_tokens": 0,
                # Table gating stats
                "table_gating": gating_stats,
                "form_pages_detected": len(form_pages),
                "yolo_figures_detected": len(figure_detections),
                "progress": {
                    "phase": "enrich",
                    "total": vlm_total_items,
                    "completed": 0,
                    "current": None,
                    "percent": 100 if vlm_total_items == 0 else 0,
                    "message": "準備 VLM 語意分析" if vlm_total_items else "沒有需要 VLM 分析的項目",
                    "updated_at": datetime.now().isoformat(),
                },
            }

            async def emit_progress(
                message: str,
                current: dict[str, Any] | None = None,
                completed_delta: int = 0,
            ) -> None:
                progress = stats["progress"]
                progress["completed"] = min(
                    int(progress.get("total", 0)),
                    int(progress.get("completed", 0)) + completed_delta,
                )
                total = int(progress.get("total", 0))
                completed = int(progress.get("completed", 0))
                progress["current"] = current
                progress["message"] = message
                progress["percent"] = 100 if total == 0 else round((completed / total) * 100, 1)
                progress["updated_at"] = datetime.now().isoformat()
                if progress_callback:
                    await progress_callback(stats)

            def progress_item(kind: str, page_idx: int | None, block_id: str | None = None) -> dict[str, Any]:
                return {
                    "kind": kind,
                    "page_idx": page_idx,
                    "block_id": block_id,
                }

            await emit_progress("準備 VLM 語意分析")

            # Process block-level enrichments
            for block, kind, table_gating in blocks_to_enrich:
                prompt_version = self.vlm_adapter.get_prompt_version(kind)
                current_item = progress_item(kind, block.page_idx, block.block_id)
                await emit_progress(
                    f"VLM 分析第 {block.page_idx + 1} 頁 {kind}",
                    current_item,
                )

                # Check cache
                if use_cache:
                    cached = await self.cache_manager.get_enrich_cache(
                        doc_id=doc_id,
                        block_id=block.block_id,
                        vlm_config_hash=vlm_config_hash,
                        prompt_version=prompt_version,
                    )
                    if cached:
                        stats["cache_hits"] += 1
                        # Read needs_review from cached output (don't override with False)
                        cached_needs_review = cached.get("needs_review", False) if isinstance(cached, dict) else False
                        enrichments.append(
                            EnrichmentEntry(
                                block_id=block.block_id,
                                kind=kind,
                                prompt_version=prompt_version,
                                model=self.vlm_config.model,
                                decode=self.vlm_config.decode_params.model_dump(),
                                input={"cached": True, "page_idx": block.page_idx},
                                output=cached,
                                quality={"needs_review": cached_needs_review},
                                evidence={
                                    "page_idx": block.page_idx,
                                    "bbox": block.bbox_norm,
                                    "asset_path": block.payload.get("img_path"),
                                },
                            )
                        )
                        await emit_progress(
                            f"第 {block.page_idx + 1} 頁 {kind} 使用快取",
                            current_item,
                            completed_delta=1,
                        )
                        continue

                # Get image path for visual enrichment.
                # Native tables may have no crop image; table summaries and form-like
                # spreadsheet tables can still use structured table text as ground truth.
                image_path = self._get_block_image(block, parse_cache_path)
                text_only_form_table = (
                    kind in ("form_asset", "form_guide") and block.type == BlockType.TABLE
                )
                if not image_path and kind != "table_summary" and not text_only_form_table:
                    await emit_progress(
                        f"第 {block.page_idx + 1} 頁 {kind} 缺少可分析影像，已略過",
                        current_item,
                        completed_delta=1,
                    )
                    continue

                # Get context from surrounding blocks
                context = self._get_context(document_ir, block)
                if text_only_form_table:
                    table_context = self._table_context_for_form(
                        block.payload.get("table_body", "")
                    )
                    if table_context:
                        context = (
                            f"{context}\n\n" if context else ""
                        ) + (
                            "[Structured spreadsheet form table; use as ground truth]\n"
                            + table_context
                        )

                # Extract table data for table_summary (reduces hallucination)
                table_body = None
                table_headers = None
                table_body_for_vlm = None  # May be truncated
                if kind == "table_summary" and block.type == BlockType.TABLE:
                    table_body = block.payload.get("table_body", "")
                    # Use truncated body if available (from gating)
                    if table_gating and table_gating.truncated and table_gating.truncated_body:
                        table_body_for_vlm = table_gating.truncated_body
                    else:
                        table_body_for_vlm = table_body
                    # Try to extract headers from original table_body
                    table_headers = self._extract_table_headers(table_body)

                # Get page thumbnail if enabled (for visual context)
                page_thumbnail_path = None
                if self.vlm_config.include_page_thumbnail:
                    page_thumbnail_path = self._get_page_thumbnail(
                        doc_id=doc_id,
                        run_id=run_id,
                        page_idx=block.page_idx,
                        run_path=run_path,
                    )

                # Call VLM with doc_id and run_id for URL building
                result = await self._enrich_block(
                    block, kind, image_path, context,
                    doc_id=doc_id, run_id=run_id,
                    table_body=table_body_for_vlm,  # Use truncated body if available
                    table_headers=table_headers,
                    page_thumbnail_path=page_thumbnail_path,
                )

                if result.success:
                    stats["enriched"] += 1
                    # Track VLM call stats
                    stats["vlm_calls_by_kind"][kind] = stats["vlm_calls_by_kind"].get(kind, 0) + 1
                    stats["vlm_total_duration_seconds"] += result.duration_seconds or 0
                    stats["vlm_total_tokens"] += result.tokens_used or 0

                    # Cache the result
                    await self.cache_manager.set_enrich_cache(
                        doc_id=doc_id,
                        block_id=block.block_id,
                        vlm_config_hash=vlm_config_hash,
                        prompt_version=prompt_version,
                        output=result.output,
                    )

                    # Build evidence from result or block
                    evidence_dict = {
                        "page_idx": block.page_idx,
                        "bbox": block.bbox_norm,
                        "asset_path": str(image_path) if image_path else None,
                    }
                    if result.evidence:
                        evidence_dict.update({
                            "page_idx": result.evidence.page_idx or block.page_idx,
                            "bbox": result.evidence.bbox or block.bbox_norm,
                            "asset_path": result.evidence.asset_path
                            or (str(image_path) if image_path else None),
                        })

                    # Build quality dict with table stats if applicable
                    quality_dict: dict[str, Any] = {
                        "needs_review": result.needs_review,
                        "tokens_used": result.tokens_used,
                        "duration_seconds": result.duration_seconds,
                    }
                    # Add table gating info for table_summary
                    if kind == "table_summary" and table_gating:
                        quality_dict["table_stats"] = table_gating.stats.to_dict()
                        if table_gating.truncated:
                            quality_dict["truncated"] = True
                            quality_dict["truncate_policy"] = table_gating.truncate_policy

                    enrichments.append(
                        EnrichmentEntry(
                            block_id=block.block_id,
                            kind=kind,
                            prompt_version=prompt_version,
                            model=self.vlm_config.model,
                            decode=self.vlm_config.decode_params.model_dump(),
                            input={
                                "asset_path": str(image_path) if image_path else None,
                                "context_preview": context[:200] if context else None,
                                "page_idx": block.page_idx,
                                "route": "vlm_text_from_mineru_table" if not image_path and kind == "table_summary" else "vlm_image",
                            },
                            output=result.output,
                            quality=quality_dict,
                            evidence=evidence_dict,
                        )
                    )
                else:
                    stats["errors"] += 1

                await emit_progress(
                    f"完成第 {block.page_idx + 1} 頁 {kind}",
                    current_item,
                    completed_delta=1,
                )

            # Export form pages as full-page assets and enrich with VLM
            if self.enrich_config.vlm_enrich_forms:
                forms_dir = run_path / "assets" / "forms"

                for page_idx in form_pages:
                    current_item = progress_item("form_asset", page_idx, f"form_page_{page_idx:04d}")
                    await emit_progress(
                        f"VLM 規劃第 {page_idx + 1} 頁表單",
                        current_item,
                    )
                    form_path = forms_dir / f"form_p{page_idx:04d}.png"
                    exported = self._export_form_page(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        output_path=form_path,
                    )
                    if exported:
                        stats["form_pages_exported"] = stats.get("form_pages_exported", 0) + 1

                        # Enrich form page with VLM if enabled
                        if self.enrich_config.enable_vlm:
                            form_block_id = f"form_page_{page_idx:04d}"
                            prompt_version = f"{self.vlm_adapter.get_prompt_version('form_asset')}:{semantic_output_language}"

                            # Check cache
                            if use_cache:
                                cached = await self.cache_manager.get_enrich_cache(
                                    doc_id=doc_id,
                                    block_id=form_block_id,
                                    vlm_config_hash=vlm_config_hash,
                                    prompt_version=prompt_version,
                                )
                                if cached:
                                    stats["cache_hits"] += 1
                                    output = cached

                                    # Apply org chart processing if needed (even for cached results)
                                    document_type = output.get("document_type", "") if isinstance(output, dict) else ""
                                    if document_type == "org_chart" and "org_chart_graph" not in output:
                                        page_blocks = [
                                            {"text": b.payload.get("text", ""), "bbox": b.bbox_norm}
                                            for b in document_ir.blocks
                                            if b.page_idx == page_idx and b.type == BlockType.TEXT
                                        ]

                                        # Fallback to VLM's all_text if MinerU blocks are insufficient
                                        MIN_BLOCKS_THRESHOLD = 10
                                        if len(page_blocks) < MIN_BLOCKS_THRESHOLD and isinstance(output, dict):
                                            vlm_all_text = output.get("all_text", [])
                                            if vlm_all_text and len(vlm_all_text) > len(page_blocks):
                                                # Use VLM's all_text as pseudo-blocks (no bbox)
                                                page_blocks = [
                                                    {"text": t, "bbox": None}
                                                    for t in vlm_all_text
                                                    if isinstance(t, str) and len(t.strip()) >= 2
                                                ]

                                        # Get original VLM structured_content for PATH parsing
                                        vlm_structured_content = output.get("structured_content", "") if isinstance(output, dict) else ""

                                        org_graph = self.org_chart_parser.parse_from_blocks(
                                            blocks=page_blocks,
                                            page_idx=page_idx,
                                            title=output.get("title", ""),
                                            date=output.get("date", ""),
                                            vlm_structured_content=vlm_structured_content,
                                        )
                                        output["org_chart_graph"] = org_graph.to_dict()
                                        graph_content = org_graph.to_structured_content()
                                        if graph_content:
                                            output["structured_content"] = graph_content
                                        if org_graph.needs_review:
                                            output["needs_review"] = True
                                            output["review_reasons"] = org_graph.review_reasons

                                    cached_needs_review = output.get("needs_review", False) if isinstance(output, dict) else False
                                    enrichments.append(
                                        EnrichmentEntry(
                                            block_id=form_block_id,
                                            kind="form_asset",
                                            prompt_version=prompt_version,
                                            model=self.vlm_config.model,
                                            decode=self.vlm_config.decode_params.model_dump(),
                                            input={
                                                "cached": True,
                                                "page_idx": page_idx,
                                                "asset_path": str(form_path),
                                            },
                                            output=output,
                                            quality={"needs_review": cached_needs_review},
                                            evidence={
                                                "page_idx": page_idx,
                                                "bbox": None,  # Full page
                                                "asset_path": str(form_path),
                                            },
                                        )
                                    )
                                    await emit_progress(
                                        f"第 {page_idx + 1} 頁表單使用快取",
                                        current_item,
                                        completed_delta=1,
                                    )
                                    continue

                            # Get rich context from page blocks (for form understanding)
                            page_blocks = [b for b in document_ir.blocks if b.page_idx == page_idx]
                            context = self._build_form_context(page_blocks)

                            # Get page thumbnail if enabled (form is already full page,
                            # but thumbnail provides context for adjacent pages)
                            form_page_thumbnail = None
                            if self.vlm_config.include_page_thumbnail:
                                form_page_thumbnail = self._get_page_thumbnail(
                                    doc_id=doc_id,
                                    run_id=run_id,
                                    page_idx=page_idx,
                                    run_path=run_path,
                                )

                            # Call VLM for form enrichment
                            result = await self._enrich_block(
                                block=None,
                                kind="form_asset",
                                image_path=form_path,
                                context=context,
                                doc_id=doc_id,
                                run_id=run_id,
                                page_idx=page_idx,
                                page_thumbnail_path=form_page_thumbnail,
                            )

                            if result.success:
                                stats["enriched"] += 1
                                # Track VLM call stats for form_asset
                                stats["vlm_calls_by_kind"]["form_asset"] = stats["vlm_calls_by_kind"].get("form_asset", 0) + 1
                                stats["vlm_total_duration_seconds"] += result.duration_seconds or 0
                                stats["vlm_total_tokens"] += result.tokens_used or 0
                                output = result.output

                                # Check if this is an org chart - use Graph-based parsing
                                document_type = output.get("document_type", "") if isinstance(output, dict) else ""
                                if document_type == "org_chart":
                                    # D4: 建立 debug bundle
                                    debug_bundle = OrgChartDebugBundle(
                                        page_idx=page_idx,
                                        doc_id=doc_id,
                                        run_id=run_id,
                                    )

                                    # 收集 VLM#1 輸出
                                    debug_bundle.vlm1_raw = result.raw_response if hasattr(result, 'raw_response') else {}
                                    debug_bundle.vlm1_validated = output.copy() if isinstance(output, dict) else {}

                                    # Get blocks for this page
                                    page_blocks = [
                                        {"text": b.payload.get("text", ""), "bbox": b.bbox_norm}
                                        for b in document_ir.blocks
                                        if b.page_idx == page_idx and b.type == BlockType.TEXT
                                    ]
                                    original_mineru_blocks = page_blocks.copy()

                                    # Fallback to VLM's all_text if MinerU blocks are insufficient
                                    MIN_BLOCKS_THRESHOLD = 10
                                    used_vlm_fallback = False
                                    if len(page_blocks) < MIN_BLOCKS_THRESHOLD and isinstance(output, dict):
                                        vlm_all_text = output.get("all_text", [])
                                        if vlm_all_text and len(vlm_all_text) > len(page_blocks):
                                            # Use VLM's all_text as pseudo-blocks (no bbox)
                                            page_blocks = [
                                                {"text": t, "bbox": None}
                                                for t in vlm_all_text
                                                if isinstance(t, str) and len(t.strip()) >= 2
                                            ]
                                            used_vlm_fallback = True

                                    # D4: 記錄 MinerU blocks
                                    debug_bundle.mineru_blocks = original_mineru_blocks

                                    # Get original VLM structured_content for PATH parsing
                                    vlm_structured_content = output.get("structured_content", "") if isinstance(output, dict) else ""

                                    # Parse org chart using B+ approach
                                    org_graph = self.org_chart_parser.parse_from_blocks(
                                        blocks=page_blocks,
                                        page_idx=page_idx,
                                        title=output.get("title", ""),
                                        date=output.get("date", ""),
                                        vlm_structured_content=vlm_structured_content,
                                    )

                                    # D4: 記錄 nodes candidates
                                    debug_bundle.nodes_candidates = [n.to_dict() for n in org_graph.nodes]

                                    # D3: VLM#2 edge selection (only if we have bbox data)
                                    has_bbox = any(b.get("bbox") for b in page_blocks)

                                    # D4: 記錄決策分支
                                    debug_bundle.decision_trace = {
                                        "mineru_blocks_count": len(original_mineru_blocks),
                                        "used_vlm_fallback": used_vlm_fallback,
                                        "final_blocks_count": len(page_blocks),
                                        "bbox_available": has_bbox,
                                        "nodes_count": len(org_graph.nodes),
                                        "vlm2_eligible": has_bbox and len(org_graph.nodes) > 0,
                                    }

                                    edge_candidates = []
                                    if has_bbox and org_graph.nodes:
                                        # Generate edge candidates from heuristics
                                        edge_candidates = self.org_chart_parser.generate_edge_candidates(
                                            org_graph.nodes
                                        )

                                        # D4: 記錄 edge candidates
                                        debug_bundle.edge_candidates = [ec.to_dict() for ec in edge_candidates]
                                        debug_bundle.decision_trace["edge_candidates_count"] = len(edge_candidates)

                                        if edge_candidates:
                                            # Call VLM#2 for edge selection
                                            try:
                                                vlm2_result = await self._select_org_chart_edges_with_debug(
                                                    doc_id=doc_id,
                                                    run_id=run_id,
                                                    image_path=form_path,
                                                    edge_candidates=edge_candidates,
                                                    nodes=org_graph.nodes,
                                                )
                                                vlm2_edges = vlm2_result.get("edges", [])

                                                # D4: 記錄 VLM#2 輸出
                                                debug_bundle.vlm2_raw = vlm2_result.get("raw", {})
                                                debug_bundle.vlm2_validated = [e.to_dict() for e in vlm2_edges]
                                                debug_bundle.decision_trace["vlm2_called"] = True
                                                debug_bundle.decision_trace["vlm2_edges_count"] = len(vlm2_edges)

                                                if vlm2_edges:
                                                    # Replace heuristic edges with VLM#2 selected edges
                                                    org_graph.edges = vlm2_edges
                                            except Exception as e:
                                                # VLM#2 failed, keep heuristic edges (or empty)
                                                org_graph.review_reasons.append(f"VLM#2 edge selection failed: {e}")
                                                debug_bundle.decision_trace["vlm2_called"] = True
                                                debug_bundle.decision_trace["vlm2_error"] = str(e)
                                        else:
                                            debug_bundle.decision_trace["skipped_vlm2"] = "no_edge_candidates"
                                    else:
                                        debug_bundle.decision_trace["skipped_vlm2"] = "no_bbox" if not has_bbox else "no_nodes"

                                    # Add graph to output
                                    output["org_chart_graph"] = org_graph.to_dict()

                                    # Use graph's structured content if available
                                    graph_content = org_graph.to_structured_content()
                                    if graph_content and len(graph_content) > len(output.get("structured_content", "")):
                                        output["structured_content"] = graph_content

                                    # Update needs_review based on graph validation
                                    if org_graph.needs_review:
                                        output["needs_review"] = True
                                        output["review_reasons"] = org_graph.review_reasons

                                    # D4: 記錄最終結果並儲存
                                    debug_bundle.canonical_graph = org_graph.to_dict()
                                    debug_bundle.render_md = graph_content
                                    debug_bundle.warnings = org_graph.review_reasons
                                    debug_bundle.save(run_path)

                                # Cache the result
                                await self.cache_manager.set_enrich_cache(
                                    doc_id=doc_id,
                                    block_id=form_block_id,
                                    vlm_config_hash=vlm_config_hash,
                                    prompt_version=prompt_version,
                                    output=output,
                                )

                                enrichments.append(
                                    EnrichmentEntry(
                                        block_id=form_block_id,
                                        kind="form_asset",
                                        prompt_version=prompt_version,
                                        model=self.vlm_config.model,
                                        decode=self.vlm_config.decode_params.model_dump(),
                                        input={
                                            "page_idx": page_idx,
                                            "asset_path": str(form_path),
                                        },
                                        output=output,
                                        quality={
                                            "needs_review": result.needs_review or output.get("needs_review", False),
                                            "tokens_used": result.tokens_used,
                                            "duration_seconds": result.duration_seconds,
                                        },
                                        evidence={
                                            "page_idx": page_idx,
                                            "bbox": None,  # Full page
                                            "asset_path": str(form_path),
                                        },
                                    )
                                )
                            else:
                                stats["errors"] += 1
                                import logging
                                logging.error(f"VLM form enrichment failed: {result.error}")

                            await emit_progress(
                                f"完成第 {page_idx + 1} 頁表單規劃",
                                current_item,
                                completed_delta=1,
                            )
                    else:
                        await emit_progress(
                            f"第 {page_idx + 1} 頁表單影像輸出失敗，已略過",
                            current_item,
                            completed_delta=1,
                        )

            # Process YOLO-detected figures/diagrams
            if self.enrich_config.enable_vlm and self.enrich_config.vlm_enrich_figures:
                figures_dir = run_path / "assets" / "figures"
                for det_idx, (page_idx, detection) in enumerate(figure_detections):
                    figure_id = f"yolo_fig_{page_idx:04d}_{det_idx:03d}"
                    prompt_version = self.vlm_adapter.get_prompt_version("figure_description")
                    current_item = progress_item("figure_caption", page_idx, figure_id)
                    await emit_progress(
                        f"VLM 分析第 {page_idx + 1} 頁圖表",
                        current_item,
                    )

                    # Check cache
                    if use_cache:
                        cached = await self.cache_manager.get_enrich_cache(
                            doc_id=doc_id,
                            block_id=figure_id,
                            vlm_config_hash=vlm_config_hash,
                            prompt_version=prompt_version,
                        )
                        if cached:
                            stats["cache_hits"] += 1
                            # Read needs_review from cached output
                            cached_needs_review = cached.get("needs_review", False) if isinstance(cached, dict) else False
                            poly = detection.get("poly", [])
                            enrichments.append(
                                EnrichmentEntry(
                                    block_id=figure_id,
                                    kind="figure_caption",  # Use consistent kind name
                                    prompt_version=prompt_version,
                                    model=self.vlm_config.model,
                                    decode=self.vlm_config.decode_params.model_dump(),
                                    input={
                                        "cached": True,
                                        "page_idx": page_idx,
                                        "asset_path": str(figures_dir / f"{figure_id}.png"),
                                    },
                                    output=cached,
                                    quality={"needs_review": cached_needs_review},
                                    evidence={
                                        "page_idx": page_idx,
                                        "bbox": poly if len(poly) == 4 else None,
                                        "asset_path": str(figures_dir / f"{figure_id}.png"),
                                    },
                                )
                            )
                            await emit_progress(
                                f"第 {page_idx + 1} 頁圖表使用快取",
                                current_item,
                                completed_delta=1,
                            )
                            continue

                    # Crop figure region from PDF
                    poly = detection.get("poly", [])
                    figure_path = figures_dir / f"{figure_id}.png"
                    cropped_image = self._crop_region_from_pdf(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        poly=poly,
                        output_path=figure_path,
                    )

                    if not cropped_image:
                        await emit_progress(
                            f"第 {page_idx + 1} 頁圖表裁切失敗，已略過",
                            current_item,
                            completed_delta=1,
                        )
                        continue

                    # Get context (title from page)
                    page_blocks = [b for b in document_ir.blocks if b.page_idx == page_idx]
                    context = ""
                    for b in page_blocks:
                        if b.type == BlockType.TEXT:
                            text = b.payload.get("text", "")
                            if text:
                                context = text[:200]
                                break

                    # Get page thumbnail if enabled (for visual context)
                    figure_page_thumbnail = None
                    if self.vlm_config.include_page_thumbnail:
                        figure_page_thumbnail = self._get_page_thumbnail(
                            doc_id=doc_id,
                            run_id=run_id,
                            page_idx=page_idx,
                            run_path=run_path,
                        )

                    # Call VLM with doc_id, run_id, and page_idx for evidence
                    result = await self._enrich_block(
                        block=None,
                        kind="figure_caption",  # Use new kind name
                        image_path=cropped_image,
                        context=context,
                        doc_id=doc_id,
                        run_id=run_id,
                        page_idx=page_idx,
                        page_thumbnail_path=figure_page_thumbnail,
                    )

                    if result.success:
                        stats["enriched"] += 1
                        # Track VLM call stats for figure_caption
                        stats["vlm_calls_by_kind"]["figure_caption"] = stats["vlm_calls_by_kind"].get("figure_caption", 0) + 1
                        stats["vlm_total_duration_seconds"] += result.duration_seconds or 0
                        stats["vlm_total_tokens"] += result.tokens_used or 0

                        # Cache the result
                        await self.cache_manager.set_enrich_cache(
                            doc_id=doc_id,
                            block_id=figure_id,
                            vlm_config_hash=vlm_config_hash,
                            prompt_version=prompt_version,
                            output=result.output,
                        )

                        enrichments.append(
                            EnrichmentEntry(
                                block_id=figure_id,
                                kind="figure_caption",  # Use new kind name
                                prompt_version=prompt_version,
                                model=self.vlm_config.model,
                                decode=self.vlm_config.decode_params.model_dump(),
                                input={
                                    "page_idx": page_idx,
                                    "yolo_score": detection.get("score"),
                                    "asset_path": str(cropped_image),
                                    "context_preview": context[:200] if context else None,
                                },
                                output=result.output,
                                quality={
                                    "needs_review": result.needs_review,
                                    "tokens_used": result.tokens_used,
                                    "duration_seconds": result.duration_seconds,
                                },
                                evidence={
                                    "page_idx": page_idx,
                                    "bbox": poly if len(poly) == 4 else None,
                                    "asset_path": str(cropped_image),
                                },
                            )
                        )
                    else:
                        stats["errors"] += 1
                        import logging
                        logging.error(f"VLM figure enrichment failed: {result.error}")

                    await emit_progress(
                        f"完成第 {page_idx + 1} 頁圖表分析",
                        current_item,
                        completed_delta=1,
                    )

            await emit_progress("VLM 語意分析完成", None)

            # Write enrichments.jsonl
            outputs_dir = run_path / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            enrichments_path = outputs_dir / "enrichments.jsonl"
            with open(enrichments_path, "w", encoding="utf-8") as f:
                for entry in enrichments:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

            return EnrichStageResult(
                success=True,
                enrichments=enrichments,
                enrichments_path=enrichments_path,
                stats=stats,
            )

        except Exception as e:
            import logging
            import traceback
            logging.error(f"Enrich stage failed: {type(e).__name__}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return EnrichStageResult(
                success=False,
                error=str(e),
            )

    def _apply_gating(
        self,
        document_ir: DocumentIR,
    ) -> tuple[list[tuple[Block, str, TableGatingResult | None]], dict[str, Any]]:
        """
        Apply gating logic to determine which blocks need enrichment.

        Returns:
            - list of (block, enrichment_kind, table_gating_result) tuples
            - gating_stats dict with skip counts and reasons

        Applies table budget to limit table enrichments.
        """
        blocks_to_enrich: list[tuple[Block, str, TableGatingResult | None]] = []
        table_count = 0
        table_budget = self.enrich_config.table_vlm_budget

        # Track gating statistics
        gating_stats = {
            "tables_evaluated": 0,
            "tables_skipped": 0,
            "tables_truncated": 0,
            "mineru_only_blocks": 0,
            "vlm_candidate_blocks": 0,
            "vlm_candidates_by_kind": {},
            "skip_reasons": {},  # {"too_small": 3, "layout_table": 2, ...}
        }

        # Check if source filename matches form patterns
        is_form_document = self._is_form_document(document_ir.source.path)

        for block in document_ir.blocks:
            kind = self._get_enrichment_kind(block, document_ir, is_form_document)
            if kind:
                gating_stats["vlm_candidate_blocks"] += 1
                gating_stats["vlm_candidates_by_kind"][kind] = (
                    gating_stats["vlm_candidates_by_kind"].get(kind, 0) + 1
                )
                table_gating = None

                # Apply table-specific gating
                if kind == "table_summary":
                    gating_stats["tables_evaluated"] += 1

                    # Evaluate table gating
                    table_body = block.payload.get("table_body", "")
                    table_gating = self._evaluate_table_gating(table_body)

                    if not table_gating.should_process:
                        # Track skip reason
                        gating_stats["tables_skipped"] += 1
                        reason = table_gating.skip_reason or "unknown"
                        gating_stats["skip_reasons"][reason] = gating_stats["skip_reasons"].get(reason, 0) + 1
                        continue

                    # Apply budget
                    if table_budget > 0 and table_count >= table_budget:
                        gating_stats["tables_skipped"] += 1
                        gating_stats["skip_reasons"]["budget"] = gating_stats["skip_reasons"].get("budget", 0) + 1
                        continue

                    table_count += 1

                    # Track truncation
                    if table_gating.truncated:
                        gating_stats["tables_truncated"] += 1

                blocks_to_enrich.append((block, kind, table_gating))
            else:
                gating_stats["mineru_only_blocks"] += 1

        return blocks_to_enrich, gating_stats

    def _is_form_document(self, source_path: str) -> bool:
        """Check if document appears to be a form based on filename."""
        filename = Path(source_path).stem.lower()
        for pattern in self.FORM_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        for pattern in self.enrich_config.form_filename_patterns:
            value = str(pattern or "").strip()
            if value and re.search(re.escape(value), filename, re.IGNORECASE):
                return True
        return False

    def _get_enrichment_kind(
        self,
        block: Block,
        document_ir: DocumentIR,
        is_form_document: bool,
    ) -> str | None:
        """
        Determine what kind of enrichment a block needs.

        Returns None if no enrichment needed.
        """
        # Skip if enrichment is disabled
        if not self.enrich_config.enable_vlm:
            return None

        if block.type == BlockType.IMAGE:
            # Enrich figures that lack good captions
            caption = block.payload.get("caption", "")
            block.payload.get("footnote", "")

            if self.enrich_config.vlm_enrich_figures:
                # Enrich if caption is missing or too short
                if not caption or len(caption) < 20:
                    return "figure_description"

        elif block.type == BlockType.TABLE:
            table_body = block.payload.get("table_body", "")

            if (
                is_form_document
                and self.enrich_config.vlm_enrich_forms
                and self._is_form_like_table(table_body)
            ):
                return "form_asset"

            if self.enrich_config.vlm_enrich_tables:
                # Enrich data tables that pass the gating criteria
                if self._should_enrich_table(table_body):
                    return "table_summary"

        # Check for form pages
        if is_form_document and self.enrich_config.vlm_enrich_forms:
            # For form documents, enrich image blocks that might be form sections
            if block.type == BlockType.IMAGE:
                # Large images on form documents are likely form sections
                bbox = block.bbox_norm
                if bbox and len(bbox) == 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    # Large blocks (covering significant page area)
                    if width > 600 or height > 400:
                        return "form_guide"

        return None

    def _analyze_table(self, table_body: str) -> TableStats:
        """
        Analyze table structure and content.

        Returns TableStats with detailed metrics.
        """
        stats = TableStats()

        if not table_body or not table_body.strip():
            return stats

        # Detect table format and parse
        is_html = "<tr" in table_body.lower() or "<td" in table_body.lower()
        is_markdown = "|" in table_body and "\n" in table_body

        rows_data: list[list[str]] = []

        if is_html:
            # Parse HTML table
            tr_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
            td_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)

            for tr_match in tr_pattern.findall(table_body):
                cells = td_pattern.findall(tr_match)
                # Clean HTML tags from cell content
                clean_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                rows_data.append(clean_cells)

        elif is_markdown:
            # Parse markdown table
            lines = table_body.strip().split("\n")
            for line in lines:
                # Skip separator rows (|---|---|)
                if re.match(r"^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?\s*$", line):
                    continue
                if "|" in line:
                    # Split by | and clean up
                    cells = [c.strip() for c in line.split("|")]
                    # Remove empty first/last cells from leading/trailing |
                    if cells and not cells[0]:
                        cells = cells[1:]
                    if cells and not cells[-1]:
                        cells = cells[:-1]
                    if cells:
                        rows_data.append(cells)
        else:
            # Fallback: treat as simple row-per-line
            lines = [line.strip() for line in table_body.strip().split("\n") if line.strip()]
            for line in lines:
                rows_data.append([line])

        # Calculate stats
        stats.rows = len(rows_data)
        stats.cols = max((len(row) for row in rows_data), default=0)
        stats.cells = stats.rows * stats.cols

        # Count non-empty cells and total characters
        non_empty = 0
        total_chars = 0
        for row in rows_data:
            for cell in row:
                total_chars += len(cell)
                if cell.strip():
                    non_empty += 1

        stats.non_empty_cells = non_empty
        stats.text_chars = total_chars
        stats.non_empty_ratio = non_empty / stats.cells if stats.cells > 0 else 0.0

        return stats

    def _truncate_table(
        self,
        table_body: str,
        head_rows: int | None = None,
        tail_rows: int | None = None,
    ) -> tuple[str, str]:
        """
        Truncate large table for VLM input.

        Keeps header row + first N rows + last N rows.
        Returns (truncated_body, policy_name).
        """
        # Use config defaults if not specified
        if head_rows is None:
            head_rows = self.enrich_config.table_truncate_head_rows
        if tail_rows is None:
            tail_rows = self.enrich_config.table_truncate_tail_rows

        is_html = "<tr" in table_body.lower()
        is_markdown = "|" in table_body and "\n" in table_body

        if is_html:
            import re
            tr_pattern = re.compile(r"<tr[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
            rows = tr_pattern.findall(table_body)

            if len(rows) <= head_rows + tail_rows + 1:
                return table_body, "no_truncation"

            # Keep header (first row) + head_rows + marker + tail_rows
            truncated_rows = (
                rows[:1 + head_rows] +  # Header + first N rows
                ["<tr><td colspan='100'>... (truncated) ...</td></tr>"] +
                rows[-tail_rows:]  # Last N rows
            )
            truncated = "\n".join(truncated_rows)
            return truncated, f"head{head_rows}+tail{tail_rows}"

        elif is_markdown:
            lines = table_body.strip().split("\n")

            # Separate header, separator, and data rows
            header_lines: list[str] = []
            data_lines: list[str] = []

            for i, line in enumerate(lines):
                if re.match(r"^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?\s*$", line):
                    # This is separator row, keep it with header
                    if not header_lines:
                        continue
                    header_lines.append(line)
                elif i <= 1:  # First two lines are usually header
                    header_lines.append(line)
                else:
                    data_lines.append(line)

            if len(data_lines) <= head_rows + tail_rows:
                return table_body, "no_truncation"

            # Truncate data rows
            truncated_data = (
                data_lines[:head_rows] +
                ["| ... (truncated) ... |"] +
                data_lines[-tail_rows:]
            )
            truncated = "\n".join(header_lines + truncated_data)
            return truncated, f"head{head_rows}+tail{tail_rows}"

        # Fallback: simple line truncation
        lines = table_body.strip().split("\n")
        if len(lines) <= head_rows + tail_rows:
            return table_body, "no_truncation"

        truncated_lines = (
            lines[:head_rows] +
            ["... (truncated) ..."] +
            lines[-tail_rows:]
        )
        return "\n".join(truncated_lines), f"head{head_rows}+tail{tail_rows}"

    def _is_layout_table(self, table_body: str, stats: TableStats) -> bool:
        """
        Detect if table is likely a layout container rather than data table.

        Layout tables typically have:
        - Very low non_empty_ratio (lots of empty cells)
        - Very few characters per cell
        - Single row/column structures
        """
        # Single row or column tables are often layout
        if stats.rows <= 1 or stats.cols <= 1:
            return True

        # Very low non-empty ratio suggests layout
        min_ratio = self.enrich_config.table_layout_min_ratio
        if stats.non_empty_ratio < min_ratio:
            return True

        # Very few characters suggests layout
        min_chars = self.enrich_config.table_layout_min_chars_per_cell
        avg_chars_per_cell = stats.text_chars / stats.cells if stats.cells > 0 else 0
        if avg_chars_per_cell < min_chars:
            return True

        return False

    def _is_form_like_table(self, table_body: str) -> bool:
        """Detect native Office/Excel layout tables that are actually fillable forms."""
        if not table_body or not table_body.strip():
            return False

        stats = self._analyze_table(table_body)
        if stats.rows < 5:
            return False
        if looks_like_reference_table(table_body):
            return False

        text = re.sub(r"<[^>]+>", " ", table_body)
        text = re.sub(r"\s+", " ", text)
        form_terms = [
            "申請", "申請人", "申請單位", "核定", "核准", "簽核", "簽章",
            "單位主管", "主任秘書", "副院長", "院長", "董事長", "出差",
            "報支", "預借", "預算審查", "變更申請", "備註", "□", "__",
            "年", "月", "日",
        ]
        hits = sum(1 for term in form_terms if term in text)
        low_density = stats.non_empty_ratio < 0.45
        has_fillable_marks = any(mark in text for mark in ["□", "___", " 年 ", " 月 ", " 日 "])
        approval_terms = ["單位主管", "主任秘書", "副院長", "院長", "董事長"]
        has_approval = any(term in text for term in approval_terms)
        return hits >= 4 and (low_density or has_fillable_marks or has_approval)

    def _table_context_for_form(self, table_body: str, max_chars: int = 6000) -> str:
        """Serialize table cells into compact text for form VLM prompts."""
        if not table_body:
            return ""

        rows: list[list[str]] = []
        tr_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
        cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
        for tr_match in tr_pattern.findall(table_body):
            cells = []
            for cell_match in cell_pattern.findall(tr_match):
                cell_text = re.sub(r"<[^>]+>", "", cell_match)
                cell_text = re.sub(r"\s+", " ", cell_text).strip()
                cells.append(cell_text)
            non_empty = [cell for cell in cells if cell]
            if non_empty:
                rows.append(cells)

        lines = []
        for row in rows:
            line = " | ".join(row).strip()
            line = re.sub(r"(\s*\|\s*){2,}", " | ", line).strip(" |")
            if line:
                lines.append(line)

        text = "\n".join(lines)
        return text[:max_chars]


    def _evaluate_table_gating(self, table_body: str) -> TableGatingResult:
        """
        Evaluate table gating criteria and return detailed result.

        Gating rules:
        1. Skip empty tables
        2. Skip layout tables (low non_empty_ratio)
        3. Skip tiny tables (< min_cells)
        4. For huge tables (> max_cells): truncate instead of skip
        """
        result = TableGatingResult()

        if not table_body or not table_body.strip():
            result.skip_reason = "no_table_body"
            return result

        # Analyze table structure
        stats = self._analyze_table(table_body)
        result.stats = stats

        # Get config thresholds
        min_cells = self.enrich_config.table_min_cells
        max_cells = self.enrich_config.table_max_cells

        # Check for layout table
        if self._is_layout_table(table_body, stats):
            result.skip_reason = "layout_table"
            return result

        # Check minimum size
        if stats.cells < min_cells:
            result.skip_reason = "too_small"
            return result

        # Check maximum size - truncate instead of skip
        if stats.cells > max_cells:
            truncated_body, policy = self._truncate_table(table_body)
            if policy != "no_truncation":
                result.truncated = True
                result.truncate_policy = policy
                result.truncated_body = truncated_body

        # Table passes gating
        result.should_process = True
        return result

    def _should_enrich_table(self, table_body: str) -> bool:
        """
        Check if table should be enriched with VLM based on gating criteria.

        Legacy wrapper for backward compatibility.
        Use _evaluate_table_gating for detailed results.
        """
        result = self._evaluate_table_gating(table_body)
        return result.should_process

    def _is_complex_table(self, table_body: str) -> bool:
        """Check if table is complex enough to need summarization (legacy)."""
        # Count rows (simple heuristic)
        row_count = table_body.count("<tr>") or table_body.count("|")
        return row_count > 5

    def _extract_table_headers(self, table_body: str) -> list[str]:
        """Extract column headers from table HTML or markdown."""
        headers = []

        # Try HTML <th> tags first
        th_pattern = re.compile(r"<th[^>]*>(.*?)</th>", re.IGNORECASE | re.DOTALL)
        th_matches = th_pattern.findall(table_body)
        if th_matches:
            for h in th_matches[:10]:  # Limit to first 10
                # Strip HTML tags from content
                clean = re.sub(r"<[^>]+>", "", h).strip()
                if clean:
                    headers.append(clean)
            if headers:
                return headers

        # Try first row of markdown table (| col1 | col2 | ...)
        lines = table_body.strip().split("\n")
        if lines:
            first_line = lines[0]
            if "|" in first_line:
                # Parse markdown table header
                parts = first_line.split("|")
                for p in parts:
                    p = p.strip()
                    if p and not re.match(r"^[-:]+$", p):  # Skip separator row
                        headers.append(p)
                if headers:
                    return headers[:10]

        # Try first <tr> row
        tr_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
        tr_matches = tr_pattern.findall(table_body)
        if tr_matches:
            first_row = tr_matches[0]
            td_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
            td_matches = td_pattern.findall(first_row)
            for td in td_matches[:10]:
                clean = re.sub(r"<[^>]+>", "", td).strip()
                if clean:
                    headers.append(clean)

        return headers

    def _build_form_context(self, page_blocks: list[Block], max_chars: int = 2000) -> str:
        """
        Build rich context for form understanding.

        Includes:
        - All text blocks (full content, not truncated)
        - Table HTML/markdown (field candidates)
        - Checkbox indicators

        This provides VLM with ground truth text to prevent hallucination.
        """
        context_parts: list[str] = []
        table_parts: list[str] = []

        for block in page_blocks:
            if block.type == BlockType.TEXT:
                text = block.payload.get("text", "").strip()
                if text:
                    context_parts.append(text)

            elif block.type == BlockType.TABLE:
                # Include table body as field candidate source
                table_body = block.payload.get("table_body", "")
                if table_body:
                    # Extract clean text from table
                    table_text = self._clean_table_for_context(table_body)
                    if table_text:
                        table_parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")

        # Combine: text first, then tables
        all_text = "\n".join(context_parts)
        all_tables = "\n".join(table_parts)

        if all_tables:
            combined = f"{all_text}\n\n{all_tables}"
        else:
            combined = all_text

        # Truncate if too long
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."

        return combined

    def _clean_table_for_context(self, table_body: str, max_chars: int = 500) -> str:
        """Extract clean text from table HTML/markdown for context."""
        import re

        # Remove HTML tags but keep text
        text = re.sub(r"<[^>]+>", " ", table_body)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text

    def _get_block_image(
        self,
        block: Block,
        parse_cache_path: Path | None,
    ) -> Path | None:
        """Get image path for a block."""
        if block.type == BlockType.IMAGE:
            img_path = block.payload.get("img_path", "")
            if img_path and parse_cache_path:
                # Try relative path
                candidate = parse_cache_path / img_path
                if candidate.exists():
                    return candidate

                # Try searching
                for p in parse_cache_path.rglob(Path(img_path).name):
                    return p

        # For other block types, we'd need to crop from page image
        # This would require page images to be available
        return None

    def _get_page_thumbnail(
        self,
        doc_id: str,
        run_id: str,
        page_idx: int,
        run_path: Path,
    ) -> Path | None:
        """
        Get page thumbnail for visual context.

        Looks for rendered page image in assets/pages/ directory.
        """
        # Check run assets/pages directory first (generated by normalize stage)
        pages_dir = run_path / "assets" / "pages"
        if pages_dir.exists():
            # Try common naming patterns
            for pattern in [
                f"p{page_idx:04d}.png",
                f"page_{page_idx:04d}.png",
                f"page_{page_idx}.png",
            ]:
                candidate = pages_dir / pattern
                if candidate.exists():
                    return candidate

        # Fallback to source directory renders
        source_dir = settings.get_doc_path(doc_id)
        for pages_subdir in ["pages", "source/pages"]:
            pages_path = source_dir / pages_subdir
            if pages_path.exists():
                for pattern in [
                    f"p{page_idx:04d}.png",
                    f"page_{page_idx:04d}.png",
                ]:
                    candidate = pages_path / pattern
                    if candidate.exists():
                        return candidate

        return None

    def _get_context(
        self,
        document_ir: DocumentIR,
        block: Block,
        window: int = 3,
    ) -> str:
        """Get context text from surrounding blocks."""
        # Find blocks on same page
        page_blocks = [
            b for b in document_ir.blocks
            if b.page_idx == block.page_idx
        ]

        # Sort by reading order
        page_blocks.sort(key=lambda b: b.reading_order)

        # Find current block index
        try:
            idx = next(
                i for i, b in enumerate(page_blocks)
                if b.block_id == block.block_id
            )
        except StopIteration:
            return ""

        # Get surrounding text blocks
        context_parts = []
        for i in range(max(0, idx - window), min(len(page_blocks), idx + window + 1)):
            if i == idx:
                continue
            b = page_blocks[i]
            if b.type == BlockType.TEXT:
                text = b.payload.get("text", "")
                if text:
                    context_parts.append(text)

        return "\n".join(context_parts)

    async def _enrich_block(
        self,
        block: Block | None,
        kind: str,
        image_path: Path | None,
        context: str,
        doc_id: str | None = None,
        run_id: str | None = None,
        page_idx: int | None = None,
        bbox: list[int] | None = None,
        table_body: str | None = None,
        table_headers: list[str] | None = None,
        page_thumbnail_path: Path | None = None,
    ) -> EnrichmentOutput:
        """
        Enrich a single block using VLM.

        Args:
            block: Block to enrich (optional for page-level enrichments)
            kind: Enrichment type (form_asset, figure_caption, table_summary, etc.)
            image_path: Path to crop image
            context: Text context from surrounding blocks
            doc_id: Document ID for URL building
            run_id: Run ID for URL building
            page_idx: Page index (0-based)
            bbox: Bounding box in MinerU 0-1000 normalized coordinates
            table_body: MinerU table HTML/markdown for table_summary
            table_headers: Extracted column headers for table_summary
            page_thumbnail_path: Optional page thumbnail for visual context
        """
        # Get page_idx and bbox from block if not provided
        if block is not None:
            if page_idx is None:
                page_idx = block.page_idx
            if bbox is None:
                bbox = block.bbox_norm

        if kind in ("form_guide", "form_asset"):
            return await self.vlm_adapter.enrich_form(
                image_path, context,
                doc_id=doc_id, run_id=run_id, page_idx=page_idx, bbox=bbox,
                page_thumbnail_path=page_thumbnail_path,
                extra_vars={"semantic_output_language": self._semantic_output_language},
            )
        elif kind in ("figure_description", "figure_caption"):
            return await self.vlm_adapter.enrich_figure(
                image_path, context,
                doc_id=doc_id, run_id=run_id, page_idx=page_idx, bbox=bbox,
                page_thumbnail_path=page_thumbnail_path,
            )
        elif kind == "table_summary":
            # Pass MinerU table data to reduce hallucination
            return await self.vlm_adapter.enrich_table(
                image_path, context,
                doc_id=doc_id, run_id=run_id, page_idx=page_idx, bbox=bbox,
                table_body=table_body, table_headers=table_headers,
                page_thumbnail_path=page_thumbnail_path,
            )
        elif kind == "page_description":
            return await self.vlm_adapter.enrich_figure(
                image_path, context,
                doc_id=doc_id, run_id=run_id, page_idx=page_idx, bbox=bbox,
                page_thumbnail_path=page_thumbnail_path,
            )
        else:
            return EnrichmentOutput(
                success=False,
                kind=kind,
                error=f"Unknown enrichment kind: {kind}",
            )

    async def _select_org_chart_edges(
        self,
        doc_id: str,
        run_id: str,
        image_path: Path,
        edge_candidates: list,
        nodes: list,
    ) -> list:
        """
        D3: Call VLM#2 to select edges from heuristic candidates.

        Args:
            doc_id: Document ID for URL building
            run_id: Run ID for URL building
            image_path: Path to org chart image
            edge_candidates: List of EdgeCandidate objects
            nodes: List of OrgNode objects

        Returns:
            List of OrgEdge objects selected by VLM#2
        """
        import json

        # Format candidates for VLM prompt
        candidates_json = json.dumps(
            [ec.to_vlm_format() for ec in edge_candidates],
            ensure_ascii=False,
            indent=2,
        )

        # Call VLM with org_chart_edges prompt
        result = await self.vlm_adapter.enrich(
            kind="org_chart_edges",
            image_path=image_path,
            context="",
            doc_id=doc_id,
            run_id=run_id,
            extra_vars={"candidates_json": candidates_json},
        )

        if not result.success:
            return []

        # Parse VLM#2 response into edges
        vlm2_edges = self.org_chart_parser.parse_vlm2_edges(
            vlm_edges_output=result.output,
            nodes=nodes,
            candidates=edge_candidates,
        )

        return vlm2_edges

    async def _select_org_chart_edges_with_debug(
        self,
        doc_id: str,
        run_id: str,
        image_path: Path,
        edge_candidates: list,
        nodes: list,
    ) -> dict[str, Any]:
        """
        D4: Call VLM#2 with debug output.

        Returns:
            dict with:
            - edges: List of OrgEdge objects
            - raw: Raw VLM response for debug
        """
        import json

        # Format candidates for VLM prompt
        candidates_json = json.dumps(
            [ec.to_vlm_format() for ec in edge_candidates],
            ensure_ascii=False,
            indent=2,
        )

        # Call VLM with org_chart_edges prompt
        result = await self.vlm_adapter.enrich(
            kind="org_chart_edges",
            image_path=image_path,
            context="",
            doc_id=doc_id,
            run_id=run_id,
            extra_vars={"candidates_json": candidates_json},
        )

        if not result.success:
            return {
                "edges": [],
                "raw": {"error": result.error, "raw_response": result.raw_response},
            }

        # Parse VLM#2 response into edges
        vlm2_edges = self.org_chart_parser.parse_vlm2_edges(
            vlm_edges_output=result.output,
            nodes=nodes,
            candidates=edge_candidates,
        )

        return {
            "edges": vlm2_edges,
            "raw": {
                "raw_response": result.raw_response,
                "parsed_output": result.output,
            },
        }

    def _is_diagram_document(self, source_path: str) -> bool:
        """Check if document appears to be a diagram based on filename."""
        filename = Path(source_path).stem.lower()
        for pattern in DIAGRAM_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return True
        return False

    def _is_visual_page(self, document_ir: DocumentIR, page_idx: int) -> bool:
        """
        Check if a page is primarily visual (diagram, chart, etc.)

        A visual page has:
        - Low total text content (short text in blocks)
        - Few text blocks
        - Or is from a diagram document
        """
        # Get blocks on this page
        page_blocks = [b for b in document_ir.blocks if b.page_idx == page_idx]

        # Count text content
        text_blocks = [b for b in page_blocks if b.type == BlockType.TEXT]
        total_text = sum(len(b.payload.get("text", "")) for b in text_blocks)

        # Visual page heuristics:
        # - Less than 200 characters of text
        # - Or filename matches diagram patterns
        if total_text < 200:
            return True

        if self._is_diagram_document(document_ir.source.path):
            return True

        return False

    def _get_visual_pages(self, document_ir: DocumentIR) -> list[int]:
        """Get list of page indices that are primarily visual."""
        visual_pages = []
        for page in document_ir.pages:
            if self._is_visual_page(document_ir, page.page_idx):
                visual_pages.append(page.page_idx)
        return visual_pages

    def _render_page_to_image(
        self,
        doc_id: str,
        page_idx: int,
        output_dir: Path,
    ) -> Path | None:
        """
        Render a PDF page to an image using PyMuPDF.

        Returns the path to the rendered image, or None if rendering fails.
        """
        if not HAS_PYMUPDF:
            return None

        # Find the source PDF
        source_dir = settings.get_doc_path(doc_id) / "source"
        pdf_path = None
        for f in source_dir.glob("original.*"):
            if f.suffix.lower() == ".pdf":
                pdf_path = f
                break

        if not pdf_path:
            return None

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"page_{page_idx:04d}.png"

            # Open PDF and render page
            doc = fitz.open(pdf_path)
            if page_idx >= len(doc):
                doc.close()
                return None

            page = doc[page_idx]
            # Render at 2x zoom for better quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            pix.save(output_path)
            doc.close()

            return output_path
        except Exception:
            return None

    def _detect_form_pages(self, document_ir: DocumentIR) -> list[int]:
        """
        Detect which pages are likely form pages.

        Form pages are identified by:
        - Filename matching form patterns (申請, 表單, etc.)
        - Pages with low text content (mostly visual structure)
        - Pages containing table structures

        Returns:
            List of page indices that are form pages
        """
        form_pages = []

        if self._is_structured_rate_table_document(document_ir):
            return []

        # Check if document is a form based on filename
        is_form_doc = self._is_form_document(document_ir.source.path)

        for page in document_ir.pages:
            page_blocks = document_ir.get_blocks_by_page(page.page_idx)
            if self._page_has_form_cues(page_blocks):
                form_pages.append(page.page_idx)
                continue

            if not is_form_doc:
                continue

            table_blocks = [b for b in page_blocks if b.type == BlockType.TABLE]
            if table_blocks:
                continue

            text_content = sum(len(b.get_text() or "") for b in page_blocks)
            if text_content < 500:
                form_pages.append(page.page_idx)

        return form_pages

    def _is_structured_rate_table_document(self, document_ir: DocumentIR) -> bool:
        """Avoid treating row-oriented reference/data tables as form pages."""
        table_blocks = [block for block in document_ir.blocks if block.type == BlockType.TABLE]
        if not table_blocks:
            return False
        reference_tables = [
            block for block in table_blocks
            if looks_like_reference_table(str(block.payload.get("table_body") or ""))
        ]
        if not reference_tables:
            return False
        whole_text = " ".join(block.get_text() or "" for block in document_ir.blocks[:30])
        if re.search(r"□|☐|☑|☒|_{3,}|＿{3,}|申請人[:：]|申請單位[:：]|簽名|簽章|請勾選", whole_text):
            return False
        return len(reference_tables) == len(table_blocks)

    def _page_has_form_cues(self, page_blocks: list[Block]) -> bool:
        """Return true for pages with explicit fillable-form language."""
        table_blocks = [block for block in page_blocks if block.type == BlockType.TABLE]
        if table_blocks and all(
            looks_like_reference_table(str(block.payload.get("table_body") or ""))
            for block in table_blocks
        ):
            return False

        text = " ".join(block.get_text() or "" for block in page_blocks)
        if not text:
            return False
        form_terms = [
            "申請人",
            "申請單位",
            "填表",
            "填寫",
            "姓名",
            "身分證",
            "身份證",
            "電話",
            "簽章",
            "簽名",
            "簽名蓋章",
            "核章",
            "審核",
            "請勾選",
            "立保證規約人",
            "保證人",
            "對保人",
            "立約",
            "name",
            "ssn",
            "social security",
            "signature",
            "date signed",
            "taxpayer",
            "requester",
            "authorization",
            "authorize",
            "consent",
            "phone number",
            "street address",
            "omb no",
            "form 4506",
            "form ssa",
            "□",
            "☐",
            "☑",
        ]
        lower_text = text.lower()
        hits = sum(1 for term in form_terms if term in text or term.lower() in lower_text)
        has_table = any(block.type == BlockType.TABLE for block in page_blocks)
        return hits >= (3 if has_table else 3)

    def _export_form_page(
        self,
        doc_id: str,
        page_idx: int,
        output_path: Path,
    ) -> Path | None:
        """
        Export a form page as a full-page image.

        Args:
            doc_id: Document ID
            page_idx: Page index
            output_path: Path to save the form image

        Returns:
            Path to the exported image, or None if failed
        """
        if not HAS_PYMUPDF:
            return None

        # Find source PDF
        source_dir = settings.get_doc_path(doc_id) / "source"
        pdf_path = None
        for f in source_dir.glob("original.*"):
            if f.suffix.lower() == ".pdf":
                pdf_path = f
                break

        if not pdf_path:
            return None

        try:
            doc = fitz.open(pdf_path)
            if page_idx >= len(doc):
                doc.close()
                return None

            page = doc[page_idx]

            # Render at form DPI (higher quality for forms)
            zoom = FORM_PAGE_DPI / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(output_path)
            doc.close()

            return output_path
        except Exception:
            return None

    def _load_yolo_detections(self, parse_cache_path: Path) -> list[dict]:
        """
        Load YOLO detection results from model.json.

        Returns list of detections per page: [{"layout_dets": [...], ...}, ...]
        """
        if not parse_cache_path:
            return []

        # Find model.json in parse cache
        for model_json in parse_cache_path.rglob("*_model.json"):
            try:
                with open(model_json, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _get_figure_detections(
        self,
        yolo_results: list[dict],
        min_score: float = YOLO_MIN_SCORE,
    ) -> list[tuple[int, dict]]:
        """
        Get figure/diagram detections from YOLO results.

        Returns list of (page_idx, detection) tuples.
        """
        figures = []
        for page_idx, page_result in enumerate(yolo_results):
            layout_dets = page_result.get("layout_dets", [])
            for det in layout_dets:
                category_id = det.get("category_id")
                score = det.get("score", 0)

                # ImageBody (figures, diagrams, charts)
                if category_id == MinerUCategoryId.ImageBody and score >= min_score:
                    figures.append((page_idx, det))

        return figures

    def _crop_region_from_pdf(
        self,
        doc_id: str,
        page_idx: int,
        poly: list[float],
        output_path: Path,
        max_size: int = VLM_IMAGE_MAX_SIZE,
    ) -> Path | None:
        """
        Crop a region from PDF page based on polygon coordinates.

        Args:
            doc_id: Document ID
            page_idx: Page index
            poly: Polygon coordinates [x1,y1,x2,y2,x3,y3,x4,y4] or [x1,y1,x2,y2]
            output_path: Where to save the cropped image
            max_size: Maximum dimension for output image

        Returns:
            Path to cropped image, or None if failed
        """
        if not HAS_PYMUPDF:
            return None

        # Find the source PDF
        source_dir = settings.get_doc_path(doc_id) / "source"
        pdf_path = None
        for f in source_dir.glob("original.*"):
            if f.suffix.lower() == ".pdf":
                pdf_path = f
                break

        if not pdf_path:
            return None

        try:
            doc = fitz.open(pdf_path)
            if page_idx >= len(doc):
                doc.close()
                return None

            page = doc[page_idx]

            # Convert poly to rect (handle both 4-point and 8-point formats)
            if not poly:
                doc.close()
                return None
            if len(poly) == 8:
                x_coords = [poly[0], poly[2], poly[4], poly[6]]
                y_coords = [poly[1], poly[3], poly[5], poly[7]]
            elif len(poly) == 4:
                x_coords = [poly[0], poly[2]]
                y_coords = [poly[1], poly[3]]
            else:
                doc.close()
                return None

            x0, y0 = min(x_coords), min(y_coords)
            x1, y1 = max(x_coords), max(y_coords)

            # Add padding (5% of bbox size)
            pad_x = (x1 - x0) * 0.05
            pad_y = (y1 - y0) * 0.05
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(page.rect.width, x1 + pad_x)
            y1 = min(page.rect.height, y1 + pad_y)

            clip_rect = fitz.Rect(x0, y0, x1, y1)

            # Calculate zoom to fit within max_size
            width = x1 - x0
            height = y1 - y0
            max_dim = max(width, height)
            zoom = min(2.0, max_size / max_dim) if max_dim > 0 else 2.0

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(output_path)
            doc.close()

            return output_path
        except Exception:
            return None
