"""
Normalize stage - Build DocumentIR from MinerU output.

Responsibilities:
- Parse MinerU content_list.json into DocumentIR
- Render PDF pages to assets/pages/ (for VLM context and Viewer)
- Build page info with dimensions
"""

import asyncio
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from PIL import Image

from app.config import PipelineConfig, settings
from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.supported_files import SPREADSHEET_NATIVE_EXTENSIONS


@dataclass
class NormalizeStageResult:
    """Result from normalize stage."""

    success: bool
    document_ir: DocumentIR | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class NormalizeStage:
    """
    Normalize stage - builds DocumentIR from MinerU content_list.json.

    Input: MinerU content_list.json
    Output: DocumentIR with normalized blocks and rendered pages
    """

    # Page render settings
    PAGE_RENDER_DPI = 150  # Balance between quality and size
    PAGE_RENDER_FORMAT = "png"

    # Text supplement settings
    TEXT_SUPPLEMENT_MIN_LENGTH = 4  # Minimum text length to consider
    TEXT_SUPPLEMENT_COVERAGE_THRESHOLD = 0.3  # Min coverage to consider "covered"

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    async def run(
        self,
        doc_id: str,
        run_id: str,
        content_list_path: Path,
        source_info: SourceInfo,
        render_pages: bool = True,
        mineru_version: str | None = None,
    ) -> NormalizeStageResult:
        """
        Run normalize stage.

        Args:
            doc_id: Document ID
            run_id: Run ID
            content_list_path: Path to MinerU content_list.json
            source_info: Source file information
            render_pages: Whether to render PDF pages to assets/pages/
            mineru_version: MinerU version for EngineInfo

        Returns:
            NormalizeStageResult with DocumentIR
        """
        try:
            # Load content_list.json
            content_list = json.loads(content_list_path.read_text(encoding="utf-8"))

            if not isinstance(content_list, list):
                return NormalizeStageResult(
                    success=False,
                    error="content_list.json is not a list",
                )

            # Build blocks
            blocks = []
            page_indices = set()

            for idx, item in enumerate(content_list):
                block = self._parse_block(item, idx)
                if block:
                    blocks.append(block)
                    page_indices.add(block.page_idx)

            # Sort by reading order
            blocks.sort(key=lambda b: (b.page_idx, b.reading_order))

            # Deduplicate overlapping blocks on same page
            blocks = self._dedup_overlapping_blocks(blocks)

            # Supplement missing text from PDF (PyMuPDF fallback)
            supplement_count = 0
            if HAS_PYMUPDF:
                blocks, supplement_count = await self._supplement_missing_text(
                    doc_id=doc_id,
                    blocks=blocks,
                    content_list_path=content_list_path,
                    source_info=source_info,
                )

            # Build page info
            pages = [
                PageInfo(page_idx=i)
                for i in sorted(page_indices)
            ]

            # Render pages and get dimensions
            run_path = settings.get_run_path(doc_id, run_id)
            pages_dir = run_path / "assets" / "pages"

            if render_pages:
                pages = await self._render_and_enrich_pages(
                    doc_id=doc_id,
                    pages=pages,
                    output_dir=pages_dir,
                    content_list_path=content_list_path,
                    source_info=source_info,
                )
            else:
                # Try to get page dimensions from MinerU images
                pages = await self._enrich_page_info(
                    doc_id=doc_id,
                    run_id=run_id,
                    pages=pages,
                    content_list_path=content_list_path,
                )

            # Build engine info
            engine = EngineInfo(
                name="mineru",
                backend=self.config.mineru.backend.value,
                version=mineru_version,
                method=self.config.mineru.method.value,
                lang=self.config.mineru.lang,
                table=self.config.mineru.table,
                formula=self.config.mineru.formula,
            )

            # Build DocumentIR
            document_ir = DocumentIR(
                doc_id=doc_id,
                run_id=run_id,
                source=source_info,
                engine=engine,
                pages=pages,
                blocks=blocks,
            )

            # Compute stats
            stats = {
                "block_count": len(blocks),
                "page_count": len(pages),
                "by_type": document_ir.count_by_type(),
                "pages_rendered": render_pages and any(p.page_image_path for p in pages),
                "pages_with_images": sum(1 for p in pages if p.page_image_path),
                "text_supplemented": supplement_count,
            }

            return NormalizeStageResult(
                success=True,
                document_ir=document_ir,
                stats=stats,
            )

        except Exception as e:
            return NormalizeStageResult(
                success=False,
                error=str(e),
            )

    def _parse_block(self, item: dict[str, Any], index: int) -> Block | None:
        """Parse a MinerU content_list item into a Block."""
        item_type = item.get("type", "")

        # Map MinerU types to BlockType
        type_map = {
            "text": BlockType.TEXT,
            "table": BlockType.TABLE,
            "image": BlockType.IMAGE,
            "equation": BlockType.EQUATION,
            "code": BlockType.CODE,
            "list": BlockType.LIST,
        }

        block_type = type_map.get(item_type)

        # Handle table_no_body_mode: table with img_path but no table_body
        if item_type == "table":
            table_body = item.get("table_body", "")
            img_path = item.get("img_path", "")

            if not table_body and img_path:
                # Table detected but only image available -> convert to image
                block_type = BlockType.IMAGE
                # Mark origin for traceability
                item["_origin"] = "table_no_body"
            elif not table_body and not img_path:
                # Table with neither body nor image -> mark as unknown
                block_type = BlockType.UNKNOWN
                item["_origin"] = "table_missing_body_and_image"

        if not block_type:
            # Unknown type, try to handle as text
            if "text" in item:
                block_type = BlockType.TEXT
            else:
                return None

        # Build block ID
        block_id = f"b{index:06d}"

        # Get bbox
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]

        # Ensure bbox values are integers
        bbox = [int(v) for v in bbox]

        # Get page index
        page_idx = item.get("page_idx", 0)

        # Build payload based on type
        payload = self._build_payload(item, block_type)

        return Block(
            block_id=block_id,
            type=block_type,
            page_idx=page_idx,
            bbox_norm=bbox,
            reading_order=index,
            payload=payload,
        )

    def _build_payload(self, item: dict[str, Any], block_type: BlockType) -> dict[str, Any]:
        """Build type-specific payload from MinerU item."""
        if block_type == BlockType.TEXT:
            return {
                "text": item.get("text", ""),
                "text_level": item.get("text_level", 0),
            }
        elif block_type == BlockType.IMAGE:
            payload = {
                "img_path": item.get("img_path", ""),
                "caption": item.get("img_caption"),
                "footnote": item.get("img_footnote"),
            }
            # Add origin marker for table-converted images
            if item.get("_origin"):
                payload["origin"] = item["_origin"]
            return payload
        elif block_type == BlockType.TABLE:
            return {
                "table_body": item.get("table_body", ""),
                "table_caption": item.get("table_caption"),
            }
        elif block_type == BlockType.EQUATION:
            return {
                "latex": item.get("latex", item.get("text", "")),
                "equation_type": item.get("equation_type"),
            }
        elif block_type == BlockType.CODE:
            return {
                "code": item.get("code", item.get("text", "")),
                "language": item.get("language"),
            }
        elif block_type == BlockType.LIST:
            # Handle list items
            items = item.get("items", [])
            if not items and "text" in item:
                # Parse text as list items (split by newlines)
                items = [line.strip() for line in item["text"].split("\n") if line.strip()]
            return {
                "items": items,
                "list_type": item.get("list_type", "unordered"),
            }
        elif block_type == BlockType.UNKNOWN:
            # Unknown block - preserve original info for debugging
            return {
                "origin": item.get("_origin", "unknown"),
                "original_type": item.get("type", ""),
                "needs_review": True,
            }

        return {}

    def _dedup_overlapping_blocks(self, blocks: list[Block]) -> list[Block]:
        """
        Remove duplicate blocks with overlapping bboxes on the same page.

        Priority: IMAGE > TABLE > others
        Uses IoU (Intersection over Union) > 0.9 as overlap threshold.
        """
        if not blocks:
            return blocks

        # Group blocks by page
        by_page: dict[int, list[Block]] = {}
        for block in blocks:
            if block.page_idx not in by_page:
                by_page[block.page_idx] = []
            by_page[block.page_idx].append(block)

        result: list[Block] = []

        for page_idx in sorted(by_page.keys()):
            page_blocks = by_page[page_idx]
            kept: list[Block] = []
            removed_ids: set[str] = set()

            for block in page_blocks:
                if block.block_id in removed_ids:
                    continue

                # Check overlap with already kept blocks
                is_duplicate = False
                for kept_block in kept:
                    if self._compute_iou(block.bbox_norm, kept_block.bbox_norm) > 0.9:
                        # Overlap detected - decide which to keep
                        # Priority: IMAGE > TABLE > others
                        block_priority = self._get_block_priority(block.type)
                        kept_priority = self._get_block_priority(kept_block.type)

                        if block_priority > kept_priority:
                            # Replace kept block with current block
                            kept.remove(kept_block)
                            removed_ids.add(kept_block.block_id)
                            kept.append(block)
                        else:
                            # Keep existing, mark current as duplicate
                            is_duplicate = True
                            removed_ids.add(block.block_id)
                        break

                if not is_duplicate and block.block_id not in removed_ids:
                    kept.append(block)

            result.extend(kept)

        # Re-sort by reading order
        result.sort(key=lambda b: (b.page_idx, b.reading_order))
        return result

    def _compute_iou(self, bbox1: list[int], bbox2: list[int]) -> float:
        """Compute Intersection over Union of two bboxes."""
        if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _get_block_priority(self, block_type: BlockType) -> int:
        """Get priority for block type (higher = more important)."""
        priority_map = {
            BlockType.IMAGE: 10,
            BlockType.TABLE: 5,
            BlockType.TEXT: 3,
            BlockType.EQUATION: 3,
            BlockType.CODE: 3,
            BlockType.LIST: 3,
            BlockType.UNKNOWN: 1,
        }
        return priority_map.get(block_type, 0)

    async def _supplement_missing_text(
        self,
        doc_id: str,
        blocks: list[Block],
        content_list_path: Path,
        source_info: SourceInfo | None = None,
    ) -> tuple[list[Block], int]:
        """
        Supplement missing text blocks from PDF using PyMuPDF.

        MinerU's layout detection may miss some text regions. This method
        extracts text directly from PDF and adds blocks for uncovered regions.

        Args:
            doc_id: Document ID
            blocks: Existing blocks from MinerU
            content_list_path: Path to MinerU content_list.json

        Returns:
            Tuple of (updated blocks list, count of supplemented blocks)
        """
        if not HAS_PYMUPDF:
            return blocks, 0
        if source_info and self._is_spreadsheet_source(source_info):
            return blocks, 0

        # Find PDF path (source or MinerU-generated)
        pdf_path = self._find_pdf_path(doc_id, content_list_path)
        if not pdf_path:
            return blocks, 0

        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return blocks, 0

        # Group existing blocks by page
        blocks_by_page: dict[int, list[Block]] = {}
        for block in blocks:
            if block.page_idx not in blocks_by_page:
                blocks_by_page[block.page_idx] = []
            blocks_by_page[block.page_idx].append(block)

        supplemented: list[Block] = []
        next_block_idx = len(blocks)  # For generating new block IDs

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_blocks = blocks_by_page.get(page_idx, [])

                # Extract text blocks from PDF
                pdf_text_blocks = self._extract_pdf_text_blocks(page)

                # Find uncovered text blocks
                for pdf_block in pdf_text_blocks:
                    text = pdf_block["text"]
                    bbox = pdf_block["bbox"]

                    # Skip short or empty text
                    if len(text.strip()) < self.TEXT_SUPPLEMENT_MIN_LENGTH:
                        continue

                    # Skip if already covered by existing blocks (text similarity)
                    if self._is_covered_by_blocks(bbox, text, page_blocks):
                        continue

                    # Skip if text appears inside a TABLE block (avoid extracting table content as text)
                    if self._is_inside_table_content(text, page_blocks):
                        continue

                    # Create supplemented block
                    block = Block(
                        block_id=f"s{next_block_idx:06d}",  # "s" prefix for supplement
                        type=BlockType.TEXT,
                        page_idx=page_idx,
                        bbox_norm=[int(v) for v in bbox],
                        reading_order=next_block_idx,
                        payload={
                            "text": text.strip(),
                            "text_level": 0,
                            "origin": "pymupdf_supplement",
                        },
                    )
                    supplemented.append(block)
                    next_block_idx += 1

        finally:
            doc.close()

        if supplemented:
            # Remove cross-page duplicates (repeated headers/footers)
            supplemented = self._remove_cross_page_duplicates(supplemented)

            # Merge and re-sort all blocks
            all_blocks = blocks + supplemented
            all_blocks.sort(key=lambda b: (b.page_idx, b.bbox_norm[1] if b.bbox_norm else 0))

            # Update reading_order based on new sort
            for i, block in enumerate(all_blocks):
                block.reading_order = i

            return all_blocks, len(supplemented)

        return blocks, 0

    def _remove_cross_page_duplicates(self, blocks: list[Block]) -> list[Block]:
        """
        Remove text blocks that appear on multiple pages (likely headers/footers).

        If the same text appears on 3+ pages, keep only the first occurrence.
        """
        # Count text occurrences across pages
        text_pages: dict[str, list[int]] = {}
        for block in blocks:
            text = (block.get_text() or "").strip().replace(" ", "").replace("\n", "")
            if len(text) < 4:
                continue
            if text not in text_pages:
                text_pages[text] = []
            text_pages[text].append(block.page_idx)

        # Find texts that appear on 3+ different pages (likely headers/footers)
        repeated_texts = {
            text for text, pages in text_pages.items()
            if len(set(pages)) >= 3
        }

        # Filter: keep only first occurrence of repeated texts
        seen_repeated: set[str] = set()
        result = []

        for block in blocks:
            text = (block.get_text() or "").strip().replace(" ", "").replace("\n", "")
            if text in repeated_texts:
                if text in seen_repeated:
                    continue  # Skip duplicate
                seen_repeated.add(text)
            result.append(block)

        return result

    def _find_pdf_path(self, doc_id: str, content_list_path: Path) -> Path | None:
        """Find PDF path for text extraction."""
        # Try source PDF first
        source_dir = settings.get_doc_path(doc_id) / "source"
        for f in source_dir.glob("original.*"):
            if f.suffix.lower() == ".pdf":
                return f

        # Fallback: MinerU-generated origin PDF (for DOCX/DOC)
        if content_list_path:
            origin_pdf = content_list_path.parent / "original_origin.pdf"
            if origin_pdf.exists():
                return origin_pdf

            # Also try layout PDF
            layout_pdf = content_list_path.parent / "original_layout.pdf"
            if layout_pdf.exists():
                return layout_pdf

        return None

    def _extract_pdf_text_blocks(self, page: Any) -> list[dict[str, Any]]:
        """
        Extract text blocks from a PDF page using PyMuPDF.

        Returns list of dicts with 'text' and 'bbox' keys.
        """
        result = []

        try:
            # Get text blocks in dict format
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") != 0:  # 0 = text block
                    continue

                bbox = block.get("bbox", [0, 0, 0, 0])
                lines = block.get("lines", [])

                # Concatenate all text in the block
                text_parts = []
                for line in lines:
                    for span in line.get("spans", []):
                        text_parts.append(span.get("text", ""))

                text = "".join(text_parts)
                if text.strip():
                    result.append({
                        "text": text,
                        "bbox": list(bbox),
                    })

        except Exception:
            pass

        return result

    def _is_inside_table_content(self, text: str, blocks: list[Block]) -> bool:
        """
        Check if text appears inside a TABLE block's content.

        This prevents extracting table cell content as separate text blocks,
        which would cause duplication with the structured TABLE output.

        Only matches if the ENTIRE text is found within table content,
        to avoid false positives from partial keyword matches.
        """
        text_clean = (text or "").strip().replace(" ", "").replace("\n", "")
        if len(text_clean) < 6:
            return False

        for block in blocks:
            if block.type != BlockType.TABLE:
                continue

            table_body = block.payload.get("table_body", "")
            if not table_body:
                continue

            # Clean table body for comparison
            table_clean = table_body.replace(" ", "").replace("\n", "")
            # Remove HTML tags for text matching
            table_text = re.sub(r'<[^>]+>', '', table_clean)

            # Check if the ENTIRE text appears in the table content
            if text_clean in table_text:
                return True

        return False

    def _is_covered_by_blocks(
        self,
        target_bbox: list[float],
        target_text: str,
        blocks: list[Block],
    ) -> bool:
        """
        Check if target text is already covered by existing blocks.

        Uses text content matching only - bbox overlap is unreliable
        due to MinerU bbox inaccuracy issues.

        Returns True if covered (should skip).
        """
        target_text_clean = (target_text or "").strip().replace(" ", "").replace("\n", "")
        if not target_text_clean or len(target_text_clean) < 4:
            return True  # Empty or too short text, skip

        # Check if text content already exists in ANY block
        for block in blocks:
            block_text = (block.get_text() or "").strip().replace(" ", "").replace("\n", "")
            if self._texts_overlap(target_text_clean, block_text):
                return True

        return False

    def _texts_overlap(self, text1: str, text2: str) -> bool:
        """
        Check if two texts have significant overlap.

        Returns True if:
        - One is substring of another
        - They share >60% common content
        """
        if not text1 or not text2:
            return False

        # Direct substring check
        if text1 in text2 or text2 in text1:
            return True

        # Check common content ratio
        shorter = text1 if len(text1) <= len(text2) else text2
        longer = text2 if len(text1) <= len(text2) else text1

        # For short texts, require exact match or substring
        if len(shorter) < 10:
            return shorter in longer

        # For longer texts, check if 60% of shorter text appears in longer
        match_count = 0
        for i in range(len(shorter) - 3):
            if shorter[i:i+4] in longer:
                match_count += 1

        if len(shorter) > 3:
            match_ratio = match_count / (len(shorter) - 3)
            if match_ratio > 0.6:
                return True

        return False

    async def _render_and_enrich_pages(
        self,
        doc_id: str,
        pages: list[PageInfo],
        output_dir: Path,
        content_list_path: Path | None = None,
        source_info: SourceInfo | None = None,
    ) -> list[PageInfo]:
        """
        Render document pages to images and enrich page info with dimensions.

        PDF sources are rendered directly. XLS/XLSX sources are first converted
        to a temporary PDF with LibreOffice so Viewer/document management can
        still show a page preview for native spreadsheet parses.
        """
        if not HAS_PYMUPDF:
            return pages

        if source_info and self._is_spreadsheet_source(source_info):
            return await self._render_spreadsheet_preview_pages(
                doc_id=doc_id,
                pages=pages,
                output_dir=output_dir,
                source_info=source_info,
            )

        pdf_path = self._find_render_pdf_path(doc_id, content_list_path)
        if not pdf_path:
            return pages

        return self._render_pdf_pages(pdf_path=pdf_path, pages=pages, output_dir=output_dir)

    def _find_render_pdf_path(self, doc_id: str, content_list_path: Path | None = None) -> Path | None:
        """Find a PDF suitable for page image rendering."""
        source_dir = settings.get_doc_path(doc_id) / "source"
        if source_dir.exists():
            for f in source_dir.glob("original.*"):
                if f.suffix.lower() == ".pdf":
                    return f

        # Fallback: MinerU/LibreOffice layout PDF for non-PDF sources.
        if content_list_path:
            layout_pdf = content_list_path.parent / "original_layout.pdf"
            if layout_pdf.exists():
                return layout_pdf

        return None

    async def _render_spreadsheet_preview_pages(
        self,
        doc_id: str,
        pages: list[PageInfo],
        output_dir: Path,
        source_info: SourceInfo,
    ) -> list[PageInfo]:
        """Convert a spreadsheet source to PDF, then render preview page images."""
        source_path = self._resolve_source_path(doc_id, source_info)
        if not source_path or not source_path.exists():
            return pages

        convert_dir = output_dir.parent / "spreadsheet_pdf"
        pdf_path, error = await self._convert_spreadsheet_to_pdf(source_path, convert_dir)
        if error or not pdf_path:
            shutil.rmtree(convert_dir, ignore_errors=True)
            return pages

        try:
            return self._render_pdf_pages(pdf_path=pdf_path, pages=pages, output_dir=output_dir)
        finally:
            shutil.rmtree(convert_dir, ignore_errors=True)

    def _render_pdf_pages(
        self,
        pdf_path: Path,
        pages: list[PageInfo],
        output_dir: Path,
    ) -> list[PageInfo]:
        """Render selected PDF pages to run-local page image assets."""
        output_dir.mkdir(parents=True, exist_ok=True)
        enriched = []

        try:
            doc = fitz.open(pdf_path)

            for page in pages:
                if page.page_idx >= len(doc):
                    enriched.append(page)
                    continue

                pdf_page = doc[page.page_idx]

                zoom = self.PAGE_RENDER_DPI / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = pdf_page.get_pixmap(matrix=mat)

                output_path = output_dir / f"p{page.page_idx:04d}.{self.PAGE_RENDER_FORMAT}"
                pix.save(output_path)

                enriched.append(
                    PageInfo(
                        page_idx=page.page_idx,
                        width_px=pix.width,
                        height_px=pix.height,
                        page_image_path=f"assets/pages/p{page.page_idx:04d}.{self.PAGE_RENDER_FORMAT}",
                    )
                )

            doc.close()
        except Exception:
            return pages

        return enriched

    async def _convert_spreadsheet_to_pdf(
        self,
        source_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, str | None]:
        """Convert XLS/XLSX to a temporary PDF for visual preview rendering."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which("libreoffice"):
            return None, "LibreOffice not installed"

        profile_dir = output_dir / "lo_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        try:
            proc = await asyncio.create_subprocess_exec(
                "libreoffice",
                "--headless",
                "--norestore",
                "--nolockcheck",
                f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
                "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(source_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                if not error_msg:
                    error_msg = stdout.decode("utf-8", errors="replace").strip()
                return None, error_msg or "LibreOffice conversion failed"

            expected = output_dir / f"{source_path.stem}.pdf"
            if expected.exists():
                return expected, None

            pdfs = sorted(output_dir.glob("*.pdf"))
            if pdfs:
                return pdfs[0], None

            return None, "PDF not generated after conversion"
        except Exception as exc:
            return None, str(exc)

    def _resolve_source_path(self, doc_id: str, source_info: SourceInfo) -> Path | None:
        """Resolve the original source file path from SourceInfo or stored document source."""
        candidates = []
        if source_info.path:
            candidates.append(Path(source_info.path))

        source_dir = settings.get_doc_path(doc_id) / "source"
        candidates.extend(source_dir.glob("original.*"))

        for candidate in candidates:
            if candidate.exists() and candidate.suffix.lower() in SPREADSHEET_NATIVE_EXTENSIONS:
                return candidate

        return None

    def _is_spreadsheet_source(self, source_info: SourceInfo) -> bool:
        """Return True when the source is a spreadsheet handled by the native parser."""
        ext = (source_info.ext or "").lower()
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        if ext in SPREADSHEET_NATIVE_EXTENSIONS:
            return True
        return Path(source_info.path or "").suffix.lower() in SPREADSHEET_NATIVE_EXTENSIONS

    async def _enrich_page_info(
        self,
        doc_id: str,
        run_id: str,
        pages: list[PageInfo],
        content_list_path: Path,
    ) -> list[PageInfo]:
        """Try to get page dimensions from MinerU output images (fallback)."""
        # Look for page images in the MinerU output
        images_dir = content_list_path.parent / "images"

        if not images_dir.exists():
            return pages

        enriched = []
        for page in pages:
            page_image = None

            # Look for page image (various naming conventions)
            for pattern in [
                f"page_{page.page_idx:04d}.png",
                f"page_{page.page_idx}.png",
                f"p{page.page_idx:04d}.png",
                f"{page.page_idx}.png",
            ]:
                candidate = images_dir / pattern
                if candidate.exists():
                    page_image = candidate
                    break

            if page_image:
                try:
                    with Image.open(page_image) as img:
                        enriched.append(
                            PageInfo(
                                page_idx=page.page_idx,
                                width_px=img.width,
                                height_px=img.height,
                                page_image_path=str(page_image.relative_to(content_list_path.parent)),
                            )
                        )
                        continue
                except Exception:
                    pass

            enriched.append(page)

        return enriched


def save_document_ir(document_ir: DocumentIR, run_path: Path) -> Path:
    """Save DocumentIR to run output directory."""
    ir_path = run_path / "document_ir.json"
    ir_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ir_path, "w", encoding="utf-8") as f:
        json.dump(document_ir.to_dict(), f, ensure_ascii=False, indent=2)

    return ir_path


def load_document_ir(run_path: Path) -> DocumentIR | None:
    """Load DocumentIR from run output directory."""
    ir_path = run_path / "document_ir.json"

    if not ir_path.exists():
        return None

    with open(ir_path, encoding="utf-8") as f:
        data = json.load(f)

    return DocumentIR.from_dict(data)
