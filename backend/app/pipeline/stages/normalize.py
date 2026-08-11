"""
Normalize stage - Build DocumentIR from MinerU output.

Responsibilities:
- Parse MinerU content_list.json into DocumentIR
- Render PDF pages to assets/pages/ (for VLM context and Viewer)
- Build page info with dimensions
"""

import asyncio
import difflib
import json
import re
import shutil
from collections.abc import Sequence
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
from app.pipeline.corpus_rules import DEFAULT_RULESET_PATH, CorpusRules
from app.pipeline.corpus_rules import get_rules as get_corpus_rules
from app.pipeline.zh_text import fix_mainland_vocab, to_taiwan_traditional
from app.supported_files import SPREADSHEET_NATIVE_EXTENSIONS

# MinerU page furniture types (running headers/footers, page numbers, margin
# notes). Kept in the IR as tagged TEXT blocks; filtering is a downstream
# (package-stage) decision.
PAGE_FURNITURE_TYPES = frozenset({"header", "footer", "page_number", "page_footnote", "aside_text"})

# A URL MinerU re-wrapped with a stray space (…qfil e&hl=en). The fragment
# after the space must itself look like a URL tail (query/param characters)
# so prose following a complete URL is never glued on.
_BROKEN_URL_RE = re.compile(
    r"(https?://[^\s]+)[ \t]+([A-Za-z0-9][A-Za-z0-9._/#-]*[&=?%][A-Za-z0-9&=?%._/#-]*)"
)


def _rejoin_broken_urls(text: str) -> str:
    if not text or "http" not in text:
        return text
    previous = None
    while previous != text:
        previous = text
        text = _BROKEN_URL_RE.sub(r"\1\2", text)
    return text


# Screenshot OCR that MinerU promotes to headings: bare number groups (a TOTP
# code 「380 671」) and push-notification account lines (「QNAP QTS:
# DOMAIN\xxx@Host-1」). The text is kept as body content — only the heading
# level is dropped, so it stays out of every chunk's heading_path.
_PURE_NUMERIC_HEADING_RE = re.compile(r"[\d\s\-:：./]+")

# A bare folio in the page margin (「21」, 「220」). Anchored full-match only —
# a year inside body prose must never qualify.
_PAGE_NUMBER_RE = re.compile(r"\d{1,4}")

# CJK and full-width punctuation: a line break inside this script carries no
# space, while Latin prose wraps on one.
_HAN_OR_FULLWIDTH_RE = re.compile(r"[一-鿿぀-ヿ？-ﾟ！-｠]")

# Han only. Used to tell an authored fragment from logo/watermark glyphs that
# the OCR swept into a title box; these journals write titles in Chinese, so a
# purely Latin/numeric fragment the running head omits is page decoration.
_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# A space the OCR invented between two Han characters. Advert copy is set in
# small display type and comes back shredded (live: 「本 書分別從政策面」,
# 「本 書 引1 別從政 策 面」), so the advert patterns are matched against a
# view with those gaps closed. Only Han-Han gaps close, so Latin words
# ("TAIWAN HYDROGEN") keep their real spacing.
_INTER_HAN_SPACE_RE = re.compile(r"(?<=[㐀-䶿一-鿿豈-﫿])\s+(?=[㐀-䶿一-鿿豈-﫿])")

_PROMOTIONAL_INSERT_MARKER = "promotional_insert_patterns"


def _compile_patterns(sources: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for source in sources:
        try:
            compiled.append(re.compile(str(source)))
        except re.error:
            continue
    return tuple(compiled)


_promotional_insert_cache: tuple[tuple[str, ...], tuple[re.Pattern[str], ...]] | None = None
_bundled_promotional_insert_sources: tuple[str, ...] | None = None


def _bundled_promotional_insert_patterns() -> tuple[str, ...]:
    global _bundled_promotional_insert_sources
    if _bundled_promotional_insert_sources is None:
        bundled = CorpusRules.from_dict(
            json.loads(DEFAULT_RULESET_PATH.read_text(encoding="utf-8"))
        )
        _bundled_promotional_insert_sources = tuple(
            bundled.marker_list(_PROMOTIONAL_INSERT_MARKER)
        )
    return _bundled_promotional_insert_sources


def _promotional_insert_patterns() -> tuple[re.Pattern[str], ...]:
    """Advert signal regexes, one per *independent* signal, from the ruleset.

    The rules are data (``document_markers.promotional_insert_patterns``), not
    code, so a different publication can retune them without a release. Two
    of them must fire before a column is dropped, which is why each entry is
    one semantic family — a hotline and an online-order line are separate
    entries because a page carrying both is unambiguously selling something.

    A corpus ruleset that predates the key falls back to the bundled default:
    ``rulesets/local.json`` replaces the whole ruleset rather than layering on
    it, so keying off "absent" instead of "empty" is what stops the fix from
    silently becoming a no-op on the corpus it was written for. An explicit
    ``[]`` still turns the pass off.
    """
    global _promotional_insert_cache
    markers = get_corpus_rules().document_markers
    if _PROMOTIONAL_INSERT_MARKER in markers:
        sources = tuple(markers[_PROMOTIONAL_INSERT_MARKER])
    else:
        sources = _bundled_promotional_insert_patterns()
    if _promotional_insert_cache is None or _promotional_insert_cache[0] != sources:
        _promotional_insert_cache = (sources, _compile_patterns(sources))
    return _promotional_insert_cache[1]


def _compact_with_offsets(raw: str) -> tuple[str, list[int]]:
    """Whitespace-free view of ``raw`` plus each kept character's original offset.

    Comparison has to ignore whitespace — the vertical column break arrives as
    a stray space (「我國頻譜 使用現況」) — while the repaired string must be
    emitted from the *original* text so real spacing (「Think Global」, the gap
    after a 眉題) survives.
    """
    chars: list[str] = []
    offsets: list[int] = []
    for offset, char in enumerate(raw):
        if not char.isspace():
            chars.append(char)
            offsets.append(offset)
    return "".join(chars), offsets


def _matched_char_count(left: str, right: str) -> int:
    """Characters the two strings agree on, in order."""
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def _normalize_for_coverage(text: str) -> str:
    """Whitespace- and escape-insensitive form for duplicate detection.

    MinerU escapes markdown metacharacters in its text layer (``1\\~10月``)
    while PyMuPDF returns the raw glyphs (``1~10月``). Comparing the two
    verbatim cost four 4-grams and dropped a genuine duplicate to 0.59 against
    a 0.60 threshold, so the fragment was re-added as an orphan line.
    """
    return re.sub(r"[\s\\]+", "", str(text or ""))


# Every payload field that carries authored prose. Captions and footnotes
# arrive as lists of strings from MinerU; table_body is HTML, and the converter
# is character-level, so the markup passes through untouched.
_ZH_TEXT_PAYLOAD_KEYS = (
    "text",
    "table_body",
    "table_caption",
    "table_footnote",
    "caption",
    "footnote",
    "chart_content",
)


def _convert_payload_value(value: Any) -> Any:
    if isinstance(value, str):
        return fix_mainland_vocab(to_taiwan_traditional(value))
    if isinstance(value, list):
        return [_convert_payload_value(item) for item in value]
    return value


def _bbox_containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Share of ``inner``'s area that lies inside ``outer`` (same coordinate space)."""
    if len(inner) < 4 or len(outer) < 4:
        return 0.0
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if area <= 0:
        return 0.0
    width = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    height = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (width * height) / area


def _join_text_lines(left: str, right: str) -> str:
    """Join two printed lines: CJK wraps without a separator, Latin needs one."""
    if not left:
        return right
    if not right:
        return left
    if _HAN_OR_FULLWIDTH_RE.match(left[-1]) or _HAN_OR_FULLWIDTH_RE.match(right[0]):
        return left + right
    return f"{left} {right}"


def _block_top(block: Block) -> int:
    """Vertical position of a block on its page, 0 when unknown."""
    bbox = block.bbox_norm or []
    return int(bbox[1]) if len(bbox) >= 2 else 0
_ACCOUNT_NOTIFICATION_RE = re.compile(r"[A-Za-z0-9]\\[A-Za-z0-9]")


def _is_ocr_noise_heading(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    if _PURE_NUMERIC_HEADING_RE.fullmatch(compact):
        return True
    return "@" in compact and bool(_ACCOUNT_NOTIFICATION_RE.search(compact))


def is_page_furniture(block: Block) -> bool:
    """Return True when a block originated from MinerU page furniture."""
    return block.payload.get("origin") == "page_furniture"


def is_promotional_insert(block: Block) -> bool:
    """Return True when a block belongs to a publisher's bound-in advert."""
    return block.payload.get("origin") == "promotional_insert"


def is_non_content(block: Block) -> bool:
    """True for every block the delivery surfaces drop as non-authored matter.

    One predicate so a new origin cannot be honoured by rag.md but forgotten
    by the chunker or — the expensive mistake — by the completeness gate,
    which would then read the deliberate drop as silent data loss.
    """
    return is_page_furniture(block) or is_promotional_insert(block)


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
    PAGE_RENDER_DPI = 200  # Match MinerU's internal render; feeds VLM crops and viewer
    PAGE_RENDER_FORMAT = "png"

    # Text supplement settings
    TEXT_SUPPLEMENT_MIN_LENGTH = 4  # Minimum text length to consider
    TEXT_SUPPLEMENT_COVERAGE_THRESHOLD = 0.3  # Min coverage to consider "covered"

    # How far a supplement may look for the block that already carries it.
    # One page covers a paragraph running over a page break; wider would let
    # repeated boilerplate mask a genuine gap.
    COVERAGE_PAGE_RADIUS = 1
    COVERAGE_GRAM_RATIO = 0.6  # share of the fragment's 4-grams the block must hold
    COVERAGE_CONTAINMENT = 0.9  # share of the candidate's box inside an existing block

    # Line merging for PyMuPDF supplements. The extractor hands back one entry
    # per printed line for these journals, so unmerged supplements arrive as
    # 17-character fragments instead of paragraphs.
    LINE_MERGE_X_OVERLAP = 0.6  # share of the narrower line's width that must align
    LINE_MERGE_GAP_RATIO = 1.2  # vertical gap, as a multiple of the line height

    # Page-furniture detection (bbox_norm is a 0-1000 space)
    FURNITURE_TOP_BAND = 140  # y1 at or above this is the page head strip
    FURNITURE_BOTTOM_BAND = 900  # y0 at or below this is the page foot strip
    FURNITURE_MAX_CHARS = 40  # a running head is a label, never a paragraph
    FURNITURE_MIN_PAGES = 2  # must recur, so one-off footers (DOI) survive

    # Column split for the bound-in-advert pass. Derived per document from the
    # vertical strip fewest blocks cross (the printed gutter), never from a
    # constant: x≈520 is what *this* corpus happens to measure, and a journal
    # with a different page grid would have every column call inverted by a
    # hardcoded value. Live over 167 documents the derived split lands in
    # 480-544, i.e. it reproduces the measured gutter from the geometry alone.
    COLUMN_SPLIT_SEARCH_LO = 0.25  # ignore the outer quarters: no gutter there
    COLUMN_SPLIT_SEARCH_HI = 0.75
    COLUMN_SPLIT_STEP = 5  # 0-1000 space, so 0.5% of the page width
    COLUMN_SPLIT_MIN_SPAN = 200  # a narrower content strip has no two columns
    COLUMN_SPLIT_MIN_BOXES = 8  # too few boxes and the "gutter" is just a gap

    # Bound-in advert detection.
    PROMO_MIN_BLOCKS = 4  # 3 blocks is a figure with a caption, not an advert
    PROMO_MIN_SIGNALS = 2  # one commerce word can occur in prose; two cannot

    # Rebuilding an OCR-damaged heading from its running head.
    # Over the 167-document store the "share of the shorter string the two
    # agree on" score is bimodal with nothing at all in between: every genuine
    # title/running-head pair scores 0.909-1.0, every coincidental one (the
    # journal name, a DOI, a folio) scores 0.5 or less. 0.75 sits in the empty
    # band, so the threshold is not tuned to either tail.
    TITLE_REPAIR_MIN_CONTAINMENT = 0.75
    # Two characters agreeing is coincidence, not a title: 「20」 in the margin
    # scored a perfect containment against a heading that contained a 2050.
    TITLE_REPAIR_MIN_MATCH_CHARS = 6
    # Context around a disagreement when asking the body which spelling it uses.
    # One character each side is enough to make 「淨零」/「浮零」 decisive while
    # still being short enough to actually occur in the prose.
    TITLE_REPAIR_CONTEXT_CHARS = 1
    # A disagreement at the very start or end of a title is ambiguous: it is
    # either a misread glyph (live: a title opening on 「至球」 where the head
    # opens on 「全球」) or two different pieces of text (a 眉題 against a
    # series label). A misread is a character or two; anything longer is
    # treated as authored text and left with the heading.
    TITLE_REPAIR_BOUNDARY_EDIT_CHARS = 3

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

            # Tag the running heads/feet MinerU typed as ordinary text, so the
            # delivery surfaces can drop them the same way they drop the
            # furniture MinerU does label.
            blocks = self._tag_layout_furniture(blocks, page_count=len(page_indices))

            # Now that the running heads are identified, use them to repair the
            # heading the vertical-title OCR damaged. Must follow the tagging
            # (it looks the heads up by tag) and precede the zh-TW pass, so
            # whatever the head contributes is normalized like everything else.
            blocks = self._repair_headings_from_running_heads(blocks)

            # Bring MinerU's OCR glyphs to zh-TW. Runs after every pass that
            # compares text with text, so the duplicate detection above
            # compared the parser's output with itself.
            blocks = self._normalize_zh_text(blocks)

            # Tag the publisher's advert bound into the last page's second
            # column. Deliberately *after* the zh-TW pass: its signals are
            # written in zh-TW and MinerU's OCR emits mainland glyphs
            # sporadically (税/质/氢 — 849 across 88 documents on 2026-08-10),
            # so matching earlier would miss the ones it mangled. It must also
            # follow the heading repair, which looks running heads up by tag
            # and would lose any head this pass re-tagged.
            blocks = self._tag_promotional_inserts(blocks)

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

        # Map MinerU types to BlockType.
        # MinerU emits code/list content as plain text (code_body/list_items),
        # and charts are image regions with optional extracted data text.
        type_map = {
            "text": BlockType.TEXT,
            "table": BlockType.TABLE,
            "image": BlockType.IMAGE,
            "equation": BlockType.EQUATION,
            "code": BlockType.TEXT,
            "list": BlockType.TEXT,
            "chart": BlockType.IMAGE,
        }
        type_map.update(dict.fromkeys(PAGE_FURNITURE_TYPES, BlockType.TEXT))

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
        item_type = item.get("type", "")

        if block_type == BlockType.TEXT:
            if item_type == "code":
                # MinerU 3.x puts code content under code_body
                payload = {
                    "text": item.get("code_body", item.get("code", item.get("text", ""))),
                    "text_level": 0,
                    "origin": "code",
                }
                if item.get("code_language"):
                    payload["code_language"] = item["code_language"]
                return payload
            if item_type == "list":
                # MinerU 3.x puts list content under list_items
                items = item.get("list_items") or item.get("items") or []
                text = "\n".join(items) if items else item.get("text", "")
                return {
                    "text": text,
                    "text_level": 0,
                    "origin": "list",
                }
            text = _rejoin_broken_urls(item.get("text", ""))
            text_level = item.get("text_level", 0)
            if text_level and _is_ocr_noise_heading(text):
                text_level = 0
            payload = {
                "text": text,
                "text_level": text_level,
            }
            if item_type in PAGE_FURNITURE_TYPES:
                payload["origin"] = "page_furniture"
            return payload
        elif block_type == BlockType.IMAGE:
            if item_type == "chart":
                payload = {
                    "img_path": item.get("img_path", ""),
                    "caption": item.get("chart_caption"),
                    "footnote": item.get("chart_footnote"),
                    "origin": "chart",
                }
                # Chart data text extracted by MinerU (may be empty)
                if item.get("content"):
                    payload["chart_content"] = item["content"]
                return payload
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
            payload = {
                "table_body": item.get("table_body", ""),
                "table_caption": item.get("table_caption"),
                # Slide reminders printed under a table arrive in
                # table_footnote, not as a separate text block — dropping the
                # field silently loses authored content.
                "table_footnote": item.get("table_footnote"),
            }
            # Keep the table crop so the parsed HTML can be verified against it
            if item.get("img_path"):
                payload["img_path"] = item["img_path"]
            return payload
        elif block_type == BlockType.EQUATION:
            return {
                "latex": item.get("latex", item.get("text", "")),
                "equation_type": item.get("equation_type"),
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

        Priority: TABLE > IMAGE > others
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
                handled_overlap = False
                for kept_block in kept:
                    if self._compute_iou(block.bbox_norm, kept_block.bbox_norm) > 0.9:
                        # Overlap detected - decide which to keep
                        # Priority: TABLE > IMAGE > others
                        block_priority = self._get_block_priority(block.type)
                        kept_priority = self._get_block_priority(kept_block.type)

                        if block_priority > kept_priority:
                            # Replace kept block with current block
                            kept.remove(kept_block)
                            removed_ids.add(kept_block.block_id)
                            kept.append(block)
                        else:
                            # Keep existing, drop current as duplicate
                            removed_ids.add(block.block_id)
                        handled_overlap = True
                        break

                if not handled_overlap:
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
        # TABLE outranks IMAGE: the parsed HTML carries the structure while an
        # overlapping image is just a raster crop of the same region.
        priority_map = {
            BlockType.TABLE: 10,
            BlockType.IMAGE: 5,
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

                page_width = float(page.rect.width)
                page_height = float(page.rect.height)
                if page_width <= 0 or page_height <= 0:
                    continue

                # Extract text blocks from PDF, rebuilding paragraphs from the
                # per-line entries the extractor returns.
                pdf_text_blocks = self._merge_adjacent_pdf_lines(
                    self._extract_pdf_text_blocks(page), page_height=page_height
                )

                # Find uncovered text blocks
                for pdf_block in pdf_text_blocks:
                    text = pdf_block["text"]
                    bbox = pdf_block["bbox"]

                    # Skip short or empty text
                    if len(text.strip()) < self.TEXT_SUPPLEMENT_MIN_LENGTH:
                        continue

                    # Convert PyMuPDF point coords to MinerU's 0-1000 space so
                    # sorting, coverage geometry and enrich crops share one
                    # coordinate system
                    bbox_norm = [
                        max(0, min(1000, int(bbox[0] * 1000 / page_width))),
                        max(0, min(1000, int(bbox[1] * 1000 / page_height))),
                        max(0, min(1000, int(bbox[2] * 1000 / page_width))),
                        max(0, min(1000, int(bbox[3] * 1000 / page_height))),
                    ]

                    # Skip if already covered by existing blocks (text or geometry)
                    if self._is_covered_by_blocks(bbox_norm, text, blocks, page_idx):
                        continue

                    # Skip if text appears inside a TABLE block (avoid extracting table content as text)
                    if self._is_inside_table_content(text, page_blocks):
                        continue

                    # Create supplemented block
                    block = Block(
                        block_id=f"s{next_block_idx:06d}",  # "s" prefix for supplement
                        type=BlockType.TEXT,
                        page_idx=page_idx,
                        bbox_norm=bbox_norm,
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

            all_blocks = self._merge_supplements_in_order(blocks, supplemented)

            return all_blocks, len(supplemented)

        return blocks, 0

    def _merge_supplements_in_order(
        self,
        blocks: list[Block],
        supplemented: list[Block],
    ) -> list[Block]:
        """Place supplements on their page without disturbing MinerU's order.

        The previous merge re-sorted *every* block by vertical position alone.
        In a two-column layout that interleaves the columns: live in
        ``a273c9e754b4a257``, 「地位。然而…」 (right column, y=597) was
        delivered ahead of the sentence it continues, 「回顧2025年…」 (left
        column, y=598), and section bodies were shuffled across headings.
        MinerU already threads the columns correctly, so its order is
        authoritative; each supplement is simply attached after the last block
        at or above it on the same page.
        """
        pending: dict[int, list[Block]] = {}
        for block in supplemented:
            pending.setdefault(block.page_idx, []).append(block)
        for page_supplements in pending.values():
            page_supplements.sort(key=_block_top)

        blocks_by_page: dict[int, list[Block]] = {}
        for block in blocks:
            blocks_by_page.setdefault(block.page_idx, []).append(block)

        merged: list[Block] = []
        for page_idx in sorted(set(blocks_by_page) | set(pending)):
            page_existing = blocks_by_page.get(page_idx, [])
            page_supplements = pending.get(page_idx, [])
            if not page_existing:
                merged.extend(page_supplements)
                continue

            leading: list[Block] = []
            attached: dict[int, list[Block]] = {}
            for supplement in page_supplements:
                anchor: int | None = None
                for idx, existing in enumerate(page_existing):
                    if _block_top(existing) <= _block_top(supplement):
                        anchor = idx
                if anchor is None:
                    leading.append(supplement)
                else:
                    attached.setdefault(anchor, []).append(supplement)

            merged.extend(leading)
            for idx, existing in enumerate(page_existing):
                merged.append(existing)
                merged.extend(attached.get(idx, []))

        for order, block in enumerate(merged):
            block.reading_order = order
        return merged

    def _normalize_zh_text(self, blocks: list[Block]) -> list[Block]:
        """Bring MinerU's text layer to zh-TW.

        The converter used to be reachable only through ``render_vlm_text``,
        so it saw the model's prose and never the parser's. MinerU's OCR
        misreads individual glyphs (稅→税, 脫→脱, 質→质, 氫→氢) and every one
        of them went out untouched: 849 across 88 of the store's 100 documents
        on 2026-08-10.

        Runs after supplementing, so the duplicate detection upstream compares
        like with like, and over every text-bearing payload field rather than
        ``text`` alone — table cells and captions carry the same misreads.
        """
        for block in blocks:
            for key in _ZH_TEXT_PAYLOAD_KEYS:
                if key in block.payload:
                    block.payload[key] = _convert_payload_value(block.payload[key])
        return blocks

    def _in_margin_band(self, block: Block) -> bool:
        """True when the block sits in the page's head or foot strip."""
        bbox = block.bbox_norm or []
        if len(bbox) < 4:
            return False
        return bbox[3] <= self.FURNITURE_TOP_BAND or bbox[1] >= self.FURNITURE_BOTTOM_BAND

    def _tag_layout_furniture(self, blocks: list[Block], page_count: int) -> list[Block]:
        """Tag running heads/feet and page numbers MinerU typed as plain text.

        MinerU only labels a fraction of the furniture (``header``/``footer``/
        ``page_number``); a journal's running head and volume line usually
        arrive as ordinary ``text``. Live evidence (2026-08-10, 100-document
        store): 1,826 such lines reached rag.md as body paragraphs and 851 of
        1,843 chunks contained at least one.

        Two signals, both requiring the block to sit in a margin band, so the
        opening page's real title — the same string, but in the body — is never
        touched:

        * the same short string recurs in the margin of several pages;
        * bare numbers occupy the margin on several pages (values differ, the
          position is what repeats).
        """
        if page_count < self.FURNITURE_MIN_PAGES:
            return blocks

        candidates: list[tuple[Block, str]] = []
        for block in blocks:
            if block.type != BlockType.TEXT or is_page_furniture(block):
                continue
            if not self._in_margin_band(block):
                continue
            compact = re.sub(r"\s+", "", str(block.payload.get("text") or ""))
            if compact and len(compact) <= self.FURNITURE_MAX_CHARS:
                candidates.append((block, compact))

        pages_by_text: dict[str, set[int]] = {}
        numbered_pages: set[int] = set()
        for block, compact in candidates:
            pages_by_text.setdefault(compact, set()).add(block.page_idx)
            if _PAGE_NUMBER_RE.fullmatch(compact):
                numbered_pages.add(block.page_idx)

        numbers_repeat = len(numbered_pages) >= self.FURNITURE_MIN_PAGES
        for block, compact in candidates:
            repeats = len(pages_by_text[compact]) >= self.FURNITURE_MIN_PAGES
            if repeats or (numbers_repeat and _PAGE_NUMBER_RE.fullmatch(compact)):
                block.payload["origin"] = "page_furniture"
        return blocks

    def _repair_headings_from_running_heads(self, blocks: list[Block]) -> list[Block]:
        """Rebuild a heading the vertical-title OCR damaged, using the running head.

        These journals set the cover title *vertically*. MinerU reads a
        vertical column glyph by glyph and drops or substitutes characters —
        live (2026-08-10, 167-document store): 「AI」 came back as 「A」, 「6G」
        as 「G」, 「全球」 as 「至球」, 「戰略」 as 「戦略」, enumeration commas
        vanished, and lettering from the cover logo was swept into the box. The
        block carries ``text_level=1``, so the damaged string becomes the
        document's H1 and reaches every delivery surface.

        The same title is printed horizontally as a running head on the inner
        pages, comes from the PDF's text layer, and is character-perfect — but
        ``_tag_layout_furniture`` (which must run first) marks it
        ``origin="page_furniture"`` and rag.md, the chunker and the
        completeness gate all drop it. So the corpus kept the broken spelling
        and discarded the clean one in 49 of 167 documents.

        The running head is therefore the character-level authority. What it is
        *not* is an authority on which parts of the title exist, and blindly
        adopting it loses or invents text, so four guards apply:

        * a heading some running head repeats verbatim (whitespace aside) is
          never rewritten — 71 documents are already correct and must stay
          byte-identical;
        * material only the heading carries *in front* of the shared core is
          kept: that is the 眉題, a four-to-ten character kicker the running
          head routinely omits;
        * material only the running head carries is never adopted — it prefixes
          series labels (「【…篇】系列N-N」) and appends subtitles the cover
          does not print;
        * where the two disagree character for character the document's own
          body prose breaks the tie, because the running head is not always the
          right one: live, every head in one document read 「淨零」 as
          「浮零」 while the cover was correct.

        Trailing material only the heading carries is dropped unless it holds
        Han: in this corpus that tail is always logo lettering the running head
        omits, while a leading Latin label (a technology generation, a standard
        name) is authored text and is kept regardless.
        """
        heads: dict[str, dict[str, Any]] = {}
        for block in blocks:
            if block.type != BlockType.TEXT or not is_page_furniture(block):
                continue
            raw = str(block.payload.get("text") or "")
            compact = re.sub(r"\s+", "", raw)
            if len(compact) < self.TITLE_REPAIR_MIN_MATCH_CHARS:
                continue
            entry = heads.setdefault(compact, {"text": raw, "pages": set()})
            entry["pages"].add(block.page_idx)
        if not heads:
            return blocks

        headings = [
            block
            for block in blocks
            if block.type == BlockType.TEXT
            and block.payload.get("text_level") == 1
            and not is_page_furniture(block)
        ]
        if not headings:
            return blocks

        # The corroboration pool: authored prose only. Headings are excluded so
        # a damaged title can never vouch for itself, furniture because the
        # running head is the very claim under test.
        body = "".join(
            re.sub(r"\s+", "", str(block.payload.get("text") or ""))
            for block in blocks
            if block.type == BlockType.TEXT
            and not is_page_furniture(block)
            and block.payload.get("text_level") != 1
        )

        for heading in headings:
            raw = str(heading.payload.get("text") or "")
            compact = re.sub(r"\s+", "", raw)
            if len(compact) < self.TITLE_REPAIR_MIN_MATCH_CHARS or compact in heads:
                continue  # too short to judge, or a running head already confirms it

            candidates = []
            for head_compact, entry in heads.items():
                matched = _matched_char_count(compact, head_compact)
                if matched < self.TITLE_REPAIR_MIN_MATCH_CHARS:
                    continue
                containment = matched / min(len(compact), len(head_compact))
                if containment < self.TITLE_REPAIR_MIN_CONTAINMENT:
                    continue
                candidates.append((len(entry["pages"]), containment, entry["text"]))
            if not candidates:
                continue

            # Pages first: where the OCR read the same head differently on
            # different pages, the reading that recurs is the one to trust.
            head_text = max(candidates, key=lambda item: (item[0], item[1], len(item[2])))[2]
            repaired = self._merge_heading_with_running_head(raw, head_text, body)
            if repaired and re.sub(r"\s+", "", repaired) != compact:
                heading.payload["text"] = repaired
        return blocks

    def _merge_heading_with_running_head(
        self,
        heading: str,
        head_text: str,
        body: str,
    ) -> str | None:
        """Splice the running head's characters into the heading's own structure.

        Within the stretch the two strings have in common the running head
        wins, character for character — that is the whole point. What sits
        *outside* that stretch belongs to whichever string prints it: the
        heading keeps its 眉題, and the head keeps its series label to itself.
        """
        heading_compact, heading_offsets = _compact_with_offsets(heading)
        head_compact, head_offsets = _compact_with_offsets(head_text)
        matcher = difflib.SequenceMatcher(None, heading_compact, head_compact, autojunk=False)
        opcodes = list(matcher.get_opcodes())
        if not any(tag == "equal" for tag, *_ in opcodes):
            return None

        # Peel the two ends off first. An edit that only one string has
        # material for is that string's own; an edit both have material for is
        # a misread, and belongs to the core where the head has the last word.
        lead = ""
        tail = ""
        tag, i1, i2, _, _ = opcodes[0]
        role = self._boundary_edit_role(tag, i2 - i1)
        if role != "core":
            opcodes.pop(0)
            if role == "heading":
                lead = heading[: heading_offsets[i2]]
        if opcodes:
            tag, i1, i2, _, _ = opcodes[-1]
            role = self._boundary_edit_role(tag, i2 - i1)
            if role != "core":
                opcodes.pop()
                if role == "heading":
                    tail = heading[heading_offsets[i1 - 1] + 1:] if i1 else heading
        if not opcodes:
            return None

        core_start, core_end = opcodes[0][1], opcodes[-1][2]
        core: list[str] = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "delete":
                continue  # glyphs bled into the column: a folio, a watermark
            if tag == "replace" and self._body_prefers_heading_spelling(
                heading_compact[max(core_start, i1 - self.TITLE_REPAIR_CONTEXT_CHARS): i1],
                heading_compact[i2: min(core_end, i2 + self.TITLE_REPAIR_CONTEXT_CHARS)],
                heading_compact[i1:i2],
                head_compact[j1:j2],
                body,
            ):
                core.append(heading_compact[i1:i2])
                continue
            core.append(head_text[head_offsets[j1]: head_offsets[j2 - 1] + 1])

        if not _HAN_RE.search(tail):
            tail = ""  # logo lettering trailing the title column
        return (lead + "".join(core) + tail).strip() or None

    def _boundary_edit_role(self, tag: str, heading_size: int) -> str:
        """Who owns the text at one end of the alignment: the heading, the head, or both.

        ``delete`` — only the heading prints it, so it is the 眉題 or the logo
        swept in after the last line. ``insert`` — only the running head prints
        it, and the cover's own scope is what the H1 records, so it is dropped.
        ``replace`` — both print something there; short enough and it is one
        misread glyph the head can correct, longer and the two are simply
        different text and the heading's own wins.
        """
        if tag == "equal":
            return "core"
        if tag == "insert":
            return "head"
        if tag == "delete":
            return "heading"
        return "core" if heading_size <= self.TITLE_REPAIR_BOUNDARY_EDIT_CHARS else "heading"

    def _body_prefers_heading_spelling(
        self,
        left: str,
        right: str,
        heading_form: str,
        head_form: str,
        body: str,
    ) -> bool:
        """Whether the body spells this fragment the heading's way, not the head's.

        Only an unambiguous vote counts: the heading's reading has to occur in
        the prose and the running head's must not. Anything else — both occur,
        neither occurs, no context to disambiguate — leaves the running head in
        charge, which is the default this whole repair rests on. Erring towards
        the heading only ever declines a repair; erring the other way is how a
        misread running head would overwrite a title that was already right.

        The full context is tried first and each one-sided pair after it, so a
        disagreement is still decided when the prose does not happen to repeat
        the surrounding phrase verbatim: 「2050淨零」 may appear nowhere while
        「淨零」 appears throughout.
        """
        candidates = [(left, right)]
        if right:
            candidates.append(("", right))
        if left:
            candidates.append((left, ""))
        for before, after in candidates:
            heading_probe = f"{before}{heading_form}{after}"
            if len(heading_probe) < 2:
                continue
            if heading_probe in body and f"{before}{head_form}{after}" not in body:
                return True
        return False

    def _column_split_x(self, blocks: list[Block]) -> float | None:
        """The x of the printed gutter, or None when the page grid is one column.

        Found the way a reader finds it: the vertical strip that the fewest
        block boxes cross. Measured over the whole document rather than one
        page, because the grid is a property of the publication and a single
        page can be dominated by a full-width figure.

        Returns None rather than guessing when there is nothing to split (too
        few boxes, too narrow a content strip). A one-column document then
        never has a "second column" for the advert pass to act on.
        """
        boxes = [
            block.bbox_norm
            for block in blocks
            if block.bbox_norm
            and len(block.bbox_norm) >= 4
            and block.bbox_norm[2] > block.bbox_norm[0]
        ]
        if len(boxes) < self.COLUMN_SPLIT_MIN_BOXES:
            return None
        x_lo = min(box[0] for box in boxes)
        x_hi = max(box[2] for box in boxes)
        span = x_hi - x_lo
        if span < self.COLUMN_SPLIT_MIN_SPAN:
            return None
        probes = list(
            range(
                int(x_lo + self.COLUMN_SPLIT_SEARCH_LO * span),
                int(x_lo + self.COLUMN_SPLIT_SEARCH_HI * span) + 1,
                self.COLUMN_SPLIT_STEP,
            )
        )
        if not probes:
            return None
        crossings = [sum(1 for box in boxes if box[0] < x < box[2]) for x in probes]
        fewest = min(crossings)
        # The widest run at the minimum, so a one-probe notch inside a column
        # never beats the real gutter; its centre is the split.
        best_width = -1
        best_span = (probes[0], probes[0])
        index = 0
        while index < len(crossings):
            if crossings[index] != fewest:
                index += 1
                continue
            end = index
            while end + 1 < len(crossings) and crossings[end + 1] == fewest:
                end += 1
            if end - index > best_width:
                best_width = end - index
                best_span = (probes[index], probes[end])
            index = end + 1
        return (best_span[0] + best_span[1]) / 2

    def _tag_promotional_inserts(self, blocks: list[Block]) -> list[Block]:
        """Tag a publisher's advert bound into the last page's second column.

        These journals sell books on the back of the article: a title, a price
        (「售價：NT$500」), a publication month, a blurb, an ordering hotline, a
        cover shot, a QR code and partner logos, filling the second column of
        the **last** page. It is not the author's text and answers no query,
        but the whole column reached rag.md and the chunks. Live evidence
        (2026-08-11, 167-document store): 32 documents, 479 blocks, of which
        105 are images that were billed as VLM figure enrichment.

        ``_tag_layout_furniture`` cannot see it — that pass wants a short
        string in a margin band recurring across pages, and an advert is a
        page-middle block appearing once. So this is a second, region-level
        path rather than a fix to the first.

        The region is a **column**, never a page. On 141 of the 167 last pages
        the second column holds something (up to 829 characters of genuine
        text where there is no advert), and the *first* column of every advert
        page still carries the article's ■參考文獻 / ■注釋 and the tail of the
        prose — 121 to 1,469 characters measured. Dropping the page would
        delete the references in all 32 cases.

        Three conditions, all needed:

        * the last page — 0 of the 32 adverts sit anywhere else;
        * the second column, split at the derived gutter (see
          ``_column_split_x``), with page furniture left to its own tag;
        * at least ``PROMO_MIN_BLOCKS`` blocks carrying at least
          ``PROMO_MIN_SIGNALS`` *different* signals from the ruleset.

        Signals that were tried and rejected, because re-adding them is the
        obvious next idea and each one is a regression:

        * a bare ``系列\\s*\\d`` — the journal numbers its own articles
          「系列3-6」/「系列1-4」 in the running head, so it matched real
          columns;
        * "the last page is short" — 141 last pages have a populated second
          column, the largest holding 829 characters of article text;
        * "mostly pictures, little text" — that is the cover page (p0).

        None of the signals that *are* shipped is safe on its own, which is
        what ``PROMO_MIN_SIGNALS`` is for: 主編 occurs in a bibliography entry,
        「\\d折」 in prose about retail pricing, and 本書 in a book review. Two of
        them agreeing, in the second column of the last page, over at least
        four blocks, is the claim being made — not any single word.

        Measured with the shipped ruleset over all 167 documents: 32 columns
        tagged, no tagged block containing 參考文獻/注釋/References, and zero
        matches over the other 1,580 page-columns in the store.
        """
        patterns = _promotional_insert_patterns()
        if not patterns:
            return blocks
        split_x = self._column_split_x(blocks)
        if split_x is None:
            return blocks

        page_indices = [block.page_idx for block in blocks]
        if not page_indices:
            return blocks
        last_page = max(page_indices)

        region = [
            block
            for block in blocks
            if block.page_idx == last_page
            and block.bbox_norm
            and len(block.bbox_norm) >= 4
            and block.bbox_norm[0] >= split_x
            and not is_page_furniture(block)
        ]
        if len(region) < self.PROMO_MIN_BLOCKS:
            return blocks

        # One line per block, so a signal can never be spelled by two
        # neighbouring blocks running together.
        region_text = "\n".join(
            _INTER_HAN_SPACE_RE.sub("", str(block.payload.get("text") or "")) for block in region
        )
        matched = sum(1 for pattern in patterns if pattern.search(region_text))
        if matched < self.PROMO_MIN_SIGNALS:
            return blocks

        for block in region:
            block.payload["origin"] = "promotional_insert"
        return blocks

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

    def _merge_adjacent_pdf_lines(
        self,
        pdf_blocks: list[dict[str, Any]],
        page_height: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rebuild paragraphs from the per-line entries PyMuPDF returns.

        ``get_text("dict")`` groups by the PDF's own text blocks, which for
        these journals is one entry per printed line. Emitting them unmerged is
        what put 「望當前國際淨零碳排趨勢，各主要國家」 into rag.md as a
        paragraph of its own. Lines join when they share a column and sit a
        single line-height apart; a column change or a paragraph gap ends the
        run.
        """
        ordered = sorted(
            (b for b in pdf_blocks if str(b.get("text") or "").strip()),
            key=lambda b: (round(float(b["bbox"][1]), 1), float(b["bbox"][0])),
        )
        merged: list[dict[str, Any]] = []
        for candidate in ordered:
            box = [float(v) for v in candidate["bbox"]]
            if (
                merged
                and not self._in_page_margin(merged[-1]["bbox"], page_height)
                and not self._in_page_margin(box, page_height)
                and self._lines_are_continuous(merged[-1]["bbox"], box)
            ):
                previous = merged[-1]
                previous["text"] = _join_text_lines(
                    previous["text"].strip(), str(candidate["text"]).strip()
                )
                previous["bbox"] = [
                    min(previous["bbox"][0], box[0]),
                    min(previous["bbox"][1], box[1]),
                    max(previous["bbox"][2], box[2]),
                    max(previous["bbox"][3], box[3]),
                ]
                continue
            merged.append({"text": str(candidate["text"]).strip(), "bbox": box})
        return merged

    def _in_page_margin(self, bbox: Sequence[float], page_height: float | None) -> bool:
        """Whether a PyMuPDF line sits in the page head or foot strip.

        Running heads, folios and the title beneath them are vertically close
        and share a column, so pure geometry merges them into one entry — which
        then cannot be dropped as furniture because half of it is content
        (「示範5-6再生能源憑證制度之發展趨勢」). Furniture is never a
        paragraph continuation, so the margins simply do not participate.
        """
        if not page_height or page_height <= 0 or len(bbox) < 4:
            return False
        top = self.FURNITURE_TOP_BAND / 1000 * page_height
        bottom = self.FURNITURE_BOTTOM_BAND / 1000 * page_height
        return bbox[3] <= top or bbox[1] >= bottom

    def _lines_are_continuous(self, previous: list[float], current: list[float]) -> bool:
        """Whether ``current`` continues the same paragraph as ``previous``."""
        overlap = max(0.0, min(previous[2], current[2]) - max(previous[0], current[0]))
        narrower = min(previous[2] - previous[0], current[2] - current[0])
        if narrower <= 0 or overlap / narrower < self.LINE_MERGE_X_OVERLAP:
            return False  # different column
        line_height = max(current[3] - current[1], 1.0)
        gap = current[1] - previous[3]
        return gap <= line_height * self.LINE_MERGE_GAP_RATIO

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
        target_bbox: Sequence[float],
        target_text: str,
        blocks: list[Block],
        page_idx: int,
    ) -> bool:
        """
        Check if target text is already covered by existing blocks.

        Uses text content matching only - bbox overlap is unreliable
        due to MinerU bbox inaccuracy issues.

        Scoped to the page and its immediate neighbours: MinerU merges a
        paragraph that runs over a page break into a single block anchored on
        the *starting* page, and PyMuPDF then re-reads the tail on the next
        page as missing text. Comparing only within the page left those tails
        in as orphan fragments (live: 2,617 across the 100-document store).
        The radius stays at one page so repeated boilerplate elsewhere in the
        document can never mask a genuine gap.

        Returns True if covered (should skip).
        """
        target_text_clean = _normalize_for_coverage(target_text)
        if not target_text_clean or len(target_text_clean) < 4:
            return True  # Empty or too short text, skip

        for block in blocks:
            if abs(block.page_idx - page_idx) > self.COVERAGE_PAGE_RADIUS:
                continue
            # Geometry settles the cases text cannot: PyMuPDF reads the
            # vertical running head 「示範政經瞭望」 in column order as
            # 「示政瞭範經望」, which matches no string MinerU produced while
            # sitting wholly inside the box MinerU already transcribed.
            # Containment, not mere overlap — adjacent boxes must not qualify.
            if block.page_idx == page_idx and _bbox_containment(
                target_bbox, block.bbox_norm or []
            ) >= self.COVERAGE_CONTAINMENT:
                return True
            block_text = block.get_text() or ""
            if isinstance(block_text, list):
                # IMAGE captions from MinerU are lists of strings
                block_text = "\n".join(str(part) for part in block_text)
            if self._block_covers_fragment(target_text_clean, _normalize_for_coverage(block_text)):
                return True

        return False

    def _block_covers_fragment(self, fragment: str, block_text: str) -> bool:
        """Whether ``block_text`` already carries everything in ``fragment``.

        Directional on purpose. The previous helper asked whether *either*
        string contained the other, so a one-character folio block ("5")
        counted as covering any fragment containing a 5 — suppressing real
        supplements for the wrong reason.
        """
        if not fragment or not block_text:
            return False
        if fragment in block_text:
            return True
        if len(fragment) < 10:
            return False  # too short to judge by overlap; containment only
        grams = [fragment[i:i + 4] for i in range(len(fragment) - 3)]
        hits = sum(1 for gram in grams if gram in block_text)
        return hits / len(grams) > self.COVERAGE_GRAM_RATIO

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

    @staticmethod
    def source_page_text_lengths(pdf_path: Path) -> list[int]:
        """Per-page char count of the PDF's own text layer (0 = scanned page)."""
        if not HAS_PYMUPDF:
            return []
        try:
            with fitz.open(pdf_path) as doc:
                return [len((page.get_text() or "").strip()) for page in doc]
        except Exception:
            return []

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
                        source_text_chars=len((pdf_page.get_text() or "").strip()),
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
