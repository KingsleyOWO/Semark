"""
Chunk stage - Split document into chunks for RAG ingestion.

Output: chunks.jsonl with semantic chunks preserving block references.
"""

import json
import math
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR

# Sections estimated below this many tokens merge with their neighbours so
# retrieval never indexes near-empty chunks.
MIN_SECTION_TOKENS = 80

# Unicode ranges counted as CJK for token estimation.
_CJK_RANGES = (
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xFF00, 0xFFEF),  # Halfwidth and fullwidth forms
)


def estimate_tokens(text: str) -> int:
    """
    Estimate embedder token count for mixed CJK/non-CJK text.

    CJK characters tokenize at roughly 1 token per 1.5 characters, other
    characters at roughly 1 token per 4 characters. A plain len(text) // 3
    undercounts CJK-heavy text by 2-3x, which silently overflows embedder
    context windows.
    """
    cjk_chars = 0
    for char in text:
        code_point = ord(char)
        if any(low <= code_point <= high for low, high in _CJK_RANGES):
            cjk_chars += 1
    other_chars = len(text) - cjk_chars
    return math.ceil(cjk_chars / 1.5) + math.ceil(other_chars / 4)


@dataclass
class Chunk:
    """A document chunk for RAG ingestion."""

    chunk_id: str
    doc_id: str
    run_id: str
    view: str  # "rag" or "dataset"
    content: str
    block_ids: list[str]
    page_indices: list[int]
    attachments: list[str] = field(default_factory=list)  # asset:// references
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "run_id": self.run_id,
            "view": self.view,
            "content": self.content,
            "block_ids": self.block_ids,
            "page_indices": self.page_indices,
            "attachments": self.attachments,
            "metadata": self.metadata,
        }


@dataclass
class ChunkStageResult:
    """Result from chunk stage."""

    success: bool
    chunks: list[Chunk] = field(default_factory=list)
    chunks_path: Path | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Section:
    """A heading-delimited run of blocks with chunking bookkeeping."""

    blocks: list[Block]
    heading_path: list[str]
    tokens: int
    mergeable: bool  # every constituent section was below MIN_SECTION_TOKENS
    heading_only: bool  # titles only, no body content


class ChunkStage:
    """
    Chunk stage - splits document into semantic chunks.

    Chunking strategy:
    1. Split by headings (respects document structure)
    2. Merge small consecutive sections (a heading is never left alone)
    3. Split large sections if exceeding max_tokens
    4. Preserve block references and heading paths for traceability

    Input: DocumentIR
    Output: chunks.jsonl
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.package_config = self.config.package
        # block_id -> figure retrieval text, loaded per run from assets_index.jsonl
        # so image blocks chunk as searchable text instead of a bare placeholder.
        self._asset_text: dict[str, str] = {}

    async def run(
        self,
        doc_id: str,
        run_id: str,
        document_ir: DocumentIR,
        run_path: Path,
    ) -> ChunkStageResult:
        """
        Run chunk stage.

        Args:
            doc_id: Document ID
            run_id: Run ID
            document_ir: Document IR with blocks
            run_path: Path to run output directory

        Returns:
            ChunkStageResult with chunks
        """
        try:
            if not self.package_config.generate_chunks:
                return ChunkStageResult(
                    success=True,
                    stats={"skipped": True, "reason": "Chunking disabled"},
                )

            chunks: list[Chunk] = []

            # Figure enrichment text (written by the package stage before chunking)
            # so image blocks become searchable body text, not a bare placeholder.
            self._asset_text = self._load_asset_retrieval_text(run_path / "outputs")

            # Generate chunks from blocks
            rag_chunks = self._chunk_blocks(
                document_ir=document_ir,
                view="rag",
                max_tokens=self.package_config.chunk_max_tokens,
                overlap_tokens=self.package_config.chunk_overlap_tokens,
            )
            chunks.extend(rag_chunks)

            # Write chunks.jsonl
            outputs_dir = run_path / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            if self._structured_chunks_should_replace(outputs_dir):
                chunks = self._load_structured_chunks(outputs_dir)

            chunks_path = outputs_dir / "chunks.jsonl"
            with open(chunks_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

            stats = {
                "total_chunks": len(chunks),
                "avg_chunk_length": (
                    sum(len(c.content) for c in chunks) / len(chunks)
                    if chunks else 0
                ),
            }

            return ChunkStageResult(
                success=True,
                chunks=chunks,
                chunks_path=chunks_path,
                stats=stats,
            )

        except Exception as e:
            return ChunkStageResult(
                success=False,
                error=str(e),
            )

    def _load_asset_retrieval_text(self, outputs_dir: Path) -> dict[str, str]:
        """Map block_id -> figure retrieval text from assets_index.jsonl.

        The package stage writes assets_index.jsonl (one AssetEntry per line, each
        carrying block_id + retrieval_text) before the chunk stage runs. Reusing it
        keeps image chunks byte-for-byte consistent with the RAG markdown, which
        renders the same retrieval_text for figures.
        """
        asset_text: dict[str, str] = {}
        index_path = outputs_dir / "assets_index.jsonl"
        if not index_path.exists():
            return asset_text
        try:
            for line in index_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                block_id = entry.get("block_id", "")
                text = (entry.get("retrieval_text") or "").strip()
                if block_id and text:
                    asset_text[block_id] = text
        except Exception:
            return {}
        return asset_text

    def _load_structured_chunks(self, outputs_dir: Path) -> list[Chunk]:
        """Use row-level structured chunks when package stage generated them."""
        chunks_path = outputs_dir / "structured_chunks.jsonl"
        if not chunks_path.exists():
            return []

        chunks: list[Chunk] = []
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=str(data["chunk_id"]),
                    doc_id=str(data["doc_id"]),
                    run_id=str(data["run_id"]),
                    view=str(data.get("view") or "structured_rag"),
                    content=str(data["content"]),
                    block_ids=[str(item) for item in data.get("block_ids", [])],
                    page_indices=[int(item) for item in data.get("page_indices", [])],
                    attachments=[str(item) for item in data.get("attachments", [])],
                    metadata=dict(data.get("metadata", {})),
                )
            )
        return chunks

    def _structured_chunks_should_replace(self, outputs_dir: Path) -> bool:
        """Return true when package-stage structured output must own final chunks."""

        chunks_path = outputs_dir / "structured_chunks.jsonl"
        if not chunks_path.exists():
            return False
        if chunks_path.read_text(encoding="utf-8").strip():
            return True

        # Empty structured chunks are not authoritative. Fall back to block chunks
        # so a run never produces an empty RAG artifact solely because review failed.
        return False

    def _chunk_blocks(
        self,
        document_ir: DocumentIR,
        view: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[Chunk]:
        """
        Chunk blocks using heading-based strategy.

        1. Group blocks by heading sections, tracking the heading path
        2. Merge small groups (headings are never emitted alone)
        3. Split large groups
        """
        chunks: list[Chunk] = []
        chunk_idx = 0

        # Group blocks by sections (split at headings)
        sections = self._build_sections(document_ir.blocks)
        sections = self._merge_small_sections(sections, max_tokens)

        for section in sections:
            if section.tokens <= max_tokens:
                # Section fits in one chunk
                chunk = self._create_chunk(
                    chunk_id=f"c{chunk_idx:06d}",
                    doc_id=document_ir.doc_id,
                    run_id=document_ir.run_id,
                    view=view,
                    blocks=section.blocks,
                    heading_path=section.heading_path,
                )
                chunks.append(chunk)
                chunk_idx += 1
            else:
                # Section too large, split further
                sub_chunks = self._split_large_section(
                    blocks=section.blocks,
                    doc_id=document_ir.doc_id,
                    run_id=document_ir.run_id,
                    view=view,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    start_idx=chunk_idx,
                    heading_path=section.heading_path,
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)

        return chunks

    def _split_by_headings(self, blocks: list[Block]) -> list[list[Block]]:
        """Split blocks into sections at heading boundaries."""
        sections: list[list[Block]] = []
        current_section: list[Block] = []

        for block in blocks:
            # Check if this is a heading
            is_heading = (
                block.type == BlockType.TEXT
                and block.payload.get("text_level", 0) > 0
            )

            if is_heading and current_section:
                # Start new section
                sections.append(current_section)
                current_section = [block]
            else:
                current_section.append(block)

        # Don't forget the last section
        if current_section:
            sections.append(current_section)

        return sections

    def _build_sections(self, blocks: list[Block]) -> list[_Section]:
        """Split blocks at headings and annotate each run with its heading path."""
        heading_stack: list[tuple[int, str]] = []
        sections: list[_Section] = []

        for section_blocks in self._split_by_headings(blocks):
            if not section_blocks:
                continue

            first = section_blocks[0]
            if first.type == BlockType.TEXT and first.payload.get("text_level", 0) > 0:
                level = int(first.payload.get("text_level", 0))
                title = str(first.payload.get("text", "")).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

            tokens = estimate_tokens(self._blocks_to_text(section_blocks))
            sections.append(
                _Section(
                    blocks=list(section_blocks),
                    heading_path=[title for _, title in heading_stack],
                    tokens=tokens,
                    mergeable=tokens < MIN_SECTION_TOKENS,
                    heading_only=self._section_is_heading_only(section_blocks),
                )
            )

        return sections

    def _section_is_heading_only(self, blocks: list[Block]) -> bool:
        """Return true when a section contains titles but no body content."""
        has_heading = False
        for block in blocks:
            if block.type == BlockType.TEXT and block.payload.get("text_level", 0) > 0:
                has_heading = True
                continue
            if self._block_to_text(block).strip():
                return False
        return has_heading

    def _merge_small_sections(
        self,
        sections: list[_Section],
        max_tokens: int,
    ) -> list[_Section]:
        """
        Merge consecutive sections that are each below MIN_SECTION_TOKENS
        (up to max_tokens per merged section), and never leave a heading-only
        section (title without body) to be emitted as its own chunk.
        """
        merged: list[_Section] = []

        for section in sections:
            if merged:
                previous = merged[-1]
                if previous.heading_only:
                    # A bare title always attaches to what follows it; adopt
                    # the follower's heading path (it already includes the
                    # title via the heading stack).
                    merged[-1] = self._merge_sections(
                        previous, section, heading_path=section.heading_path
                    )
                    continue
                if (
                    previous.mergeable
                    and section.mergeable
                    and previous.tokens + section.tokens <= max_tokens
                ):
                    merged[-1] = self._merge_sections(
                        previous, section, heading_path=previous.heading_path
                    )
                    continue
            merged.append(section)

        # A trailing bare title has nothing following it; fold it backward.
        if len(merged) >= 2 and merged[-1].heading_only:
            last = merged.pop()
            merged[-1] = self._merge_sections(
                merged[-1], last, heading_path=merged[-1].heading_path
            )

        return merged

    def _merge_sections(
        self,
        first: _Section,
        second: _Section,
        heading_path: list[str],
    ) -> _Section:
        """Combine two adjacent sections into one."""
        return _Section(
            blocks=first.blocks + second.blocks,
            heading_path=list(heading_path),
            tokens=first.tokens + second.tokens,
            mergeable=first.mergeable and second.mergeable,
            heading_only=first.heading_only and second.heading_only,
        )

    def _split_large_section(
        self,
        blocks: list[Block],
        doc_id: str,
        run_id: str,
        view: str,
        max_tokens: int,
        overlap_tokens: int,
        start_idx: int,
        heading_path: list[str] | None = None,
    ) -> list[Chunk]:
        """Split a large section into smaller chunks."""
        chunks: list[Chunk] = []
        current_blocks: list[Block] = []
        current_length = 0
        chunk_idx = start_idx

        for block in blocks:
            block_tokens = estimate_tokens(self._block_to_text(block))

            if current_length + block_tokens > max_tokens and current_blocks:
                # Create chunk with current blocks
                chunk = self._create_chunk(
                    chunk_id=f"c{chunk_idx:06d}",
                    doc_id=doc_id,
                    run_id=run_id,
                    view=view,
                    blocks=current_blocks,
                    heading_path=heading_path,
                    continuation=bool(chunks),
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Start new chunk with overlap
                # Keep last block(s) for context overlap
                overlap_blocks = self._get_overlap_blocks(
                    current_blocks, overlap_tokens
                )
                current_blocks = overlap_blocks + [block]
                current_length = sum(
                    estimate_tokens(self._block_to_text(b)) for b in current_blocks
                )
            else:
                current_blocks.append(block)
                current_length += block_tokens

        # Final chunk
        if current_blocks:
            chunk = self._create_chunk(
                chunk_id=f"c{chunk_idx:06d}",
                doc_id=doc_id,
                run_id=run_id,
                view=view,
                blocks=current_blocks,
                heading_path=heading_path,
                continuation=bool(chunks),
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_blocks(
        self,
        blocks: list[Block],
        overlap_tokens: int,
    ) -> list[Block]:
        """Get blocks for overlap from the end."""
        overlap_blocks: list[Block] = []
        current_tokens = 0

        for block in reversed(blocks):
            block_tokens = estimate_tokens(self._block_to_text(block))
            if current_tokens + block_tokens > overlap_tokens:
                break
            overlap_blocks.insert(0, block)
            current_tokens += block_tokens

        return overlap_blocks

    def _create_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        run_id: str,
        view: str,
        blocks: list[Block],
        heading_path: list[str] | None = None,
        continuation: bool = False,
    ) -> Chunk:
        """Create a chunk from blocks."""
        heading_path = list(heading_path or [])
        content = self._blocks_to_text(blocks)
        if continuation and heading_path:
            # Continuation of a split section: repeat the heading path so
            # retrievers see the section context the heading blocks carry.
            content = f"{' > '.join(heading_path)}（續）\n\n{content}"
        block_ids = [b.block_id for b in blocks]
        page_indices = list(set(b.page_idx for b in blocks))

        # Extract attachments (asset references)
        attachments: list[str] = []
        for block in blocks:
            if block.type == BlockType.IMAGE:
                img_path = block.payload.get("img_path", "")
                if img_path:
                    attachments.append(f"asset://{img_path}")

        # Metadata
        metadata: dict[str, Any] = {
            "block_count": len(blocks),
            "has_table": any(b.type == BlockType.TABLE for b in blocks),
            "has_image": any(b.type == BlockType.IMAGE for b in blocks),
            "heading_path": heading_path,
        }

        # Preserve original table HTML (content renders it as markdown)
        table_html = [
            str(b.payload.get("table_body", ""))
            for b in blocks
            if b.type == BlockType.TABLE and b.payload.get("table_body")
        ]
        if table_html:
            metadata["table_html"] = table_html

        # Add heading info if first block is heading
        if blocks and blocks[0].type == BlockType.TEXT:
            level = blocks[0].payload.get("text_level", 0)
            if level > 0:
                metadata["heading"] = blocks[0].payload.get("text", "")
                metadata["heading_level"] = level

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            run_id=run_id,
            view=view,
            content=content,
            block_ids=block_ids,
            page_indices=page_indices,
            attachments=attachments,
            metadata=metadata,
        )

    def _blocks_to_text(self, blocks: list[Block]) -> str:
        """Convert blocks to text."""
        parts = [self._block_to_text(b) for b in blocks]
        return "\n\n".join(p for p in parts if p)

    def _block_to_text(self, block: Block) -> str:
        """Convert a single block to text."""
        if block.type == BlockType.TEXT:
            text = block.payload.get("text", "")
            level = block.payload.get("text_level", 0)
            if level > 0:
                prefix = "#" * min(level, 6)
                return f"{prefix} {text}"
            return text

        elif block.type == BlockType.TABLE:
            caption = block.payload.get("table_caption", "")
            body = _table_body_to_markdown(block.payload.get("table_body", ""))
            if caption:
                return f"**{caption}**\n\n{body}"
            return body

        elif block.type == BlockType.IMAGE:
            caption = block.payload.get("caption", "")
            # Prefer the VLM figure retrieval text so the figure's information is
            # searchable in the chunk body, rather than a model-unreadable
            # ``[Image: caption]`` placeholder. Falls back to the placeholder when
            # no enrichment text was produced for this block.
            retrieval_text = self._asset_text.get(block.block_id, "")
            if retrieval_text:
                return retrieval_text
            return f"[Image: {caption}]" if caption else "[Image]"

        elif block.type == BlockType.EQUATION:
            latex = block.payload.get("latex", "")
            return f"$${latex}$$"

        elif block.type == BlockType.CODE:
            code = block.payload.get("code", "")
            lang = block.payload.get("language", "")
            return f"```{lang}\n{code}\n```"

        elif block.type == BlockType.LIST:
            items = block.payload.get("items", [])
            list_type = block.payload.get("list_type", "unordered")
            lines = []
            for i, item in enumerate(items):
                if list_type == "ordered":
                    lines.append(f"{i + 1}. {item}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)

        return ""


class _HTMLTableParser(HTMLParser):
    """Collect (text, rowspan, colspan) cells for each table row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._row: list[tuple[str, int, int]] | None = None
        self._cell_parts: list[str] | None = None
        self._rowspan = 1
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            attr_map = dict(attrs)
            self._rowspan = self._parse_span(attr_map.get("rowspan"))
            self._colspan = self._parse_span(attr_map.get("colspan"))
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell_parts is not None:
            self._row.append(
                ("".join(self._cell_parts), self._rowspan, self._colspan)
            )
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    @staticmethod
    def _parse_span(value: str | None) -> int:
        try:
            return max(int(str(value)), 1)
        except (TypeError, ValueError):
            return 1


def _strip_html_tags(html: str) -> str:
    """Strip tags and collapse whitespace."""
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def _table_body_to_markdown(body: str) -> str:
    """
    Convert an HTML table to a markdown pipe table.

    Rowspan/colspan cells repeat their value across the spanned grid cells.
    Markup that does not parse into table rows has its tags stripped instead;
    tag-free bodies pass through unchanged.
    """
    if "<" not in body:
        return body

    parser = _HTMLTableParser()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        return _strip_html_tags(body)
    if not parser.rows:
        return _strip_html_tags(body)

    # Expand rowspan/colspan into a rectangular grid by repeating values.
    grid: list[list[str]] = []
    carry: dict[int, tuple[str, int]] = {}
    for cells in parser.rows:
        row: list[str] = []
        col = 0
        for text, rowspan, colspan in cells:
            while col in carry:
                carried_text, remaining = carry.pop(col)
                if remaining > 1:
                    carry[col] = (carried_text, remaining - 1)
                row.append(carried_text)
                col += 1
            clean = " ".join(text.split()).replace("|", "\\|")
            for _ in range(colspan):
                row.append(clean)
                if rowspan > 1:
                    carry[col] = (clean, rowspan - 1)
                col += 1
        while col in carry:
            carried_text, remaining = carry.pop(col)
            if remaining > 1:
                carry[col] = (carried_text, remaining - 1)
            row.append(carried_text)
            col += 1
        grid.append(row)

    width = max(len(row) for row in grid)
    lines: list[str] = []
    for i, row in enumerate(grid):
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)
