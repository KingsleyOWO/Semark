"""
Chunk stage - Split document into chunks for RAG ingestion.

Output: chunks.jsonl with semantic chunks preserving block references.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR


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


class ChunkStage:
    """
    Chunk stage - splits document into semantic chunks.

    Chunking strategy:
    1. Split by headings (respects document structure)
    2. Merge small consecutive blocks
    3. Split large blocks if exceeding max_tokens
    4. Preserve block references for traceability

    Input: DocumentIR
    Output: chunks.jsonl
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
            structured_chunks = self._load_structured_chunks(outputs_dir)
            if structured_chunks:
                if self._structured_chunks_should_replace(outputs_dir):
                    chunks = structured_chunks
                else:
                    chunks.extend(structured_chunks)

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
        """Return true when structured chunks should replace generic chunks."""

        plan_path = outputs_dir / "document_plan.json"
        if not plan_path.exists():
            return True
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return plan.get("document_type") != "form_collection"

    def _chunk_blocks(
        self,
        document_ir: DocumentIR,
        view: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[Chunk]:
        """
        Chunk blocks using heading-based strategy.

        1. Group blocks by heading sections
        2. Merge small groups
        3. Split large groups
        """
        chunks: list[Chunk] = []
        chunk_idx = 0

        # Group blocks by sections (split at headings)
        sections = self._split_by_headings(document_ir.blocks)

        for section_blocks in sections:
            if not section_blocks:
                continue

            # Estimate token count (rough: 1 token ≈ 2 chars for CJK, 4 chars for English)
            section_text = self._blocks_to_text(section_blocks)
            estimated_tokens = len(section_text) // 3  # Average

            if estimated_tokens <= max_tokens:
                # Section fits in one chunk
                chunk = self._create_chunk(
                    chunk_id=f"c{chunk_idx:06d}",
                    doc_id=document_ir.doc_id,
                    run_id=document_ir.run_id,
                    view=view,
                    blocks=section_blocks,
                )
                chunks.append(chunk)
                chunk_idx += 1
            else:
                # Section too large, split further
                sub_chunks = self._split_large_section(
                    blocks=section_blocks,
                    doc_id=document_ir.doc_id,
                    run_id=document_ir.run_id,
                    view=view,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    start_idx=chunk_idx,
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

    def _split_large_section(
        self,
        blocks: list[Block],
        doc_id: str,
        run_id: str,
        view: str,
        max_tokens: int,
        overlap_tokens: int,
        start_idx: int,
    ) -> list[Chunk]:
        """Split a large section into smaller chunks."""
        chunks: list[Chunk] = []
        current_blocks: list[Block] = []
        current_length = 0
        chunk_idx = start_idx

        for block in blocks:
            block_text = self._block_to_text(block)
            block_tokens = len(block_text) // 3

            if current_length + block_tokens > max_tokens and current_blocks:
                # Create chunk with current blocks
                chunk = self._create_chunk(
                    chunk_id=f"c{chunk_idx:06d}",
                    doc_id=doc_id,
                    run_id=run_id,
                    view=view,
                    blocks=current_blocks,
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
                    len(self._block_to_text(b)) // 3 for b in current_blocks
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
            block_tokens = len(self._block_to_text(block)) // 3
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
    ) -> Chunk:
        """Create a chunk from blocks."""
        content = self._blocks_to_text(blocks)
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
        }

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
            body = block.payload.get("table_body", "")
            if caption:
                return f"**{caption}**\n\n{body}"
            return body

        elif block.type == BlockType.IMAGE:
            caption = block.payload.get("caption", "")
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
