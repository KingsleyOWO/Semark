"""
Document IR (Intermediate Representation) model.

Based on MinerU content_list.json structure.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class BlockType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    EQUATION = "equation"
    CODE = "code"
    LIST = "list"
    UNKNOWN = "unknown"  # For unrecognized or problematic blocks


class SourceInfo(BaseModel):
    """Source file information."""

    path: str
    ext: str
    sha256: str
    size_bytes: int


class EngineInfo(BaseModel):
    """Parsing engine information."""

    name: str = "mineru"
    backend: str
    version: str | None = None
    method: str
    lang: str | None = None
    table: bool = True
    formula: bool = True


class PageInfo(BaseModel):
    """Page information."""

    page_idx: int
    width_px: int | None = None
    height_px: int | None = None
    page_image_path: str | None = None


class TextPayload(BaseModel):
    """Payload for text blocks."""

    text: str
    text_level: int = 0  # 0=body, 1/2/...=heading level


class ImagePayload(BaseModel):
    """Payload for image blocks."""

    img_path: str
    caption: str | None = None
    footnote: str | None = None


class TablePayload(BaseModel):
    """Payload for table blocks."""

    table_body: str  # HTML or markdown
    table_caption: str | None = None


class EquationPayload(BaseModel):
    """Payload for equation blocks."""

    latex: str
    equation_type: str | None = None  # inline or display


class CodePayload(BaseModel):
    """Payload for code blocks."""

    code: str
    language: str | None = None


class ListPayload(BaseModel):
    """Payload for list blocks."""

    items: list[str]
    list_type: str = "unordered"  # ordered or unordered


class Block(BaseModel):
    """
    A content block in the document.

    bbox_norm is [x0, y0, x1, y1] in 0-1000 normalized coordinates.
    """

    block_id: str
    type: BlockType
    page_idx: int
    bbox_norm: list[int] = Field(default_factory=list)  # [x0, y0, x1, y1] 0-1000
    reading_order: int = 0

    # Type-specific payload
    payload: dict[str, Any] = Field(default_factory=dict)

    # Enrichment reference (if enriched)
    enrichment_ref: str | None = None

    def get_text(self) -> str:
        """Extract text content from block."""
        if self.type == BlockType.TEXT:
            return self.payload.get("text", "")
        elif self.type == BlockType.TABLE:
            return self.payload.get("table_body", "")
        elif self.type == BlockType.IMAGE:
            return self.payload.get("caption", "")
        elif self.type == BlockType.EQUATION:
            return self.payload.get("latex", "")
        elif self.type == BlockType.CODE:
            return self.payload.get("code", "")
        elif self.type == BlockType.LIST:
            return "\n".join(self.payload.get("items", []))
        return ""

    def get_heading_level(self) -> int | None:
        """Get heading level if this is a heading block."""
        if self.type == BlockType.TEXT:
            level = self.payload.get("text_level", 0)
            return level if level > 0 else None
        return None


class DocumentIR(BaseModel):
    """
    Document Intermediate Representation.

    This is the normalized representation of a parsed document,
    built from MinerU content_list.json.
    """

    doc_id: str
    run_id: str
    source: SourceInfo
    engine: EngineInfo
    pages: list[PageInfo] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)

    def get_blocks_by_page(self, page_idx: int) -> list[Block]:
        """Get all blocks on a specific page."""
        return [b for b in self.blocks if b.page_idx == page_idx]

    def get_blocks_by_type(self, block_type: BlockType) -> list[Block]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks if b.type == block_type]

    def get_block(self, block_id: str) -> Block | None:
        """Get a specific block by ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def get_text_blocks(self) -> list[Block]:
        """Get all text blocks."""
        return self.get_blocks_by_type(BlockType.TEXT)

    def get_headings(self) -> list[Block]:
        """Get all heading blocks."""
        return [
            b for b in self.blocks
            if b.type == BlockType.TEXT and b.payload.get("text_level", 0) > 0
        ]

    def count_by_type(self) -> dict[str, int]:
        """Count blocks by type."""
        counts: dict[str, int] = {}
        for block in self.blocks:
            counts[block.type.value] = counts.get(block.type.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentIR":
        """Create from dictionary."""
        return cls.model_validate(data)
