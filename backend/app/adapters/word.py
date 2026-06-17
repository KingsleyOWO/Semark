"""Native Word parser for DOCX inputs.

The parser keeps paragraph/table/image order before the document enters the
MinerU-oriented normalization stage. It emits MinerU-compatible content_list
items so later stages can stay parser-agnostic.
"""

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


@dataclass
class WordParseResult:
    success: bool
    output_dir: Path
    content_list_path: Path | None = None
    markdown_path: Path | None = None
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ImageExport:
    rel_id: str
    relative_path: str
    filename: str


def parse_word_document(input_path: Path, output_dir: Path) -> WordParseResult:
    """Parse a DOCX document into MinerU-compatible content_list JSON."""
    output_root = output_dir / input_path.stem / "native"
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        document = Document(input_path)
    except Exception as exc:
        return WordParseResult(
            success=False,
            output_dir=output_root,
            error=f"Word parse failed: {exc}",
        )

    content_items: list[dict[str, Any]] = []
    markdown_lines: list[str] = []
    paragraph_count = 0
    table_count = 0
    image_count = 0
    recent_texts: list[str] = []
    exported_images: dict[str, _ImageExport] = {}

    for block_idx, block in enumerate(_iter_block_items(document)):
        if isinstance(block, Paragraph):
            text = _paragraph_text(block)
            if text:
                level = _paragraph_level(block)
                content_items.append(
                    {
                        "type": "text",
                        "text": text,
                        "text_level": level,
                        "page_idx": 0,
                        "bbox": [0, 0, 0, 0],
                        "metadata": {
                            "source": "native_docx",
                            "block_index": block_idx,
                            "style": block.style.name if block.style else "",
                        },
                    }
                )
                prefix = "#" * min(level, 6) if level else ""
                markdown_lines.append(f"{prefix} {text}".strip())
                markdown_lines.append("")
                paragraph_count += 1
                recent_texts.append(text)
                recent_texts = recent_texts[-3:]

            for rel_id in _paragraph_image_rel_ids(block):
                image = _export_image(document, rel_id, output_root, exported_images, image_count)
                if not image:
                    continue
                image_count += 1
                caption = _image_caption(recent_texts, image_count)
                content_items.append(
                    {
                        "type": "image",
                        "img_path": image.relative_path,
                        "img_caption": [caption] if caption else [],
                        "img_footnote": [],
                        "page_idx": 0,
                        "bbox": [0, 0, 0, 0],
                        "metadata": {
                            "source": "native_docx",
                            "block_index": block_idx,
                            "rel_id": rel_id,
                        },
                    }
                )
                alt_text = caption or image.filename
                markdown_lines.append(f"![{alt_text}]({image.relative_path})")
                markdown_lines.append("")
            continue

        if isinstance(block, Table):
            rows = _table_rows(block)
            if not rows:
                continue

            table_count += 1
            caption = _table_caption(recent_texts, table_count)
            table_html = _rows_to_html_table(rows)
            content_items.append(
                {
                    "type": "table",
                    "table_caption": caption,
                    "table_body": table_html,
                    "page_idx": 0,
                    "bbox": [0, 0, 0, 0],
                    "metadata": {
                        "source": "native_docx",
                        "block_index": block_idx,
                        "row_count": len(rows),
                        "column_count": max((len(row) for row in rows), default=0),
                    },
                }
            )
            markdown_lines.append(f"**{caption}**")
            markdown_lines.append("")
            markdown_lines.append(table_html)
            markdown_lines.append("")

    if not content_items:
        return WordParseResult(
            success=False,
            output_dir=output_root,
            error="Word document contains no readable paragraphs, tables, or images",
        )

    content_list_path = output_root / f"{input_path.stem}_content_list.json"
    content_list_path.write_text(
        json.dumps(content_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_path = output_root / f"{input_path.stem}.md"
    markdown_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")

    return WordParseResult(
        success=True,
        output_dir=output_root,
        content_list_path=content_list_path,
        markdown_path=markdown_path,
        stats={
            "parser": "native_docx",
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "image_count": image_count,
        },
    )


def _iter_block_items(parent: DocumentType | _Cell):
    """Yield paragraphs and tables in document order."""
    if isinstance(parent, DocumentType):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

    for child in parent_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def _paragraph_text(paragraph: Paragraph) -> str:
    return " ".join(part.strip() for part in paragraph.text.splitlines() if part.strip()).strip()


def _paragraph_level(paragraph: Paragraph) -> int:
    style_name = paragraph.style.name if paragraph.style else ""
    if style_name.startswith("Heading"):
        suffix = style_name.replace("Heading", "").strip()
        if suffix.isdigit():
            return max(1, min(int(suffix), 6))
        return 1
    if style_name.startswith("標題"):
        suffix = style_name.replace("標題", "").strip()
        if suffix.isdigit():
            return max(1, min(int(suffix), 6))
        return 1
    return 0


def _paragraph_image_rel_ids(paragraph: Paragraph) -> list[str]:
    rel_ids: list[str] = []
    for element in paragraph._p.iter():
        if not element.tag.endswith("}blip"):
            continue
        rel_id = element.get(qn("r:embed")) or element.get(qn("r:link"))
        if rel_id and rel_id not in rel_ids:
            rel_ids.append(rel_id)
    return rel_ids


def _export_image(
    document: DocumentType,
    rel_id: str,
    output_root: Path,
    exported_images: dict[str, _ImageExport],
    image_count: int,
) -> _ImageExport | None:
    if rel_id in exported_images:
        return exported_images[rel_id]

    image_part = document.part.related_parts.get(rel_id)
    if not image_part or not hasattr(image_part, "blob"):
        return None

    suffix = Path(str(getattr(image_part, "partname", ""))).suffix.lower()
    if not suffix:
        suffix = _suffix_from_content_type(getattr(image_part, "content_type", ""))
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}:
        suffix = ".png"

    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"docx_img{image_count:04d}{suffix}"
    destination = images_dir / filename
    destination.write_bytes(image_part.blob)

    export = _ImageExport(
        rel_id=rel_id,
        relative_path=f"images/{filename}",
        filename=filename,
    )
    exported_images[rel_id] = export
    return export


def _suffix_from_content_type(content_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
    }.get(content_type.lower(), ".png")


def _table_rows(table: Table) -> list[list[str]]:
    rows: list[list[str]] = []
    max_width = 0

    for row in table.rows:
        values = [_cell_text(cell) for cell in row.cells]
        while values and not values[-1]:
            values.pop()
        if any(values):
            max_width = max(max_width, len(values))
            rows.append(values)

    return [row + [""] * (max_width - len(row)) for row in rows]


def _cell_text(cell: _Cell) -> str:
    parts: list[str] = []
    for paragraph in cell.paragraphs:
        text = _paragraph_text(paragraph)
        if text:
            parts.append(text)
    return " / ".join(parts).strip()


def _table_caption(recent_texts: list[str], table_count: int) -> str:
    for text in reversed(recent_texts):
        compact = text.replace(" ", "")
        if len(text) <= 100 and any(token in compact for token in ("表", "Table", "附件")):
            return text
    return f"Word 表格 {table_count}"


def _image_caption(recent_texts: list[str], image_count: int) -> str:
    for text in reversed(recent_texts):
        compact = text.replace(" ", "")
        image_tokens = ("圖", "Figure", "流程", "架構", "組織")
        if len(text) <= 100 and any(token in compact for token in image_tokens):
            return text
    return f"Word 圖片 {image_count}"


def _rows_to_html_table(rows: list[list[str]]) -> str:
    if not rows:
        return "<table></table>"

    lines = ["<table>"]
    header_idx = _header_row_index(rows)
    for row_idx, row in enumerate(rows):
        tag = "th" if row_idx == header_idx else "td"
        lines.append("  <tr>")
        for cell in row:
            lines.append(f"    <{tag}>{html.escape(cell)}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _header_row_index(rows: list[list[str]]) -> int:
    for idx, row in enumerate(rows[:6]):
        non_empty = [cell for cell in row if cell]
        if len(non_empty) >= 2:
            return idx
    return 0
