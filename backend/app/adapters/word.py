"""Native Word parser for DOCX inputs.

The parser keeps paragraph/table/image order before the document enters the
MinerU-oriented normalization stage. It emits MinerU-compatible content_list
items so later stages can stay parser-agnostic.
"""

import hashlib
import html
import json
import re
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

    content_controls = _unwrap_content_controls(document)

    content_items: list[dict[str, Any]] = []
    markdown_lines: list[str] = []
    paragraph_count = 0
    table_count = 0
    image_count = 0
    recent_texts: list[str] = []
    exported_images: dict[str, _ImageExport] = {}

    def append_image(rel_id, block_idx, caption=None, **location):
        nonlocal image_count
        image = _export_image(document, rel_id, output_root, exported_images, image_count)
        if not image:
            return
        image_count += 1
        caption = caption or _image_caption(recent_texts, image_count)
        location["caption_is_context"] = True
        location["caption_origin"] = "table_location" if "cell_path" in location else "neighboring_text"
        content_items.append({
            "type": "image", "img_path": image.relative_path,
            "img_caption": [caption], "img_footnote": [], "page_idx": 0, "bbox": [0, 0, 0, 0],
            "metadata": {"source": "native_docx", "block_index": block_idx, "rel_id": rel_id,
                         "image_sha256": hashlib.sha256((output_root / image.relative_path).read_bytes()).hexdigest(),
                         **location},
        })
        markdown_lines.extend([f"![{caption}]({image.relative_path})", ""])

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
                append_image(rel_id, block_idx)
            continue


        if isinstance(block, Table):
            rows = _table_rows(block)
            pictures = list(_table_image_refs(block))
            if not rows and not pictures:
                continue

            table_count += 1
            caption = _table_caption(recent_texts, table_count)
            table_html = _table_to_html(block)
            table_content_index = len(content_items)
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
                        "row_count": len(block.rows),
                        "column_count": len(block.columns),
                    },
                }
            )
            markdown_lines.append(f"**{caption}**")
            markdown_lines.append("")
            markdown_lines.append(table_html)
            markdown_lines.append("")
            for number, (rel_id, cell_path, row_label) in enumerate(pictures, 1):
                position = " > ".join(f"第{cell['row'] + 1}列第{cell['column'] + 1}欄" for cell in cell_path)
                label = f"{caption}／{position}／圖片{number}"
                if row_label:
                    label += f"（{row_label}）"
                append_image(rel_id, block_idx, label, parent_content_index=table_content_index,
                             table_caption=caption, cell_path=cell_path, row_label=row_label,
                             caption_is_context=True)

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
            "content_controls_unwrapped": content_controls,
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "image_count": image_count,
        },
    )


def _unwrap_content_controls(document: DocumentType) -> int:
    """Expose visible SDT content to python-docx without changing the source file.

    Controls can wrap whole blocks, table cells, or runs inside a paragraph.
    Only sdtContent belongs to the document; control properties are metadata.
    """
    controls = list(document.element.body.iter(qn("w:sdt")))
    count = 0
    for control in reversed(controls):
        content = control.find(qn("w:sdtContent"))
        if content is None:
            continue
        parent = control.getparent()
        index = parent.index(control)
        for child in list(content):
            parent.insert(index, child)
            index += 1
        parent.remove(control)
        count += 1
    return count


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
    pictures = []
    for element in paragraph._p.iter(qn("a:blip")):
        rel_id = element.get(qn("r:embed")) or element.get(qn("r:link"))
        if not rel_id:
            continue
        anchor = next((a for a in element.iterancestors() if a.tag == qn("wp:anchor")), None)
        placement = None
        if anchor is not None:
            horizontal, vertical = anchor.find(qn("wp:positionH")), anchor.find(qn("wp:positionV"))
            extent = anchor.find(qn("wp:extent"))
            if horizontal is not None and vertical is not None and extent is not None:
                x, y = horizontal.find(qn("wp:posOffset")), vertical.find(qn("wp:posOffset"))
                try:
                    if x is not None and y is not None:
                        placement = (horizontal.get("relativeFrom"), vertical.get("relativeFrom"),
                                     int(x.text), int(y.text), int(extent.get("cy")))
                except (TypeError, ValueError):
                    pass
        pictures.append((rel_id, placement))
    # Only compare floating positions in the same paragraph and reference frame.
    # Inline pictures, alignment rules and mixed reference frames retain XML order.
    positions = [p for _, p in pictures]
    if (len(pictures) > 1 and all(p and p[0] and p[1] and p[4] > 0 for p in positions)
            and len({p[:2] for p in positions}) == 1):
        rows = []
        for picture in sorted(pictures, key=lambda item: item[1][3]):
            _, (_, _, x, y, height) = picture
            row = next((r for r in rows if
                        min(r[0] + r[1], y + height) - max(r[0], y) >= min(r[1], height) * 0.5), None)
            if row is None:
                rows.append([y, height, [picture]])
            else:
                row[2].append(picture)
        pictures = [picture for _, _, row in rows for picture in sorted(row, key=lambda item: item[1][2])]
    return [rel_id for rel_id, _ in pictures]


def _table_image_refs(table, path=(), table_index=0):
    """Visit actual XML cells once, including images in nested tables.

    row.cells repeats merged cells; XML cells preserve each drawing occurrence.
    Grid columns (including spans/omissions) and nested table indices are zero based.
    """
    for row_index, row in enumerate(table.rows):
        column = int(getattr(row, "grid_cols_before", 0) or 0)
        row_label = ""
        for tc in row._tr.tc_lst:
            cell = _Cell(tc, table)
            current_path = (*path, {"table_index": table_index, "row": row_index, "column": column})
            label = row_label or _cell_text(cell)[:160]
            nested_index = 0
            for block in _iter_block_items(cell):
                if isinstance(block, Paragraph):
                    for rel_id in _paragraph_image_rel_ids(block):
                        yield rel_id, list(current_path), label
                else:
                    yield from _table_image_refs(block, current_path, nested_index)
                    nested_index += 1
            if not row_label:
                row_label = _cell_text(cell)[:160]
            column += int(tc.grid_span)


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
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".emf", ".wmf", ".svg"}:
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


def _table_to_html(table: Table) -> str:
    """Preserve Word gridSpan/vMerge without copying merged cell text."""
    rows: list[list[dict[str, Any]]] = []
    active: dict[int, dict[str, Any]] = {}
    for row in table.rows:
        cells: list[dict[str, Any]] = []
        next_active: dict[int, dict[str, Any]] = {}
        column = int(getattr(row, "grid_cols_before", 0) or 0)
        for tc in row._tr.tc_lst:
            span = int(tc.grid_span)
            merge = tc.vMerge
            if merge == "continue" and column in active:
                cell = active[column]
                cell["rowspan"] += 1
                next_active[column] = cell
            else:
                cell = {"text": _cell_text(_Cell(tc, table)), "colspan": span, "rowspan": 1}
                cells.append(cell)
                if merge == "restart":
                    next_active[column] = cell
            column += span
        active = next_active
        rows.append(cells)
    lines = ['<table data-source-grid="1">']
    for idx, row in enumerate(rows):
        # Word tblHeader means repeat this display row on subsequent pages.
        # It is not a semantic column role (often it contains dates or a title).
        properties = table.rows[idx]._tr.trPr
        header = properties.find(qn("w:tblHeader")) if properties is not None else None
        is_header = header is not None and header.get(qn("w:val"), "1") not in {"0", "false", "off"}
        lines.append('<tr data-repeat-header="1">' if is_header else "<tr>")
        # A repeated full-width date/title is table context, not N column names.
        labels = {cell["text"].strip() for cell in row if cell["text"].strip()}
        date_range = any(re.search(r"\d.*(?:至|到|[~～–—]).*\d", label) for label in labels)
        tag = "th" if is_header and len(labels) > 1 and not date_range else "td"
        for cell in row:
            attrs = "".join(
                f' {key}="{cell[key]}"' for key in ("rowspan", "colspan") if cell[key] > 1
            )
            lines.append(f"<{tag}{attrs}>{html.escape(cell['text'])}</{tag}>")
        lines.append("</tr>")
    return "\n".join(lines + ["</table>"])


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
