"""Native Word parser for DOCX inputs.

The parser keeps paragraph/table/image order before the document enters the
MinerU-oriented normalization stage. It emits MinerU-compatible content_list
items so later stages can stay parser-agnostic.
"""

import contextvars
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
    token = _NUMBER_LABELS.set({})
    try:
        return _parse_word_document(input_path, output_dir)
    finally:
        _NUMBER_LABELS.reset(token)


def _parse_word_document(input_path: Path, output_dir: Path) -> WordParseResult:
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
    # Cell text is read several times per table (rows, picture labels, HTML),
    # so list numbers are counted once, in document order, before any of it.
    _NUMBER_LABELS.set(_number_labels(document))

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

            for box_text in _text_box_texts(block):
                content_items.append({
                    "type": "text", "text": box_text, "text_level": 0, "page_idx": 0, "bbox": [0, 0, 0, 0],
                    "metadata": {"source": "native_docx", "block_index": block_idx, "text_box": True},
                })
                markdown_lines.extend([box_text, ""])
                paragraph_count += 1

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
    text = " ".join(part.strip() for part in paragraph.text.splitlines() if part.strip()).strip()
    label = _NUMBER_LABELS.get().get(paragraph._p)
    return f"{label}{text}".strip() if label and text else text


def _text_box_texts(paragraph: Paragraph) -> list[str]:
    """Paragraphs of the text boxes anchored in ``paragraph``, each box once.

    A modern box is stored twice: mc:Choice (wps:txbx) and an mc:Fallback VML
    copy for older readers. Boxes nested in a box belong to that box's own
    paragraphs, which this function visits recursively.
    """
    texts: list[str] = []
    for box in paragraph._p.iter(qn("w:txbxContent")):
        between = []
        for ancestor in box.iterancestors():
            if ancestor is paragraph._p:
                break
            between.append(ancestor.tag)
        if _MC_FALLBACK in between or qn("w:txbxContent") in between or qn("w:p") in between:
            continue
        for child in box.iterchildren(qn("w:p")):
            inner = Paragraph(child, paragraph._parent)
            text = _paragraph_text(inner)
            if text:
                texts.append(text)
            texts.extend(_text_box_texts(inner))
    return texts


_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_NUMBER_LABELS: contextvars.ContextVar[dict[Any, str]] = contextvars.ContextVar("word_number_labels", default={})


def _number_labels(document: DocumentType) -> dict[Any, str]:
    """Render Word's automatic list numbers, which never appear in the runs.

    Counts follow Word: lists sharing an abstract definition continue each
    other unless a w:num restarts a level with startOverride, and a level
    restarts whenever a shallower level advances.
    """
    try:
        numbering = document.part.numbering_part.element
    except (KeyError, NotImplementedError):
        return {}
    abstracts = {a.get(qn("w:abstractNumId")): a for a in numbering.iterchildren(qn("w:abstractNum"))}
    instances: dict[str, tuple[str, dict[int, int]]] = {}
    for num in numbering.iterchildren(qn("w:num")):
        abstract_ref = num.find(qn("w:abstractNumId"))
        if abstract_ref is None:
            continue
        overrides = {}
        for override in num.iterchildren(qn("w:lvlOverride")):
            start = override.find(qn("w:startOverride"))
            if start is not None:
                overrides[int(override.get(qn("w:ilvl"), "0"))] = int(start.get(qn("w:val"), "0"))
        instances[num.get(qn("w:numId"))] = (abstract_ref.get(qn("w:val")), overrides)

    counters: dict[str, dict[int, int]] = {}
    started: set[str] = set()
    labels: dict[Any, str] = {}
    for p in document.element.body.iter(qn("w:p")):
        if any(ancestor.tag == _MC_FALLBACK for ancestor in p.iterancestors()):
            continue
        num_id, ilvl = _paragraph_num_ref(p, document)
        if not num_id or num_id == "0" or num_id not in instances:
            continue
        abstract_id, overrides = instances[num_id]
        abstract = abstracts.get(abstract_id)
        if abstract is None:
            continue
        levels = {int(lvl.get(qn("w:ilvl"), "0")): lvl for lvl in abstract.iterchildren(qn("w:lvl"))}
        if ilvl not in levels:
            continue
        counts = counters.setdefault(abstract_id, {})
        if num_id not in started:
            started.add(num_id)
            for level, start in overrides.items():
                counts[level] = start - 1
        level = levels[ilvl]
        counts[ilvl] = counts.get(ilvl, _level_int(level, "w:start", 0) - 1) + 1
        for deeper in [lvl for lvl in counts if lvl > ilvl]:
            del counts[deeper]
        labels[p] = _render_number_label(level, levels, counts)
    return labels


def _paragraph_num_ref(p, document: DocumentType) -> tuple[str | None, int]:
    """numId/ilvl set on the paragraph, else inherited through its style chain."""
    ppr = p.pPr
    num_pr = ppr.numPr if ppr is not None else None
    if num_pr is not None and num_pr.numId is not None:
        return str(num_pr.numId.val), int(num_pr.ilvl.val) if num_pr.ilvl is not None else 0
    style_id = ppr.pStyle.val if ppr is not None and ppr.pStyle is not None else None
    seen = set()
    while style_id and style_id not in seen:
        seen.add(style_id)
        style = document.styles.element.get_by_id(style_id)
        if style is None:
            break
        style_num = style.pPr.numPr if style.pPr is not None else None
        if style_num is not None and style_num.numId is not None:
            return str(style_num.numId.val), int(style_num.ilvl.val) if style_num.ilvl is not None else 0
        based_on = style.basedOn
        style_id = based_on.val if based_on is not None else None
    return None, 0


def _level_int(level, tag: str, default: int) -> int:
    node = level.find(qn(tag))
    try:
        return int(node.get(qn("w:val"))) if node is not None else default
    except (TypeError, ValueError):
        return default


def _render_number_label(level, levels: dict[int, Any], counts: dict[int, int]) -> str:
    fmt_node = level.find(qn("w:numFmt"))
    fmt = fmt_node.get(qn("w:val")) if fmt_node is not None else "decimal"
    text_node = level.find(qn("w:lvlText"))
    template = text_node.get(qn("w:val"), "") if text_node is not None else ""
    if fmt == "none" or not template:
        return ""
    if fmt == "bullet":
        label = "•"
    else:
        legal = level.find(qn("w:isLgl")) is not None

        def placeholder(match):
            ref = int(match.group(1)) - 1
            ref_level = levels.get(ref)
            ref_fmt_node = ref_level.find(qn("w:numFmt")) if ref_level is not None else None
            ref_fmt = ref_fmt_node.get(qn("w:val")) if ref_fmt_node is not None else "decimal"
            return _format_number(counts.get(ref, 1), "decimal" if legal else ref_fmt)

        label = re.sub(r"%([1-9])", placeholder, template)
    suffix_node = level.find(qn("w:suff"))
    suffix = suffix_node.get(qn("w:val")) if suffix_node is not None else "tab"
    return label + ("" if suffix == "nothing" else " ")


_CJK_DIGITS = "〇一二三四五六七八九"


def _cjk_number(value: int) -> str:
    if value <= 0 or value >= 10000:
        return str(value)
    parts = []
    for unit_value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        digit, value = divmod(value, unit_value)
        if digit:
            parts.append(("" if unit == "十" and digit == 1 and not parts else _CJK_DIGITS[digit]) + unit)
        elif parts and value:
            if parts[-1] != "零":
                parts.append("零")
    return "".join(parts)


def _roman(value: int) -> str:
    if value <= 0 or value >= 4000:
        return str(value)
    out = ""
    for number, symbol in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                           (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while value >= number:
            out += symbol
            value -= number
    return out


def _format_number(value: int, fmt: str) -> str:
    if fmt in ("taiwaneseCountingThousand", "taiwaneseCounting", "chineseCounting", "chineseCountingThousand",
               "chineseLegalSimplified", "japaneseCounting"):
        return _cjk_number(value)
    if fmt == "ideographDigital":
        return "".join(_CJK_DIGITS[int(d)] for d in str(value))
    if fmt == "ideographTraditional" and 1 <= value <= 10:
        return "甲乙丙丁戊己庚辛壬癸"[value - 1]
    if fmt == "ideographZodiac" and 1 <= value <= 12:
        return "子丑寅卯辰巳午未申酉戌亥"[value - 1]
    if fmt in ("upperRoman", "lowerRoman"):
        roman = _roman(value)
        return roman if fmt == "upperRoman" else roman.lower()
    if fmt in ("upperLetter", "lowerLetter") and value > 0:
        letter = chr(ord("A") + (value - 1) % 26) * ((value - 1) // 26 + 1)
        return letter if fmt == "upperLetter" else letter.lower()
    if fmt == "decimalZero":
        return f"{value:02d}"
    if fmt == "decimalFullWidth":
        return str(value).translate(str.maketrans("0123456789", "０１２３４５６７８９"))
    if fmt == "decimalEnclosedCircle" and 1 <= value <= 20:
        return chr(0x2460 + value - 1)
    return str(value)


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
        parts.extend(_text_box_texts(paragraph))
    return " / ".join(parts).strip()


# Placeholder name for a table the document never named; not source text.
GENERATED_TABLE_CAPTION = "Word 表格 {}"


def _table_caption(recent_texts: list[str], table_count: int) -> str:
    for text in reversed(recent_texts):
        compact = text.replace(" ", "")
        if len(text) <= 100 and any(token in compact for token in ("表", "Table", "附件")):
            return text
    return GENERATED_TABLE_CAPTION.format(table_count)


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
