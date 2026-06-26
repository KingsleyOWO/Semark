"""Pure rendering helpers used by the package stage."""

import re
from typing import Any

from app.pipeline.semantic.language import normalize_semantic_output_language, page_label
from app.pipeline.structured_rag import parse_html_table

# LaTeX symbol patterns for cleanup
LATEX_CHECKBOX_PATTERNS = [
    # Common checkbox/checkmark LaTeX patterns from DOCX/MinerU
    (r'\$\\?\s*fint\s*\$', '☑'),  # $\ fint$ -> ☑
    (r'\$\\?\s*square\s*\$', '☐'),  # $\square$ -> ☐
    (r'\$\\?\s*boxtimes\s*\$', '☒'),  # $\boxtimes$ -> ☒
    (r'\$\\?\s*checkmark\s*\$', '✓'),  # $\checkmark$ -> ✓
    (r'\$\\?\s*times\s*\$', '✗'),  # $\times$ -> ✗
    # Inline math with checkbox symbols
    (r'\$\s*\\Box\s*\$', '☐'),
    (r'\$\s*\\CheckedBox\s*\$', '☑'),
]

# Compiled patterns for efficiency
_LATEX_CHECKBOX_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in LATEX_CHECKBOX_PATTERNS]


def clean_latex_symbols(text: str) -> str:
    """
    Convert common LaTeX symbols to Unicode equivalents.

    Primarily handles checkbox symbols from DOCX documents that MinerU
    outputs as LaTeX math notation.
    """
    if not text:
        return text

    result = text
    for pattern, replacement in _LATEX_CHECKBOX_COMPILED:
        result = pattern.sub(replacement, result)

    return result


def html_table_to_text(html: str, caption: str | list | None = None) -> str:
    """
    Convert HTML table to RAG-friendly serialized text format.

    Output format:
        TABLE: <caption>
        COLUMNS: col1 | col2 | col3
        ROW: val1 | val2 | val3
        ROW: val4 | val5 | val6

    Features:
    - Preserves row/column structure semantically
    - Folds consecutive empty rows (>2 becomes "...(N empty rows)")
    - Cleans up LaTeX symbols
    """
    if not html or not html.strip():
        return ""

    # Ensure caption is a string (MinerU may return list)
    if isinstance(caption, list):
        caption = " ".join(str(x) for x in caption if x)

    # Parse HTML table
    # Simple regex-based parsing for <tr> and <td>/<th>
    rows: list[list[str]] = []

    # Find all rows
    row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)

    for row_match in row_pattern.finditer(html):
        row_content = row_match.group(1)
        cells = []
        for cell_match in cell_pattern.finditer(row_content):
            cell_text = cell_match.group(1)
            # Strip HTML tags from cell content
            cell_text = re.sub(r'<[^>]+>', '', cell_text)
            cell_text = cell_text.strip()
            cell_text = clean_latex_symbols(cell_text)
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # Build output
    lines: list[str] = []

    # Add caption if provided
    if caption:
        lines.append(f"TABLE: {clean_latex_symbols(caption)}")
    else:
        lines.append("TABLE:")

    # Detect header row (first row with content)
    header_row = None
    data_start = 0
    for i, row in enumerate(rows):
        if any(cell for cell in row):
            header_row = row
            data_start = i + 1
            break

    if header_row:
        lines.append(f"COLUMNS: {' | '.join(header_row)}")

    # Process data rows with empty row folding
    empty_count = 0
    for row in rows[data_start:]:
        # Check if row is empty (all cells empty)
        is_empty = not any(cell for cell in row)

        if is_empty:
            empty_count += 1
        else:
            # Flush empty rows if > 2 consecutive
            if empty_count > 2:
                lines.append(f"...({empty_count} empty rows)")
            elif empty_count > 0:
                # Add individual empty rows
                for _ in range(empty_count):
                    lines.append("ROW: (empty)")
            empty_count = 0

            # Add this row
            lines.append(f"ROW: {' | '.join(row)}")

    # Flush remaining empty rows
    if empty_count > 2:
        lines.append(f"...({empty_count} empty rows)")

    return "\n".join(lines)




def infer_table_asset_title(
    *,
    caption: str | list | None,
    source_title: str = "",
    page_idx: int | None = None,
    table_idx: int = 0,
    semantic_output_language: str = "zh-TW",
) -> str:
    """Infer a human-readable title for split table documents."""

    if isinstance(caption, list):
        caption_text = " ".join(str(x) for x in caption if str(x).strip())
    else:
        caption_text = str(caption or "")
    caption_text = clean_latex_symbols(re.sub(r"\s+", " ", caption_text).strip())
    if caption_text and not _is_generic_table_title(caption_text):
        return caption_text[:100]

    language = _resolve_render_language(semantic_output_language)
    table_label = "Table" if language == "en" else "表格"
    source = clean_latex_symbols(re.sub(r"\s+", " ", str(source_title or "")).strip())
    if source and not _is_generic_table_title(source):
        suffix = f"{table_label} {table_idx + 1}"
        if page_idx is not None:
            return f"{source} {page_label(page_idx, language)} {suffix}"[:100]
        return f"{source} {suffix}"[:100]

    return f"{table_label} {table_idx + 1}"


def _is_generic_table_title(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or "")).lower()
    return bool(
        not compact
        or compact in {"table", "表格", "表", "none", "null", "[]"}
        or re.fullmatch(r"table\d*", compact)
        or re.fullmatch(r"table[一二三四五六七八九十0-9]+", compact)
        or re.fullmatch(r"表格[一二三四五六七八九十0-9]*", compact)
        or re.fullmatch(r"\*?[a-z0-9]{6,}\*?(?:page\d+)?(?:table\d+)?", compact)
        or (
            "page" in compact
            and "table" in compact
            and bool(re.fullmatch(r"[*a-z0-9_-]+page\d+table\d+", compact))
        )
    )

def semantic_table_to_text(
    html: str,
    caption: str | list | None = None,
    semantic_output_language: str = "zh-TW",
) -> str:
    """
    Convert a source table into row-level semantic Markdown for retrieval.

    This keeps the original table as one asset, but serializes each data row as
    a self-contained record so RAG can answer by row keys such as classification
    number, item name, amount, owner unit, or retention period.
    """
    rows = parse_html_table(html)
    if not rows:
        return ""

    language = _resolve_render_language(semantic_output_language)
    if isinstance(caption, list):
        caption = " ".join(str(x) for x in caption if x)
    title = clean_latex_symbols(str(caption or "").strip()) or ("Table" if language == "en" else "表格")

    if table_rows_are_low_confidence(rows):
        return low_confidence_table_to_text(rows, title, semantic_output_language=language)

    header_idx = _detect_semantic_table_header_index(rows)
    if header_idx is None:
        return table_fragment_to_text(rows, title, semantic_output_language=language)

    header = [_clean_table_cell(cell) for cell in rows[header_idx]]
    context_rows = rows[:header_idx]
    data_rows = rows[header_idx + 1 :]
    width = len(header)

    lines: list[str] = [
        f"## {title}",
        "",
        f"{_table_label('table_name', language)}{title}",
        f"{_table_label('columns', language)}{_join_cells((cell for cell in header if cell), language)}",
    ]

    contexts = [_row_to_inline_text(row) for row in context_rows]
    contexts = [text for text in contexts if text]
    if contexts:
        lines.append(f"{_table_label('context', language)}{_join_cells(contexts, language)}")

    lines.extend(["", f"## {_table_heading('rows', language)}"])

    last_values: dict[int, str] = {}
    record_idx = 0
    for raw_row in data_rows:
        row = [_clean_table_cell(cell) for cell in raw_row]
        if not any(row):
            continue
        if len(row) < width:
            row.extend([""] * (width - len(row)))
        elif len(row) > width:
            row = row[:width]
        if _is_repeated_table_header(row, header):
            continue
        if _is_table_section_row(row):
            section_text = _row_to_inline_text(row)
            if section_text and section_text not in lines:
                lines.append(f"{_table_label('context', language)}{section_text}")
            continue

        filled = list(row)
        for idx, value in enumerate(row):
            if value:
                last_values[idx] = value
            elif idx < 2 and last_values.get(idx):
                filled[idx] = last_values[idx]

        data_pairs = [
            (header[idx] or _fallback_column_name(idx, language), value)
            for idx, value in enumerate(filled)
            if value
        ]
        if not data_pairs:
            continue

        record_idx += 1
        heading = _semantic_table_record_heading(data_pairs, record_idx)
        separator = ": " if language == "en" else "："
        lines.extend(["", f"### {heading}"])
        for key, value in data_pairs:
            lines.append(f"- {key}{separator}{value}")

    if record_idx == 0:
        return table_fragment_to_text(rows, title, semantic_output_language=language)

    return "\n".join(lines).strip()


def table_rows_are_low_confidence(rows: list[list[str]]) -> bool:
    """Return True when a parsed table is mostly OCR/layout noise."""

    clean_rows = [
        [_clean_table_cell(cell) for cell in row]
        for row in rows
        if any(_clean_table_cell(cell) for cell in row)
    ]
    cells = [cell for row in clean_rows for cell in row if cell]
    if len(cells) < 8:
        return False

    weak_cells = [cell for cell in cells if _is_weak_table_cell(cell)]
    meaningful_cells = [cell for cell in cells if _is_meaningful_table_cell(cell)]
    weak_ratio = len(weak_cells) / len(cells)
    meaningful_ratio = len(meaningful_cells) / len(cells)

    max_width = max((len(row) for row in clean_rows), default=0)
    sparse_wide_table = max_width >= 5 and meaningful_ratio < 0.35 and weak_ratio >= 0.45
    mostly_weak_cells = weak_ratio >= 0.65 and meaningful_ratio < 0.45
    has_garbled_marker = any(_looks_like_garbled_table_cell(cell) for cell in cells)
    if has_garbled_marker and (sparse_wide_table or weak_ratio >= 0.45):
        return True
    return sparse_wide_table or mostly_weak_cells


def low_confidence_table_to_text(
    rows: list[list[str]],
    title: str,
    semantic_output_language: str = "zh-TW",
) -> str:
    """Render noisy OCR tables without inventing columns or row facts."""

    language = _resolve_render_language(semantic_output_language)
    visible_lines: list[str] = []
    seen: set[str] = set()
    for row in rows:
        cells = [_clean_table_cell(cell) for cell in row if _clean_table_cell(cell)]
        useful = [
            cell
            for cell in cells
            if _is_meaningful_table_cell(cell) and not _looks_like_garbled_table_cell(cell)
        ]
        if not useful:
            continue
        text = (" | " if language == "en" else "；").join(useful)
        key = re.sub(r"\s+", "", text).lower()
        if key and key not in seen:
            seen.add(key)
            visible_lines.append(text)
        if len(visible_lines) >= 12:
            break

    if language == "en":
        lines = [
            f"## {title}",
            "",
            f"{_table_label('table_name', language)}{title}",
            "Content type: low-confidence table OCR",
            "The table OCR is too noisy to generate reliable field-level rows. Only recognizable text is retained.",
        ]
        if visible_lines:
            lines.extend(["", "## Recognizable Text"])
            lines.extend(f"- {line}" for line in visible_lines)
        return "\n".join(lines).strip()

    lines = [
        f"## {title}",
        "",
        f"{_table_label('table_name', language)}{title}",
        "內容類型：低可信度表格 OCR",
        "表格 OCR 品質不足，未產生欄位化資料列；以下僅保留較可信的可辨識文字。",
    ]
    if visible_lines:
        lines.extend(["", "## 可辨識文字"])
        lines.extend(f"- {line}" for line in visible_lines)
    return "\n".join(lines).strip()


def table_fragment_to_text(
    rows: list[list[str]],
    title: str,
    semantic_output_language: str = "zh-TW",
) -> str:
    """Render table fragments without a reliable header as readable notes."""

    clean_rows = [
        [_clean_table_cell(cell) for cell in row]
        for row in rows
        if any(_clean_table_cell(cell) for cell in row)
    ]
    if not clean_rows:
        return ""

    language = _resolve_render_language(semantic_output_language)
    if language == "en":
        return _english_table_fragment_to_sections(clean_rows, title)

    row_separator = " | " if language == "en" else "；"
    row_prefix = "Row" if language == "en" else "第"
    row_suffix = "" if language == "en" else " 列"
    row_colon = ": " if language == "en" else "："

    lines: list[str] = [
        f"## {title}",
        "",
        f"{_table_label('table_name', language)}{title}",
        f"{_table_label('content_type', language)}{_table_heading('fragment', language)}",
        "",
        f"## {_table_heading('content', language)}",
    ]
    for idx, row in enumerate(clean_rows, start=1):
        cells = [cell for cell in row if cell]
        if not cells:
            continue
        if len(cells) == 1:
            lines.append(f"- {row_prefix} {idx}{row_suffix}{row_colon}{cells[0]}")
        else:
            joined = row_separator.join(cells)
            lines.append(f"- {row_prefix} {idx}{row_suffix}{row_colon}{joined}")

    return "\n".join(lines).strip()




def _english_table_fragment_to_sections(rows: list[list[str]], title: str) -> str:
    """Render headerless English form tables as sections instead of Row N dumps."""

    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_items: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_items
        if current_title or current_items:
            sections.append((current_title or "Table Content", current_items))
        current_title = ""
        current_items = []

    for row in rows:
        cells = [_clean_table_cell(cell) for cell in row if _clean_table_cell(cell)]
        if not cells:
            continue
        row_text = _clean_english_table_fragment_text(" | ".join(cells))
        if not row_text or _is_low_value_english_table_row(row_text):
            continue

        step_match = re.search(r"\b(Step\s+\d+\s*:\s*[^|.]+)", row_text, flags=re.IGNORECASE)
        if step_match:
            flush()
            current_title = _clean_english_table_fragment_text(step_match.group(1).rstrip(" :"))
            remainder = _clean_english_table_fragment_text(
                row_text[: step_match.start()] + " " + row_text[step_match.end() :]
            )
            if remainder and not _is_low_value_english_table_row(remainder):
                current_items.append(remainder)
            continue

        current_items.append(row_text)

    flush()

    lines: list[str] = [f"## {title}", ""]
    if not sections:
        lines.append("## Content")
        for row in rows:
            text = _clean_english_table_fragment_text(" | ".join(_clean_table_cell(cell) for cell in row if cell))
            if text and not _is_low_value_english_table_row(text):
                lines.append(f"- {text}")
        return "\n".join(lines).strip()

    lines.append("Content type: form/table fragment")
    for section_title, items in sections:
        lines.extend(["", f"### {section_title}"])
        for item in _dedupe_table_items(items)[:12]:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


def _is_weak_table_cell(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    if not compact:
        return True
    if re.fullmatch(r"[\d.,:：;；/\-]+", compact):
        return True
    if re.fullmatch(r"[□☐☑✓✔✕✗.。·•、,，:：;；_\-]+", compact):
        return True
    if len(compact) <= 1 and not re.search(r"[\u4e00-\u9fff]", compact):
        return True
    if len(compact) <= 2 and re.fullmatch(r"[A-Za-z0-9.]+", compact):
        return True
    return False


def _is_meaningful_table_cell(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    if not compact or _is_weak_table_cell(compact):
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
    alpha_count = len(re.findall(r"[A-Za-z]", compact))
    if cjk_count >= 2 or alpha_count >= 3:
        return True
    return len(compact) >= 4 and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", compact))


def _looks_like_garbled_table_cell(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    if not compact:
        return False
    suspicious_chars = sum(1 for ch in compact if ch in "圖图隊队阅閱样樣精验驗〇○昇鑑箇")
    if suspicious_chars >= 2 and len(compact) <= 24:
        return True
    if re.search(r"[圖图隊队].{0,8}[验驗]|[验驗].{0,8}[圖图隊队]", compact):
        return True
    return False


def _clean_english_table_fragment_text(value: str) -> str:
    text = re.sub(r"\s+", " ", clean_latex_symbols(value or "")).strip()
    text = re.sub(r"\b([A-Z][a-z]+)([A-Z][a-z]+)\b", r"\1 \2", text)
    text = re.sub(r"\s+\|\s+(?=[.)]?\d{1,2}\s*$)", " ", text)
    return text.strip(" |")


def _is_low_value_english_table_row(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    if not compact:
        return True
    if re.fullmatch(r"\*?[a-z0-9]{6,}\*?", compact):
        return True
    if re.fullmatch(r"[0-9.]+", compact):
        return True
    return compact in {"wholedollarsonly", "electroniconlyonecopy"}


def _dedupe_table_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = re.sub(r"\W+", "", item).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _resolve_render_language(semantic_output_language: str) -> str:
    language = normalize_semantic_output_language(semantic_output_language)
    return "en" if language == "en" else "zh-TW"


def _table_label(key: str, language: str) -> str:
    if language == "en":
        return {
            "table_name": "Table name: ",
            "columns": "Columns: ",
            "context": "Category or scope: ",
            "content_type": "Content type: ",
        }[key]
    return {
        "table_name": "表格名稱：",
        "columns": "欄位：",
        "context": "分類或範圍：",
        "content_type": "內容類型：",
    }[key]


def _table_heading(key: str, language: str) -> str:
    if language == "en":
        return {
            "rows": "Rows",
            "fragment": "table fragment or continuation data",
            "content": "Content",
        }[key]
    return {
        "rows": "資料列",
        "fragment": "表格片段或續接資料",
        "content": "內容",
    }[key]


def _fallback_column_name(idx: int, language: str) -> str:
    return f"Column {idx + 1}" if language == "en" else f"欄位{idx + 1}"


def _join_cells(values: Any, language: str) -> str:
    separator = ", " if language == "en" else "、"
    return separator.join(str(value) for value in values if str(value).strip())


def _detect_semantic_table_header_index(rows: list[list[str]]) -> int | None:
    known_header_tokens = {
        "分類號",
        "項目",
        "內容描述",
        "保存年限",
        "文件保管單位",
        "備註",
        "職稱",
        "職級",
        "交通費",
        "每日費用",
        "宿費",
        "雜費",
        "欄位",
        "名稱",
        "說明",
        "金額",
        "單位",
    }
    for idx, row in enumerate(rows[:5]):
        normalized = [_clean_table_cell(cell) for cell in row]
        non_empty = [cell for cell in normalized if cell]
        if len(non_empty) < 2:
            continue
        compact_cells = [re.sub(r"\s+", "", cell) for cell in non_empty]
        token_hits = sum(
            1
            for cell in compact_cells
            if any(token in cell for token in known_header_tokens)
        )
        if token_hits >= 2:
            return idx
        if idx <= 2 and len(non_empty) >= 3 and _next_rows_look_like_data(rows[idx + 1 : idx + 4]):
            return idx
    return None


def _is_repeated_table_header(row: list[str], header: list[str]) -> bool:
    row_compact = [re.sub(r"\s+", "", cell) for cell in row if cell]
    header_compact = [re.sub(r"\s+", "", cell) for cell in header if cell]
    if not row_compact or not header_compact:
        return False
    matches = sum(1 for cell in row_compact if cell in header_compact)
    return matches >= min(3, len(header_compact))


def _is_table_section_row(row: list[str]) -> bool:
    non_empty = [cell for cell in row if cell]
    if len(non_empty) != 1:
        return False
    text = non_empty[0]
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 12:
        return False
    if re.match(r"^\d{5}$", compact):
        return False
    return bool(re.search(r"(?:類|項目|包含|管理)", compact))


def _next_rows_look_like_data(rows: list[list[str]]) -> bool:
    useful_rows = 0
    for row in rows:
        non_empty = [_clean_table_cell(cell) for cell in row if _clean_table_cell(cell)]
        if len(non_empty) >= 2 and any(re.search(r"\d", cell) for cell in non_empty):
            useful_rows += 1
    return useful_rows >= 1


def _clean_table_cell(value: Any) -> str:
    text = clean_latex_symbols(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _row_to_inline_text(row: list[str]) -> str:
    cells = [_clean_table_cell(cell) for cell in row]
    return " ".join(cell for cell in cells if cell).strip()


def _semantic_table_record_heading(pairs: list[tuple[str, str]], record_idx: int) -> str:
    values = {re.sub(r"\s+", "", key): value for key, value in pairs}
    for first_key, second_key in (("分類號", "項目"), ("職稱/職級別", "每日費用"), ("項目", "內容描述")):
        first = values.get(first_key)
        second = values.get(second_key)
        if first and second:
            return f"{first} {second}"[:80]
    for key in ("分類號", "項目", "名稱", "職稱", "職級", "內容描述"):
        value = values.get(key)
        if value:
            return value[:80]
    first_value = pairs[0][1] if pairs else ""
    return (first_value or f"資料列 {record_idx}")[:80]


def clean_html_table(html: str) -> str:
    """
    Clean HTML table by removing excessive empty rows.

    Keeps HTML structure but folds consecutive empty rows into a comment.
    """
    if not html or not html.strip():
        return ""

    # Find all rows
    row_pattern = re.compile(r'<tr[^>]*>.*?</tr>', re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)

    rows = list(row_pattern.finditer(html))
    if not rows:
        return html

    # Classify rows as empty or not
    row_info: list[tuple[str, bool]] = []  # (row_html, is_empty)
    for row_match in rows:
        row_html = row_match.group(0)
        cells = cell_pattern.findall(row_html)
        is_empty = all(not re.sub(r'<[^>]+>', '', c).strip() for c in cells)
        row_info.append((row_html, is_empty))

    # Build result with empty row folding
    result_rows: list[str] = []
    empty_count = 0

    for row_html, is_empty in row_info:
        if is_empty:
            empty_count += 1
        else:
            # Flush empty rows
            if empty_count > 2:
                result_rows.append(f"<!-- {empty_count} empty rows omitted -->")
            elif empty_count > 0:
                # Keep individual empty rows (up to 2)
                for _ in range(empty_count):
                    result_rows.append(row_info[0][0] if row_info else "<tr><td></td></tr>")
            empty_count = 0
            result_rows.append(row_html)

    # Handle trailing empty rows
    if empty_count > 2:
        result_rows.append(f"<!-- {empty_count} empty rows omitted -->")

    # Reconstruct HTML: replace original rows with cleaned rows
    # Find table start and end
    table_start = html.find("<table")
    table_end = html.rfind("</table>")

    if table_start == -1 or table_end == -1:
        # No table tags, just return joined rows
        return "\n".join(result_rows)

    # Find the content between <table...> and </table>
    table_open_end = html.find(">", table_start) + 1

    return html[:table_open_end] + "\n".join(result_rows) + html[table_end:]
