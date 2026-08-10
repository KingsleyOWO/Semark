"""Structured RAG planning and row-level extraction helpers."""

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.models.document_ir import BlockType, DocumentIR, coerce_payload_text
from app.pipeline.corpus_rules import get_rules as get_corpus_rules
from app.pipeline.delivery import atomic_write_json, atomic_write_jsonl, atomic_write_text
from app.pipeline.semantic.language import (
    display_form_section,
    form_template_sections,
    get_form_language_pack,
    resolve_semantic_output_language,
)
from app.pipeline.semantic.language import (
    page_label as semantic_page_label,
)
from app.pipeline.semantic.language import (
    requirement_label as semantic_requirement_label,
)
from app.pipeline.semantic.normalizer import (
    clean_title_noise as semantic_clean_title_noise,
)
from app.pipeline.semantic.normalizer import (
    fields_to_dicts as semantic_fields_to_dicts,
)
from app.pipeline.semantic.normalizer import (
    normalize_fields as semantic_normalize_fields,
)
from app.pipeline.semantic.normalizer import (
    normalize_notes as semantic_normalize_notes,
)
from app.pipeline.semantic.normalizer import (
    source_title_from_path as semantic_source_title_from_path,
)

_LATEX_CHECKBOX_PATTERNS = [
    (r"\$\\?\s*fint\s*\$", "☑"),
    (r"\$\\?\s*square\s*\$", "☐"),
    (r"\$\\?\s*boxtimes\s*\$", "☒"),
    (r"\$\\?\s*checkmark\s*\$", "✓"),
    (r"\$\\?\s*times\s*\$", "✗"),
    (r"\$\s*\\Box\s*\$", "☐"),
    (r"\$\s*\\CheckedBox\s*\$", "☑"),
]
_LATEX_CHECKBOX_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _LATEX_CHECKBOX_PATTERNS
]


def clean_latex_symbols(text: str) -> str:
    """Convert common LaTeX checkbox symbols to Unicode."""

    result = text or ""
    for pattern, replacement in _LATEX_CHECKBOX_COMPILED:
        result = pattern.sub(replacement, result)
    return result


@dataclass
class DocumentPlan:
    """Planner output for downstream structured extraction."""

    document_type: str
    title: str
    effective_date: str | None = None
    currency: str | None = None
    query_granularity: str = "block"
    primary_entities: list[str] = field(default_factory=list)
    recommended_schema: dict[str, str] = field(default_factory=dict)
    rag_strategy: str = "default_chunks"
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "title": self.title,
            "effective_date": self.effective_date,
            "currency": self.currency,
            "query_granularity": self.query_granularity,
            "primary_entities": self.primary_entities,
            "recommended_schema": self.recommended_schema,
            "rag_strategy": self.rag_strategy,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class StructuredRagOutput:
    """All structured RAG artifacts generated from a DocumentIR."""

    plan: DocumentPlan
    records: list[dict[str, Any]]
    rag_markdown: str
    chunks: list[dict[str, Any]]
    stats: dict[str, Any]


class _TableHTMLParser(HTMLParser):
    """Small table parser that preserves rowspan/colspan attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, Any]]] = []
        self._current_row: list[dict[str, Any]] | None = None
        self._current_cell: dict[str, Any] | None = None
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            attr_map = {key.lower(): value for key, value in attrs}
            self._current_cell = {
                "rowspan": _safe_int(attr_map.get("rowspan"), 1),
                "colspan": _safe_int(attr_map.get("colspan"), 1),
            }
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            text = clean_latex_symbols(" ".join(self._cell_parts))
            text = re.sub(r"\s+", " ", text).strip()
            self._current_cell["text"] = text
            self._current_row.append(self._current_cell)
            self._current_cell = None
            self._cell_parts = []
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None


def build_structured_rag(
    document_ir: DocumentIR,
    semantic_output_language: str = "auto",
) -> StructuredRagOutput:
    """Build planner output, normalized records, and row-level RAG artifacts."""

    language = resolve_semantic_output_language(semantic_output_language, document_ir)
    plan = plan_document(document_ir)
    records: list[dict[str, Any]] = []

    if plan.document_type == "travel_daily_allowance_table":
        records = extract_travel_allowance_records(document_ir, plan)
    elif plan.document_type == "travel_domestic_expense_rate_table":
        records = extract_domestic_travel_rate_records(document_ir, plan)

    records.extend(_collect_adjacent_table_note_records(document_ir, plan, records, len(records)))

    rag_markdown = render_structured_rag_markdown(plan, records, semantic_output_language=language)
    chunks = build_structured_chunks(document_ir, records)

    stats = {
        "document_type": plan.document_type,
        "record_count": len(records),
        "chunk_count": len(chunks),
        "records_by_block": _count_by(records, "block_id"),
        "needs_review_count": sum(1 for record in records if record.get("needs_review")),
        "semantic_output_language": language,
    }
    return StructuredRagOutput(
        plan=plan,
        records=records,
        rag_markdown=rag_markdown,
        chunks=chunks,
        stats=stats,
    )


def build_form_documents_rag(
    document_ir: DocumentIR,
    enrichments: dict[str, dict[str, Any]],
    semantic_output_language: str = "auto",
) -> StructuredRagOutput:
    """Build sub-document RAG artifacts for forms detected inside one source PDF."""

    language = resolve_semantic_output_language(
        semantic_output_language,
        document_ir,
        json.dumps(enrichments, ensure_ascii=False),
    )
    source_plan = plan_document(document_ir)
    if source_plan.document_type in {"travel_daily_allowance_table", "travel_domestic_expense_rate_table"}:
        return StructuredRagOutput(plan=source_plan, records=[], rag_markdown="", chunks=[], stats={})

    form_outputs = _collect_form_outputs(enrichments)
    form_outputs = _dedupe_form_outputs_by_page(form_outputs)
    if not form_outputs:
        form_outputs = _collect_form_like_table_outputs(document_ir)
    if not form_outputs:
        form_outputs = _collect_form_like_text_page_outputs(
            document_ir,
            semantic_output_language=language,
        )
    form_outputs = _merge_spreadsheet_single_form_outputs(document_ir, form_outputs)
    if not form_outputs:
        plan = DocumentPlan(
            document_type="generic_document",
            title=document_ir.source.path,
            confidence=0.0,
        )
        return StructuredRagOutput(plan=plan, records=[], rag_markdown="", chunks=[], stats={})

    single_form_document = _should_merge_forms_as_single_source_document(
        document_ir=document_ir,
        form_outputs=form_outputs,
    )
    plan = DocumentPlan(
        document_type="form_document" if single_form_document else "form_collection",
        title=_form_title_from_source_path(document_ir.source.path) if single_form_document else document_ir.source.path,
        query_granularity="whole_document_semantic_sections" if single_form_document else "one_chunk_per_form_unit",
        primary_entities=["form", "section", "field", "workflow", "attachment_rule"],
        rag_strategy="single_form_document_semantic_chunks" if single_form_document else "form_subdocuments_with_parent_context",
        confidence=0.88 if single_form_document else 0.85,
        evidence={
            "form_page_count": len(form_outputs),
            "single_source_form_document": single_form_document,
        },
    )

    records: list[dict[str, Any]] = []
    for form_idx, form in enumerate(form_outputs):
        records.extend(
            _records_from_form_output(
                document_ir,
                form,
                form_idx,
                len(records),
                semantic_output_language=language,
            )
        )
    records = _filter_low_value_form_records(records)
    if single_form_document and records:
        records = _collapse_form_records_to_single_source_document(
            document_ir=document_ir,
            records=records,
            semantic_output_language=language,
        )
        plan.title = str(records[0].get("form_name") or plan.title)
    if not records:
        empty_plan = DocumentPlan(
            document_type="generic_document",
            title=document_ir.source.path,
            confidence=0.0,
        )
        return StructuredRagOutput(
            plan=empty_plan,
            records=[],
            rag_markdown="",
            chunks=[],
            stats={
                "document_type": empty_plan.document_type,
                "form_count": 0,
                "record_count": 0,
                "chunk_count": 0,
                "records_by_subdoc": {},
                "needs_review_count": 0,
                "semantic_output_language": language,
            },
        )
    plan.evidence["form_page_count"] = len({record["subdoc_id"] for record in records})
    if single_form_document:
        plan.evidence["source_page_indices"] = _available_page_indices(document_ir)

    supporting_sections = [] if single_form_document else _collect_supporting_page_sections(document_ir, records, language)
    rag_markdown = render_form_documents_markdown(plan, records, semantic_output_language=language)
    supporting_markdown = _render_supporting_page_sections(supporting_sections, language)
    if supporting_markdown:
        rag_markdown = f"{rag_markdown.rstrip()}\n\n{supporting_markdown}"
    chunks = build_form_chunks(document_ir, records)
    chunks.extend(_build_supporting_page_chunks(document_ir, supporting_sections, language))
    stats = {
        "document_type": plan.document_type,
        "form_count": len({record["subdoc_id"] for record in records}),
        "supporting_page_count": len(supporting_sections),
        "record_count": len(records),
        "chunk_count": len(chunks),
        "records_by_subdoc": _count_by(records, "subdoc_id"),
        "needs_review_count": sum(1 for record in records if record.get("needs_review")),
        "semantic_output_language": language,
    }
    return StructuredRagOutput(
        plan=plan,
        records=records,
        rag_markdown=rag_markdown,
        chunks=chunks,
        stats=stats,
    )


async def build_structured_rag_with_vlm_fallback(
    document_ir: DocumentIR,
    run_path: Path,
    vlm_adapter: Any,
    max_pages: int = 5,
    semantic_output_language: str = "auto",
) -> StructuredRagOutput:
    """Build structured RAG and use VLM page extraction for MinerU-missed table pages."""

    language = resolve_semantic_output_language(semantic_output_language, document_ir)
    base_output = build_structured_rag(document_ir, semantic_output_language=language)
    if base_output.plan.document_type != "travel_daily_allowance_table":
        return base_output

    fallback_pages = select_vlm_fallback_pages(document_ir, base_output.records, max_pages)
    if not fallback_pages:
        return base_output

    records = list(base_output.records)
    seq = len(records)
    vlm_stats = {
        "attempted_pages": [],
        "successful_pages": [],
        "failed_pages": [],
        "record_count": 0,
    }

    for page_idx in fallback_pages:
        page_image = _resolve_page_image(document_ir, run_path, page_idx)
        if page_image is None:
            vlm_stats["failed_pages"].append({"page_idx": page_idx, "reason": "missing_page_image"})
            continue

        vlm_stats["attempted_pages"].append(page_idx)
        result = await vlm_adapter.extract_structured_table_records(
            image_path=page_image,
            document_plan=base_output.plan.to_dict(),
            context_text=_page_context(document_ir, page_idx),
            doc_id=document_ir.doc_id,
            run_id=document_ir.run_id,
            page_idx=page_idx,
        )
        if not result.success:
            vlm_stats["failed_pages"].append({"page_idx": page_idx, "reason": result.error})
            continue

        added = normalize_vlm_table_records(
            output=result.output,
            document_ir=document_ir,
            plan=base_output.plan,
            page_idx=page_idx,
            seq_start=seq,
            needs_review=result.needs_review,
        )
        records.extend(added)
        seq += len(added)
        vlm_stats["successful_pages"].append(page_idx)
        vlm_stats["record_count"] += len(added)

    rag_markdown = render_structured_rag_markdown(base_output.plan, records, semantic_output_language=language)
    chunks = build_structured_chunks(document_ir, records)
    stats = {
        "document_type": base_output.plan.document_type,
        "record_count": len(records),
        "chunk_count": len(chunks),
        "records_by_block": _count_by(records, "block_id"),
        "needs_review_count": sum(1 for record in records if record.get("needs_review")),
        "vlm_fallback": vlm_stats,
        "semantic_output_language": language,
    }
    return StructuredRagOutput(
        plan=base_output.plan,
        records=records,
        rag_markdown=rag_markdown,
        chunks=chunks,
        stats=stats,
    )


def plan_document(document_ir: DocumentIR) -> DocumentPlan:
    """Plan a document-specific extraction schema from headings and table headers."""

    texts = [
        str(block.payload.get("text", ""))
        for block in document_ir.blocks[:12]
        if block.type == BlockType.TEXT
    ]
    title = _clean_document_title(
        next(
            (text for text in texts if "表" in text),
            texts[0] if texts else document_ir.source.path,
        )
    )
    context = "\n".join(texts)
    table_headers = " ".join(
        _plain_text(str(block.payload.get("table_body", "")))[:500]
        for block in document_ir.blocks
        if block.type == BlockType.TABLE
    )
    evidence_text = f"{context}\n{table_headers}"

    corpus_rules = get_corpus_rules()
    is_allowance = (
        any(term in evidence_text for term in corpus_rules.keyword_list("daily_allowance_amount_terms"))
        and any(term in evidence_text for term in corpus_rules.keyword_list("daily_allowance_location_terms"))
        and re.search("|".join(corpus_rules.keyword_list("daily_allowance_rate_patterns")), evidence_text)
    )
    is_domestic_expense_rate = _looks_like_domestic_expense_rate_table(evidence_text)

    effective_date = _extract_effective_date(evidence_text)
    currency = "USD" if ("美元" in evidence_text or "USD" in evidence_text.upper()) else None

    if is_domestic_expense_rate:
        return DocumentPlan(
            document_type="travel_domestic_expense_rate_table",
            title=title,
            effective_date=effective_date,
            currency="TWD",
            query_granularity="one_record_per_role_rate",
            primary_entities=["role_title", "transport_fee_rule", "lodging_fee", "miscellaneous_fee"],
            recommended_schema={
                "role_title": "string",
                "transport_fee_rule": "string|null",
                "lodging_weekday_twd": "number|null",
                "lodging_holiday_twd": "number|null",
                "miscellaneous_twd": "number|null",
                "source_page": "number",
            },
            rag_strategy="row_level_chunks_with_parent_context",
            confidence=0.88,
            evidence={
                "matched_terms": ["職稱/職級別", "交通費", "宿費", "雜費"],
                "sample_title": title,
            },
        )

    if is_allowance:
        return DocumentPlan(
            document_type="travel_daily_allowance_table",
            title=title,
            effective_date=effective_date,
            currency=currency,
            query_granularity="one_record_per_location_rate",
            primary_entities=["region", "country", "city", "allowance_amount"],
            recommended_schema={
                "region": "string|null",
                "country_zh": "string|null",
                "country_en": "string|null",
                "city_zh": "string|null",
                "city_en": "string|null",
                "location_label": "string",
                "location_type": "region|country|city|other|condition",
                "rate_usd": "number|null",
                "condition": "string|null",
                "source_page": "number",
            },
            rag_strategy="row_level_chunks_with_parent_context",
            confidence=0.9,
            evidence={
                "matched_terms": ["生活費", "日支數額", "地區/國家/城市"],
                "sample_title": title,
            },
        )

    return DocumentPlan(
        document_type="generic_document",
        title=title,
        effective_date=effective_date,
        currency=currency,
        confidence=0.4,
    )


def extract_travel_allowance_records(
    document_ir: DocumentIR,
    plan: DocumentPlan,
) -> list[dict[str, Any]]:
    """Extract row-level travel allowance records from MinerU table blocks."""

    records: list[dict[str, Any]] = []
    seq = 0
    for block in document_ir.blocks:
        if block.type != BlockType.TABLE:
            continue
        table_body = str(block.payload.get("table_body", "") or "")
        rows = parse_html_table(table_body)
        if not rows:
            continue

        context = {
            "region": None,
            "country_zh": None,
            "country_en": None,
            "last_location_label": None,
        }
        for row in rows:
            record = _record_from_allowance_row(
                row=row,
                context=context,
                block_id=block.block_id,
                page_idx=block.page_idx,
                doc_id=document_ir.doc_id,
                run_id=document_ir.run_id,
                title=plan.title,
                effective_date=plan.effective_date,
                currency=plan.currency,
                seq=seq,
            )
            if record:
                records.append(record)
                seq += 1

    return records


def extract_domestic_travel_rate_records(
    document_ir: DocumentIR,
    plan: DocumentPlan,
) -> list[dict[str, Any]]:
    """Extract row-level domestic travel expense rates by role/grade."""

    records: list[dict[str, Any]] = []
    seq = 0
    for block in document_ir.blocks:
        if block.type != BlockType.TABLE:
            continue
        rows = parse_html_table(str(block.payload.get("table_body", "") or ""))
        for row in rows:
            record = _record_from_domestic_travel_rate_row(
                row=row,
                block_id=block.block_id,
                page_idx=block.page_idx,
                doc_id=document_ir.doc_id,
                run_id=document_ir.run_id,
                title=plan.title,
                effective_date=plan.effective_date,
                seq=seq,
            )
            if record:
                records.append(record)
                seq += 1
    return records


def _collect_adjacent_table_note_records(
    document_ir: DocumentIR,
    plan: DocumentPlan,
    existing_records: list[dict[str, Any]],
    seq_start: int,
) -> list[dict[str, Any]]:
    """Attach explanatory notes that immediately follow structured table blocks."""

    if plan.document_type not in {"travel_daily_allowance_table", "travel_domestic_expense_rate_table"}:
        return []

    table_block_ids = {str(record.get("block_id") or "") for record in existing_records}
    if not table_block_ids:
        return []

    block_by_id = {block.block_id: block for block in document_ir.blocks}
    note_records: list[dict[str, Any]] = []
    seq = seq_start
    seen_notes: set[str] = set()

    for block_id in sorted(table_block_ids):
        table_block = block_by_id.get(block_id)
        if table_block is None:
            continue
        notes = _collect_notes_after_block(document_ir, table_block.block_id, plan.title)
        if not notes:
            continue
        note_text = "\n".join(notes).strip()
        normalized_note = re.sub(r"\s+", "", note_text)
        if not normalized_note or normalized_note in seen_notes:
            continue
        seen_notes.add(normalized_note)
        note_records.append(
            {
                "record_id": f"note{seq:06d}",
                "document_type": "table_note",
                "parent_document_type": plan.document_type,
                "doc_id": document_ir.doc_id,
                "run_id": document_ir.run_id,
                "block_id": table_block.block_id,
                "source_page_idx": table_block.page_idx,
                "source_title": plan.title,
                "effective_date": plan.effective_date,
                "currency": plan.currency,
                "note_text": note_text,
                "needs_review": False,
                "review_reasons": [],
            }
        )
        seq += 1

    return note_records


def _collect_notes_after_block(document_ir: DocumentIR, block_id: str, title: str) -> list[str]:
    try:
        start_idx = next(idx for idx, block in enumerate(document_ir.blocks) if block.block_id == block_id)
    except StopIteration:
        return []

    title_compact = re.sub(r"\s+", "", title or "")
    notes: list[str] = []
    started = False
    scanned = 0

    for block in document_ir.blocks[start_idx + 1:]:
        if block.type == BlockType.TABLE:
            break
        if block.type != BlockType.TEXT:
            continue
        scanned += 1
        if scanned > 40:
            break

        text = _normalize_note_line(str(block.payload.get("text") or ""))
        if not text or _is_page_noise(text, title_compact):
            continue

        is_note_start = _looks_like_note_start(text)
        if is_note_start:
            started = True
        elif not started:
            continue

        if started:
            notes.append(text)

    return _merge_note_lines(notes)


def _normalize_note_line(text: str) -> str:
    text = clean_latex_symbols(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_note_start(text: str) -> bool:
    return bool(re.match(r"^(?:備註|備注|註[:：]|[一二三四五六七八九十]+、|\d+[.、])", text))


def _is_page_noise(text: str, title_compact: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if compact.isdigit() or re.match(r"^[壹貳參肆伍陸柒捌玖拾一二三四五六七八九十\-－_—\s\d]+$", text):
        return True
    if title_compact and compact == title_compact:
        return True
    if (
        not _looks_like_note_start(text)
        and not re.search(r"[。；;，,：:]", text)
        and len(compact) <= 40
        and re.search(r"(?:辦法|規程|規章|要點|準則)$", compact)
    ):
        return True
    if len(compact) <= 8 and not _looks_like_note_start(text) and not re.search(r"[。；;]", text):
        return True
    return False


def _merge_note_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if _looks_like_note_start(line) or not merged:
            merged.append(line)
        else:
            merged[-1] = f"{merged[-1]}{line}"
    return merged


def parse_html_table(table_body: str) -> list[list[str]]:
    """Parse table HTML into a rectangular-ish grid, expanding row/col spans."""

    if "<tr" not in table_body.lower():
        return _parse_plain_rows(table_body)

    parser = _TableHTMLParser()
    parser.feed(table_body)

    grid: list[list[str]] = []
    rowspans: dict[int, tuple[int, str]] = {}
    for parsed_row in parser.rows:
        row: list[str] = []
        col_idx = 0
        for cell in parsed_row:
            while col_idx in rowspans:
                remaining, text = rowspans[col_idx]
                row.append(text)
                if remaining <= 1:
                    del rowspans[col_idx]
                else:
                    rowspans[col_idx] = (remaining - 1, text)
                col_idx += 1

            text = str(cell.get("text", "")).strip()
            rowspan = max(1, int(cell.get("rowspan") or 1))
            colspan = max(1, int(cell.get("colspan") or 1))
            for offset in range(colspan):
                row.append(text if offset == 0 else "")
                if rowspan > 1:
                    rowspans[col_idx + offset] = (rowspan - 1, text if offset == 0 else "")
            col_idx += colspan

        while col_idx in rowspans:
            remaining, text = rowspans[col_idx]
            row.append(text)
            if remaining <= 1:
                del rowspans[col_idx]
            else:
                rowspans[col_idx] = (remaining - 1, text)
            col_idx += 1

        normalized = [_normalize_cell(cell) for cell in row]
        if any(normalized):
            grid.append(normalized)

    return grid


def looks_like_reference_table(table_body: str) -> bool:
    """Return true for dense data/reference tables that should not become form assets."""

    rows = parse_html_table(table_body)
    if len(rows) < 3:
        return False

    text = _plain_text(table_body)
    if _has_strong_fillable_form_markers(text):
        return False

    widths = [len(row) for row in rows if row]
    max_width = max(widths or [0])
    if max_width < 3:
        return False

    non_empty_cells = sum(1 for row in rows for cell in row if cell.strip())
    total_cells = sum(max_width for _ in rows)
    density = non_empty_cells / total_cells if total_cells else 0.0

    data_rows = [row for row in rows if sum(1 for cell in row if cell.strip()) >= 2]
    repeated_shape_rows = [row for row in rows if len(row) >= 3 and sum(1 for cell in row if cell.strip()) >= 3]
    numeric_cells = sum(1 for row in rows for cell in row if re.search(r"\d", cell))
    amount_or_unit_cells = sum(
        1
        for row in rows
        for cell in row
        if re.search(r"\d[\d,]*(?:\.\d+)?", cell) or any(unit in cell for unit in ["元", "美元", "USD", "%"])
    )

    has_header_like_row = any(
        sum(1 for cell in row if cell.strip()) >= 3
        and not any(_has_blank_placeholder(cell) for cell in row)
        for row in rows[:3]
    )

    return bool(
        len(data_rows) >= 3
        and has_header_like_row
        and (density >= 0.45 or len(repeated_shape_rows) >= 3)
        and (amount_or_unit_cells >= 2 or numeric_cells >= 3 or len(repeated_shape_rows) >= 4)
    )


def _has_strong_fillable_form_markers(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if any(mark in text for mark in ["□", "☐", "☑", "☒"]):
        return True
    if re.search(r"_{3,}|＿{3,}|-{5,}", text):
        return True
    if re.search(r"[：:]\s*(?:年\s*月\s*日|\s{3,})", text):
        return True
    strong_terms = ["申請人", "申請單位", "填寫", "填表", "請勾選", "簽名", "簽章", "簽核欄"]
    if any(term in compact for term in strong_terms):
        return True
    return False


def _has_blank_placeholder(text: str) -> bool:
    return bool(re.search(r"_{3,}|＿{3,}|-{5,}|\s{4,}", text))


def render_structured_rag_markdown(
    plan: DocumentPlan,
    records: list[dict[str, Any]],
    semantic_output_language: str = "zh-TW",
) -> str:
    language = resolve_semantic_output_language(semantic_output_language)
    lines = [f"# {plan.title}", ""]
    if plan.effective_date:
        label = "Effective date" if language == "en" else "生效日期"
        lines.append(f"{label}：{plan.effective_date}")
    if plan.currency:
        label = "Currency" if language == "en" else "幣別"
        lines.append(f"{label}：{plan.currency}")
    if plan.effective_date or plan.currency:
        lines.append("")

    plan_title_compact = re.sub(r"\s+", "", _clean_form_title(plan.title))
    for record in records:
        text = record_to_rag_text(record)
        text_compact = re.sub(r"\s+", "", _clean_form_title(text.rstrip("。")))
        if text_compact == plan_title_compact:
            continue
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_structured_chunks(
    document_ir: DocumentIR,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one RAG chunk per normalized record."""

    chunks = []
    for idx, record in enumerate(records):
        chunks.append(
            {
                "chunk_id": f"sr{idx:06d}",
                "doc_id": document_ir.doc_id,
                "run_id": document_ir.run_id,
                "view": "structured_rag",
                "content": record_to_rag_text(record),
                "block_ids": [record["block_id"]],
                "page_indices": [record["source_page_idx"]],
                "attachments": [],
                "metadata": {
                    "record_id": record["record_id"],
                    "document_type": record["document_type"],
                    "region": record.get("region"),
                    "country_zh": record.get("country_zh"),
                    "city_zh": record.get("city_zh"),
                    "rate_usd": record.get("rate_usd"),
                    "role_title": record.get("role_title"),
                    "lodging_weekday_twd": record.get("lodging_weekday_twd"),
                    "lodging_holiday_twd": record.get("lodging_holiday_twd"),
                    "miscellaneous_twd": record.get("miscellaneous_twd"),
                    "needs_review": record.get("needs_review", False),
                },
            }
        )
    return chunks


def render_form_documents_markdown(
    plan: DocumentPlan,
    records: list[dict[str, Any]],
    semantic_output_language: str = "zh-TW",
) -> str:
    language = resolve_semantic_output_language(semantic_output_language)
    pack = get_form_language_pack(language)
    lines = [f"# {plan.title}", ""]
    skip_sections = {"Use Cases", "適用場景"}
    for _, subdoc_records in _group_form_records(records):
        if not subdoc_records:
            continue
        first = subdoc_records[0]
        form_name = str(first.get("form_name") or plan.title)
        page_title = "Page" if language == "en" else "頁碼"
        lines.append(f"## {form_name}")
        separator = ": " if language == "en" else "："
        lines.append(f"{page_title}{separator}{first['page_label']}")
        lines.append("")

        summary = next(
            (
                record
                for record in subdoc_records
                if record.get("content_type") == "form_summary"
            ),
            None,
        )
        if summary:
            summary_text = _clean_rendered_form_record_content(
                str(summary["content"]),
                form_name=form_name,
                section=str(summary.get("section") or ""),
                language=language,
            )
            lines.append(_compact_text_no_ellipsis(summary_text, 900))
            lines.append("")

        section_records = [
            record
            for record in subdoc_records
            if record.get("content_type")
            in {"form_section", "form_workflow", "form_attachment_rule"}
        ]
        if plan.document_type == "form_document":
            section_records = _merge_single_form_document_section_records(
                section_records,
                form_name=form_name,
                language=language,
            )
        for record in section_records:
            section_name = str(record.get("section") or "")
            if plan.document_type == "form_document" and section_name in skip_sections:
                continue
            lines.append(f"### {section_name}")
            section_limit = 2800 if any(term in section_name for term in ("注意事項", "備註", "Notes")) else 900
            content = _clean_rendered_form_record_content(
                str(record["content"]),
                form_name=form_name,
                section=section_name,
                language=language,
            )
            lines.append(_compact_text_no_ellipsis(content, section_limit))
            lines.append("")

        field_records = [
            record for record in subdoc_records if record.get("content_type") == "form_field"
        ]
        if field_records:
            lines.append(f"### {pack.field_descriptions}")
            for section, section_fields in _group_fields_by_section(field_records):
                field_items = []
                for record in section_fields:
                    requirement = str(record.get("requirement") or ("required" if record.get("required") else "situational"))
                    field_type = record.get("input_type") or "text"
                    requirement_text = semantic_requirement_label(requirement, language)
                    field_items.append(f"{record['field_name']}({requirement_text}, {field_type})")
                separator = ", " if language == "en" else "、"
                suffix = "." if language == "en" else "。"
                section_label = display_form_section(str(section), language)
                lines.append(f"- {section_label}: {separator.join(field_items)}{suffix}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _clean_rendered_form_record_content(text: str, *, form_name: str, section: str, language: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    if language == "en":
        cleaned = re.sub(r"^Form:\s*[^.]{1,220}\.\s*Section:\s*[^.]{1,140}\.\s*", "", cleaned)
        cleaned = re.sub(r"^Form:\s*[^.]{1,220}\.\s*Field:\s*[^.]{1,140}\.\s*Section:\s*[^.]{1,140}\.\s*", "", cleaned)
        cleaned = re.sub(r"^Form:\s*[^.]{1,220}\.\s*", "", cleaned)
        return cleaned.strip()
    cleaned = re.sub(r"^表單：[^。]{1,220}。區塊：[^。]{1,140}。", "", cleaned)
    cleaned = re.sub(r"^表單：[^。]{1,220}。欄位：[^。]{1,140}。區塊：[^。]{1,140}。", "", cleaned)
    cleaned = re.sub(r"^表單：[^。]{1,220}。", "", cleaned)
    return cleaned.strip()


def _merge_single_form_document_section_records(
    records: list[dict[str, Any]],
    *,
    form_name: str,
    language: str,
) -> list[dict[str, Any]]:
    """Render official multi-page forms as one semantic document, not page dumps."""

    raw_source_sections = {
        "Source Extracted Text",
        "Original Extraction Supplement",
    } if language == "en" else set()
    keep_first_only = {"RAG Query Summary", "RAG 查詢摘要"}
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    seen_by_section: dict[str, set[str]] = {}

    for record in records:
        section = str(record.get("section") or "").strip()
        if not section or section in raw_source_sections:
            continue
        content = _clean_rendered_form_record_content(
            str(record.get("content") or ""),
            form_name=form_name,
            section=section,
            language=language,
        )
        if not content:
            continue
        if section in keep_first_only and section in grouped:
            continue
        normalized = re.sub(r"\s+", " ", content).strip().lower()
        if not normalized:
            continue
        seen = seen_by_section.setdefault(section, set())
        if normalized in seen:
            continue
        seen.add(normalized)
        if section not in grouped:
            grouped[section] = dict(record)
            grouped[section]["content"] = content
            order.append(section)
        else:
            grouped[section]["content"] = f"{grouped[section]['content']}\n\n{content}"

    return [grouped[section] for section in order]


def build_form_chunks(
    document_ir: DocumentIR,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build RAG chunks for form sub-documents."""

    chunks = []
    for idx, record in enumerate(records):
        language = str(record.get("semantic_output_language") or "zh-TW")
        section = str(record.get("section") or "")
        section_label = display_form_section(section, language) if section else section
        chunks.append(
            {
                "chunk_id": f"sf{idx:06d}",
                "doc_id": str(record.get("logical_doc_id") or document_ir.doc_id),
                "run_id": document_ir.run_id,
                "view": "structured_form",
                "content": record["content"],
                "block_ids": [record["block_id"]],
                "page_indices": record["page_indices"],
                "attachments": [],
                "metadata": {
                    "record_id": record["record_id"],
                    "document_type": record["document_type"],
                    "logical_doc_id": record.get("logical_doc_id"),
                    "parent_doc_id": document_ir.doc_id,
                    "source_doc_id": document_ir.doc_id,
                    "subdoc_id": record["subdoc_id"],
                    "subdoc_type": "form",
                    "content_type": record["content_type"],
                    "form_name": record["form_name"],
                    "section": section_label,
                    "field_name": record.get("field_name"),
                    "input_type": record.get("input_type"),
                    "required": record.get("required"),
                    "requirement": record.get("requirement"),
                    "needs_review": record.get("needs_review", False),
                },
            }
        )
    return chunks



def _covered_form_pages(records: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for page_idx in record.get("page_indices") or []:
            try:
                pages.add(int(page_idx))
            except (TypeError, ValueError):
                continue
    return pages


def _collect_supporting_page_sections(
    document_ir: DocumentIR,
    records: list[dict[str, Any]],
    semantic_output_language: str,
) -> list[dict[str, Any]]:
    covered_pages = _covered_form_pages(records)
    sections: list[dict[str, Any]] = []
    for page in document_ir.pages:
        page_idx = int(page.page_idx)
        if page_idx in covered_pages:
            continue
        block_ids: list[str] = []
        parts: list[str] = []
        for block in document_ir.get_blocks_by_page(page_idx):
            if block.type not in {BlockType.TEXT, BlockType.TABLE}:
                continue
            text = _plain_text(block.get_text())
            if not text:
                continue
            block_ids.append(block.block_id)
            parts.append(text)
        content = _compact_text_no_ellipsis("\n".join(parts), 5000)
        if len(content) < 120:
            continue
        label = semantic_page_label(page_idx, semantic_output_language)
        title = "Supporting Information" if semantic_output_language == "en" else "補充說明"
        sections.append(
            {
                "page_idx": page_idx,
                "page_label": label,
                "title": f"{title} - {label}",
                "content": content,
                "block_ids": block_ids,
            }
        )
    return sections


def _render_supporting_page_sections(sections: list[dict[str, Any]], semantic_output_language: str) -> str:
    if not sections:
        return ""
    heading = "Supporting Information" if semantic_output_language == "en" else "補充說明"
    lines = [f"## {heading}", ""]
    for section in sections:
        lines.append(f"### {section['page_label']}")
        lines.append(str(section.get("content") or ""))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_supporting_page_chunks(
    document_ir: DocumentIR,
    sections: list[dict[str, Any]],
    semantic_output_language: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx, section in enumerate(sections):
        content = str(section.get("content") or "").strip()
        if not content:
            continue
        chunks.append(
            {
                "chunk_id": f"support{idx:06d}",
                "doc_id": document_ir.doc_id,
                "run_id": document_ir.run_id,
                "view": "structured_supporting_page",
                "content": content,
                "block_ids": [str(item) for item in section.get("block_ids") or []],
                "page_indices": [int(section.get("page_idx", idx))],
                "attachments": [],
                "metadata": {
                    "document_type": "supporting_page",
                    "content_type": "supporting_information",
                    "section": "Supporting Information" if semantic_output_language == "en" else "補充說明",
                    "source_doc_id": document_ir.doc_id,
                    "page_label": section.get("page_label"),
                    "semantic_output_language": semantic_output_language,
                },
            }
        )
    return chunks


def _collect_form_outputs(enrichments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    forms: list[dict[str, Any]] = []
    for block_id, enrichment in enrichments.items():
        if enrichment.get("kind") not in {"form_asset", "form_guide"}:
            continue
        output = enrichment.get("output") or {}
        if not isinstance(output, dict):
            continue
        if output.get("document_type") not in {None, "", "form"}:
            continue
        page_idx = enrichment.get("input", {}).get("page_idx")
        if page_idx is None:
            page_idx = enrichment.get("evidence", {}).get("page_idx")
        forms.append(
            {
                "block_id": block_id,
                "page_idx": int(page_idx or 0),
                "output": output,
                "quality": enrichment.get("quality") or {},
            }
        )
    return sorted(forms, key=lambda item: (item["page_idx"], item["block_id"]))


def _dedupe_form_outputs_by_page(forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(forms) <= 1:
        return forms
    grouped: dict[int, list[dict[str, Any]]] = {}
    for form in forms:
        grouped.setdefault(int(form.get("page_idx") or 0), []).append(form)
    result: list[dict[str, Any]] = []
    for page_idx, items in sorted(grouped.items()):
        if len(items) == 1:
            result.append(items[0])
            continue
        def score(item: dict[str, Any]) -> tuple[int, int, int]:
            output = item.get("output") or {}
            fields = output.get("field_schema") or []
            guide = str(output.get("filling_guide") or "")
            title = str(output.get("title") or "")
            non_generic_title = 0 if re.fullmatch(r"Form Page \d+|Form \d+", title, re.I) else 1
            return (len(fields), len(guide), non_generic_title)
        result.append(max(items, key=score))
    return sorted(result, key=lambda item: (int(item.get("page_idx") or 0), str(item.get("block_id") or "")))


def _collect_form_like_table_outputs(document_ir: DocumentIR) -> list[dict[str, Any]]:
    """Create synthetic form outputs for table/spreadsheet pages that are clearly forms."""

    forms: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    for block in document_ir.blocks:
        if block.type != BlockType.TABLE or block.page_idx in seen_pages:
            continue
        rows = parse_html_table(str(block.payload.get("table_body") or ""))
        page_rows = rows + _same_page_text_rows(document_ir, block.page_idx, exclude_block_id=block.block_id)
        if not is_form_like_document(document_ir, page_rows):
            continue
        title = _infer_form_title_from_rows(page_rows, document_ir.source.path)
        forms.append(
            {
                "block_id": block.block_id,
                "page_idx": int(block.page_idx or 0),
                "output": {
                    "title": title,
                    "document_type": "form",
                    "field_schema": [],
                    "filling_guide": "",
                    "retrieval_text": "",
                    "needs_review": True,
                    "_fallback": "form_like_table_detector",
                },
                "quality": {"needs_review": True},
            }
        )
        seen_pages.update(_spreadsheet_form_page_indices(document_ir, int(block.page_idx or 0)))
    return sorted(forms, key=lambda item: (item["page_idx"], item["block_id"]))




def _collect_form_like_text_page_outputs(
    document_ir: DocumentIR,
    semantic_output_language: str = "auto",
) -> list[dict[str, Any]]:
    """Create synthetic form outputs for PDF pages parsed as text-only form layouts."""

    forms: list[dict[str, Any]] = []
    table_pages = {block.page_idx for block in document_ir.blocks if block.type == BlockType.TABLE}
    page_indices = sorted({block.page_idx for block in document_ir.blocks if block.type == BlockType.TEXT})
    for page_idx in page_indices:
        if page_idx in table_pages:
            continue
        rows = _text_rows_for_page(document_ir, page_idx)
        if not rows or not is_form_like_document(document_ir, rows):
            continue
        fields = _infer_form_fields(rows)
        if len(fields) < 3 and _form_text_signal_score(rows) < 8:
            continue
        title = _infer_form_title_from_rows(rows, document_ir.source.path)
        sections = _infer_form_sections(rows)
        notes = _infer_form_notes(rows)
        approval_fields = [
            field for field in fields
            if str(field.get("type") or "") == "signature"
            or _infer_field_type(str(field.get("name") or "")) == "signature"
        ]
        guide = "\n".join(
            _build_form_semantic_guide(
                title=title,
                source_path=document_ir.source.path,
                sections=sections,
                fields=fields,
                notes=notes,
                approval_fields=approval_fields,
                semantic_output_language=semantic_output_language,
            )
        ).strip()
        all_text = [" | ".join(cell for cell in row if cell) for row in rows]
        generic_triggers = (
            ["form", "application", "authorization"]
            if resolve_semantic_output_language(semantic_output_language) == "en"
            else ["表單", "申請", "填寫規則", "簽核流程"]
        )
        triggers = _dedupe_strings(
            [
                title,
                Path(document_ir.source.path).stem,
                *generic_triggers,
                *[str(field.get("name") or "") for field in fields[:30]],
            ]
        )
        block_id = _first_text_block_id_for_page(document_ir, page_idx) or f"text_form_page_{page_idx:04d}"
        retrieval_text = _compact_text_no_ellipsis(
            " ".join(
                [
                    title,
                    " ".join(triggers[:30]),
                    " ".join(str(field.get("name") or "") for field in fields[:40]),
                ]
            ),
            1400,
        )
        forms.append(
            {
                "block_id": block_id,
                "page_idx": int(page_idx or 0),
                "output": {
                    "title": title,
                    "document_type": "form",
                    "triggers": triggers,
                    "all_text": all_text,
                    "field_schema": fields,
                    "filling_guide": guide,
                    "retrieval_text": retrieval_text,
                    "semantic_template": "form_text_page_fallback",
                    "needs_review": True,
                    "_fallback": "form_like_text_page_detector",
                },
                "quality": {"needs_review": True},
            }
        )
    return sorted(forms, key=lambda item: (item["page_idx"], item["block_id"]))


def _text_rows_for_page(document_ir: DocumentIR, page_idx: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for block in document_ir.blocks:
        if block.page_idx != page_idx or block.type != BlockType.TEXT:
            continue
        text = str(block.payload.get("text") or "").strip()
        if not text or _looks_like_noisy_header_text(text):
            continue
        rows.append([text])
    return rows


def _first_text_block_id_for_page(document_ir: DocumentIR, page_idx: int) -> str:
    for block in document_ir.blocks:
        if block.page_idx == page_idx and block.type == BlockType.TEXT:
            return block.block_id
    return ""


_COLON_LABEL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9 /&().'’,-]{1,45}\s*:")
_URI_SCHEME_TAIL_RE = re.compile(r"(?:https?|ftps?|doi|urn|isbn|issn)\s*:$", re.IGNORECASE)
# A fillable label is a short noun phrase ("Date signed:"), not a sentence.
_MAX_LABEL_WORDS = 3


def _colon_label_hits(text: str) -> list[str]:
    """Colon-terminated phrases that could plausibly be fillable field labels.

    A reference list turns this heuristic against itself: `https:` and `DOI:`
    are locators, and a colon-terminated citation title («The General Agreement
    on Trade in Services:») is a sentence. Both are dropped so a report's
    bibliography stops reading as a form's field labels.
    """
    hits: list[str] = []
    for match in _COLON_LABEL_RE.findall(text):
        label = match.strip()
        if _URI_SCHEME_TAIL_RE.search(label):
            continue
        if len(label.rstrip(":").split()) > _MAX_LABEL_WORDS:
            continue
        hits.append(label)
    return hits


def _form_text_signal_score(rows: list[list[str]]) -> int:
    text = _plain_text(" ".join(" ".join(str(cell or "") for cell in row) for row in rows))
    checkbox_score = text.count("□") + text.count("☐") + text.count("☑")
    colon_label_score = len(_colon_label_hits(text))
    english_hits = _english_form_field_hits(text)
    signature_score = len(re.findall(r"\b(signature|signed|date signed|applicant|payee|taxpayer)\b", text, re.I))
    return checkbox_score * 2 + min(colon_label_score, 12) + len(english_hits) * 2 + min(signature_score, 6)

def is_form_like_document(document_ir: DocumentIR, rows: list[list[str]] | None = None) -> bool:
    """Return true when source and parser evidence point to a fillable form."""

    ext = document_ir.source.ext.lower()
    source_text = f"{Path(document_ir.source.path).stem} {Path(document_ir.source.path).name}"
    row_text = " ".join(
        " ".join(str(cell or "") for cell in row)
        for row in (rows or _document_table_rows(document_ir))
    )
    text = _plain_text(f"{source_text} {row_text}")
    # 申請單(?!位): 申請單位 is "applicant unit" — a department, written in every
    # procedural notice — and matching its first three characters made any
    # document that names a department score a form name.
    form_name_score = len(re.findall(r"申請單(?!位)|請領單|報支單|出差單|核銷單|請款單|申報單|異動單|增加單|移轉單|報廢單|申請表|登記表|檢核表|授權書|同意書|委託書", text))
    english_form_score = len(
        re.findall(
            r"\b(application|authorization|authorisation|consent|claim|request|reimbursement|transcript|tax return)\b|\bform\b",
            text,
            re.IGNORECASE,
        )
    )
    field_score = len(re.findall(r"申請人|申請單位|申請日期|姓名|員工編號|事由|起訖|起始地點|到達地點|金額|合計|單位主管|簽名|簽章|領款人|核定", text))
    english_field_score = len(_english_form_field_hits(text))
    colon_label_score = len(_colon_label_hits(text))
    checkbox_score = text.count("□") + text.count("☐") + text.count("☑")
    if looks_like_reference_table(row_text) and form_name_score == 0 and english_form_score == 0 and checkbox_score == 0:
        return False
    if form_name_score >= 1 and field_score >= 2:
        return True
    if english_form_score >= 1 and (english_field_score >= 4 or colon_label_score >= 5 or checkbox_score >= 2):
        return True
    if ext in {"xls", "xlsx", "ods", "doc", "docx"} and field_score >= 5:
        return True
    if ext in {"pdf", "png", "jpg", "jpeg"} and english_form_score >= 1 and english_field_score >= 3:
        return True
    if checkbox_score >= 2 and (field_score >= 3 or english_field_score >= 3):
        return True
    return False


def _english_form_field_hits(text: str) -> set[str]:
    lowered = text.lower()
    terms = {
        "name",
        "date",
        "date signed",
        "signature",
        "address",
        "phone",
        "email",
        "ssn",
        "social security number",
        "taxpayer",
        "identification number",
        "employer identification number",
        "payee",
        "department",
        "amount",
        "account",
        "routing",
        "business purpose",
        "applicant",
        "authorized representative",
        "birth",
        "passport",
        "nationality",
        "purpose",
        "records",
        "disclosed",
        "recipient",
        "authorization",
    }
    return {term for term in terms if re.search(rf"\b{re.escape(term)}s?\b", lowered)}


def _document_table_rows(document_ir: DocumentIR) -> list[list[str]]:
    rows: list[list[str]] = []
    for block in document_ir.blocks:
        if block.type == BlockType.TABLE:
            rows.extend(parse_html_table(str(block.payload.get("table_body") or "")))
        elif block.type == BlockType.TEXT:
            text = str(block.payload.get("text") or "").strip()
            if text:
                rows.append([text])
    return rows



def _merge_spreadsheet_single_form_outputs(
    document_ir: DocumentIR,
    forms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(forms) <= 1:
        return forms
    if document_ir.source.ext.lower() not in {"xls", "xlsx", "ods"}:
        return forms
    if not _source_name_has_form_keyword(document_ir.source.path):
        return forms

    primary = dict(forms[0])
    output = dict(primary.get("output") or {})
    source_title = _form_title_from_source_path(document_ir.source.path)
    current_title = _clean_form_title(output.get("title"))
    if _is_weak_form_title(current_title) or not current_title:
        output["title"] = source_title
    output["needs_review"] = True
    output["_merged_form_pages"] = [int(form.get("page_idx") or 0) for form in forms]
    primary["output"] = output
    quality = dict(primary.get("quality") or {})
    quality["needs_review"] = True
    primary["quality"] = quality
    return [primary]


def _available_page_indices(document_ir: DocumentIR) -> list[int]:
    pages = sorted({int(page.page_idx) for page in document_ir.pages})
    if pages:
        return pages
    return sorted({int(block.page_idx) for block in document_ir.blocks if block.page_idx is not None})


def _form_output_page_indices(document_ir: DocumentIR, form: dict[str, Any]) -> set[int]:
    output = dict(form.get("output") or {})
    merged = output.get("_merged_form_pages")
    if isinstance(merged, list):
        pages: set[int] = set()
        for page in merged:
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                continue
        if pages:
            return pages
    try:
        page_idx = int(form.get("page_idx") or 0)
    except (TypeError, ValueError):
        page_idx = 0
    return _spreadsheet_form_page_indices(document_ir, page_idx)


def _source_name_suggests_form_collection(source_path: str) -> bool:
    name = Path(source_path).stem
    return bool(re.search(r"表單[彙汇]整|表單集|合集|套表|附件集|範本集|懶人包|懒人包|附件包|templates?|packet|collection", name, re.IGNORECASE))


def _form_outputs_have_conflicting_titles(document_ir: DocumentIR, form_outputs: list[dict[str, Any]]) -> bool:
    source_key = _normalized_form_title_key(_form_title_from_source_path(document_ir.source.path))
    keys: list[str] = []
    for form in form_outputs:
        output = dict(form.get("output") or {})
        title = _clean_form_title(output.get("title"))
        if _is_weak_form_title(title):
            continue
        key = _normalized_form_title_key(title)
        if not key:
            continue
        if source_key and (key in source_key or source_key in key):
            continue
        if key not in keys:
            keys.append(key)
    if len(keys) <= 1:
        return False
    if _source_name_has_form_keyword(document_ir.source.path) and not _source_name_suggests_form_collection(document_ir.source.path):
        return False
    return True


def _normalized_form_title_key(title: str) -> str:
    text = _clean_form_title(title)
    text = re.sub(r"(?i)\b(?:page|p)\s*\d+\b", " ", text)
    text = re.sub(r"第\s*\d+\s*頁", " ", text)
    text = re.sub(r"\.(?:pdf|docx?|xlsx?|ods|png|jpe?g)$", " ", text, flags=re.IGNORECASE)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text).lower()


def _should_merge_forms_as_single_source_document(
    *,
    document_ir: DocumentIR,
    form_outputs: list[dict[str, Any]],
) -> bool:
    if not form_outputs:
        return False
    ext = str(document_ir.source.ext or "").lower()
    if ext in {"xls", "xlsx", "ods"}:
        return False
    if _source_name_suggests_form_collection(document_ir.source.path):
        return False

    source_pages = _available_page_indices(document_ir)
    covered_pages: set[int] = set()
    for form in form_outputs:
        covered_pages.update(_form_output_page_indices(document_ir, form))
    if not source_pages:
        source_pages = sorted(covered_pages)
    if not source_pages or not covered_pages:
        return False

    coverage = len(set(source_pages) & covered_pages) / max(len(source_pages), 1)
    source_has_form_keyword = _source_name_has_form_keyword(document_ir.source.path)
    form_like_source = is_form_like_document(document_ir)
    if _form_outputs_have_conflicting_titles(document_ir, form_outputs):
        return False
    if len(source_pages) <= 1 and (source_has_form_keyword or form_like_source):
        return True
    if (
        len(form_outputs) == 1
        and form_like_source
        and _uncovered_pages_are_single_form_supporting_information(document_ir, source_pages, covered_pages)
    ):
        return True
    if coverage >= 0.8 and (source_has_form_keyword or form_like_source):
        return True
    if source_has_form_keyword and coverage >= 0.6:
        return True
    return False


def _uncovered_pages_are_single_form_supporting_information(
    document_ir: DocumentIR,
    source_pages: list[int],
    covered_pages: set[int],
) -> bool:
    """Treat instruction/privacy pages as part of a single official form."""

    uncovered = [page_idx for page_idx in source_pages if page_idx not in covered_pages]
    if not uncovered:
        return True
    if len(uncovered) > 6:
        return False

    support_patterns = [
        r"\binstructions?\b",
        r"\bgeneral instructions?\b",
        r"\bspecific instructions?\b",
        r"\bexplanation of form\b",
        r"\bpurpose of form\b",
        r"\bwhere to file\b",
        r"\bsignature and date\b",
        r"\bfuture developments?\b",
        r"\bwhat'?s new\b",
        r"\bprivacy act\b",
        r"\bpaperwork reduction act\b",
        r"\broutine uses\b",
        r"\bsection references\b",
        r"\bmail or fax\b",
        r"\bchart for\b",
        r"\bdo not send\b",
        r"填寫說明|填表說明|申請須知|注意事項|隱私權|個人資料|補充說明|說明事項|備註",
    ]

    for page_idx in uncovered:
        parts: list[str] = []
        for block in document_ir.get_blocks_by_page(page_idx):
            if block.type not in {BlockType.TEXT, BlockType.TABLE}:
                continue
            page_text = _plain_text(block.get_text())
            if page_text:
                parts.append(page_text)
        page_text = "\n".join(parts).strip()
        if len(re.sub(r"\s+", "", page_text)) < 80:
            continue
        marker_count = sum(1 for pattern in support_patterns if re.search(pattern, page_text, re.IGNORECASE))
        if marker_count <= 0:
            return False
        if _has_strong_fillable_form_markers(page_text) and marker_count < 2:
            return False
    return True


def _page_range_label(page_indices: list[int], semantic_output_language: str) -> str:
    language = resolve_semantic_output_language(semantic_output_language)
    pages = sorted({int(page) for page in page_indices})
    if not pages:
        return "Pages unknown" if language == "en" else "頁面未知"
    if len(pages) == 1:
        return semantic_page_label(pages[0], language)
    display = [page + 1 for page in pages]
    if display == list(range(display[0], display[-1] + 1)):
        return f"Pages {display[0]}-{display[-1]}" if language == "en" else f"第 {display[0]}-{display[-1]} 頁"
    joined = ", ".join(str(page) for page in display)
    return f"Pages {joined}" if language == "en" else f"第 {joined} 頁"


def _best_single_source_form_title(document_ir: DocumentIR, records: list[dict[str, Any]]) -> str:
    source_title = _form_title_from_source_path(document_ir.source.path)
    if _source_name_has_form_keyword(document_ir.source.path):
        return source_title
    source_key = _normalized_form_title_key(source_title)
    for record in records:
        candidate = _clean_form_title(record.get("form_name"))
        if _is_weak_form_title(candidate):
            continue
        candidate_key = _normalized_form_title_key(candidate)
        if candidate_key and candidate_key != source_key:
            return candidate
    return source_title


def _collapse_form_records_to_single_source_document(
    *,
    document_ir: DocumentIR,
    records: list[dict[str, Any]],
    semantic_output_language: str,
) -> list[dict[str, Any]]:
    form_name = _best_single_source_form_title(document_ir, records)
    subdoc_id = f"form:0000:{_slugify(form_name)}"
    logical_doc_id = f"{document_ir.doc_id}::{subdoc_id}"
    pages = _available_page_indices(document_ir) or sorted(_covered_form_pages(records))
    page_label = _page_range_label(pages, semantic_output_language)
    collapsed: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["subdoc_id"] = subdoc_id
        item["logical_doc_id"] = logical_doc_id
        item["parent_doc_id"] = document_ir.doc_id
        item["form_name"] = form_name
        item["page_indices"] = pages
        item["page_label"] = page_label
        item["needs_review"] = True
        collapsed.append(item)
    return collapsed


def _source_name_has_form_keyword(source_path: str) -> bool:
    name = Path(source_path).stem
    return bool(re.search(r"申請單|請領單|報支單|出差單|核銷單|請款單|申報單|異動單|增加單|移轉單|報廢單|申請表|登記表|檢核表|報告單|紀錄單|意見表|審查表|評分表|說明書|授權書|約定書|契約書|同意書|委託書|保證規約|規約|\b(application|authorization|authorisation|consent|claim|request|reimbursement|form)\b", name, re.IGNORECASE))


def _form_title_from_source_path(source_path: str) -> str:
    return semantic_source_title_from_path(source_path) or Path(source_path).stem[:80] or "表單"


def _is_weak_form_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", title or "")
    if not compact:
        return True
    weak_exact = {"單位主管", "單位主管核定", "領款人簽章", "出差人簽名", "主管核定", "合計金額"}
    if compact in weak_exact:
        return True
    if re.fullmatch(r"form(?:page)?\d*|form\d+|page\d+form", compact, re.IGNORECASE):
        return True
    if re.fullmatch(r"[0-9a-f]{8,}(?:-[0-9a-f]{4,})+", compact, re.IGNORECASE):
        return True
    if re.fullmatch(r"[0-9a-f]{2,8}(?:-[0-9a-f]{4,}){2,}", compact, re.IGNORECASE):
        return True
    if re.fullmatch(r"[0-9a-f]{16,}", compact, re.IGNORECASE):
        return True
    if re.search(r"事件編號|表單編號|申請日期|填表日期|xxx|xx\(|序號", compact, re.IGNORECASE):
        return True
    if len(compact) <= 8 and re.search(r"主管|簽章|簽名|核定|合計|金額|申請人|事由|地點", compact):
        return True
    if _looks_like_clause_fragment_title(compact):
        return True
    if len(compact) <= 12 and not re.search(r"表|單|申請|報告|紀錄|審查|評分|檢核|說明|流程|辦法|規約|授權|約定|契約|同意書|委託書|附件|領據|收據|切結書", compact):
        return True
    return False


def _looks_like_clause_fragment_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", title or "")
    if not compact:
        return False
    if compact.startswith(("附件", "附表", "表")):
        return False
    if not re.match(r"^(?:[一二三四五六七八九十]{1,3}|\d{1,2})[、.．]", compact):
        return False
    if re.search(r"(?:申請表|申請單|請領單|報支單|出差單|核銷單|請款單|申報單|異動單|增加單|移轉單|報廢單|報告單|紀錄單|意見表|審查表|評分表|說明書)$", compact):
        return False
    return bool(re.search(r"因|應|得|須|均|不得|本院|奉派|申請[（(]奉派[）)]|未服務|賠償|同意", compact))


def _clean_document_title(value: Any) -> str:
    text = _clean_form_title(value)
    text = re.sub(r"[昇鑑](?=台灣|臺灣|國內|國外|大台北|大臺北)", "", text)
    return text


def _build_form_semantic_guide(
    *,
    title: str,
    source_path: str,
    sections: list[str],
    fields: list[dict[str, Any]],
    notes: list[str],
    approval_fields: list[dict[str, Any]],
    semantic_output_language: str = "zh-TW",
) -> list[str]:
    language = resolve_semantic_output_language(semantic_output_language)
    pack = get_form_language_pack(language)
    source_name = Path(source_path).name
    base_sections = sections or ["基本資料", "填寫內容", "簽核"]
    section_names = [display_form_section(section, language) for section in base_sections]
    grouped_fields = _group_field_dicts_by_section(fields)
    conditional_fields = [
        field for field in fields
        if field.get("requirement") == "conditional" or str(field.get("name") or "").startswith(("□", "☐", "☑"))
    ]

    if language == "en":
        guide_parts = [
            f"## {pack.form_purpose}",
            f"\"{title}\" is a form from source file \"{source_name}\". Use it to capture, submit, authorize, verify, or record the listed information.",
            "",
            f"## {pack.use_cases}",
            f"Retrieve this document when users ask about the purpose, completion method, fields, approval flow, or notes for \"{title}\".",
            "",
            f"## {pack.form_structure}",
        ]
        guide_parts.extend(f"- {section}" for section in section_names[:12])

        if grouped_fields:
            guide_parts.extend(["", f"## {pack.filling_guidance}"])
            for section, section_fields in grouped_fields:
                names = [field["name"] for field in section_fields[:8] if field.get("name")]
                if names:
                    guide_parts.append(
                        f"- {display_form_section(section, language)}: fill in or verify {', '.join(names)}."
                    )

        if conditional_fields:
            guide_parts.extend(["", f"## {pack.conditional_fields}"])
            for field in conditional_fields[:8]:
                guide_parts.append(
                    f"- {field['name']}: complete this field only when the selected option or actual case applies; do not treat it as always required."
                )

        if approval_fields:
            guide_parts.extend(["", f"## {pack.approval_flow}"])
            guide_parts.append(" -> ".join(field["name"] for field in approval_fields[:10]))

        clean_notes, version = semantic_normalize_notes(notes)
        if version.raw:
            guide_parts.extend(["", f"## {pack.version_info}", f"Version: {version.raw}"])
        if clean_notes:
            guide_parts.extend(["", f"## {pack.notes}"])
            guide_parts.extend(f"- {note}" for note in clean_notes[:12])

        guide_parts.extend([
            "",
            f"## {pack.rag_summary}",
            f"This document can answer questions about the purpose, applicable scenario, required fields, conditional fields, approval fields, and notes for \"{title}\".",
        ])
        return guide_parts

    guide_parts = [
        f"## {pack.form_purpose}",
        f"「{title}」是來源檔案「{source_name}」中的表單，用於辦理、申請、核定或記錄表單所列事項。",
        "",
        f"## {pack.use_cases}",
        f"當使用者需要查詢「{title}」的用途、填寫方式、欄位內容、簽核流程或注意事項時，應召回本文件。",
        "",
        f"## {pack.form_structure}",
    ]
    guide_parts.extend(f"- {section}" for section in section_names[:12])

    if grouped_fields:
        guide_parts.extend(["", f"## {pack.filling_guidance}"])
        for section, section_fields in grouped_fields:
            names = [field["name"] for field in section_fields[:8] if field.get("name")]
            if names:
                guide_parts.append(f"- {display_form_section(section, language)}：主要填寫或確認{'、'.join(names)}。")

    if conditional_fields:
        guide_parts.extend(["", f"## {pack.conditional_fields}"])
        for field in conditional_fields[:8]:
            guide_parts.append(f"- {field['name']}：此欄位通常依勾選項目或實際情境填寫，不應一律視為必填。")

    if approval_fields:
        guide_parts.extend(["", f"## {pack.approval_flow}"])
        guide_parts.append(" → ".join(field["name"] for field in approval_fields[:10]))

    clean_notes, version = semantic_normalize_notes(notes)
    if version.raw:
        guide_parts.extend(["", f"## {pack.version_info}", f"版本：{version.raw}"])
    if clean_notes:
        guide_parts.extend(["", f"## {pack.notes}"])
        guide_parts.extend(f"- {note}" for note in clean_notes[:12])

    guide_parts.extend([
        "",
        f"## {pack.rag_summary}",
        f"本文件可回答「{title}」的用途、適用情境、應填欄位、條件欄位、簽核欄位與注意事項。",
    ])
    return guide_parts


def _group_field_dicts_by_section(fields: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field_item in fields:
        name = str(field_item.get("name") or "").strip()
        if not name:
            continue
        section = str(field_item.get("section") or _infer_field_section(name))
        grouped.setdefault(section, []).append(field_item)
    return sorted(grouped.items(), key=lambda item: _field_section_sort_key(item[0]))


def _field_section_sort_key(section: str) -> int:
    order = {
        "申請/基本資料": 0,
        "出差/行程資訊": 1,
        "費用/報支資訊": 2,
        "附件/佐證資料": 4,
        "簽核/用印": 5,
        "表單欄位": 9,
    }
    return order.get(section, 8)


def _fallback_form_output_from_table(
    document_ir: DocumentIR,
    form: dict[str, Any],
    output: dict[str, Any],
    semantic_output_language: str = "zh-TW",
) -> dict[str, Any]:
    """Build usable form semantics when VLM returns an empty form output."""

    block_id = str(form.get("block_id") or "")
    block = next((item for item in document_ir.blocks if item.block_id == block_id), None)
    if not block or block.type != BlockType.TABLE:
        return output

    rows = parse_html_table(str(block.payload.get("table_body") or ""))
    if not rows:
        return output

    page_text_rows = _same_page_text_rows(document_ir, block.page_idx, exclude_block_id=block.block_id)
    semantic_rows = rows + page_text_rows

    title = _infer_form_title_from_rows(semantic_rows, document_ir.source.path)
    sections = _infer_form_sections(semantic_rows)
    fields = _infer_form_fields(semantic_rows)
    notes = _infer_form_notes(semantic_rows)
    approval_fields = [field for field in fields if field.get("type") == "signature"]

    guide_parts = _build_form_semantic_guide(
        title=title,
        source_path=document_ir.source.path,
        sections=sections,
        fields=fields,
        notes=notes,
        approval_fields=approval_fields,
        semantic_output_language=semantic_output_language,
    )

    generic_triggers = (
        ["form", "application", "completion guidance", "approval flow"]
        if semantic_output_language == "en"
        else ["表單", "申請", "填寫規則", "簽核流程"]
    )
    triggers = sorted({
        title,
        Path(document_ir.source.path).stem,
        *generic_triggers,
        *[field["name"] for field in fields[:20]],
    })
    retrieval_text = " ".join([
        title,
        " ".join(triggers[:30]),
        " ".join(section for section in sections[:10]),
    ])

    fallback = dict(output)
    fallback.update(
        {
            "title": title,
            "document_type": "form",
            "triggers": triggers,
            "all_text": [" | ".join(cell for cell in row if cell) for row in semantic_rows],
            "field_schema": fields,
            "filling_guide": "\n".join(guide_parts).strip(),
            "retrieval_text": retrieval_text,
            "semantic_template": "form_v2",
            "needs_review": True,
            "_fallback": "table_form_semanticizer",
        }
    )
    return fallback


def _augment_form_output_from_ir_tables(
    document_ir: DocumentIR,
    form: dict[str, Any],
    output: dict[str, Any],
    semantic_output_language: str = "zh-TW",
) -> dict[str, Any]:
    """Merge parser table evidence into form output when it improves reliability.

    Native spreadsheet/DOCX forms often need parser rows to recover labels and
    footnotes. For PDF/image forms, a complete VLM field schema is usually more
    precise than table-derived heuristics, so parser rows should add notes/text
    but not noisy guessed fields.
    """

    page_idx = int(form.get("page_idx") or 0)
    rows = _form_semantic_rows_from_ir(document_ir, form)
    if not rows:
        return output

    title = _resolve_form_title_from_output_and_rows(output.get("title"), rows, document_ir.source.path)
    existing_fields = _dedupe_form_fields(output.get("field_schema", []))
    inferred_fields = _infer_form_fields(rows)
    parser_notes = _dedupe_strings(_infer_form_notes(rows) + _collect_form_page_notes(document_ir, page_idx))
    vlm_notes = _infer_form_notes([[str(item)] for item in output.get("all_text", []) if str(item).strip()])
    notes = _merge_form_notes_prefer_vlm(parser_notes, vlm_notes)
    sections = _infer_form_sections(rows)

    if _should_trust_vlm_form_schema(document_ir, output, existing_fields):
        supplemental_fields = [
            field for field in inferred_fields
            if str(field.get("type") or "") == "signature"
            or _infer_field_type(str(field.get("name") or "")) == "signature"
        ]
        supplemental_fields.extend(_extract_signature_fields_from_rows(rows))
        trusted_fields = _dedupe_form_fields(existing_fields + supplemental_fields)
        all_text = [str(item) for item in output.get("all_text", []) if str(item).strip()]
        if notes:
            all_text.extend(notes)
        if supplemental_fields:
            all_text.extend(str(field.get("evidence_text") or field.get("name") or "") for field in supplemental_fields)
        triggers = [str(item) for item in output.get("triggers", []) if str(item).strip()]
        triggers.extend([title, Path(document_ir.source.path).stem, *[field["name"] for field in trusted_fields[:32]]])
        retrieval_parts = [
            str(output.get("retrieval_text") or ""),
            title,
            " ".join(field["name"] for field in trusted_fields[:40] if field.get("name")),
            " ".join(notes[:6]),
        ]
        guide = _append_parser_notes_to_form_guide(
            str(output.get("filling_guide") or ""),
            notes,
            semantic_output_language=semantic_output_language,
        )
        augmented = dict(output)
        augmented.update(
            {
                "title": title,
                "document_type": "form",
                "field_schema": trusted_fields,
                "filling_guide": guide,
                "all_text": _dedupe_strings(all_text),
                "triggers": _dedupe_strings(triggers),
                "retrieval_text": _compact_text_no_ellipsis(" ".join(part for part in retrieval_parts if part), 1400),
                "semantic_template": "form_vlm_schema_trusted_with_signature_supplement",
                "needs_review": bool(output.get("needs_review")),
            }
        )
        return augmented

    if not inferred_fields:
        current_title = _clean_form_title(output.get("title"))
        if title and title != current_title:
            augmented = dict(output)
            triggers = [str(item) for item in output.get("triggers", []) if str(item).strip()]
            triggers.extend([title, Path(document_ir.source.path).stem])
            retrieval_parts = [str(output.get("retrieval_text") or ""), title]
            augmented.update(
                {
                    "title": title,
                    "document_type": "form",
                    "triggers": _dedupe_strings(triggers),
                    "retrieval_text": _compact_text_no_ellipsis(
                        " ".join(part for part in retrieval_parts if part),
                        1200,
                    ),
                }
            )
            return augmented
        return output

    merged_fields = _dedupe_form_fields(existing_fields + inferred_fields)
    approval_fields = [
        field for field in merged_fields
        if str(field.get("type") or "") == "signature" or _infer_field_type(str(field.get("name") or "")) == "signature"
    ]
    guide = "\n".join(
        _build_form_semantic_guide(
            title=title,
            source_path=document_ir.source.path,
            sections=sections,
            fields=merged_fields,
            notes=notes,
            approval_fields=approval_fields,
            semantic_output_language=semantic_output_language,
        )
    ).strip()

    all_text = [str(item) for item in output.get("all_text", []) if str(item).strip()]
    all_text.extend(" | ".join(cell for cell in row if cell) for row in rows)
    triggers = [str(item) for item in output.get("triggers", []) if str(item).strip()]
    triggers.extend([title, Path(document_ir.source.path).stem, *[field["name"] for field in merged_fields[:30]]])

    retrieval_parts = [
        str(output.get("retrieval_text") or ""),
        title,
        " ".join(field["name"] for field in merged_fields[:40] if field.get("name")),
        " ".join(notes[:8]),
    ]
    augmented = dict(output)
    augmented.update(
        {
            "title": title,
            "document_type": "form",
            "field_schema": merged_fields,
            "filling_guide": guide,
            "all_text": _dedupe_strings(all_text),
            "triggers": _dedupe_strings(triggers),
            "retrieval_text": _compact_text_no_ellipsis(" ".join(part for part in retrieval_parts if part), 1200),
            "semantic_template": "form_v2_ir_augmented",
            "needs_review": bool(output.get("needs_review")) or page_idx in _spreadsheet_form_page_indices(document_ir, page_idx),
        }
    )
    return augmented

def _extract_signature_fields_from_rows(rows: list[list[str]]) -> list[dict[str, Any]]:
    # Recover merged approval/signature roles from compact signature rows.
    role_patterns = [
        ("申請人", r"申請人"),
        ("主任(組長)", r"主任[（(]組長[）)]"),
        ("單位副主管", r"單位副主管"),
        ("單位主管", r"單位主管"),
        ("主任秘書", r"主任秘書"),
        ("副院長", r"副院長"),
        ("院長", r"院長"),
        ("董事長", r"董事長"),
        ("人事", r"人事"),
        ("副處長", r"副處長"),
        ("處長", r"(?<!副)處長"),
        ("經辦人", r"經辦人"),
        ("總務負責人", r"總務負責人"),
        ("總務組負責人", r"總務組負責人"),
        ("行政處副處長", r"行政處副處長"),
        ("行政處處長", r"行政處處長"),
    ]
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        raw = " | ".join(str(cell or "").strip() for cell in row if str(cell or "").strip())
        compact = re.sub(r"\s+", "", raw)
        if not compact:
            continue
        if len(compact) > 45 and not re.search(r"[:：]", raw):
            continue
        if re.search(r"[。；;]", raw) and not re.search(r"[:：]", raw):
            continue
        for name, pattern in role_patterns:
            if name in seen:
                continue
            if re.search(pattern, compact):
                seen.add(name)
                fields.append({
                    "name": name,
                    "type": "signature",
                    "required": False,
                    "requirement": "situational",
                    "section": "簽核/用印",
                    "aliases": [],
                    "evidence_text": raw,
                })
    return fields

def _note_list_marker(text: str) -> str | None:
    match = re.match(r"^\s*(?:註[:：]?\s*)?([0-9]{1,2}|[一二三四五六七八九十]{1,3})[.．、]", str(text or ""))
    return match.group(1) if match else None


def _merge_form_notes_prefer_vlm(parser_notes: list[str], vlm_notes: list[str]) -> list[str]:
    parser_notes = _canonicalize_form_notes(parser_notes)
    vlm_notes = _canonicalize_form_notes(vlm_notes)
    if not vlm_notes:
        return parser_notes
    merged: list[str] = []
    seen: set[str] = set()
    marker_index: dict[str, int] = {}

    for note in vlm_notes:
        key = re.sub(r"\s+", "", note)
        if key and key not in seen:
            seen.add(key)
            marker = _note_list_marker(note)
            if marker:
                marker_index[marker] = len(merged)
            merged.append(note)

    for note in parser_notes:
        marker = _note_list_marker(note)
        if marker and marker in marker_index:
            existing_idx = marker_index[marker]
            existing = merged[existing_idx]
            existing_key = _note_similarity_key(existing)
            parser_key = _note_similarity_key(note)
            if len(parser_key) > len(existing_key) and existing_key in parser_key:
                seen.discard(re.sub(r"\s+", "", existing))
                merged[existing_idx] = note
                seen.add(re.sub(r"\s+", "", note))
            continue
        key = re.sub(r"\s+", "", note)
        if key and key not in seen:
            seen.add(key)
            if marker:
                marker_index[marker] = len(merged)
            merged.append(note)

    clean_notes, _version = semantic_normalize_notes(merged)
    return _dedupe_similar_notes(clean_notes)


def _append_parser_notes_to_form_guide(
    guide: str,
    notes: list[str],
    semantic_output_language: str = "zh-TW",
) -> str:
    guide = _normalize_form_guide_versions(guide)
    clean_notes, version = semantic_normalize_notes(_canonicalize_form_notes(notes or []))
    language = resolve_semantic_output_language(semantic_output_language)
    additions: list[str] = []
    if version.raw and version.raw not in guide:
        if language == "en":
            additions.extend(["", "## Version Information", f"Version: {version.raw}"])
        else:
            additions.extend(["", "## 版本資訊", f"版本：{version.raw}"])
    guide_compact = re.sub(r"\s+", "", guide or "")
    guide_notes = _canonicalize_form_notes(_extract_note_like_lines(guide))
    guide_markers = {_note_list_marker(note) for note in guide_notes}
    guide_markers.discard(None)
    missing_notes = []
    for note in clean_notes:
        marker = _note_list_marker(note)
        if marker and marker in guide_markers:
            guide_note = next((item for item in guide_notes if _note_list_marker(item) == marker), "")
            guide_key = _note_similarity_key(guide_note)
            note_key = _note_similarity_key(note)
            if guide_key and (note_key in guide_key or len(note_key) <= len(guide_key) or guide_key not in note_key):
                continue
        if re.sub(r"\s+", "", note) in guide_compact:
            continue
        missing_notes.append(note)
    if missing_notes:
        heading = "Complete Source Notes" if language == "en" else "來源完整注意事項"
        additions.extend(["", f"## {heading}"])
        additions.extend(f"- {note}" for note in missing_notes[:12])
    if not additions:
        return guide
    return (guide.rstrip() + "\n" + "\n".join(additions)).strip()


def _extract_note_like_lines(text: str) -> list[str]:
    notes: list[str] = []
    for line in str(text or "").splitlines():
        cleaned = line.strip().lstrip("-*• ").strip()
        if re.match(r"^(?:註[:：]?|備註[:：]?|\d+[.．、]|[一二三四五六七八九十]+[.．、])", cleaned):
            notes.append(cleaned)
    return notes


def _canonicalize_form_notes(notes: list[str]) -> list[str]:
    expanded: list[str] = []
    for note in notes or []:
        expanded.extend(_split_note_items(str(note or "")))
    clean_notes, _version = semantic_normalize_notes(_dedupe_strings(expanded))
    return _dedupe_similar_notes(clean_notes)


def _dedupe_similar_notes(notes: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    marker_index: dict[str, int] = {}
    for note in notes or []:
        text = re.sub(r"\s+", " ", str(note or "")).strip()
        if not text:
            continue
        key = _note_similarity_key(text)
        if key in seen:
            continue
        marker = _note_list_marker(text)
        if marker and marker in marker_index:
            existing_idx = marker_index[marker]
            existing = result[existing_idx]
            existing_key = _note_similarity_key(existing)
            if len(key) > len(existing_key) and existing_key in key:
                seen.discard(existing_key)
                result[existing_idx] = text
                seen.add(key)
            continue
        contained_idx = next((idx for idx, existing in enumerate(result) if _note_is_contained_duplicate(text, existing)), None)
        if contained_idx is not None:
            existing = result[contained_idx]
            if len(key) > len(_note_similarity_key(existing)):
                seen.discard(_note_similarity_key(existing))
                result[contained_idx] = text
                seen.add(key)
                if marker:
                    marker_index[marker] = contained_idx
            continue
        seen.add(key)
        if marker:
            marker_index[marker] = len(result)
        result.append(text)
    return result


def _note_similarity_key(text: str) -> str:
    compact = re.sub(r"\s+", "", text or "")
    compact = re.sub(r"^[*•-]+", "", compact)
    compact = re.sub(r"^註[:：]?", "", compact)
    compact = compact.replace("，", "").replace("、", "").replace("；", "").replace("。", "")
    return compact


def _note_is_contained_duplicate(candidate: str, existing: str) -> bool:
    left = _note_similarity_key(candidate)
    right = _note_similarity_key(existing)
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return len(short) >= 12 and short in long


def _should_trust_vlm_form_schema(
    document_ir: DocumentIR,
    output: dict[str, Any],
    existing_fields: list[dict[str, Any]],
) -> bool:
    """Return true when VLM schema is strong enough to avoid parser-field merge."""

    if output.get("document_type") not in {None, "", "form"}:
        return False
    if output.get("_fallback") or output.get("_salvaged") or output.get("_error"):
        return False
    ext = document_ir.source.ext.lower().lstrip(".")
    if ext in {"xls", "xlsx", "ods", "doc", "docx"}:
        return False
    if len(existing_fields) < 8:
        return False
    field_names = " ".join(str(field.get("name") or "") for field in existing_fields)
    has_signature = any(str(field.get("type") or "") == "signature" for field in existing_fields)
    has_core_fields = bool(re.search(r"姓名|申請|日期|地點|事由|電話|身分證|身份證", field_names))
    guide = str(output.get("filling_guide") or "")
    return has_core_fields and (has_signature or len(existing_fields) >= 12) and len(guide) >= 30


def _form_semantic_rows_from_ir(document_ir: DocumentIR, form: dict[str, Any]) -> list[list[str]]:
    block_id = str(form.get("block_id") or "")
    page_idx = int(form.get("page_idx") or 0)
    page_indices = _spreadsheet_form_page_indices(document_ir, page_idx)
    rows: list[list[str]] = []

    for block in document_ir.blocks:
        if block.page_idx not in page_indices:
            continue
        if block.type == BlockType.TABLE:
            rows.extend(parse_html_table(str(block.payload.get("table_body") or "")))
            continue
        if block.type == BlockType.TEXT and (
            block.block_id == block_id
            or block.page_idx == page_idx
            or document_ir.source.ext.lower() in {"xls", "xlsx", "ods"}
        ):
            text_value = str(block.payload.get("text") or "").strip()
            if text_value and not _looks_like_noisy_header_text(text_value):
                rows.append([text_value])
    return rows


def _spreadsheet_form_page_indices(document_ir: DocumentIR, page_idx: int) -> set[int]:
    """Return pages that probably belong to the same printed spreadsheet form."""

    ext = document_ir.source.ext.lower()
    if ext not in {"xls", "xlsx", "ods"}:
        return {page_idx}
    available = {page.page_idx for page in document_ir.pages} or {block.page_idx for block in document_ir.blocks}
    if _source_name_has_form_keyword(document_ir.source.path) and available:
        return set(sorted(available))
    page_indices = {page_idx}
    # Spreadsheet print areas often split one visual form into a continued page.
    if page_idx + 1 in available:
        page_indices.add(page_idx + 1)
    return page_indices


def _same_page_text_rows(
    document_ir: DocumentIR,
    page_idx: int,
    exclude_block_id: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for candidate in document_ir.blocks:
        if candidate.block_id == exclude_block_id:
            continue
        if candidate.page_idx != page_idx or candidate.type != BlockType.TEXT:
            continue
        text = str(candidate.payload.get("text") or "").strip()
        if not text:
            continue
        if _looks_like_noisy_header_text(text):
            continue
        rows.append([text])
    return rows


def _looks_like_noisy_header_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if compact in get_corpus_rules().marker_list("noisy_header_exact"):
        return True
    if re.fullmatch(r"[年月日飛機其他雜費幣別匯率]+", compact):
        return True
    if len(compact) <= 2:
        return True
    return False


def _infer_form_title_from_rows(rows: list[list[str]], source_path: str) -> str:
    source_title = _form_title_from_source_path(source_path)
    source_has_form_keyword = _source_name_has_form_keyword(source_path)
    candidates: list[str] = []
    for row in rows[:12]:
        cells = [cell.strip() for cell in row if cell.strip()]
        for cell in cells:
            candidate_text = _clean_form_title_candidate_text(cell)
            compact = candidate_text.replace(" ", "").replace("　", "")
            if (len(compact) < 4 and compact not in {"委託書", "領據", "收據"}) or len(compact) > 120:
                continue
            if _looks_like_clause_fragment_title(compact):
                continue
            has_zh_title_term = any(
                term in compact
                for term in [
                    "表",
                    "單",
                    "申請",
                    "請領",
                    "增加",
                    "異動",
                    "檢核",
                    "規約",
                    "授權",
                    "約定",
                    "契約",
                    "同意書",
                    "委託書",
                    "附件",
                    "領據",
                    "收據",
                    "報告書",
                    "成果報告",
                    "切結書",
                ]
            )
            has_en_title_term = bool(
                re.search(
                    r"\b(application|authorization|authorisation|consent|claim|request|reimbursement|release|transcript|tax return|form)\b",
                    candidate_text,
                    re.IGNORECASE,
                )
            )
            if not (has_zh_title_term or has_en_title_term):
                continue
            cleaned = _clean_form_title(candidate_text[:120])
            if (
                cleaned
                and _looks_like_form_title_candidate(cleaned)
                and not _is_weak_form_title(cleaned)
                and not _looks_like_field_title_candidate(cleaned)
            ):
                candidates.append(cleaned)
    best_candidate = _best_form_title_candidate(candidates)
    if best_candidate and not source_has_form_keyword:
        return best_candidate
    strong_title_terms = ["申請單", "請領單", "報支單", "出差單", "申請表", "檢核表", "授權書", "同意書", "委託書", "增加單", "異動單", "領據", "收據", "報告書", "成果報告", "切結書"]
    if best_candidate and source_has_form_keyword and (
        any(term in best_candidate for term in strong_title_terms)
        or bool(re.search(r"^附件[一二三四五六七八九十0-9]", best_candidate))
        or bool(re.search(r"\b(application|authorization|authorisation|consent|claim|request|reimbursement|release|transcript|tax return|form)\b", best_candidate, re.IGNORECASE))
    ):
        return best_candidate
    return source_title


def _clean_form_title_candidate_text(value: str) -> str:
    text = re.sub(r"\s+", " ", clean_latex_symbols(str(value or ""))).strip()
    text = re.sub(r"\bForm and receipts must be submitted\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDo not sign this form\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bRequest may be rejected\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\bRequest for Transcript of Tax Return)\s+Form\s*$", r"\1", text, flags=re.IGNORECASE)
    return text.strip("#:： -")


def _best_form_title_candidate(candidates: list[str]) -> str:
    clean_candidates = _dedupe_strings([candidate for candidate in candidates if candidate])
    if not clean_candidates:
        return ""
    return max(clean_candidates, key=_form_title_candidate_score)


def _form_title_candidate_score(title: str) -> int:
    text = _clean_form_title(title)
    compact = re.sub(r"\s+", "", text)
    lowered = text.lower()
    score = 0
    if any(term in compact for term in ["申請單", "申請表", "檢核表", "授權書", "同意書", "委託書", "領據", "收據", "報告書", "切結書"]):
        score += 35
    if re.search(r"\b(authorization|authorisation|release of information|request for transcript|reimbursement claim|claim form|application form|consent)\b", lowered):
        score += 35
    if re.search(r"\bform\s+[A-Z0-9-]+\b", text, re.IGNORECASE):
        score += 12
    if re.search(r"\b(page|omb|discontinue prior editions|rev\.|revision)\b", lowered):
        score -= 18
    if re.match(r"^\d+[a-z]?\s+", text, re.IGNORECASE):
        score -= 45
    if re.search(r"[□☐☑]", text):
        score -= 35
    if 8 <= len(compact) <= 70:
        score += 6
    return score


def _resolve_form_title_from_output_and_rows(
    output_title: Any,
    rows: list[list[str]],
    source_path: str,
) -> str:
    current_title = _clean_form_title(output_title)
    inferred_title = _infer_form_title_from_rows(rows, source_path)
    source_title = _form_title_from_source_path(source_path)
    if _is_weak_form_title(current_title):
        return inferred_title
    if _should_prefer_local_form_title(current_title, inferred_title, source_title):
        return inferred_title
    return current_title or inferred_title


def _should_prefer_local_form_title(current_title: str, inferred_title: str, source_title: str) -> bool:
    if not current_title or not inferred_title:
        return False
    current_key = _title_compare_key(current_title)
    inferred_key = _title_compare_key(inferred_title)
    source_key = _title_compare_key(source_title)
    if not current_key or not inferred_key or inferred_key == current_key:
        return False
    if inferred_key == source_key:
        return False
    current_is_source_level = current_key == source_key or current_key in source_key or source_key in current_key
    if not current_is_source_level:
        return False
    return _looks_like_form_title_candidate(inferred_title) and not _is_weak_form_title(inferred_title)


def _title_compare_key(title: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", _clean_form_title(title)).lower()


def _looks_like_form_title_candidate(title: str) -> bool:
    compact = re.sub(r"\s+", "", title or "")
    if not compact:
        return False
    if compact.startswith(("□", "☐", "☑")):
        return False
    if _looks_like_clause_fragment_title(compact):
        return False
    if re.search(r"[。；;]", compact) and not re.search(r"附件[一二三四五六七八九十0-9]", compact):
        return False
    strong_terms = (
        "申請表", "申請單", "請領單", "報支單", "出差單", "核銷單", "請款單",
        "申報單", "異動單", "增加單", "移轉單", "報廢單", "報告單", "紀錄單",
        "意見表", "審查表", "評分表", "檢核表", "說明書", "授權書", "約定書", "契約書", "同意書", "委託書", "保證規約",
    )
    if any(term in compact for term in strong_terms):
        return True
    if re.search(r"^附件[一二三四五六七八九十0-9].{2,60}$", compact):
        return True
    if not re.search(r"[\u4e00-\u9fff]", compact):
        text = _clean_form_title(title)
        if re.search(
            r"\b(application|authorization|authorisation|consent|claim|request|reimbursement|release|transcript|tax return|form)\b",
            text,
            re.IGNORECASE,
        ) and not re.search(r"\b(please|return completed|mail completed|where to send)\b", text, re.IGNORECASE):
            return True
    return False

def _looks_like_field_title_candidate(title: str) -> bool:
    text = str(title or "").strip()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if re.match(r"^\d+[a-z]?\s+", text, re.IGNORECASE):
        return True
    if re.search(r"[□☐☑]", text) and len(compact) > 20:
        return True
    fieldish = {"請購單位", "使用保管單位", "承辦單位", "申請日期", "主管", "副主管", "備註"}
    if compact.strip("：:") in fieldish:
        return True
    if len(compact) <= 6 and re.search(r"單位|日期|主管|備註|姓名|金額", compact):
        return True
    return False


def _infer_form_sections(rows: list[list[str]]) -> list[str]:
    sections: list[str] = []
    for row in rows:
        for cell in row:
            text = cell.strip()
            if re.match(r"^[一二三四五六七八九十]+[、．.]", text):
                sections.append(text[:80])
    return _dedupe_strings(sections) or ["基本資料", "填寫內容", "簽核"]


def _infer_form_fields(rows: list[list[str]]) -> list[dict[str, Any]]:
    field_names: list[str] = []
    label_terms = [
        "單位", "申請人", "日期", "姓名", "員工編號", "職級", "職稱", "身份證",
        "地點", "事由", "期間", "代理人", "主管", "主任秘書", "副院長", "院長",
        "董事長", "報支單位", "預估費用", "預借金額", "預算審查", "計畫名稱",
        "預算科目", "審查人員", "行政處總務", "變更事由", "備註", "簽名",
        "簽章", "核定", "工作紀要", "交通費", "飛機", "汽車", "火車", "高鐵",
        "宿費", "膳雜費", "生活費", "辦公費", "雜費", "幣別", "匯率", "折合台幣",
        "合計", "小計", "單據編號", "起訖", "受款人", "領款人", "應繳回", "應補發", "沖預借", "起始地點", "到達地點", "合計金額",
    ]
    english_label_terms = {
        "name", "date", "signature", "address", "phone", "email", "ssn",
        "social security", "taxpayer", "identification", "employer", "payee",
        "department", "amount", "account", "routing", "business purpose",
        "applicant", "birth", "passport", "nationality", "purpose", "records",
        "recipient", "authorized", "representative", "preparer", "vendor", "student",
    }
    for row in rows:
        row_text = " ".join(cell for cell in row if cell)
        for cell in row:
            raw_text = re.sub(r"\s+", " ", clean_latex_symbols(str(cell or ""))).strip()
            labels = [_clean_inferred_field_label(label) for label in _extract_form_field_labels(raw_text)]
            labels = [label for label in labels if label]
            field_names.extend(labels)
            text = _clean_inferred_field_label(_normalize_field_label(raw_text))
            if labels and (raw_text.count(":") >= 2 or len(raw_text) > 50):
                continue
            if not text or len(text) > 80:
                continue
            if text.startswith(("註", "Taiwan", "台 灣", "•", "- ")):
                continue
            if re.match(r"^\d+[.．、]", text):
                continue
            if text in get_corpus_rules().marker_list("form_section_skip_headings"):
                continue
            lower_text = text.lower()
            if (
                any(term in text for term in label_terms)
                or any(term in lower_text for term in english_label_terms)
                or "□" in row_text
                or "☐" in row_text
            ):
                field_names.append(text)

    fields: list[dict[str, Any]] = []
    for name in _dedupe_strings(field_names):
        requirement = _infer_requirement(name)
        fields.append(
            {
                "name": name,
                "type": _infer_field_type(name),
                "required": requirement == "required",
                "requirement": requirement,
                "section": _infer_field_section(name),
                "aliases": [],
                "evidence_text": name,
            }
        )
    return fields[:60]

def _normalize_field_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().strip("：:")


def _clean_inferred_field_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("：:")
    if not text:
        return ""
    text = re.sub(r"^\d+[a-z]?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(if not [^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(Dept Specific$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\s*If\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\s*Note\b.*$", "", text, flags=re.IGNORECASE)
    corpus_rules = get_corpus_rules()
    for label_fix in corpus_rules.field_label_fixes:
        text = label_fix.sub(text)
    for label_override in corpus_rules.field_label_overrides:
        if label_override.pattern.search(text):
            text = label_override.replacement
    text = re.sub(r"\s+", " ", text).strip(" -/:：.")
    return text


def _extract_form_field_labels(text: str) -> list[str]:
    labels: list[str] = []
    if not text:
        return labels
    segments = [part.strip() for part in re.split(r"\s+\|\s+|[;；]", text) if part.strip()] or [text]
    for segment in segments:
        segment = re.sub(r"^[-*•]+\s*", "", segment).strip()
        for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9 /&().'’,-]{1,45})\s*:", segment):
            label = _normalize_field_label(match.group(1))
            if _looks_like_english_field_label(label):
                labels.append(label)
        for match in re.finditer(r"\b([A-Za-z][A-Za-z /&().'’,-]{2,45})\s*(?:_{2,}|☐|□)", segment):
            label = _normalize_field_label(match.group(1))
            if _looks_like_english_field_label(label):
                labels.append(label)
    return _dedupe_strings(labels)


def _looks_like_english_field_label(label: str) -> bool:
    text = re.sub(r"\s+", " ", label or "").strip(" -/:：")
    if not text or len(text) > 48:
        return False
    words = re.findall(r"[A-Za-z]{2,}", text)
    if not words or len(words) > 7:
        return False
    lowered = text.lower()
    bad = {"if", "for", "or", "and", "yes", "no", "other", "page", "form"}
    if lowered in bad:
        return False
    return bool(
        re.search(
            r"\b(name|date|signature|address|phone|email|ssn|social security|taxpayer|identification|employer|payee|department|amount|account|routing|business purpose|applicant|birth|passport|nationality|purpose|records|recipient|authorized|representative|preparer|vendor|student)\b",
            lowered,
        )
    )


def _infer_field_type(name: str) -> str:
    normalized = _normalize_field_label(name)
    lowered = normalized.lower()
    signature_names = {"單位主管", "主任秘書", "副院長", "院長", "董事長", "出差人簽名", "出差人簽章"}
    if normalized in signature_names or re.fullmatch(r".+人簽(名|章)", normalized):
        return "signature"
    if re.search(r"\b(signature|signed|authorized representative)\b", lowered):
        return "signature"
    if normalized.startswith(("□", "☐", "☑")) or any(term in normalized for term in ["保險", "報支單位"]):
        return "checkbox"
    if re.search(r"\b(yes|no|student|vendor|employee|other)\b", lowered) and len(lowered) <= 32:
        return "checkbox"
    if any(term in normalized for term in ["日期", "期間", "年月日"]) or re.search(r"\b(date|birth|birthday|expires)\b", lowered):
        return "date"
    if any(term in normalized for term in ["金額", "費用", "交通費", "宿費", "膳雜費", "生活費", "辦公費", "匯率", "折合台幣", "合計", "小計"]) or re.search(r"\b(amount|total|expense|fee|fare|mileage|rate|account|routing)\b", lowered):
        return "number"
    if "身份證" in normalized or re.search(r"\b(ssn|social security|identification|taxpayer|employer identification)\b", lowered):
        return "id"
    if "姓名" in normalized or "申請人" in normalized or re.search(r"\b(name|applicant|payee|preparer)\b", lowered):
        return "name"
    if re.search(r"\b(email|phone|address)\b", lowered):
        return "text"
    return "text"


def _infer_requirement(name: str) -> str:
    normalized = _normalize_field_label(name)
    lowered = normalized.lower()
    if normalized.startswith(("□", "☐", "☑")):
        return "conditional"
    conditional_terms = [
        "預估費用", "預借金額", "需用日期", "報支單位", "付款對象", "支票抬頭",
        "戶名", "銀行", "帳號", "預算審查", "計畫名稱", "預算科目", "已耗用",
        "審查人員", "保險", "申根", "其他", "變更", "備註", "代理人", "行政處總務",
        "主任秘書", "副院長", "院長", "董事長",
    ]
    if any(term in normalized for term in conditional_terms):
        return "conditional"
    if re.search(r"\b(other|optional|if applicable|yes|no|vendor|student)\b", lowered):
        return "conditional"
    required_terms = [
        "申請單位", "申請人", "申請日期", "姓名", "職級", "職稱", "出差地點",
        "出差事由", "出差期間", "單位主管", "出差人簽名", "出差人簽章",
    ]
    if any(term in normalized for term in required_terms):
        return "required"
    if re.search(r"\b(name|date|signature|signed|address|applicant|payee|taxpayer|ssn)\b", lowered):
        return "required"
    return "situational"


def _infer_required(name: str) -> bool:
    return _infer_requirement(name) == "required"


def _infer_form_notes(rows: list[list[str]]) -> list[str]:
    notes: list[str] = []
    for row in rows:
        text = " ".join(cell for cell in row if cell).strip()
        if not text:
            continue
        if re.match(r"^(註[:：]?|註\d+[:：]|\d+[.．、])", text):
            notes.extend(_split_note_items(text))
    normalized_notes, _ = semantic_normalize_notes(_dedupe_strings(notes))
    return normalized_notes


def _collect_form_page_notes(document_ir: DocumentIR, page_idx: int) -> list[str]:
    notes: list[str] = []
    blocks = list(document_ir.blocks)
    for idx, block in enumerate(blocks):
        if block.page_idx != page_idx or block.type != BlockType.TABLE:
            continue
        started = False
        lines: list[str] = []
        for next_block in blocks[idx + 1: idx + 45]:
            if next_block.page_idx != page_idx:
                break
            if next_block.type == BlockType.TABLE:
                break
            if next_block.type != BlockType.TEXT:
                continue
            text = re.sub(r"\s+", " ", str(next_block.payload.get("text") or "")).strip()
            if not text:
                continue
            is_heading = bool(re.match(r"^(?:備註|備注|註[:：]|註\d+[:：])", text))
            is_item = bool(re.match(r"^(?:[一二三四五六七八九十]+、|\d+[.、])", text))
            if is_heading:
                started = True
            elif not started:
                continue
            elif re.match(r"^第[一二三四五六七八九十0-9]+條", text):
                break
            elif not is_item and len(re.sub(r"\s+", "", text)) > 220:
                break
            lines.append(text)
        notes.extend(_split_note_items(" ".join(lines)))
    clean_notes, _version = semantic_normalize_notes(_dedupe_strings(notes))
    return clean_notes


def _is_meaningful_note_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    if re.fullmatch(r"(?:註\d*[:：]?|\d+[.．、]?|[一二三四五六七八九十]+[、．.]?)", compact):
        return False
    normalized = re.sub(r"[\s.．、:：]", "", compact)
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", normalized)) and len(normalized) >= 2

def _split_note_items(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", clean_latex_symbols(text or "")).strip()
    if not text:
        return []

    numbered_note_matches = list(re.finditer(r"註\d+[:：]", text))
    if len(numbered_note_matches) > 1:
        items: list[str] = []
        for idx, match in enumerate(numbered_note_matches):
            end = numbered_note_matches[idx + 1].start() if idx + 1 < len(numbered_note_matches) else len(text)
            item = text[match.start():end].strip()
            if _is_meaningful_note_text(item):
                items.append(item)
        return items

    had_note_prefix = bool(re.match(r"^(?:註|備註|備注)[:：]", text))
    text = re.sub(r"^(?:註|備註|備注)[:：]\s*", "", text).strip()
    item_pattern = re.compile(
        r"(?:(?<=^)|(?<=\s))(?P<marker>(?:\d{1,2}|[一二三四五六七八九十]{1,3})[.．、])\s*(?=[A-Za-z\u4e00-\u9fff「『（(])"
    )
    matches = list(item_pattern.finditer(text))
    if matches:
        items = []
        for idx, match in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            item = text[match.start():end].strip()
            if idx == 0 and had_note_prefix:
                item = f"註：{item}"
            if _is_meaningful_note_text(item):
                items.append(item)
        return items

    return [text] if _is_meaningful_note_text(text) else []


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = re.sub(r"\s+", "", item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item.strip())
    return result


def _build_form_display_summary(
    *,
    form_name: str,
    source_path: str,
    fields: list[dict[str, Any]],
    notes: list[str] | None = None,
    semantic_output_language: str = "zh-TW",
) -> str:
    language = resolve_semantic_output_language(semantic_output_language)
    grouped = _group_field_dicts_by_section(fields)
    source_name = Path(source_path).name
    if language == "en":
        parts = [
            f"\"{form_name}\" is a form from source file \"{source_name}\". Use it to capture, authorize, confirm, or record the listed items.",
        ]
        focus_parts: list[str] = []
        for section, section_fields in grouped:
            names = [
                str(field.get("name") or "").strip()
                for field in section_fields
                if str(field.get("name") or "").strip()
            ]
            if names:
                focus_parts.append(f"{display_form_section(section, language)} includes {', '.join(names[:6])}")
        if focus_parts:
            parts.append("Primary field groups: " + "; ".join(focus_parts[:6]) + ".")
        clean_notes, version = semantic_normalize_notes(notes or [])
        if version.raw:
            parts.append(f"Version information: {version.raw}.")
        if clean_notes:
            parts.append("Notes include: " + "; ".join(clean_notes[:4]) + ".")
        return _compact_text_no_ellipsis(" ".join(parts), 780)

    parts = [
        f"「{form_name}」是來源檔案「{source_name}」中的表單，用於辦理、授權、確認或記錄表單所列事項。",
    ]
    focus_parts: list[str] = []
    for section, section_fields in grouped:
        names = [
            str(field.get("name") or "").strip()
            for field in section_fields
            if str(field.get("name") or "").strip()
        ]
        if names:
            focus_parts.append(f"{display_form_section(section, language)}包含{'、'.join(names[:6])}")
    if focus_parts:
        parts.append("主要欄位分組：" + "；".join(focus_parts[:6]) + "。")
    clean_notes, version = semantic_normalize_notes(notes or [])
    if version.raw:
        parts.append(f"版本資訊：{version.raw}。")
    if clean_notes:
        parts.append("注意事項包含：" + "；".join(clean_notes[:4]) + "。")
    return _compact_text_no_ellipsis(" ".join(parts), 780)


def _records_from_form_output(
    document_ir: DocumentIR,
    form: dict[str, Any],
    form_idx: int,
    seq_start: int,
    semantic_output_language: str = "zh-TW",
) -> list[dict[str, Any]]:
    language = resolve_semantic_output_language(semantic_output_language, document_ir)
    pack = get_form_language_pack(language)
    output = dict(form["output"] or {})
    page_idx = int(form["page_idx"])
    if not output.get("filling_guide") and not output.get("field_schema"):
        output = _fallback_form_output_from_table(
            document_ir,
            form,
            output,
            semantic_output_language=language,
        )
    output = _augment_form_output_from_ir_tables(
        document_ir,
        form,
        output,
        semantic_output_language=language,
    )
    form_name = _clean_form_title(output.get("title"))
    source_form_name = _form_title_from_source_path(document_ir.source.path)
    if _is_weak_form_title(form_name):
        form_name = source_form_name
    elif _source_name_has_form_keyword(document_ir.source.path) and not re.search(
        r"申請單|請領單|報支單|出差單|申請表|增加單|移轉單|報廢單|報告單|紀錄單|意見表|審查表|評分表|說明書|保證規約",
        form_name,
    ) and not _looks_like_form_title_candidate(form_name):
        form_name = source_form_name
    fallback_name = f"Form page {page_idx + 1}" if language == "en" else f"表單第 {page_idx + 1} 頁"
    form_name = form_name or fallback_name
    subdoc_id = f"form:{form_idx:04d}:{_slugify(form_name)}"
    logical_doc_id = f"{document_ir.doc_id}::{subdoc_id}"
    page_label = semantic_page_label(page_idx, language)
    common = {
        "document_type": "form_document",
        "doc_id": document_ir.doc_id,
        "logical_doc_id": logical_doc_id,
        "parent_doc_id": document_ir.doc_id,
        "run_id": document_ir.run_id,
        "subdoc_id": subdoc_id,
        "form_name": form_name,
        "source_title": document_ir.source.path,
        "page_indices": sorted(_spreadsheet_form_page_indices(document_ir, page_idx)),
        "page_label": page_label,
        "block_id": str(form["block_id"]),
        "semantic_output_language": language,
        "needs_review": bool(
            output.get("needs_review") or form.get("quality", {}).get("needs_review")
        ),
    }

    fields = _filter_form_fields_for_display(
        _dedupe_form_fields(output.get("field_schema", [])),
        form_name=form_name,
    )
    output["filling_guide"] = _normalize_form_guide_versions(_ensure_form_semantic_guide(
        form_name=form_name,
        source_path=document_ir.source.path,
        filling_guide=str(output.get("filling_guide") or ""),
        fields=fields,
        semantic_output_language=language,
    ))

    parser_notes = _infer_form_notes(_form_semantic_rows_from_ir(document_ir, form))
    vlm_notes = _infer_form_notes([[str(item)] for item in output.get("all_text", []) if str(item).strip()])
    notes = _merge_form_notes_prefer_vlm(parser_notes, vlm_notes)

    records: list[dict[str, Any]] = []
    summary_text = _build_form_display_summary(
        form_name=form_name,
        source_path=document_ir.source.path,
        fields=fields,
        notes=notes,
        semantic_output_language=language,
    )
    triggers = [str(item) for item in output.get("triggers", []) if str(item).strip()]
    if language == "en":
        summary_parts = [
            f"Form: {form_name}.",
            f"Source: {document_ir.source.path}, {page_label}.",
        ]
        if summary_text:
            summary_parts.append(f"Purpose and filling guidance: {_compact_text_no_ellipsis(summary_text, 700)}")
        if triggers:
            summary_parts.append(f"Common query keywords: {', '.join(triggers[:12])}.")
        summary_section = pack.form_summary
    else:
        summary_parts = [
            f"表單：{form_name}。",
            f"來源：{document_ir.source.path}，{page_label}。",
        ]
        if summary_text:
            summary_parts.append(f"用途與填寫重點：{_compact_text_no_ellipsis(summary_text, 700)}")
        if triggers:
            summary_parts.append(f"常見查詢關鍵字：{'、'.join(triggers[:12])}。")
        summary_section = pack.form_summary
    records.append(
        {
            **common,
            "record_id": f"formrec{seq_start + len(records):06d}",
            "content_type": "form_summary",
            "section": summary_section,
            "content": " ".join(summary_parts),
        }
    )

    all_text_items = _dedupe_strings([
        str(item).strip()
        for item in output.get("all_text", [])
        if str(item).strip()
    ])
    supplementary_lines = _form_supplementary_fact_lines(
        all_text_items,
        fields=fields,
        output=output,
    )
    if supplementary_lines:
        if language == "en":
            source_text_section = "Supplementary References & Values"
            source_text_content = (
                f"Form: {form_name}. Section: {source_text_section}. "
                f"Supplementary references and values from source: "
                f"{_compact_text_no_ellipsis('; '.join(supplementary_lines), 3200)}"
            )
        else:
            source_text_section = "補充法規依據與數值"
            source_text_content = (
                f"表單：{form_name}。區塊：{source_text_section}。"
                f"補充法規依據與數值：{_compact_text_no_ellipsis('；'.join(supplementary_lines), 3200)}"
            )
        records.append(
            {
                **common,
                "record_id": f"formrec{seq_start + len(records):06d}",
                "content_type": "form_section",
                "section": source_text_section,
                "content": source_text_content,
            }
        )

    guide_sections = _split_form_guide_sections(
        str(output.get("filling_guide") or ""),
        semantic_output_language=language,
    )
    for section_title, section_text in guide_sections:
        section_limit = 2600 if any(term in section_title for term in ("注意事項", "備註", "Notes")) else 900
        if language == "en":
            content = (
                f"Form: {form_name}. Section: {section_title}. "
                f"{_compact_text_no_ellipsis(section_text, section_limit)}"
            )
        else:
            content = (
                f"表單：{form_name}。區塊：{section_title}。"
                f"{_compact_text_no_ellipsis(section_text, section_limit)}"
            )
        records.append(
            {
                **common,
                "record_id": f"formrec{seq_start + len(records):06d}",
                "content_type": _classify_form_section(section_title, section_text),
                "section": section_title,
                "content": content,
            }
        )

    for field_order, field_data in enumerate(fields):
        if not isinstance(field_data, dict):
            continue
        field_name = str(field_data.get("name") or "").strip()
        if not field_name:
            continue
        field_type = str(field_data.get("type") or "text")
        requirement = str(field_data.get("requirement") or "").strip()
        required = bool(field_data.get("required")) or requirement == "required"
        if not requirement:
            requirement = "required" if required else "situational"
        section = str(field_data.get("section") or _infer_field_section(field_name))
        section_label = display_form_section(section, language)
        aliases = [str(item) for item in field_data.get("aliases", []) if str(item).strip()]
        evidence = str(field_data.get("evidence_text") or "").strip()
        if language == "en":
            content = (
                f"Form: {form_name}. Field: {field_name}. Section: {section_label}. "
                f"Type: {field_type}. Requirement: {semantic_requirement_label(requirement, language)}. "
                f"Purpose: fill in or verify \"{field_name}\"."
            )
            if aliases:
                content += f" Aliases: {', '.join(aliases[:8])}."
            if evidence:
                content += f" Source text: {_compact_text_no_ellipsis(evidence, 160)}"
        else:
            content = (
                f"表單：{form_name}。欄位：{field_name}。區塊：{section_label}。"
                f"型態：{field_type}。填寫條件：{semantic_requirement_label(requirement, language)}。"
                f"用途：填寫或確認「{field_name}」。"
            )
            if aliases:
                content += f"別名：{'、'.join(aliases[:8])}。"
            if evidence:
                content += f"來源文字：{_compact_text_no_ellipsis(evidence, 160)}"
        records.append(
            {
                **common,
                "record_id": f"formrec{seq_start + len(records):06d}",
                "content_type": "form_field",
                "section": section,
                "field_name": field_name,
                "input_type": field_type,
                "required": required,
                "requirement": requirement,
                "field_order": field_order,
                "aliases": aliases,
                "content": content,
            }
        )

    return records


def _form_supplementary_fact_lines(
    all_text_items: list[str],
    *,
    fields: list[dict[str, Any]],
    output: dict[str, Any],
) -> list[str]:
    # Keep only source-text lines that carry a value fact (number/date/amount or
    # legal reference like 第八條) the extracted fields/guide do not already
    # cover. Blank field labels (申請日期： 年 月 日) carry no value and are
    # dropped, so a well-extracted form sheds the noisy "來源抽取文字" dump while
    # a legal reference or amount seen only in the OCR text survives — under a
    # semantic heading, not a residue label.
    from app.pipeline.repair_guard import extract_value_tokens

    lines = [str(item).strip() for item in (all_text_items or []) if str(item).strip()]
    if not lines:
        return []
    guide = str(output.get("filling_guide") or "")
    covered = extract_value_tokens(guide + " " + " ".join(str(field.get("name") or "") for field in fields))
    kept: list[str] = []
    for line in lines:
        line_values = extract_value_tokens(line)
        if line_values and (line_values - covered):
            kept.append(line)
    return kept


def _clean_evidence_text(value: str) -> str:
    text = re.sub(r"\s+", " ", clean_latex_symbols(value or "")).strip()
    text = text.replace("...", "").replace("…", "")
    return text.strip(" ，；、。")


def _normalize_form_guide_versions(guide: str) -> str:
    text = clean_latex_symbols(guide or "").strip()
    if not text:
        return text
    versions = []
    def repl(match: re.Match[str]) -> str:
        version = match.group(1).replace("．", ".")
        if version not in versions:
            versions.append(version)
        return ""
    text = re.sub(
        r"^[ \t>*-]*(?:本表單)?版本(?:日期)?(?:為|：|:)?\s*([0-9]{2,3}[.．][0-9]{1,2}[.．][0-9]{1,2}(?:核定|修正|修訂)?版)[。.]?\s*$",
        repl,
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if versions and not re.search(r"^#{1,6}\s*版本資訊", text, re.MULTILINE):
        text = "## 版本資訊\n" + "\n".join(f"版本：{version}" for version in versions) + "\n\n" + text
    return text


def _requirement_label(requirement: str) -> str:
    labels = {
        "required": "明確必填",
        "conditional": "條件填寫",
        "situational": "依情境填寫",
        "optional": "選填",
    }
    return labels.get(requirement, "依情境填寫")



def _ensure_form_semantic_guide(
    *,
    form_name: str,
    source_path: str,
    filling_guide: str,
    fields: list[dict[str, Any]],
    semantic_output_language: str = "zh-TW",
) -> str:
    language = resolve_semantic_output_language(semantic_output_language)
    pack = get_form_language_pack(language)
    guide = clean_latex_symbols(filling_guide or "").strip()
    required_sections = form_template_sections(language)
    if all(section in guide for section in required_sections):
        return guide

    core_sections = [
        pack.form_purpose,
        pack.use_cases,
        pack.form_structure,
        pack.filling_guidance,
        pack.approval_flow,
        pack.notes,
    ]
    if guide and sum(1 for section in core_sections if section in guide) >= 2:
        additions: list[str] = []
        if pack.rag_summary not in guide:
            if language == "en":
                additions.extend([
                    "",
                    f"## {pack.rag_summary}",
                    f"This document can answer questions about the purpose, applicable scenario, required fields, conditional fields, approval fields, and notes for \"{form_name}\".",
                ])
            else:
                additions.extend([
                    "",
                    f"## {pack.rag_summary}",
                    f"本文件可回答「{form_name}」的用途、適用情境、應填欄位、條件欄位、簽核欄位與注意事項。",
                ])
        return (guide + "\n" + "\n".join(additions)).strip() if additions else guide

    field_dicts = []
    for field_item in fields:
        if not isinstance(field_item, dict):
            continue
        name = str(field_item.get("name") or "").strip()
        if not name:
            continue
        copied = dict(field_item)
        copied.setdefault("section", _infer_field_section(name))
        copied.setdefault("requirement", "required" if copied.get("required") else "situational")
        field_dicts.append(copied)

    signature_fields = [
        field_item for field_item in field_dicts
        if str(field_item.get("type") or "") == "signature" or _infer_field_type(str(field_item.get("name") or "")) == "signature"
    ]
    synthetic = _build_form_semantic_guide(
        title=form_name,
        source_path=source_path,
        sections=_dedupe_strings([str(field.get("section") or "表單欄位") for field in field_dicts]) or ["基本資料", "填寫內容", "簽核"],
        fields=field_dicts,
        notes=[],
        approval_fields=signature_fields,
        semantic_output_language=language,
    )
    if guide:
        synthetic.extend(["", f"## {pack.original_extraction}", guide])
    return "\n".join(synthetic).strip()


def _looks_like_note_field_name(name: str) -> bool:
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    if not text:
        return False
    if not re.match(r"^(?:註[:：]?|備註[:：]?|\d{1,2}[.．、]|[一二三四五六七八九十]+[.．、])", text):
        if not re.search(r"[。；;]", text):
            return False
        if text.startswith(("□", "☐", "☑")) or re.search(r"[:：]", text):
            return False
        return bool(re.search(r"依本院|應|不得|不受|辦理|報備|檢附|須經|規定|規範", text))
    return bool(re.search(r"[。；;]|辦理|填寫|檢附|不得|應於|核銷|報支", text))


def _filter_form_fields_for_display(fields: list[dict[str, Any]], form_name: str) -> list[dict[str, Any]]:
    form_key = re.sub(r"\s+", "", form_name or "")
    filtered: list[dict[str, Any]] = []
    for form_field in fields:
        name = str(form_field.get("name") or "").strip()
        if not name or _looks_like_note_field_name(name):
            continue
        name_key = re.sub(r"\s+", "", name)
        if form_key and name_key in {form_key, f"附件二{form_key}", f"附件一{form_key}"}:
            continue
        filtered.append(form_field)
    return filtered


def _dedupe_form_fields(fields: Any) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for field_data in fields or []:
        if not isinstance(field_data, dict):
            continue
        field_name = _clean_inferred_field_label(str(field_data.get("name") or "").strip())
        if not field_name or _looks_like_note_field_name(field_name):
            continue
        field_type = str(field_data.get("type") or "text").strip() or "text"
        key = (re.sub(r"\s+", "", field_name).lower(), field_type.lower())
        if key not in seen:
            copied = dict(field_data)
            copied["name"] = field_name
            copied["type"] = field_type
            if copied.get("evidence_text"):
                copied["evidence_text"] = _clean_evidence_text(str(copied.get("evidence_text") or ""))
            deduped.append(copied)
            seen[key] = copied
            continue

        existing = seen[key]
        existing["required"] = bool(existing.get("required") or field_data.get("required"))
        existing_aliases = [
            str(item) for item in existing.get("aliases", []) if str(item).strip()
        ]
        incoming_aliases = [
            str(item) for item in field_data.get("aliases", []) if str(item).strip()
        ]
        merged_aliases = list(dict.fromkeys(existing_aliases + incoming_aliases))
        if merged_aliases:
            existing["aliases"] = merged_aliases
        if not existing.get("evidence_text") and field_data.get("evidence_text"):
            existing["evidence_text"] = _clean_evidence_text(str(field_data["evidence_text"]))
    normalized = semantic_fields_to_dicts(semantic_normalize_fields(deduped))
    normalized_keys = {
        (re.sub(r"\s+", "", str(field.get("name") or "")).lower(), str(field.get("type") or "text").lower())
        for field in normalized
    }
    for original_field in deduped:
        name = str(original_field.get("name") or "")
        key = (re.sub(r"\s+", "", name).lower(), str(original_field.get("type") or "text").lower())
        if key not in normalized_keys and re.search(r"[A-Za-z]", name):
            normalized.append(original_field)
            normalized_keys.add(key)
    return normalized




def _filter_low_value_form_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for _, subdoc_records in _group_form_records(records):
        if not _is_low_value_form_record_group(subdoc_records):
            filtered.extend(subdoc_records)
    return filtered


def _is_low_value_form_record_group(records: list[dict[str, Any]]) -> bool:
    if not records:
        return True
    if any(record.get("content_type") == "form_field" for record in records):
        return False

    content_types = {str(record.get("content_type") or "") for record in records}
    if content_types - {"form_summary", "form_section", "form_workflow"}:
        return False

    combined = "\n".join(str(record.get("content") or "") for record in records)
    combined_lower = combined.lower()
    if re.search(r"\bblank page\b|no visible text|no visible fields|empty page", combined_lower):
        return True

    sections = {str(record.get("section") or "").strip() for record in records}
    specific_sections = {
        "Form Fields",
        "Form Fields and Field Descriptions",
        "Filling Guidance",
        "Conditional Fields",
        "Approval Flow",
        "Notes",
        "Attachments",
        "Required Supporting Documents",
        "表單欄位",
        "填寫重點",
        "條件欄位",
        "簽核流程",
        "注意事項",
        "附件/佐證資料",
    }
    if any(section in specific_sections for section in sections):
        return False

    first = records[0]
    form_name = _clean_form_title(first.get("form_name"))
    source_title = _clean_form_title(first.get("source_title"))
    source_stem = _clean_form_title(Path(source_title).stem)
    form_key = re.sub(r"\W+", "", form_name).lower()
    source_keys = {
        re.sub(r"\W+", "", item).lower()
        for item in (source_title, source_stem)
        if item
    }
    title_is_source_stub = bool(form_key and form_key in source_keys)
    title_is_generic = bool(re.fullmatch(r"formpage\d+|form\d*|表單第\d+頁|表單", form_key, re.IGNORECASE))

    template_markers = [
        "source file",
        "use cases",
        "form structure",
        "rag query summary",
        "來源檔案",
        "適用場景",
        "表單結構",
        "rag 查詢摘要",
    ]
    marker_hits = sum(1 for marker in template_markers if marker in combined_lower)
    has_only_template_sections = content_types <= {"form_summary", "form_section", "form_workflow"} and marker_hits >= 2
    compact = re.sub(r"\s+", "", _clean_form_title(combined))

    return bool((title_is_source_stub or title_is_generic) and has_only_template_sections and len(compact) < 2600)


def _group_form_records(
    records: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        subdoc_id = str(record.get("subdoc_id") or "")
        if subdoc_id not in index:
            index[subdoc_id] = []
            grouped.append((subdoc_id, index[subdoc_id]))
        index[subdoc_id].append(record)
    return grouped


def _group_fields_by_section(
    field_records: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    section_order = {
        "申請/基本資料": 0,
        "進修/訓練資訊": 1,
        "保證人/商號資料": 2,
        "出差/行程資訊": 1,
        "費用/報支資訊": 2,
        "附件/佐證資料": 4,
        "簽核/用印": 5,
        "表單欄位": 9,
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in field_records:
        grouped.setdefault(str(record.get("section") or "表單欄位"), []).append(record)

    result = sorted(grouped.items(), key=lambda item: section_order.get(item[0], 99))
    for _, records in result:
        records.sort(key=lambda record: int(record.get("field_order") or 0))
    return result


def select_vlm_fallback_pages(
    document_ir: DocumentIR,
    records: list[dict[str, Any]],
    max_pages: int = 5,
) -> list[int]:
    """Select pages where MinerU likely missed large tables."""

    pages_with_records = {int(record["source_page_idx"]) for record in records}
    candidates: list[int] = []
    for page in document_ir.pages:
        page_idx = page.page_idx
        if page_idx in pages_with_records:
            continue
        page_blocks = document_ir.get_blocks_by_page(page_idx)
        has_unknown = any(block.type == BlockType.UNKNOWN for block in page_blocks)
        has_table = any(block.type == BlockType.TABLE for block in page_blocks)
        text = " ".join(_block_search_text(block) for block in page_blocks)
        looks_like_allowance = bool(
            re.search(r"生活費|日支|地區|國家|城市|美元|USD", text, re.IGNORECASE)
        )
        has_page_image = bool(page.page_image_path)
        if has_page_image and looks_like_allowance and (has_unknown or has_table):
            candidates.append(page_idx)
    return candidates[:max_pages]


def normalize_vlm_table_records(
    output: dict[str, Any],
    document_ir: DocumentIR,
    plan: DocumentPlan,
    page_idx: int,
    seq_start: int = 0,
    needs_review: bool = False,
) -> list[dict[str, Any]]:
    """Normalize VLM JSON rows into the same record shape as MinerU table rows."""

    records = []
    seen: set[tuple[Any, ...]] = set()
    for offset, item in enumerate(output.get("records", []) or []):
        rate = item.get("rate_usd")
        try:
            rate = int(rate) if rate is not None and str(rate).strip() else None
        except (TypeError, ValueError):
            rate = None

        if rate is None:
            continue

        location_label = str(item.get("location_label") or "").strip()
        city_zh = _clean_optional(item.get("city_zh"))
        city_en = _clean_optional(item.get("city_en"))
        country_zh = _clean_optional(item.get("country_zh"))
        country_en = _clean_optional(item.get("country_en"))
        if not location_label:
            location_label = _join_zh_en(city_zh, city_en) or _join_zh_en(country_zh, country_en)

        dedupe_key = (
            page_idx,
            country_zh,
            city_zh,
            item.get("condition"),
            rate,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        confidence = item.get("confidence", 0.8)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        review_reasons = []
        if needs_review or confidence_value < 0.75:
            review_reasons.append("vlm_low_confidence")

        records.append(
            {
                "record_id": f"rec{seq_start + len(records):06d}",
                "document_type": plan.document_type,
                "doc_id": document_ir.doc_id,
                "run_id": document_ir.run_id,
                "block_id": f"vlm-page-{page_idx:04d}",
                "source_page_idx": page_idx,
                "source_title": plan.title,
                "effective_date": plan.effective_date,
                "currency": plan.currency or "USD",
                "region": _clean_optional(item.get("region")),
                "country_zh": country_zh,
                "country_en": country_en,
                "city_zh": city_zh,
                "city_en": city_en,
                "location_label": location_label,
                "location_type": str(item.get("location_type") or "city"),
                "condition": _clean_optional(item.get("condition")),
                "rate_usd": rate,
                "raw_cells": [str(item.get("evidence_text") or location_label)],
                "extraction_route": "vlm_page_image_unknown_table",
                "confidence": confidence_value,
                "needs_review": bool(review_reasons),
                "review_reasons": review_reasons,
            }
        )
    return records


def _block_search_text(block: Any) -> str:
    payload = getattr(block, "payload", {}) or {}
    parts = [
        block.get_text() if hasattr(block, "get_text") else "",
        payload.get("text", ""),
        payload.get("table_body", ""),
        payload.get("caption", ""),
    ]
    return _plain_text(" ".join(coerce_payload_text(part) for part in parts))


def _clean_form_title(value: Any) -> str:
    return semantic_clean_title_noise(value)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value).strip("-")
    return slug[:48] or "form"


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _compact_text(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", clean_latex_symbols(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _compact_text_no_ellipsis(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", clean_latex_symbols(value or "")).strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    boundaries = [
        truncated.rfind(". "),
        truncated.rfind("; "),
        truncated.rfind(", "),
        truncated.rfind("。"),
        truncated.rfind("；"),
        truncated.rfind("，"),
        truncated.rfind(" "),
    ]
    boundary = max(boundaries)
    if boundary >= int(max_chars * 0.72):
        truncated = truncated[: boundary + 1]
    truncated = truncated.rstrip(" ,;，；、。.")
    zh = sum(1 for ch in truncated if "一" <= ch <= "鿿")
    ascii_letters = sum(1 for ch in truncated if ch.isascii() and ch.isalpha())
    suffix = "." if ascii_letters > zh * 1.5 else "。"
    if suffix == ".":
        truncated = re.sub(
            r"(?:,?\s+\b(?:and|or|including|include|includes|not|to|of|for|with|by|from|as|the|a|an|in|on|at)\b)+$",
            "",
            truncated,
            flags=re.IGNORECASE,
        ).rstrip(" ,;")
    return truncated + suffix


def _split_form_guide_sections(
    guide: str,
    semantic_output_language: str = "zh-TW",
) -> list[tuple[str, str]]:
    guide = clean_latex_symbols(guide or "").strip()
    if not guide:
        return []

    language = resolve_semantic_output_language(semantic_output_language)
    pack = get_form_language_pack(language)
    known_headings = {
        pack.filling_rules,
        pack.approval_flow,
        pack.field_descriptions,
        pack.use_cases,
        pack.notes,
        pack.form_purpose,
        pack.form_structure,
        pack.filling_guidance,
        pack.conditional_fields,
        pack.rag_summary,
        pack.version_info,
        pack.original_extraction,
        "填寫規則",
        "簽核流程",
        "表單欄位",
        "附件",
        "適用場景",
        "注意事項",
        "Form Purpose",
        "Use Cases",
        "Form Structure",
        "Filling Guidance",
        "Conditional Fields",
        "Approval Flow",
        "Notes",
        "RAG Query Summary",
    }
    sections: list[tuple[str, list[str]]] = []
    current_title = pack.filling_rules
    current_lines: list[str] = []
    for raw_line in guide.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.sub(r"^#{1,6}\s*", "", line).strip()
        heading = heading.rstrip(":：")
        is_heading = bool(line.startswith("#") or heading in known_headings)
        if is_heading:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = heading
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, " ".join(lines)) for title, lines in sections]


def _classify_form_section(title: str, text: str) -> str:
    haystack = f"{title} {text}"
    if re.search(r"簽核|核可|簽署|主管|院長|approval|signature|reviewer", haystack, re.IGNORECASE):
        return "form_workflow"
    if re.search(r"附件|檢附|合約|切結書|證明|attachment|supporting|evidence", haystack, re.IGNORECASE):
        return "form_attachment_rule"
    if re.search(r"欄位|姓名|身分證|電話|日期|field|name|date|phone", haystack, re.IGNORECASE):
        return "form_section"
    return "form_section"


def _infer_field_section(field_name: str) -> str:
    mapping = [
        (r"法定代理人|委任代理人|代理人姓名|代理人電話|代理人地址|authorized representative|representative", "申請/基本資料"),
        (r"出差地點|出差事由|出差期間|職務代理人|代理職務|出差.*代理人|變更|起訖地點|工作紀要|business purpose|trip|travel|itinerary|lodging|car", "出差/行程資訊"),
        (r"□|☐|保險|報支單位|預估費用|預借金額|金額|費用|交通費|宿費|膳雜費|生活費|辦公費|匯率|幣別|折合台幣|合計|小計|預算|付款|支票|匯款|受款人|應繳回|應補發|沖預借|單據編號|amount|expense|fee|fare|payment|payee|account|routing|total|reimbursement|mileage|rate", "費用/報支資訊"),
        (r"主任|主管|秘書|副院長|院長|人事|處長|簽|章|核|對保|signature|signed|approval|approved|certification", "簽核/用印"),
        (r"保證|保證人|商號|營業|資本|負責人|被保人|關係", "保證人/商號資料"),
        (r"學校|系所|科系|學位|進修|選修|課程|學科|減免|受訓|訓練|education|school", "進修/訓練資訊"),
        (
            r"authorization|authorize|authorizing|disclosure|disclose|consent|purpose|record|records|transcript|tax return|tax form|medical records|education records|from whom|to whom|of what|source|recipient",
            "授權/揭露範圍",
        ),
        (
            r"姓名|出生|身分證|電話|手機|E-?mail|地址|緊急|申請|日期|單位|職級|職稱|員工|name|date|birth|birthday|ssn|social security|taxpayer|identification|phone|email|address|city|state|zip|department|applicant|preparer|vendor|student|passport|nationality|guardian|parent of minor|representative",
            "申請/基本資料",
        ),
        (r"附件|合約|切結|證明|預算|attachment|certificate|document|supporting|evidence", "附件/佐證資料"),
    ]
    for pattern, section in mapping:
        if re.search(pattern, field_name, re.IGNORECASE):
            return section
    return "表單欄位"


def record_to_rag_text(record: dict[str, Any]) -> str:
    """Render a single structured record as a self-contained retrieval text."""

    parts = [str(record.get("source_title") or "").strip()]
    if record.get("effective_date"):
        parts.append(f"自 {record['effective_date']} 生效")
    if record.get("currency"):
        parts.append(f"單位 {record['currency']}")

    if record.get("document_type") == "table_note":
        if record.get("note_text"):
            parts.append(f"表格備註：{record['note_text']}")
    elif record.get("document_type") == "travel_domestic_expense_rate_table":
        if record.get("role_title"):
            parts.append(f"職稱/職級別：{record['role_title']}")
        if record.get("transport_fee_rule"):
            parts.append(f"交通費：{record['transport_fee_rule']}")
        if record.get("lodging_weekday_twd") is not None:
            parts.append(f"宿費平日每日 {record['lodging_weekday_twd']} 元")
        if record.get("lodging_holiday_twd") is not None:
            parts.append(f"宿費假日每日 {record['lodging_holiday_twd']} 元")
        if record.get("miscellaneous_twd") is not None:
            parts.append(f"雜費每日 {record['miscellaneous_twd']} 元")
    else:
        location_parts = [
            record.get("region"),
            _join_zh_en(record.get("country_zh"), record.get("country_en")),
            _join_zh_en(record.get("city_zh"), record.get("city_en")) or record.get("location_label"),
        ]
        location = "，".join(str(part) for part in location_parts if part)
        if location:
            parts.append(location)
        if record.get("condition"):
            parts.append(f"條件：{record['condition']}")
        if record.get("rate_usd") is not None:
            parts.append(f"生活費日支數額為 {record['rate_usd']} 美元")
    if record.get("source_page_idx") is not None:
        parts.append(f"來源頁碼：第 {int(record['source_page_idx']) + 1} 頁")

    return "。".join(part for part in parts if part) + "。"


def write_structured_rag_outputs(output: StructuredRagOutput, outputs_dir: Any) -> dict[str, str]:
    """Write structured RAG artifacts to an outputs directory."""

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "document_plan": outputs_dir / "document_plan.json",
        "structured_records": outputs_dir / "structured_records.jsonl",
        "structured_rag": outputs_dir / "structured_rag.md",
        "structured_chunks": outputs_dir / "structured_chunks.jsonl",
    }

    atomic_write_json(paths["document_plan"], output.plan.to_dict(), ensure_ascii=False, indent=2)
    language = str(output.stats.get("semantic_output_language") or "zh-TW")
    atomic_write_jsonl(
        paths["structured_records"],
        (_localized_record_for_output(record, language) for record in output.records),
    )
    atomic_write_text(paths["structured_rag"], output.rag_markdown, encoding="utf-8")
    atomic_write_jsonl(
        paths["structured_chunks"],
        (_localized_chunk_for_output(chunk, language) for chunk in output.chunks),
    )

    if output.plan.document_type == "form_collection":
        paths.update(_write_form_subdocument_outputs(output, outputs_dir))

    return {key: str(path) for key, path in paths.items()}


def _localized_record_for_output(record: dict[str, Any], language: str) -> dict[str, Any]:
    if resolve_semantic_output_language(language) != "en":
        return record
    result = dict(record)
    section = str(result.get("section") or "")
    if section:
        result["section"] = display_form_section(section, "en")
    result.pop("section_key", None)
    return result


def _localized_chunk_for_output(chunk: dict[str, Any], language: str) -> dict[str, Any]:
    if resolve_semantic_output_language(language) != "en":
        return chunk
    result = dict(chunk)
    metadata = dict(result.get("metadata") or {})
    section = str(metadata.get("section_key") or metadata.get("section") or "")
    if section:
        metadata["section"] = display_form_section(section, "en")
    metadata.pop("section_key", None)
    result["metadata"] = metadata
    return result


def _write_form_subdocument_outputs(
    output: StructuredRagOutput,
    outputs_dir: Path,
) -> dict[str, Path]:
    forms_dir = outputs_dir / "forms"
    forms_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_subdoc: dict[str, list[dict[str, Any]]] = {}
    for chunk in output.chunks:
        subdoc_id = str(chunk.get("metadata", {}).get("subdoc_id") or "")
        if subdoc_id:
            chunks_by_subdoc.setdefault(subdoc_id, []).append(chunk)

    index: list[dict[str, Any]] = []
    paths: dict[str, Path] = {
        "forms_index": outputs_dir / "forms_index.json",
        "forms_dir": forms_dir,
    }
    for form_idx, (subdoc_id, subdoc_records) in enumerate(_group_form_records(output.records)):
        form_key = f"form_{form_idx:04d}"
        markdown_path = forms_dir / f"{form_key}.md"
        fields_path = forms_dir / f"{form_key}.fields.jsonl"
        chunks_path = forms_dir / f"{form_key}.chunks.jsonl"
        first = subdoc_records[0]
        field_records = [
            record for record in subdoc_records if record.get("content_type") == "form_field"
        ]

        form_output = StructuredRagOutput(
            plan=output.plan,
            records=subdoc_records,
            rag_markdown=render_form_documents_markdown(
                output.plan,
                subdoc_records,
                semantic_output_language=str(output.stats.get("semantic_output_language") or "zh-TW"),
            ),
            chunks=chunks_by_subdoc.get(subdoc_id, []),
            stats={},
        )
        atomic_write_text(markdown_path, form_output.rag_markdown, encoding="utf-8")
        language = str(output.stats.get("semantic_output_language") or "zh-TW")
        atomic_write_jsonl(
            fields_path,
            (_localized_record_for_output(record, language) for record in field_records),
        )
        atomic_write_jsonl(
            chunks_path,
            (_localized_chunk_for_output(chunk, language) for chunk in form_output.chunks),
        )

        index.append(
            {
                "form_id": form_key,
                "subdoc_id": subdoc_id,
                "logical_doc_id": first.get("logical_doc_id"),
                "parent_doc_id": first.get("parent_doc_id") or first.get("doc_id"),
                "title": first.get("form_name"),
                "page_indices": first.get("page_indices", []),
                "page_label": first.get("page_label"),
                "record_count": len(subdoc_records),
                "field_count": len(field_records),
                "files": {
                    "markdown": str(markdown_path),
                    "fields": str(fields_path),
                    "chunks": str(chunks_path),
                },
            }
        )

    atomic_write_json(paths["forms_index"], index, ensure_ascii=False, indent=2)
    return paths


def _looks_like_domestic_expense_rate_table(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    corpus_rules = get_corpus_rules()
    return bool(
        any(term in compact for term in corpus_rules.keyword_list("domestic_rate_role_terms"))
        and all(term in compact for term in corpus_rules.keyword_list("domestic_rate_required_terms"))
        and any(term in compact for term in corpus_rules.keyword_list("domestic_rate_lodging_terms"))
        and any(term in compact for term in corpus_rules.keyword_list("domestic_rate_travel_terms"))
    )


def _record_from_domestic_travel_rate_row(
    row: list[str],
    block_id: str,
    page_idx: int,
    doc_id: str,
    run_id: str,
    title: str,
    effective_date: str | None,
    seq: int,
) -> dict[str, Any] | None:
    cells = [_normalize_cell(cell) for cell in row]
    cells = cells + [""] * max(0, 5 - len(cells))
    role = cells[0]
    joined = " ".join(cell for cell in cells if cell)
    if not role or _is_header_row(joined):
        return None
    if not any(_parse_twd_amount(cell) is not None for cell in cells[2:5]):
        return None

    transport_fee_rule = cells[1] or None
    lodging_weekday = _parse_twd_amount(cells[2])
    lodging_holiday = _parse_twd_amount(cells[3])
    miscellaneous = _parse_twd_amount(cells[4])
    if lodging_weekday is None and lodging_holiday is None and miscellaneous is None:
        return None

    noisy = _looks_ocr_noisy(joined)
    return {
        "record_id": f"rec{seq:06d}",
        "document_type": "travel_domestic_expense_rate_table",
        "doc_id": doc_id,
        "run_id": run_id,
        "block_id": block_id,
        "source_page_idx": page_idx,
        "source_title": title,
        "effective_date": effective_date,
        "currency": "TWD",
        "role_title": role,
        "transport_fee_rule": transport_fee_rule,
        "lodging_weekday_twd": lodging_weekday,
        "lodging_holiday_twd": lodging_holiday,
        "miscellaneous_twd": miscellaneous,
        "raw_cells": [cell for cell in cells if cell],
        "needs_review": noisy,
        "review_reasons": ["ocr_noise"] if noisy else [],
    }


def _parse_twd_amount(value: str) -> int | None:
    digits = _normalize_digits(value)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _record_from_allowance_row(
    row: list[str],
    context: dict[str, Any],
    block_id: str,
    page_idx: int,
    doc_id: str,
    run_id: str,
    title: str,
    effective_date: str | None,
    currency: str | None,
    seq: int,
) -> dict[str, Any] | None:
    cells = [_normalize_cell(cell) for cell in row]
    cells = cells + [""] * max(0, 4 - len(cells))
    compact = [cell for cell in cells if cell]
    joined = " ".join(compact)

    if not compact:
        return None
    if _is_header_row(joined):
        return None

    amount_idx = _find_amount_cell(cells)
    has_amount = amount_idx is not None
    location_idx = _find_location_cell(cells, amount_idx)
    location_label = cells[location_idx] if location_idx is not None else ""

    if _is_region_row(cells, has_amount):
        context["region"] = location_label or compact[-1]
        context["country_zh"] = None
        context["country_en"] = None
        context["last_location_label"] = None
        return None

    if not has_amount and location_label:
        if _has_leading_code(cells, location_idx):
            context["last_location_label"] = location_label
            return None
        zh, en = _split_zh_en(location_label)
        context["country_zh"] = zh or location_label
        context["country_en"] = en
        context["last_location_label"] = location_label
        return None

    if not has_amount:
        return None

    rate = int(_normalize_digits(cells[amount_idx or 0]))
    if not location_label and context.get("last_location_label"):
        location_label = str(context["last_location_label"])

    condition = None
    if location_label.startswith("(") and location_label.endswith(")"):
        condition = location_label
        location_label = str(context.get("last_location_label") or "")

    zh, en = _split_zh_en(location_label)
    location_type = "city"
    country_zh = context.get("country_zh")
    country_en = context.get("country_en")
    city_zh = zh or location_label
    city_en = en

    if _is_other_location(location_label):
        location_type = "other"
        city_zh = "其他"
        city_en = "Other"
    elif not country_zh and location_label:
        location_type = "country"
        country_zh = zh or location_label
        country_en = en
        city_zh = None
        city_en = None

    context["last_location_label"] = location_label or context.get("last_location_label")

    needs_review = not location_label or _looks_ocr_noisy(joined)
    return {
        "record_id": f"rec{seq:06d}",
        "document_type": "travel_daily_allowance_table",
        "doc_id": doc_id,
        "run_id": run_id,
        "block_id": block_id,
        "source_page_idx": page_idx,
        "source_title": title,
        "effective_date": effective_date,
        "currency": currency or "USD",
        "region": context.get("region"),
        "country_zh": country_zh,
        "country_en": country_en,
        "city_zh": city_zh,
        "city_en": city_en,
        "location_label": location_label,
        "location_type": location_type,
        "condition": condition,
        "rate_usd": rate,
        "raw_cells": [cell for cell in cells if cell],
        "needs_review": needs_review,
        "review_reasons": ["ocr_noise"] if needs_review else [],
    }


def _parse_plain_rows(table_body: str) -> list[list[str]]:
    rows = []
    for line in table_body.splitlines():
        if "|" in line:
            rows.append([_normalize_cell(cell) for cell in line.split("|")])
    return rows


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _normalize_cell(value: str) -> str:
    value = clean_latex_symbols(value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("0ther", "Other").replace("O ther", "Other")


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _normalize_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _find_amount_cell(cells: list[str]) -> int | None:
    last_non_empty = None
    for idx, cell in enumerate(cells):
        if cell.strip():
            last_non_empty = idx
    if last_non_empty is None:
        return None
    if re.fullmatch(r"\d{2,4}", _normalize_digits(cells[last_non_empty])):
        return last_non_empty
    return None


def _find_location_cell(cells: list[str], amount_idx: int | None) -> int | None:
    candidates = range(0, amount_idx if amount_idx is not None else len(cells))
    for idx in reversed(list(candidates)):
        cell = cells[idx].strip()
        if cell and not re.fullmatch(r"[+\-一二三四五六七八九十百0-9]+", cell):
            return idx
    return None


def _has_leading_code(cells: list[str], location_idx: int | None) -> bool:
    if location_idx is None:
        return False
    for cell in cells[:location_idx]:
        if re.fullmatch(r"\d{1,4}", _normalize_digits(cell)):
            return True
    return False


def _is_header_row(joined: str) -> bool:
    return (
        "編號" in joined
        or "编號" in joined
        or "日支" in joined
        or joined in {"地區、國家 城市或其他", "地區、國家 名稱 城市或其他"}
    )


def _is_region_row(cells: list[str], has_amount: bool) -> bool:
    if has_amount:
        return False
    joined = " ".join(cell for cell in cells if cell)
    return bool(re.fullmatch(r"[A-Z]", cells[0] if cells else "")) or "地區" in joined


def _is_other_location(value: str) -> bool:
    return "其他" in value or "Other" in value


def _split_zh_en(value: str) -> tuple[str | None, str | None]:
    value = value.strip()
    match = re.search(r"^(.*?)\((.*?)\)\s*$", value)
    if not match:
        return (value or None), None
    zh = match.group(1).strip() or None
    en = match.group(2).strip() or None
    return zh, en


def _join_zh_en(zh: Any, en: Any) -> str:
    if zh and en:
        return f"{zh}({en})"
    return str(zh or en or "")


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _page_context(document_ir: DocumentIR, page_idx: int, max_chars: int = 4000) -> str:
    parts = []
    for block in document_ir.get_blocks_by_page(page_idx):
        text = _plain_text(block.get_text())
        if text:
            parts.append(text)
    context = "\n".join(parts)
    if len(context) > max_chars:
        return context[:max_chars] + "..."
    return context


def _resolve_page_image(document_ir: DocumentIR, run_path: Path, page_idx: int) -> Path | None:
    page = next((item for item in document_ir.pages if item.page_idx == page_idx), None)
    if page is None or not page.page_image_path:
        return None

    image_path = Path(page.page_image_path)
    candidates = []
    if image_path.is_absolute():
        candidates.append(image_path)
    else:
        candidates.append(run_path / image_path)
        candidates.append(run_path / "assets" / image_path)
        candidates.append(run_path / "assets" / "pages" / image_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_effective_date(text: str) -> str | None:
    pattern = (
        r"自\s*([0-9一二三四五六七八九十百]+年"
        r"[0-9一二三四五六七八九十]+月"
        r"[0-9一二三四五六七八九十]+日)\s*生效"
    )
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def _looks_ocr_noisy(text: str) -> bool:
    return any(token in text for token in ["0ther", " +七", "一九0", "二00"])


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts
