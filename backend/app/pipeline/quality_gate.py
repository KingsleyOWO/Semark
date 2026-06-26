"""Quality gate for RAG-ready document outputs.

This module checks the final semantic output against parser evidence before the
run is considered RAG-ready. Rules are deliberately conservative: they flag
risk and route to VLM audit instead of rewriting content blindly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.adapters.vlm import VLMAdapter
from app.models.document_ir import Block, BlockType, DocumentIR
from app.pipeline.semantic.language import (
    form_template_sections,
    prompt_form_sections,
    prompt_language_instruction,
    resolve_semantic_output_language,
)
from app.pipeline.semantic.normalizer import is_version_text, split_merged_field_label
from app.pipeline.structured_rag import is_form_like_document, looks_like_reference_table

_QUALITY_GATE_MESSAGES_EN = {
    "source_preview_missing": "Some source pages are missing preview images, so the viewer can only show text structure without visual comparison.",
    "possible_wrong_asset_kind": "A dense data or reference table may have been exported as a form, which can route RAG documents incorrectly.",
    "table_notes_missing": "Notes or annotations after a table are present in the source but missing from the final semantic document.",
    "html_table_without_semantic_text": "A table subdocument has insufficient semantic description and may contain only raw table content or a short summary.",
    "semantic_output_too_short": "Figure or form semantic output is too short and may not support RAG queries well.",
    "semantic_template_incomplete": "The form semantic document is missing required RAG template sections and may be only field lists or summary text.",
    "semantic_summary_too_dense": "The form summary is too dense; fields, keywords, and workflow may be mixed into one paragraph and hard to read after retrieval.",
    "form_like_document_not_structured": "The source appears to be a fillable form, but the final output was not split into semantic form documents.",
    "possible_over_split_form": "A single spreadsheet form may have been split into multiple low-information subforms and should likely be merged.",
    "form_signature_fields_missing": "The source page contains signature, manager, or approval fields that appear to be missing from the final semantic document.",
    "field_name_too_long": "Some field names are too long and may contain merged fields or raw OCR labels.",
    "merged_field_detected": "Some fields appear to combine multiple labels, which can reduce RAG retrieval precision.",
    "too_many_generic_fields": "Too many fields remain in the generic form-field category, so semantic grouping may be insufficient.",
    "version_misclassified_as_note": "Version information appears to be placed in notes and should be represented as document version metadata.",
    "summary_contains_ellipsis": "Semantic output contains ellipses, which may cause RAG retrieval to return incomplete sentences.",
    "raw_parser_residue": "The final output appears to contain raw parser table markers and should be converted into semantic sentences or structured fields.",
    "ocr_title_noise": "The final output still appears to contain OCR noise in headings.",
    "english_noise_high": "A Traditional Chinese output contains an unusually high English ratio and may include untranslated VLM captions or summaries.",
    "structured_output_empty": "The structured semantic output is empty even though source text exists.",
    "vlm_enrichment_parse_failed": "A VLM/LLM enrichment response could not be parsed, so semantic content may be missing.",
    "target_language_mismatch": "The semantic output contains text from the wrong output language.",
    "vlm_audit_failed": "The VLM audit failed, so rule-based quality findings were preserved.",
    "vlm_audit_missing_items": "After comparing the source page, the VLM audit reported possible omissions in the final semantic document.",
}


def _quality_gate_message_for_language(code: str, message: str, semantic_output_language: str) -> str:
    if resolve_semantic_output_language(semantic_output_language) == "en":
        return _QUALITY_GATE_MESSAGES_EN.get(code, message)
    return message


@dataclass
class QualityGateIssue:
    code: str
    severity: str
    message: str
    page_idx: int | None = None
    block_id: str | None = None
    document_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, semantic_output_language: str = "zh-TW") -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": _quality_gate_message_for_language(self.code, self.message, semantic_output_language),
            "page_idx": self.page_idx,
            "block_id": self.block_id,
            "document_id": self.document_id,
            "evidence": self.evidence,
        }


@dataclass
class QualityGateResult:
    status: str
    score: float
    issues: list[QualityGateIssue] = field(default_factory=list)
    vlm_audit_candidates: list[dict[str, Any]] = field(default_factory=list)
    vlm_audits: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        semantic_output_language = str(self.stats.get("semantic_output_language") or "zh-TW")
        return {
            "status": self.status,
            "score": self.score,
            "issues": [issue.to_dict(semantic_output_language) for issue in self.issues],
            "vlm_audit_candidates": self.vlm_audit_candidates,
            "vlm_audits": self.vlm_audits,
            "stats": self.stats,
        }


async def run_quality_gate(
    *,
    document_ir: DocumentIR,
    source_md: str,
    assets: list[Any],
    structured_output: Any,
    enrichments: dict[str, dict[str, Any]],
    run_path: Path,
    vlm_adapter: VLMAdapter | None = None,
    max_vlm_audits: int = 2,
    semantic_output_language: str = "auto",
) -> QualityGateResult:
    """Run rule-based checks and optional VLM audit for risky pages."""

    issues: list[QualityGateIssue] = []
    final_text = _final_semantic_text(source_md, structured_output)
    structured_text = _structured_semantic_text(structured_output)
    language = resolve_semantic_output_language(semantic_output_language, document_ir, final_text)
    source_text = _compact(final_text)
    assets_by_block = {str(getattr(asset, "block_id", "")): asset for asset in assets}
    structured_record_blocks = {
        str(record.get("block_id") or "")
        for record in getattr(structured_output, "records", []) or []
        if record.get("block_id")
    }

    issues.extend(_check_source_page_images(document_ir))
    issues.extend(_check_structured_output_presence(document_ir, structured_output, structured_text, source_md))
    issues.extend(_check_enrichment_failures(enrichments))
    issues.extend(_check_table_classification(document_ir, assets_by_block))
    issues.extend(_check_table_notes(document_ir, source_text))
    issues.extend(_check_assets_semantics(assets, structured_record_blocks, final_text))
    issues.extend(_check_semantic_template(structured_output, final_text, language))
    issues.extend(_check_form_like_document_not_structured(document_ir, structured_output))
    issues.extend(_check_possible_over_split_form(document_ir, structured_output))
    issues.extend(_check_form_signatures(document_ir, source_text, enrichments))
    issues.extend(_check_rag_readiness(structured_output, final_text))
    issues.extend(_check_language_noise(source_text, language, structured_text=structured_text))

    candidates = _build_vlm_audit_candidates(document_ir, issues, max_candidates=max_vlm_audits)
    vlm_audits: list[dict[str, Any]] = []
    if vlm_adapter is not None and candidates:
        for candidate in candidates[:max_vlm_audits]:
            audit = await _run_vlm_audit(
                vlm_adapter=vlm_adapter,
                document_ir=document_ir,
                run_path=run_path,
                candidate=candidate,
                source_md=source_md,
                structured_text=structured_text,
                semantic_output_language=language,
            )
            vlm_audits.append(audit)
            issues.extend(_issues_from_vlm_audit(audit))

    status = _status_from_issues(issues)
    score = _score_from_issues(issues)
    semantic_quality = _semantic_quality_stats(issues)
    stats = {
        "issue_count": len(issues),
        "issues_by_severity": _count_by(issue.severity for issue in issues),
        "issues_by_code": _count_by(issue.code for issue in issues),
        "vlm_audit_candidate_count": len(candidates),
        "vlm_audit_count": len(vlm_audits),
        "structured_document_type": getattr(getattr(structured_output, "plan", None), "document_type", None),
        "structured_record_count": len(getattr(structured_output, "records", []) or []),
        "semantic_quality": semantic_quality,
        "semantic_output_language": language,
    }
    return QualityGateResult(
        status=status,
        score=score,
        issues=issues,
        vlm_audit_candidates=candidates,
        vlm_audits=vlm_audits,
        stats=stats,
    )


def write_quality_gate(result: QualityGateResult, outputs_dir: Path) -> Path:
    path = outputs_dir / "quality_gate.json"
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path



def _final_semantic_text(source_md: str, structured_output: Any) -> str:
    parts = [source_md or ""]
    rag_markdown = str(getattr(structured_output, "rag_markdown", "") or "")
    if rag_markdown and rag_markdown not in parts[0]:
        parts.append(rag_markdown)
    for chunk in getattr(structured_output, "chunks", []) or []:
        content = str(chunk.get("content") or "") if isinstance(chunk, dict) else ""
        if content:
            parts.append(content)
    return "\n".join(parts)



def _structured_semantic_text(structured_output: Any) -> str:
    parts: list[str] = []
    rag_markdown = str(getattr(structured_output, "rag_markdown", "") or "")
    if rag_markdown:
        parts.append(rag_markdown)
    for chunk in getattr(structured_output, "chunks", []) or []:
        content = str(chunk.get("content") or "") if isinstance(chunk, dict) else ""
        if content:
            parts.append(content)
    return "\n".join(parts)


def _check_structured_output_presence(
    document_ir: DocumentIR,
    structured_output: Any,
    structured_text: str,
    source_md: str,
) -> list[QualityGateIssue]:
    if _compact(structured_text):
        return []
    if len(_compact(source_md)) < 40:
        return []

    plan = getattr(structured_output, "plan", None)
    document_type = str(getattr(plan, "document_type", "") or "")
    records = [record for record in getattr(structured_output, "records", []) or [] if isinstance(record, dict)]
    chunks = [chunk for chunk in getattr(structured_output, "chunks", []) or [] if isinstance(chunk, dict)]
    has_semantic_candidate = any(block.type in {BlockType.TABLE, BlockType.IMAGE} for block in document_ir.blocks)
    severity = "high" if has_semantic_candidate or not records else "medium"
    return [
        QualityGateIssue(
            code="structured_output_empty",
            severity=severity,
            message="來源已有可解析內容，但 structured_rag / structured_chunks 為空，最終語意輸出不可用。",
            evidence={
                "source_text_length": len(_compact(source_md)),
                "structured_document_type": document_type,
                "record_count": len(records),
                "chunk_count": len(chunks),
                "has_table_or_image": has_semantic_candidate,
            },
        )
    ]


def _check_enrichment_failures(enrichments: dict[str, dict[str, Any]]) -> list[QualityGateIssue]:
    issues: list[QualityGateIssue] = []
    for block_id, enrichment in (enrichments or {}).items():
        output = enrichment.get("output") or {}
        if not isinstance(output, dict):
            continue
        error = str(output.get("_error") or "")
        if "JSON_PARSE_FAILED" not in error:
            continue
        evidence = dict(enrichment.get("evidence") or {})
        input_info = dict(enrichment.get("input") or {})
        page_idx = evidence.get("page_idx", input_info.get("page_idx"))
        issues.append(
            QualityGateIssue(
                code="vlm_enrichment_parse_failed",
                severity="high",
                message="VLM/LLM enrichment 回傳無法解析的 JSON，可能導致圖表、表單或表格語意內容缺漏。",
                page_idx=page_idx if isinstance(page_idx, int) else None,
                block_id=str(block_id),
                evidence={
                    "kind": enrichment.get("kind"),
                    "error": error[:300],
                    "tokens_used": (enrichment.get("quality") or {}).get("tokens_used"),
                },
            )
        )
    return issues

def _check_source_page_images(document_ir: DocumentIR) -> list[QualityGateIssue]:
    issues = []
    if not document_ir.pages:
        return issues
    missing = [page.page_idx for page in document_ir.pages if not page.page_image_path]
    if missing:
        severity = "warning" if document_ir.source.ext.lower() in {"xls", "xlsx", "ods"} else "medium"
        issues.append(
            QualityGateIssue(
                code="source_preview_missing",
                severity=severity,
                message="部分來源頁沒有頁面圖，Viewer/文件管理只能顯示文字結構，無法做視覺對照。",
                evidence={"missing_page_indices": missing[:20], "source_ext": document_ir.source.ext},
            )
        )
    return issues


def _check_table_classification(document_ir: DocumentIR, assets_by_block: dict[str, Any]) -> list[QualityGateIssue]:
    issues = []
    for block in document_ir.blocks:
        if block.type != BlockType.TABLE:
            continue
        table_body = str(block.payload.get("table_body") or "")
        asset = assets_by_block.get(block.block_id)
        if asset and getattr(asset, "type", "") == "form_asset" and looks_like_reference_table(table_body):
            issues.append(
                QualityGateIssue(
                    code="possible_wrong_asset_kind",
                    severity="high",
                    message="密集資料/參照表疑似被輸出成表單，可能造成 RAG 文件分流錯誤。",
                    page_idx=block.page_idx,
                    block_id=block.block_id,
                    evidence={"asset_id": getattr(asset, "asset_id", None), "asset_type": getattr(asset, "type", None)},
                )
            )
    return issues


def _check_table_notes(document_ir: DocumentIR, source_text: str) -> list[QualityGateIssue]:
    issues = []
    for block in document_ir.blocks:
        if block.type != BlockType.TABLE:
            continue
        notes = _collect_notes_after_block(document_ir, block)
        if not notes:
            continue
        missing = [note for note in notes if _compact(note) not in source_text]
        if missing:
            issues.append(
                QualityGateIssue(
                    code="table_notes_missing",
                    severity="high",
                    message="表格後方有備註/註解，但最終語意文件沒有完整包含。",
                    page_idx=block.page_idx,
                    block_id=block.block_id,
                    evidence={"missing_notes": missing[:6]},
                )
            )
    return issues


def _check_assets_semantics(assets: list[Any], structured_record_blocks: set[str], final_text: str) -> list[QualityGateIssue]:
    issues = []
    final_compact = _compact(final_text)
    for asset in assets:
        retrieval_text = str(getattr(asset, "retrieval_text", "") or "")
        asset_type = str(getattr(asset, "type", ""))
        if asset_type == "form_asset":
            field_names = " ".join(str(field.get("name") or "") for field in (getattr(asset, "field_schema", []) or []) if isinstance(field, dict))
            retrieval_text = " ".join(part for part in [retrieval_text, str(getattr(asset, "filling_guide", "") or ""), field_names] if part)
        elif asset_type == "figure_asset":
            figure_parts = [
                retrieval_text,
                str(getattr(asset, "semantic_caption", "") or ""),
                str(getattr(asset, "structured_content", "") or ""),
                " ".join(str(item) for item in (getattr(asset, "facts", []) or [])),
                " ".join(str(item) for item in (getattr(asset, "keywords", []) or [])),
            ]
            retrieval_text = " ".join(part for part in figure_parts if part)
        compact = _compact(retrieval_text)
        if asset_type == "form_asset" and str(getattr(asset, "block_id", "")) in structured_record_blocks:
            continue
        if asset_type == "form_asset":
            title_compact = _compact(str(getattr(asset, "title", "") or ""))
            if len(title_compact) >= 4 and title_compact in final_compact:
                continue
        if asset_type == "table_asset" and _table_asset_is_semanticized(retrieval_text, final_compact):
            continue
        if asset_type == "table_asset" and ("<table" in retrieval_text.lower() or len(compact) < 80):
            issues.append(
                QualityGateIssue(
                    code="html_table_without_semantic_text",
                    severity="medium",
                    message="表格子文件缺少足夠語意化描述，可能只剩原始表格或過短摘要。",
                    page_idx=getattr(asset, "page_idx", None),
                    block_id=getattr(asset, "block_id", None),
                    document_id=getattr(asset, "asset_id", None),
                    evidence={"text_length": len(compact), "asset_type": asset_type},
                )
            )
        if asset_type in {"figure_asset", "form_asset"} and len(compact) < 120:
            if _is_low_value_empty_figure_asset(asset, retrieval_text):
                continue
            issues.append(
                QualityGateIssue(
                    code="semantic_output_too_short",
                    severity="medium",
                    message="圖示/表單語意輸出過短，可能不足以支援 RAG 查詢。",
                    page_idx=getattr(asset, "page_idx", None),
                    block_id=getattr(asset, "block_id", None),
                    document_id=getattr(asset, "asset_id", None),
                    evidence={"text_length": len(compact), "asset_type": asset_type},
                )
            )
    return issues



def _is_low_value_empty_figure_asset(asset: Any, retrieval_text: str) -> bool:
    if str(getattr(asset, "type", "")) != "figure_asset":
        return False
    title = str(getattr(asset, "title", "") or "")
    content_parts = [
        retrieval_text,
        str(getattr(asset, "semantic_caption", "") or ""),
        str(getattr(asset, "structured_content", "") or ""),
        " ".join(str(item) for item in (getattr(asset, "facts", []) or [])),
        " ".join(str(item) for item in (getattr(asset, "keywords", []) or [])),
        " ".join(str(item) for item in (getattr(asset, "triggers", []) or [])),
    ]
    combined = " ".join(part for part in content_parts if part).strip()
    if re.search(r"流程圖|流程|flowchart|diagram|chart|graph|table|表格|圖表|架構|structure|org chart", combined, re.I):
        return False
    if not _is_generic_figure_title(title):
        return False
    meaningful = re.sub(r"(?i)\b(?:figure|fig|image)\s*\d+\b", " ", combined)
    meaningful = re.sub(r"圖\s*\d+|圖片\s*\d+", " ", meaningful)
    meaningful = meaningful.replace(title, " ")
    return len(_compact(meaningful)) < 8


def _is_generic_figure_title(title: str) -> bool:
    compact = _compact(title).lower()
    return bool(re.fullmatch(r"(?:figure|fig|image)\d+|圖\d+|圖片\d+", compact))


def _table_asset_is_semanticized(retrieval_text: str, final_compact: str) -> bool:
    """Return true when a short table asset is already represented in final semantic text."""

    labels = []
    text = re.sub(r"<[^>]+>", " ", retrieval_text or "")
    for raw in re.split(r"[|｜,，\n:：\s]+", text):
        label = _compact(raw)
        if not label:
            continue
        if label.upper() in {"TABLE", "COLUMNS", "ROW"}:
            continue
        if len(label) <= 1:
            continue
        if label not in labels:
            labels.append(label)
    if not labels:
        return False
    matched = [label for label in labels if label in final_compact]
    return len(matched) >= min(2, len(labels))


def _check_semantic_template(structured_output: Any, final_text: str, semantic_output_language: str = "zh-TW") -> list[QualityGateIssue]:
    """Check whether structured semantic outputs follow the expected RAG template."""
    plan = getattr(structured_output, "plan", None)
    document_type = str(getattr(plan, "document_type", "") or "")
    if document_type not in {"form_collection", "form_document"}:
        return []

    required_sections = form_template_sections(semantic_output_language, include_field_descriptions=True)
    present = [section for section in required_sections if section in final_text]
    missing = [section for section in required_sections if section not in final_text]
    issues: list[QualityGateIssue] = []
    if len(present) < 4:
        issues.append(
            QualityGateIssue(
                code="semantic_template_incomplete",
                severity="high",
                message="表單語意文件缺少固定 RAG 模板章節，可能只剩欄位堆疊或摘要文字。",
                evidence={"missing_sections": missing, "present_sections": present},
            )
        )

    if re.search(r"用途與填寫重點[^\n]{500,}", final_text) and "填寫重點" not in present:
        issues.append(
            QualityGateIssue(
                code="semantic_summary_too_dense",
                severity="medium",
                message="表單摘要過度密集，欄位、關鍵字與流程可能混在同一段，RAG 召回後不易判讀。",
                evidence={"text_length": len(_compact(final_text))},
            )
        )
    return issues


def _check_form_like_document_not_structured(document_ir: DocumentIR, structured_output: Any) -> list[QualityGateIssue]:
    plan = getattr(structured_output, "plan", None)
    document_type = str(getattr(plan, "document_type", "") or "")
    if document_type in {"form_collection", "form_document"}:
        return []
    if not is_form_like_document(document_ir):
        return []
    return [
        QualityGateIssue(
            code="form_like_document_not_structured",
            severity="high",
            message="來源看起來是可填寫表單，但最終輸出沒有拆成語意化表單文件。",
            evidence={
                "source_path": document_ir.source.path,
                "source_ext": document_ir.source.ext,
                "structured_document_type": document_type,
            },
        )
    ]


def _check_possible_over_split_form(document_ir: DocumentIR, structured_output: Any) -> list[QualityGateIssue]:
    plan = getattr(structured_output, "plan", None)
    document_type = str(getattr(plan, "document_type", "") or "")
    if document_type != "form_collection":
        return []
    if str(document_ir.source.ext or "").lower() not in {"xls", "xlsx", "ods"}:
        return []
    source_name = Path(document_ir.source.path).stem
    if not re.search(r"申請單|請領單|報支單|出差單|核銷單|請款單|申報單|異動單|申請表|登記表", source_name):
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for record in getattr(structured_output, "records", []) or []:
        if not isinstance(record, dict) or record.get("document_type") != "form_document":
            continue
        subdoc_id = str(record.get("subdoc_id") or "")
        item = grouped.setdefault(subdoc_id, {"title": record.get("form_name"), "fields": 0})
        if record.get("content_type") == "form_field":
            item["fields"] += 1
    if len(grouped) <= 1:
        return []

    weak = [
        item for item in grouped.values()
        if int(item.get("fields") or 0) <= 3 or _looks_like_field_title(str(item.get("title") or ""))
    ]
    if len(weak) < 1:
        return []
    return [
        QualityGateIssue(
            code="possible_over_split_form",
            severity="medium",
            message="單一表單型 spreadsheet 疑似被拆成多個低資訊量子表單，可能應合併成同一份表單文件。",
            evidence={"source_path": document_ir.source.path, "forms": grouped},
        )
    ]


def _looks_like_field_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", title or "")
    if compact in {"單位主管", "單位主管核定", "領款人簽章", "出差人簽名", "主管核定", "合計金額"}:
        return True
    return len(compact) <= 8 and bool(re.search(r"主管|簽章|簽名|核定|合計|金額|申請人|事由|地點", compact))


def _check_form_signatures(
    document_ir: DocumentIR,
    source_text: str,
    enrichments: dict[str, dict[str, Any]],
) -> list[QualityGateIssue]:
    issues = []
    raw_by_page: dict[int, str] = {}
    for block in document_ir.blocks:
        if block.type in {BlockType.TEXT, BlockType.TABLE}:
            raw_by_page.setdefault(block.page_idx, "")
            raw_by_page[block.page_idx] += "\n" + _plain_block_text(block)

    for page_idx, raw_text in raw_by_page.items():
        raw_compact = _compact(raw_text)
        if not re.search(r"(?:簽名|簽章|核章|單位主管|主任秘書|副院長|院長|董事長|申請人)", raw_compact):
            continue
        missing_terms = [term for term in _extract_signature_terms(raw_text) if term not in source_text]
        if not missing_terms:
            continue
        issues.append(
            QualityGateIssue(
                code="form_signature_fields_missing",
                severity="high",
                message="原始頁面出現簽名/主管/核章欄位，但最終語意文件疑似沒有保留。",
                page_idx=page_idx,
                evidence={"raw_terms": _extract_signature_terms(raw_text), "missing_terms": missing_terms},
            )
        )
    return issues


def _check_rag_readiness(structured_output: Any, final_text: str) -> list[QualityGateIssue]:
    issues: list[QualityGateIssue] = []
    records = [record for record in getattr(structured_output, "records", []) or [] if isinstance(record, dict)]

    long_fields = []
    merged_fields = []
    generic_fields = 0
    form_fields = 0
    for record in records:
        if record.get("content_type") != "form_field":
            continue
        form_fields += 1
        field_name = str(record.get("field_name") or "").strip()
        section = str(record.get("section") or "")
        if section == "表單欄位":
            generic_fields += 1
        if len(field_name) > 32:
            long_fields.append(field_name)
        if len(split_merged_field_label(field_name)) > 1:
            merged_fields.append(field_name)

    if long_fields:
        issues.append(
            QualityGateIssue(
                code="field_name_too_long",
                severity="warning",
                message="部分欄位名稱過長，可能是多個欄位黏連或 raw OCR label。",
                evidence={"fields": long_fields[:8]},
            )
        )
    if merged_fields:
        issues.append(
            QualityGateIssue(
                code="merged_field_detected",
                severity="warning",
                message="部分欄位疑似多個欄位黏在一起，會降低 RAG 查詢精準度。",
                evidence={"fields": merged_fields[:8]},
            )
        )
    if form_fields and generic_fields > max(4, form_fields * 0.35):
        issues.append(
            QualityGateIssue(
                code="too_many_generic_fields",
                severity="warning",
                message="太多欄位仍停留在 generic 表單欄位分類，語意分組可能不夠精準。",
                evidence={"generic_fields": generic_fields, "form_fields": form_fields},
            )
        )

    note_version_lines = []
    version_in_note_section = re.compile(
        r"^#{2,3}\s*(?:注意事項|備註)\s*$[\s\S]{0,180}?([0-9]{2,3}[.．][0-9]{1,2}[.．][0-9]{1,2}(?:核定|修正|修訂)?版)",
        re.MULTILINE,
    )
    for match in version_in_note_section.finditer(final_text):
        candidate = match.group(1)
        if is_version_text(candidate):
            note_version_lines.append(candidate)
    if note_version_lines:
        issues.append(
            QualityGateIssue(
                code="version_misclassified_as_note",
                severity="warning",
                message="版本資訊疑似被放進注意事項，應改放文件版本 metadata。",
                evidence={"versions": note_version_lines[:6]},
            )
        )

    if "..." in final_text:
        issues.append(
            QualityGateIssue(
                code="summary_contains_ellipsis",
                severity="warning",
                message="語意輸出含截斷符號，可能讓 RAG 召回到不完整句子。",
                evidence={},
            )
        )
    if re.search(r"<table|</tr>|</td>|\bTABLE:|\bROW:", final_text, re.IGNORECASE):
        issues.append(
            QualityGateIssue(
                code="raw_parser_residue",
                severity="warning",
                message="最終輸出疑似殘留 parser 原始表格標記，應轉成語意句或結構化欄位。",
                evidence={},
            )
        )
    if re.search(r"^#{1,3}\s*.*表[一二三四五六七八九十0-9]+[〇○昇鑑箇]", final_text, re.MULTILINE):
        issues.append(
            QualityGateIssue(
                code="ocr_title_noise",
                severity="warning",
                message="最終輸出仍有標題 OCR 雜訊。",
                evidence={},
            )
        )
    return issues


def _semantic_quality_stats(issues: list[QualityGateIssue]) -> dict[str, Any]:
    correctness_codes = {
        "source_preview_missing",
        "possible_wrong_asset_kind",
        "table_notes_missing",
        "html_table_without_semantic_text",
        "semantic_output_too_short",
        "semantic_template_incomplete",
        "form_like_document_not_structured",
        "possible_over_split_form",
        "form_signature_fields_missing",
        "english_noise_high",
        "structured_output_empty",
        "vlm_enrichment_parse_failed",
        "target_language_mismatch",
    }
    readiness_codes = {
        "semantic_summary_too_dense",
        "field_name_too_long",
        "merged_field_detected",
        "too_many_generic_fields",
        "version_misclassified_as_note",
        "summary_contains_ellipsis",
        "raw_parser_residue",
        "ocr_title_noise",
        "target_language_mismatch",
    }
    correctness_penalty = 0.0
    readiness_penalty = 0.0
    repairs: set[str] = set()
    repair_map = {
        "field_name_too_long": "split_merged_fields",
        "merged_field_detected": "split_merged_fields",
        "too_many_generic_fields": "classify_fields",
        "version_misclassified_as_note": "move_version_to_metadata",
        "summary_contains_ellipsis": "compress_summary_without_ellipsis",
        "raw_parser_residue": "semanticize_raw_table",
        "ocr_title_noise": "clean_title_noise",
        "structured_output_empty": "generate_structured_semantic_output",
        "vlm_enrichment_parse_failed": "repair_unparsed_enrichment_output",
        "target_language_mismatch": "rewrite_in_target_language",
    }
    for issue in issues:
        severity_penalty = {"high": 0.2, "medium": 0.12, "warning": 0.04}.get(issue.severity, 0.04)
        if issue.code in correctness_codes:
            correctness_penalty += severity_penalty
        if issue.code in readiness_codes:
            readiness_penalty += max(0.06, severity_penalty)
        if issue.code in repair_map:
            repairs.add(repair_map[issue.code])
    return {
        "correctness_score": max(0.0, round(1.0 - correctness_penalty, 3)),
        "rag_readiness_score": max(0.0, round(1.0 - readiness_penalty, 3)),
        "recommended_repairs": sorted(repairs),
    }


def _check_language_noise(
    source_text: str,
    semantic_output_language: str = "zh-TW",
    *,
    structured_text: str | None = None,
) -> list[QualityGateIssue]:
    check_text = _compact(structured_text) if structured_text is not None else source_text
    if not check_text:
        return []

    zh = sum(1 for ch in check_text if "一" <= ch <= "鿿")
    ascii_letters = sum(1 for ch in check_text if ch.isascii() and ch.isalpha())
    if semantic_output_language == "zh-TW":
        if zh >= 80 and ascii_letters > zh * 1.2:
            return [
                QualityGateIssue(
                    code="english_noise_high",
                    severity="medium",
                    message="中文文件輸出中英文比例異常偏高，可能混入 VLM 英文 caption 或未本地化摘要。",
                    evidence={"chinese_chars": zh, "ascii_letters": ascii_letters},
                )
            ]
        return []

    if semantic_output_language == "en":
        chinese_template_terms = [
            term
            for term in ("填寫規則", "申請", "簽核流程", "表單", "注意事項", "來源頁面")
            if term in check_text
        ]
        if zh >= 4 or chinese_template_terms:
            return [
                QualityGateIssue(
                    code="target_language_mismatch",
                    severity="medium",
                    message="英文語意輸出混入中文模板詞或中文片段，應改寫為一致的英文輸出。",
                    evidence={
                        "chinese_chars": zh,
                        "ascii_letters": ascii_letters,
                        "chinese_template_terms": chinese_template_terms[:10],
                    },
                )
            ]
    return []


def _build_vlm_audit_candidates(
    document_ir: DocumentIR,
    issues: list[QualityGateIssue],
    max_candidates: int,
) -> list[dict[str, Any]]:
    # VLM audit is the main judge for high-risk visual/semantic cases, but it is
    # fed MinerU evidence and current structured output so it cannot free-write.
    if max_candidates <= 0:
        return []

    risky_codes = {
        "possible_wrong_asset_kind",
        "form_signature_fields_missing",
        "html_table_without_semantic_text",
        "structured_output_empty",
        "vlm_enrichment_parse_failed",
        "target_language_mismatch",
        "form_like_document_not_structured",
        "semantic_template_incomplete",
    }
    doc_level_codes = {
        "structured_output_empty",
        "target_language_mismatch",
        "form_like_document_not_structured",
        "semantic_template_incomplete",
    }
    page_to_reasons: dict[int, list[str]] = {}
    pages_with_images = [page for page in document_ir.pages if page.page_image_path]

    for issue in issues:
        if issue.code not in risky_codes:
            continue
        if issue.page_idx is not None:
            page = next((item for item in document_ir.pages if item.page_idx == issue.page_idx), None)
            if page is None or not page.page_image_path:
                continue
            page_to_reasons.setdefault(issue.page_idx, []).append(issue.code)
            continue
        if issue.code in doc_level_codes:
            for page in pages_with_images[:max_candidates]:
                page_to_reasons.setdefault(page.page_idx, []).append(issue.code)

    candidates = []
    for page_idx, reasons in sorted(page_to_reasons.items()):
        page = next(item for item in document_ir.pages if item.page_idx == page_idx)
        candidates.append(
            {
                "page_idx": page_idx,
                "page_image_path": page.page_image_path,
                "reasons": sorted(set(reasons)),
            }
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


async def _run_vlm_audit(
    *,
    vlm_adapter: VLMAdapter,
    document_ir: DocumentIR,
    run_path: Path,
    candidate: dict[str, Any],
    source_md: str,
    structured_text: str,
    semantic_output_language: str = "zh-TW",
) -> dict[str, Any]:
    page_idx = int(candidate["page_idx"])
    image_path = _resolve_page_image(run_path, str(candidate.get("page_image_path") or ""))
    if image_path is None:
        return {"success": False, "page_idx": page_idx, "error": "missing_page_image"}

    context = _audit_context(
        document_ir,
        page_idx,
        source_md,
        structured_text,
        candidate.get("reasons") or [],
    )
    result = await vlm_adapter.enrich(
        kind="quality_audit",
        image_path=image_path,
        context=context,
        doc_id=document_ir.doc_id,
        run_id=document_ir.run_id,
        page_idx=page_idx,
        bbox=None,
        extra_vars={
            "semantic_output_language": semantic_output_language,
            "semantic_output_language_instruction": prompt_language_instruction(semantic_output_language),
            "semantic_template_sections": prompt_form_sections(semantic_output_language),
        },
    )
    return {
        "success": result.success,
        "page_idx": page_idx,
        "reasons": candidate.get("reasons") or [],
        "output": result.output,
        "error": result.error,
        "tokens_used": result.tokens_used,
        "duration_seconds": result.duration_seconds,
        "needs_review": result.needs_review,
    }


def _issues_from_vlm_audit(audit: dict[str, Any]) -> list[QualityGateIssue]:
    if not audit.get("success"):
        return [
            QualityGateIssue(
                code="vlm_audit_failed",
                severity="warning",
                message="VLM audit 執行失敗，已保留規則型檢查結果。",
                page_idx=audit.get("page_idx"),
                evidence={"error": audit.get("error")},
            )
        ]
    output = audit.get("output") or {}
    status = str(output.get("status") or "").lower()
    issues = []
    missing_items = output.get("missing_items") or []
    if status in {"needs_fix", "fail", "failed"} or missing_items:
        issues.append(
            QualityGateIssue(
                code="vlm_audit_missing_items",
                severity="high",
                message="VLM 對照來源頁後指出最終語意文件可能有缺漏。",
                page_idx=audit.get("page_idx"),
                evidence={
                    "missing_items": missing_items[:10] if isinstance(missing_items, list) else missing_items,
                    "wrong_classification": output.get("wrong_classification"),
                    "confidence": output.get("confidence"),
                },
            )
        )
    return issues


def _audit_context(
    document_ir: DocumentIR,
    page_idx: int,
    source_md: str,
    structured_text: str,
    reasons: list[str],
) -> str:
    blocks = document_ir.get_blocks_by_page(page_idx)
    mineru_text = "\n".join(_plain_block_text(block) for block in blocks)[:7000]
    current_semantic = (structured_text or "").strip()[:9000]
    raw_source = (source_md or "").strip()[:3000]
    if not current_semantic:
        current_semantic = "(empty structured semantic output)"
    return (
        "請審核這一頁是否被正確轉成 RAG 語意文件。請以頁面圖像與 MinerU 解析文字為主要 evidence，"
        "只判斷 evidence 支援的缺漏，不要自行補內容。\n"
        f"風險原因：{', '.join(reasons)}\n\n"
        "[MinerU OCR / layout / table evidence]\n"
        f"{mineru_text}\n\n"
        "[Current structured semantic output to audit]\n"
        f"{current_semantic}\n\n"
        "[Raw source markdown fallback, lower priority]\n"
        f"{raw_source}\n"
    )

def _collect_notes_after_block(document_ir: DocumentIR, table_block: Block) -> list[str]:
    try:
        start_idx = next(idx for idx, block in enumerate(document_ir.blocks) if block.block_id == table_block.block_id)
    except StopIteration:
        return []
    lines = []
    started = False
    for block in document_ir.blocks[start_idx + 1:start_idx + 45]:
        if block.type == BlockType.TABLE:
            break
        if block.type != BlockType.TEXT:
            continue
        text = re.sub(r"\s+", " ", str(block.payload.get("text") or "")).strip()
        if not text or _is_noise_line(text):
            continue
        is_note_heading = bool(re.match(r"^(?:備註|備注|註[:：]|註\d+[:：])", text))
        is_note_item = bool(re.match(r"^(?:[一二三四五六七八九十]+、|\d+[.、])", text))
        if is_note_heading:
            started = True
        elif not started:
            continue
        elif re.match(r"^第[一二三四五六七八九十0-9]+條", text):
            break
        elif started and re.match(r"^[一二三四五六七八九十]+、", text) and re.search(r"申請|報支|核定|核銷|變更|填寫|作業|流程", text):
            break
        elif not is_note_item and len(_compact(text)) > 180:
            break
        if started:
            lines.append(text)
    notes = _merge_note_lines(lines)
    return [note for note in notes if not is_version_text(note)]


def _merge_note_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if re.match(r"^(?:備註|備注|註[:：]|註\d+[:：]|[一二三四五六七八九十]+、|\d+[.、])", line) or not merged:
            merged.append(line)
        else:
            merged[-1] += line
    return merged


def _is_noise_line(text: str) -> bool:
    compact = _compact(text)
    if compact.isdigit():
        return True
    if len(compact) <= 8 and not re.search(r"[。；;]", text):
        return True
    if not re.search(r"[。；;，,：:]", text) and len(compact) <= 40 and re.search(r"(?:辦法|規程|規章|要點|準則)$", compact):
        return True
    return False


def _plain_block_text(block: Block) -> str:
    if block.type == BlockType.TEXT:
        return str(block.payload.get("text") or "")
    if block.type == BlockType.TABLE:
        return re.sub(r"<[^>]+>", " ", str(block.payload.get("table_body") or ""))
    if block.type == BlockType.IMAGE:
        return " ".join(str(block.payload.get(key) or "") for key in ("caption", "footnote"))
    return block.get_text()


def _extract_signature_terms(text: str) -> list[str]:
    seen = []
    for line in str(text or "").splitlines():
        raw = line.strip()
        compact = _compact(raw)
        if not compact:
            continue
        signature_like_line = len(compact) <= 45 or bool(re.search(r"[:：]", raw))
        if not signature_like_line:
            continue
        if re.search(r"[。；;]", raw) and not re.search(r"[:：]", raw):
            continue
        for term in re.findall(r"申請人|簽名|簽章|核章|單位主管|主任秘書|副院長|院長|董事長", compact):
            explicit_label = bool(re.search(re.escape(term) + r"[:：]", compact))
            if len(compact) > 60 and not explicit_label:
                continue
            if term not in seen:
                seen.append(term)
    return seen


def _resolve_page_image(run_path: Path, page_image_path: str) -> Path | None:
    if not page_image_path:
        return None
    path = Path(page_image_path)
    candidates = [path, run_path / path]
    if str(page_image_path).startswith("assets/"):
        candidates.append(run_path / page_image_path)
    else:
        candidates.append(run_path / "assets" / page_image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _status_from_issues(issues: list[QualityGateIssue]) -> str:
    if any(issue.severity == "high" for issue in issues):
        return "needs_review"
    if any(issue.severity == "medium" for issue in issues):
        return "warning"
    return "pass"


def _score_from_issues(issues: list[QualityGateIssue]) -> float:
    penalty = 0.0
    for issue in issues:
        if issue.severity == "high":
            penalty += 0.25
        elif issue.severity == "medium":
            penalty += 0.12
        elif issue.severity == "warning":
            penalty += 0.05
    return max(0.0, round(1.0 - penalty, 3))


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")
