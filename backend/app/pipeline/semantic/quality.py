"""Semantic quality scoring for RAG readiness."""

from __future__ import annotations

import re

from .normalizer import is_version_text, split_merged_field_label
from .schema import SemanticDocument, SemanticIssue, SemanticQualityReport


def evaluate_semantic_quality(document: SemanticDocument, rendered_text: str = "") -> SemanticQualityReport:
    issues: list[SemanticIssue] = []
    repairs: list[str] = []

    if not document.title or re.search(r"[昇鑑〇○]", document.title):
        issues.append(SemanticIssue("ocr_title_noise", "warning", "標題疑似仍有 OCR 雜訊。", {"title": document.title}))
        repairs.append("clean_title_noise")
    version_notes = [note for note in document.notes if is_version_text(note)]
    if version_notes:
        issues.append(SemanticIssue("version_misclassified_as_note", "warning", "版本資訊不應放在注意事項。", {"notes": version_notes}))
        repairs.append("move_version_to_metadata")
    long_fields = [field.name for field in document.fields if len(field.name) > 24]
    if long_fields:
        issues.append(SemanticIssue("field_name_too_long", "warning", "部分欄位名稱過長，可能是欄位黏連。", {"fields": long_fields[:8]}))
        repairs.append("split_merged_fields")
    merged = [field.name for field in document.fields if len(split_merged_field_label(field.name)) > 1]
    if merged:
        issues.append(SemanticIssue("merged_field_detected", "warning", "部分欄位疑似多個欄位黏在一起。", {"fields": merged[:8]}))
        repairs.append("split_merged_fields")
    if document.document_type == "form_document":
        if not document.usage_scenarios and not document.purpose:
            issues.append(SemanticIssue("missing_usage_scenario", "warning", "表單缺少明確用途或使用情境。"))
        if not document.approval_flow and any(re.search(r"簽|章|核|主管|院長", field.name) for field in document.fields):
            issues.append(SemanticIssue("missing_approval_flow", "warning", "表單有簽核欄位但缺少簽核流程。"))
        if sum(1 for field in document.fields if field.section == "表單欄位") > max(5, len(document.fields) * 0.35):
            issues.append(SemanticIssue("too_many_generic_fields", "warning", "太多欄位未被分入具體區塊。"))
            repairs.append("classify_fields")
    if rendered_text and "..." in rendered_text:
        issues.append(SemanticIssue("summary_contains_ellipsis", "warning", "輸出摘要包含截斷符號，可能遺失語意。"))
        repairs.append("compress_summary_without_ellipsis")

    correctness_penalty = sum(0.12 for issue in issues if issue.severity in {"high", "medium"}) + sum(0.04 for issue in issues if issue.severity == "warning")
    readiness_penalty = sum(0.08 for issue in issues if issue.code in {"field_name_too_long", "merged_field_detected", "too_many_generic_fields", "summary_contains_ellipsis"})
    return SemanticQualityReport(
        correctness_score=max(0.0, round(1.0 - correctness_penalty, 3)),
        rag_readiness_score=max(0.0, round(1.0 - readiness_penalty, 3)),
        issues=issues,
        recommended_repairs=sorted(set(repairs)),
    )
