"""Semantic output language helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

SemanticLanguage = Literal["zh-TW", "en"]
SemanticLanguageSelection = Literal["auto", "zh-TW", "en"]
VALID_SEMANTIC_OUTPUT_LANGUAGES = {"auto", "zh-TW", "en"}


@dataclass(frozen=True)
class FormLanguagePack:
    form_purpose: str
    use_cases: str
    form_structure: str
    filling_guidance: str
    conditional_fields: str
    approval_flow: str
    version_info: str
    notes: str
    rag_summary: str
    field_descriptions: str
    form_summary: str
    original_extraction: str
    filling_rules: str
    source_page_prefix: str


FORM_LANGUAGE_PACKS: dict[SemanticLanguage, FormLanguagePack] = {
    "zh-TW": FormLanguagePack(
        form_purpose="表單用途",
        use_cases="適用場景",
        form_structure="表單結構",
        filling_guidance="填寫重點",
        conditional_fields="條件欄位",
        approval_flow="簽核流程",
        version_info="版本資訊",
        notes="注意事項",
        rag_summary="RAG 查詢摘要",
        field_descriptions="表單欄位與欄位說明",
        form_summary="表單摘要",
        original_extraction="原始抽取補充",
        filling_rules="填寫規則",
        source_page_prefix="第",
    ),
    "en": FormLanguagePack(
        form_purpose="Form Purpose",
        use_cases="Use Cases",
        form_structure="Form Structure",
        filling_guidance="Filling Guidance",
        conditional_fields="Conditional Fields",
        approval_flow="Approval Flow",
        version_info="Version Information",
        notes="Notes",
        rag_summary="RAG Query Summary",
        field_descriptions="Form Fields and Field Descriptions",
        form_summary="Form Summary",
        original_extraction="Original Extraction Supplement",
        filling_rules="Filling Rules",
        source_page_prefix="Page",
    ),
}

SECTION_TRANSLATIONS = {
    "基本資料": "Basic Information",
    "填寫內容": "Entry Details",
    "簽核": "Approval",
    "申請/基本資料": "Applicant / Basic Information",
    "出差/行程資訊": "Travel / Itinerary",
    "費用/報支資訊": "Expense / Payment Information",
    "附件/佐證資料": "Attachments / Supporting Evidence",
    "簽核/用印": "Approval / Signature",
    "表單欄位": "Form Fields",
    "保證人/商號資料": "Guarantor / Business Information",
    "進修/訓練資訊": "Study / Training Information",
}

REQUIREMENT_LABELS = {
    "zh-TW": {
        "required": "明確必填",
        "conditional": "條件填寫",
        "situational": "依情境填寫",
        "optional": "選填",
        "default": "依情境填寫",
    },
    "en": {
        "required": "required",
        "conditional": "conditional",
        "situational": "situational",
        "optional": "optional",
        "default": "situational",
    },
}


def normalize_semantic_output_language(value: Any) -> SemanticLanguageSelection:
    text = str(value or "auto").strip()
    if text in {"zh", "zh-tw", "zh_TW", "zh_tw", "traditional_chinese"}:
        return "zh-TW"
    if text in {"english", "en-US", "en_us"}:
        return "en"
    if text in VALID_SEMANTIC_OUTPUT_LANGUAGES:
        return text  # type: ignore[return-value]
    return "auto"


def resolve_semantic_output_language(
    selection: Any = "auto",
    document_ir: Any | None = None,
    text: str | None = None,
) -> SemanticLanguage:
    normalized = normalize_semantic_output_language(selection)
    if normalized in {"zh-TW", "en"}:
        return normalized  # type: ignore[return-value]

    evidence = text or ""
    if document_ir is not None:
        evidence = "\n".join(part for part in [evidence, _document_text(document_ir)] if part)

    zh_chars = sum(1 for ch in evidence if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in evidence if ch.isascii() and ch.isalpha())
    source_path = str(getattr(getattr(document_ir, "source", None), "path", "") or "")
    if zh_chars >= 12 and zh_chars >= ascii_letters * 0.08:
        return "zh-TW"
    if zh_chars >= 4 and re.search(r"[\u4e00-\u9fff]", source_path):
        return "zh-TW"
    return "en"


def get_form_language_pack(language: Any) -> FormLanguagePack:
    normalized = resolve_semantic_output_language(language) if language == "auto" else normalize_semantic_output_language(language)
    if normalized == "auto":
        normalized = "zh-TW"
    return FORM_LANGUAGE_PACKS[normalized]  # type: ignore[index]


def display_form_section(section: str, language: Any) -> str:
    if normalize_semantic_output_language(language) == "en":
        return SECTION_TRANSLATIONS.get(section, section)
    return section


def requirement_label(requirement: str, language: Any) -> str:
    lang = normalize_semantic_output_language(language)
    if lang == "auto":
        lang = "zh-TW"
    labels = REQUIREMENT_LABELS[lang]  # type: ignore[index]
    return labels.get(requirement, labels["default"])


def form_template_sections(language: Any, *, include_field_descriptions: bool = False) -> list[str]:
    pack = get_form_language_pack(language)
    sections = [
        pack.form_purpose,
        pack.use_cases,
        pack.form_structure,
        pack.filling_guidance,
        pack.rag_summary,
    ]
    if include_field_descriptions:
        sections.insert(4, pack.field_descriptions)
    return sections


def prompt_language_instruction(language: Any) -> str:
    lang = normalize_semantic_output_language(language)
    if lang == "en":
        return "Write generated semantic descriptions, section headings, filling guides, captions, and summaries in English. Preserve source field names and quoted source text exactly when needed."
    if lang == "zh-TW":
        return "使用繁體中文撰寫語意描述、章節標題、填寫說明、caption 與摘要；必要時保留來源欄位名稱與原文。"
    return "Choose the output language from the source document: English documents use English semantic scaffolding; Traditional Chinese documents use Traditional Chinese semantic scaffolding. Preserve source field names and quoted source text when needed."


def prompt_form_sections(language: Any) -> str:
    pack = get_form_language_pack(language)
    return ", ".join(
        f"## {section}"
        for section in [
            pack.form_purpose,
            pack.use_cases,
            pack.form_structure,
            pack.filling_guidance,
            pack.approval_flow,
            pack.notes,
        ]
    )


def page_label(page_idx: int, language: Any) -> str:
    if normalize_semantic_output_language(language) == "en":
        return f"Page {page_idx + 1}"
    return f"第 {page_idx + 1} 頁"


def _document_text(document_ir: Any) -> str:
    parts = [str(getattr(getattr(document_ir, "source", None), "path", "") or "")]
    for block in getattr(document_ir, "blocks", []) or []:
        payload = getattr(block, "payload", {}) or {}
        if isinstance(payload, dict):
            for key in ("text", "table_body", "caption"):
                value = payload.get(key)
                if value:
                    parts.append(_strip_html(str(value)))
    return "\n".join(parts)


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)
