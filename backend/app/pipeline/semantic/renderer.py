"""Render semantic documents into compact RAG-oriented Markdown."""

from __future__ import annotations

from collections import defaultdict

from .schema import SemanticDocument, SemanticField


def render_semantic_markdown(document: SemanticDocument) -> str:
    if document.document_type == "form_document":
        return _render_form(document)
    if document.document_type == "reference_table":
        return _render_reference_table(document)
    return _render_generic(document)


def _render_form(document: SemanticDocument) -> str:
    lines = [f"# {document.title}", ""]
    lines.extend([
        f"來源檔案：{document.source.file_name}",
        f"來源頁面：{', '.join('第 ' + str(page + 1) + ' 頁' for page in document.source.pages)}" if document.source.pages else "",
        "文件類型：表單",
    ])
    if document.version.raw:
        lines.append(f"版本：{document.version.raw}")
    lines.append("")
    if document.purpose:
        lines.extend(["## 文件定位", document.purpose, ""])
    if document.usage_scenarios:
        lines.append("## 何時使用")
        lines.extend(f"- {item}" for item in document.usage_scenarios)
        lines.append("")
    grouped = _group_fields(document.fields)
    if grouped:
        lines.append("## 主要填寫內容")
        for section, fields in grouped.items():
            names = "、".join(field.normalized_name for field in fields[:12])
            if len(fields) > 12:
                names += f"，等 {len(fields)} 個欄位"
            lines.append(f"- {section}：{names}")
        lines.append("")
    if document.approval_flow:
        lines.extend(["## 簽核與送件流程", " → ".join(document.approval_flow), ""])
    if document.notes:
        lines.append("## 注意事項")
        lines.extend(f"- {note}" for note in document.notes)
        lines.append("")
    if document.fields:
        lines.append("## 表單欄位")
        for section, fields in grouped.items():
            items = [f"{field.normalized_name}({_requirement_label(field.requirement)}, {field.type})" for field in fields]
            lines.append(f"- {section}：{'、'.join(items)}。")
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def _render_reference_table(document: SemanticDocument) -> str:
    lines = [f"# {document.title}", "", f"來源檔案：{document.source.file_name}", "文件類型：表格", ""]
    if document.purpose:
        lines.extend(["## 文件定位", document.purpose, ""])
    if document.records:
        lines.append("## 表格資料")
        for record in document.records:
            parts = [f"{key}：{value}" for key, value in record.items() if value not in {None, ""}]
            lines.append("- " + "；".join(parts))
        lines.append("")
    if document.notes:
        lines.append("## 備註")
        lines.extend(f"- {note}" for note in document.notes)
    return "\n".join(lines).strip() + "\n"


def _render_generic(document: SemanticDocument) -> str:
    lines = [f"# {document.title}", "", f"來源檔案：{document.source.file_name}", f"文件類型：{document.document_type}"]
    if document.purpose:
        lines.extend(["", document.purpose])
    return "\n".join(lines).strip() + "\n"


def _group_fields(fields: list[SemanticField]) -> dict[str, list[SemanticField]]:
    grouped: dict[str, list[SemanticField]] = defaultdict(list)
    for field in fields:
        grouped[field.section or "表單欄位"].append(field)
    order = ["申請/基本資料", "出差/行程資訊", "交通工具", "費用/報支資訊", "進修/訓練資訊", "附件/佐證資料", "簽核/用印", "表單欄位"]
    ordered = {section: grouped[section] for section in order if section in grouped}
    ordered.update({section: values for section, values in grouped.items() if section not in order})
    return ordered


def _requirement_label(requirement: str) -> str:
    return {"required": "明確必填", "conditional": "條件填寫", "situational": "依情境填寫", "optional": "選填"}.get(requirement, "依情境填寫")
