"""A verified interpretation has one delivery surface; the original IR is evidence."""
from app.models.document_ir import BlockType


def replacement_text(block) -> str:
    evidence = block.payload.get("agentic_evidence") or {}
    if (evidence.get("tool_version") != "visual-repair-v2"
            or evidence.get("status") != "complete"):
        return ""
    semantics = str(block.payload.get("verified_semantics") or "").strip()
    if not semantics:
        return ""
    caption = block.payload.get("table_caption") or block.payload.get("caption") or ""
    if isinstance(caption, list):
        caption = " ".join(map(str, caption))
    notes = block.payload.get("table_footnote") or block.payload.get("footnote") or []
    if isinstance(notes, str):
        notes = [notes]
    return "\n\n".join(str(v) for v in [caption, semantics, *notes] if v)


def delivery_view(ir):
    """Render accepted blocks as prose without re-extracting the superseded table.

    This is a copy. Source table HTML, images, and the accepted cell mapping
    remain unchanged in document_ir.json and dataset.md for audit.
    """
    result = ir.model_copy(deep=True)
    for block in result.blocks:
        text = replacement_text(block)
        if text:
            block.type = BlockType.TEXT
            block.payload = {"text": text, "text_level": 0, "agentic_delivery": True}
    return result


def filter_recovery_enrichments(ir, enrichments):
    """Candidate and committed packaging must discard the same superseded enrichments."""
    recovered = [b for b in ir.blocks if b.payload.get("agentic_recovery_tool")]
    if not recovered:
        return enrichments
    valid = {b.block_id for b in ir.blocks}
    pages = {p for b in recovered for p in b.payload.get("agentic_source_pages", [])}
    return {
        key: value for key, value in enrichments.items()
        if not (key.startswith(("b", "recovery_")) and key not in valid)
        and not (ir.has_physical_pages() and value.get("kind") in {"form_asset", "form_guide"}
                 and value.get("input", {}).get("page_idx") in pages)
    }
