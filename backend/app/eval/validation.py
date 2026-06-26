"""Lightweight schema checks for benchmark artifacts."""

from __future__ import annotations

from typing import Any

REQUIRED_ASSET_FIELDS = {
    "type",
    "asset_id",
    "doc_id",
    "run_id",
    "title",
    "page_idx",
    "asset_path",
    "block_id",
    "retrieval_text",
    "needs_review",
}

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "doc_id",
    "run_id",
    "view",
    "content",
    "block_ids",
    "page_indices",
    "attachments",
    "metadata",
}


def validate_asset_entry(entry: dict[str, Any]) -> list[str]:
    """Return validation errors for one assets_index.jsonl entry."""
    errors: list[str] = []
    missing = sorted(REQUIRED_ASSET_FIELDS - set(entry))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    if entry.get("type") not in {"form_asset", "figure_asset", "table_asset"}:
        errors.append("type must be form_asset, figure_asset, or table_asset")

    if not isinstance(entry.get("asset_id"), str) or not entry.get("asset_id"):
        errors.append("asset_id must be a non-empty string")

    if not isinstance(entry.get("retrieval_text"), str) or not entry.get("retrieval_text", "").strip():
        errors.append("retrieval_text must be non-empty")

    if not isinstance(entry.get("page_idx"), int):
        errors.append("page_idx must be an integer")

    if "triggers" in entry and not isinstance(entry.get("triggers"), list):
        errors.append("triggers must be a list when present")

    if "field_schema" in entry and not isinstance(entry.get("field_schema"), list):
        errors.append("field_schema must be a list when present")

    return errors


def validate_chunk_entry(entry: dict[str, Any]) -> list[str]:
    """Return validation errors for one chunks.jsonl entry."""
    errors: list[str] = []
    missing = sorted(REQUIRED_CHUNK_FIELDS - set(entry))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    if not isinstance(entry.get("content"), str) or not entry.get("content", "").strip():
        errors.append("content must be non-empty")

    for key in ("block_ids", "page_indices", "attachments"):
        if key in entry and not isinstance(entry.get(key), list):
            errors.append(f"{key} must be a list")

    return errors


def validate_source_map(data: dict[str, Any]) -> list[str]:
    """Return validation errors for source_map.json."""
    errors: list[str] = []
    anchors = data.get("md_anchors")
    if not isinstance(anchors, list):
        return ["md_anchors must be a list"]

    for idx, anchor in enumerate(anchors):
        if not isinstance(anchor, dict):
            errors.append(f"anchor {idx} must be an object")
            continue

        if not anchor.get("anchor_id"):
            errors.append(f"anchor {idx} missing anchor_id")

        md_range = anchor.get("md_range")
        if (
            not isinstance(md_range, list)
            or len(md_range) != 2
            or not all(isinstance(v, int) for v in md_range)
            or md_range[0] > md_range[1]
        ):
            errors.append(f"anchor {idx} has invalid md_range")

        block_ids = anchor.get("block_ids")
        if not isinstance(block_ids, list) or not block_ids:
            errors.append(f"anchor {idx} must reference at least one block")

    return errors

