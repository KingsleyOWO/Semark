"""
Settings API routes.

Provides endpoints for viewing and updating VLM and pipeline configuration.
Settings are stored in the database and override environment defaults.
"""

import asyncio
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.adapters.mineru import MinerUAdapter
from app.config import (
    PROFILES,
    MinerUBackend,
    MinerUMethod,
    ProfileName,
    VLMApiMode,
    VLMImageMode,
    settings,
)
from app.db.database import Database, get_db
from app.db.repositories import SettingsRepository

router = APIRouter(prefix="/settings", tags=["settings"])


# Request/Response models
class VLMSettingsUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    api_mode: str | None = None
    image_mode: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    request_timeout_seconds: float | None = None


class ProfileOverrideUpdate(BaseModel):
    """Request model for updating profile-specific overrides."""

    # MinerU
    method: str | None = None  # auto, txt, ocr
    formula: bool | None = None

    # Enrich
    enable_vlm: bool | None = None
    vlm_enrich_forms: bool | None = None
    vlm_enrich_figures: bool | None = None
    vlm_enrich_tables: bool | None = None
    table_vlm_budget: int | None = None
    table_min_cells: int | None = None
    table_max_cells: int | None = None

    # Package
    chunk_max_tokens: int | None = None
    chunk_overlap_tokens: int | None = None


class VLMSettingsResponse(BaseModel):
    base_url: str
    api_key: str  # Masked
    model: str
    api_mode: str
    image_mode: str
    decode_params: dict[str, Any]
    request_timeout_seconds: float
    available_modes: list[str]
    available_image_modes: list[str]


# Helper to get merged VLM settings (DB overrides env)
async def get_vlm_config(db: Database) -> dict[str, Any]:
    """Get enrichment VLM config merged from env defaults and DB overrides."""
    return await _get_vlm_config_from_settings_key(
        db,
        settings_key="vlm",
        defaults={
            "base_url": settings.vlm_base_url,
            "api_key": settings.vlm_api_key,
            "model": settings.vlm_model,
        },
    )


async def get_review_vlm_config(db: Database) -> dict[str, Any]:
    """Get reviewer VLM config. Defaults to the enrichment VLM unless explicitly configured."""
    enrich_config = await get_vlm_config(db)
    return await _get_vlm_config_from_settings_key(
        db,
        settings_key="review_vlm",
        defaults={
            "base_url": settings.review_vlm_base_url or enrich_config["base_url"],
            "api_key": settings.review_vlm_api_key or enrich_config["api_key"],
            "model": settings.review_vlm_model or enrich_config["model"],
            "api_mode": enrich_config["api_mode"],
            "image_mode": enrich_config["image_mode"],
            "request_timeout_seconds": enrich_config["request_timeout_seconds"],
            "decode_params": enrich_config["decode_params"],
        },
    )


async def _get_vlm_config_from_settings_key(
    db: Database,
    *,
    settings_key: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Get VLM config merged from defaults and one DB settings key."""
    repo = SettingsRepository(db)
    db_settings = await repo.get(settings_key) or {}

    default_vlm = PROFILES[ProfileName.ACCURATE].vlm
    default_decode = defaults.get("decode_params") or {}

    return {
        "base_url": db_settings.get("base_url", defaults.get("base_url", settings.vlm_base_url)),
        "api_key": db_settings.get("api_key", defaults.get("api_key", settings.vlm_api_key)),
        "model": db_settings.get("model", defaults.get("model", settings.vlm_model)),
        "api_mode": db_settings.get("api_mode", defaults.get("api_mode", default_vlm.api_mode.value)),
        "image_mode": db_settings.get(
            "image_mode", defaults.get("image_mode", default_vlm.image_mode.value)
        ),
        "request_timeout_seconds": db_settings.get(
            "request_timeout_seconds",
            defaults.get("request_timeout_seconds", default_vlm.request_timeout_seconds),
        ),
        "decode_params": {
            "temperature": db_settings.get(
                "temperature", default_decode.get("temperature", default_vlm.decode_params.temperature)
            ),
            "top_p": db_settings.get("top_p", default_decode.get("top_p", default_vlm.decode_params.top_p)),
            "top_k": db_settings.get("top_k", default_decode.get("top_k", default_vlm.decode_params.top_k)),
            "max_tokens": db_settings.get(
                "max_tokens", default_decode.get("max_tokens", default_vlm.decode_params.max_tokens)
            ),
            "repetition_penalty": db_settings.get(
                "repetition_penalty",
                default_decode.get("repetition_penalty", default_vlm.decode_params.repetition_penalty),
            ),
        },
    }


def _mask_vlm_settings_response(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_url": config["base_url"],
        "api_key": "***" if config["api_key"] else "",
        "model": config["model"],
        "api_mode": config["api_mode"],
        "image_mode": config["image_mode"],
        "decode_params": config["decode_params"],
        "request_timeout_seconds": config["request_timeout_seconds"],
        "available_modes": [m.value for m in VLMApiMode],
        "available_image_modes": [m.value for m in VLMImageMode],
    }


@router.get("/vlm")
async def get_vlm_settings(
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Get current enrichment VLM settings (merged from env defaults and DB overrides).
    """
    config = await get_vlm_config(db)
    return _mask_vlm_settings_response(config)


@router.get("/review-vlm")
async def get_review_vlm_settings(
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Get current reviewer VLM settings."""
    config = await get_review_vlm_config(db)
    return _mask_vlm_settings_response(config)


async def _update_vlm_settings_key(
    settings_key: str,
    update: VLMSettingsUpdate,
    db: Database,
) -> dict[str, Any]:
    repo = SettingsRepository(db)
    current = await repo.get(settings_key) or {}

    if update.base_url is not None:
        current["base_url"] = update.base_url
    if update.api_key is not None:
        current["api_key"] = update.api_key
    if update.model is not None:
        current["model"] = update.model
    if update.api_mode is not None:
        if update.api_mode not in [m.value for m in VLMApiMode]:
            raise HTTPException(400, f"Invalid api_mode: {update.api_mode}")
        current["api_mode"] = update.api_mode
    if update.image_mode is not None:
        if update.image_mode not in [m.value for m in VLMImageMode]:
            raise HTTPException(400, f"Invalid image_mode: {update.image_mode}")
        current["image_mode"] = update.image_mode
    if update.temperature is not None:
        current["temperature"] = update.temperature
    if update.top_p is not None:
        current["top_p"] = update.top_p
    if update.top_k is not None:
        current["top_k"] = update.top_k
    if update.max_tokens is not None:
        current["max_tokens"] = update.max_tokens
    if update.repetition_penalty is not None:
        if not 1.0 <= update.repetition_penalty <= 2.0:
            raise HTTPException(400, "repetition_penalty must be between 1.0 and 2.0")
        current["repetition_penalty"] = update.repetition_penalty
    if update.request_timeout_seconds is not None:
        current["request_timeout_seconds"] = update.request_timeout_seconds

    await repo.set(settings_key, current)
    return current


@router.put("/vlm")
async def update_vlm_settings(
    update: VLMSettingsUpdate,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Update enrichment VLM settings. Only provided fields will be updated.
    """
    current = await _update_vlm_settings_key("vlm", update, db)
    return {"message": "VLM settings updated", "settings": current}


@router.put("/review-vlm")
async def update_review_vlm_settings(
    update: VLMSettingsUpdate,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Update reviewer VLM settings. Only provided fields will be updated."""
    current = await _update_vlm_settings_key("review_vlm", update, db)
    return {"message": "Reviewer VLM settings updated", "settings": current}


@router.delete("/vlm")
async def reset_vlm_settings(
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """
    Reset enrichment VLM settings to defaults (remove DB overrides).
    """
    repo = SettingsRepository(db)
    await repo.delete("vlm")
    return {"message": "VLM settings reset to defaults"}


@router.delete("/review-vlm")
async def reset_review_vlm_settings(
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """Reset reviewer VLM settings to defaults."""
    repo = SettingsRepository(db)
    await repo.delete("review_vlm")
    return {"message": "Reviewer VLM settings reset to defaults"}


# ==================== MinerU Settings ====================


class MinerUSettingsUpdate(BaseModel):
    """Request model for updating MinerU settings."""

    method: str | None = None
    backend: str | None = None
    lang: str | None = None
    table: bool | None = None
    formula: bool | None = None
    api_url: str | None = None
    vlm_url: str | None = None
    vlm_model_name: str | None = None
    vlm_api_key: str | None = None
    model_source: str | None = None
    pdf_render_timeout: int | None = None
    pdf_render_threads: int | None = None
    table_merge_enable: bool | None = None
    processing_window_size: int | None = None
    api_max_concurrent_requests: int | None = None


# Helper to get merged MinerU settings
async def get_mineru_config(db: Database) -> dict[str, Any]:
    """Get MinerU config merged from env defaults and DB overrides."""
    repo = SettingsRepository(db)
    db_settings = await repo.get("mineru") or {}

    # Use the quality profile as the global MinerU defaults.
    default_mineru = PROFILES[ProfileName.ACCURATE].mineru
    backend = db_settings.get("backend", default_mineru.backend.value)
    if backend == "vlm":
        backend = MinerUBackend.VLM_AUTO_ENGINE.value

    return {
        "method": db_settings.get("method", default_mineru.method.value),
        "backend": backend,
        "lang": db_settings.get("lang", default_mineru.lang),
        "table": db_settings.get("table", default_mineru.table),
        "formula": db_settings.get("formula", default_mineru.formula),
        "api_url": db_settings.get("api_url", settings.mineru_api_url),
        "vlm_url": db_settings.get("vlm_url", settings.mineru_vlm_url),
        "vlm_model_name": db_settings.get("vlm_model_name", settings.mineru_vlm_model_name),
        "model_source": db_settings.get("model_source", settings.mineru_model_source),
        "pdf_render_timeout": db_settings.get(
            "pdf_render_timeout", default_mineru.pdf_render_timeout
        ),
        "pdf_render_threads": db_settings.get(
            "pdf_render_threads", default_mineru.pdf_render_threads
        ),
        "table_merge_enable": db_settings.get(
            "table_merge_enable", default_mineru.table_merge_enable
        ),
        "processing_window_size": db_settings.get(
            "processing_window_size", default_mineru.processing_window_size
        ),
        "api_max_concurrent_requests": db_settings.get(
            "api_max_concurrent_requests", default_mineru.api_max_concurrent_requests
        ),
    }


@router.get("/mineru")
async def get_mineru_settings(
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Get current MinerU settings (merged from env defaults and DB overrides).

    Returns:
        - method: auto | txt | ocr
        - backend: pipeline | vlm
        - lang: OCR language (e.g., chinese_cht)
        - table: Enable table parsing
        - formula: Enable formula parsing
        - api_url: Existing mineru-api/router URL
        - vlm_url: OpenAI-compatible URL for MinerU *-http-client backends
        - model_source: huggingface | modelscope | local
        - pdf_render_timeout: Timeout for PDF rendering
        - pdf_render_threads: PDF render concurrency
        - table_merge_enable: Enable table cell merging
        - available_methods: List of valid method values
        - available_backends: List of valid backend values
    """
    config = await get_mineru_config(db)

    return {
        **config,
        "available_methods": [m.value for m in MinerUMethod],
        "available_backends": [b.value for b in MinerUBackend],
    }


@router.put("/mineru")
async def update_mineru_settings(
    update: MinerUSettingsUpdate,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Update MinerU settings. Only provided fields will be updated.
    """
    repo = SettingsRepository(db)

    # Get current settings
    current = await repo.get("mineru") or {}

    # Update only provided fields with validation
    if update.method is not None:
        if update.method not in [m.value for m in MinerUMethod]:
            raise HTTPException(400, f"Invalid method: {update.method}")
        current["method"] = update.method

    if update.backend is not None:
        backend = "vlm-auto-engine" if update.backend == "vlm" else update.backend
        if backend not in [b.value for b in MinerUBackend]:
            raise HTTPException(400, f"Invalid backend: {update.backend}")
        current["backend"] = backend

    if update.lang is not None:
        current["lang"] = update.lang

    if update.table is not None:
        current["table"] = update.table

    if update.formula is not None:
        current["formula"] = update.formula

    if update.api_url is not None:
        current["api_url"] = update.api_url

    if update.vlm_url is not None:
        current["vlm_url"] = update.vlm_url

    if update.vlm_model_name is not None:
        current["vlm_model_name"] = update.vlm_model_name

    if update.vlm_api_key is not None:
        current["vlm_api_key"] = update.vlm_api_key

    if update.model_source is not None:
        if update.model_source not in {"huggingface", "modelscope", "local"}:
            raise HTTPException(400, "model_source must be huggingface, modelscope, or local")
        current["model_source"] = update.model_source

    if update.pdf_render_timeout is not None:
        current["pdf_render_timeout"] = update.pdf_render_timeout

    if update.pdf_render_threads is not None:
        current["pdf_render_threads"] = update.pdf_render_threads

    if update.table_merge_enable is not None:
        current["table_merge_enable"] = update.table_merge_enable

    if update.processing_window_size is not None:
        current["processing_window_size"] = update.processing_window_size

    if update.api_max_concurrent_requests is not None:
        current["api_max_concurrent_requests"] = update.api_max_concurrent_requests

    # Save
    await repo.set("mineru", current)

    return {"message": "MinerU settings updated", "settings": current}


@router.delete("/mineru")
async def reset_mineru_settings(
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """
    Reset MinerU settings to defaults (remove DB overrides).
    """
    repo = SettingsRepository(db)
    await repo.delete("mineru")
    return {"message": "MinerU settings reset to defaults"}


@router.get("/mineru/probe")
async def probe_mineru(
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Probe MinerU CLI and the configured long-running MinerU API URL.
    """
    adapter = MinerUAdapter()
    available, version_or_error = await adapter.check_available()
    mineru_config = await get_mineru_config(db)
    api_url = mineru_config.get("api_url")
    api_probe = await _probe_tcp_url(api_url) if api_url else {
        "configured": False,
        "available": False,
        "url": api_url,
        "error": None,
    }

    return {
        "available": available,
        "version": version_or_error if available else None,
        "error": None if available else version_or_error,
        "cli_path": settings.mineru_cli_path,
        "api_url": api_url,
        "api_probe": api_probe,
        "fallback_enabled": True,
    }


async def _probe_tcp_url(url: str | None) -> dict[str, Any]:
    if not url:
        return {"configured": False, "available": False, "url": url, "error": None}

    parsed = urlparse(url)
    if not parsed.hostname:
        return {
            "configured": True,
            "available": False,
            "url": url,
            "error": "Invalid URL",
        }

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(parsed.hostname, port),
            timeout=1.0,
        )
        writer.close()
        await writer.wait_closed()
        return {
            "configured": True,
            "available": True,
            "url": url,
            "host": parsed.hostname,
            "port": port,
            "error": None,
        }
    except Exception as exc:
        return {
            "configured": True,
            "available": False,
            "url": url,
            "host": parsed.hostname,
            "port": port,
            "error": str(exc) or exc.__class__.__name__,
        }


@router.get("/profiles")
async def get_profiles() -> dict[str, Any]:
    """
    Get available pipeline profiles and their configurations.
    """
    profiles = {}
    for name, config in PROFILES.items():
        profiles[name.value] = {
            "mineru": {
                "method": config.mineru.method.value,
                "backend": config.mineru.backend.value,
                "lang": config.mineru.lang,
                "table": config.mineru.table,
                "formula": config.mineru.formula,
                "model_source": config.mineru.model_source or settings.mineru_model_source,
            },
            "enrich": {
                "enable_vlm": config.enrich.enable_vlm,
                "vlm_enrich_forms": config.enrich.vlm_enrich_forms,
                "vlm_enrich_figures": config.enrich.vlm_enrich_figures,
                "vlm_enrich_tables": config.enrich.vlm_enrich_tables,
                "table_vlm_budget": config.enrich.table_vlm_budget,
                "table_min_cells": config.enrich.table_min_cells,
                "table_max_cells": config.enrich.table_max_cells,
                "table_truncate_head_rows": config.enrich.table_truncate_head_rows,
                "table_truncate_tail_rows": config.enrich.table_truncate_tail_rows,
                "table_layout_min_ratio": config.enrich.table_layout_min_ratio,
            },
            "package": {
                "generate_dataset_md": config.package.generate_dataset_md,
                "generate_rag_md": config.package.generate_rag_md,
                "chunk_max_tokens": config.package.chunk_max_tokens,
                "chunk_overlap_tokens": config.package.chunk_overlap_tokens,
            },
        }

    return {
        "profiles": profiles,
        "default": settings.default_profile.value,
    }


@router.get("/system")
async def get_system_settings() -> dict[str, Any]:
    """
    Get system settings and paths.
    """
    return {
        "workspace_path": str(settings.workspace_path),
        "database_path": str(settings.database_path),
        "mineru_cli_path": settings.mineru_cli_path,
        "host": settings.host,
        "port": settings.port,
        "debug": settings.debug,
    }


@router.get("/vlm/probe")
async def probe_vlm(
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Probe enrichment VLM endpoint for availability and capabilities.
    Uses the current (merged) VLM settings.
    """
    config = await get_vlm_config(db)
    return await _probe_vlm_config(config)


@router.get("/review-vlm/probe")
async def probe_review_vlm(
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Probe reviewer VLM endpoint for availability and capabilities."""
    config = await get_review_vlm_config(db)
    return await _probe_vlm_config(config)


async def _probe_vlm_config(config: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from app.adapters.vlm import VLMAdapter
    from app.config import VLMConfig, VLMDecodeParams

    try:
        vlm_config = VLMConfig(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            api_mode=VLMApiMode(config["api_mode"]),
            image_mode=VLMImageMode(config["image_mode"]),
            request_timeout_seconds=config["request_timeout_seconds"],
            decode_params=VLMDecodeParams(
                temperature=config["decode_params"]["temperature"],
                top_p=config["decode_params"]["top_p"],
                max_tokens=config["decode_params"]["max_tokens"],
            ),
        )
    except ValidationError as e:
        raise HTTPException(400, f"Invalid VLM settings: {e}")

    try:
        adapter = VLMAdapter(vlm_config)
        result = await adapter.probe_capabilities(force=True)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(500, f"VLM probe failed: {e}")


# ==================== Profile Overrides ====================

# Profile descriptions in Chinese
PROFILE_DESCRIPTIONS = {
    "fast": {
        "name": "FAST (快速模式)",
        "description": "最快的處理速度，跳過所有 VLM 增強。適合快速預覽或文字為主的文件。",
        "features": ["不啟用 VLM", "不解析公式", "自動選擇解析方法"],
    },
    "accurate": {
        "name": "ACCURATE (精確模式)",
        "description": "最高輸出質量，使用 OCR 和完整 VLM 增強。適合需要高精度的正式文件。",
        "features": ["強制 OCR", "完整 VLM 增強（包含表格）", "更大的表格處理預算"],
    },
}


async def get_profile_with_overrides(
    profile_name: str, db: Database
) -> dict[str, Any]:
    """Get profile config merged with user overrides."""
    # Validate profile name
    try:
        profile_enum = ProfileName(profile_name)
    except ValueError:
        raise HTTPException(404, f"Profile not found: {profile_name}")

    base_config = PROFILES[profile_enum]
    repo = SettingsRepository(db)

    # Get user overrides
    overrides = await repo.get(f"profile_override:{profile_name}") or {}

    # Build merged config
    return {
        "name": profile_name,
        "description": PROFILE_DESCRIPTIONS.get(profile_name, {}),
        "is_default": profile_name == settings.default_profile.value,
        "has_overrides": bool(overrides),
        "config": {
            "mineru": {
                "method": overrides.get("method", base_config.mineru.method.value),
                "backend": base_config.mineru.backend.value,
                "lang": base_config.mineru.lang,
                "table": base_config.mineru.table,
                "formula": overrides.get("formula", base_config.mineru.formula),
                "model_source": base_config.mineru.model_source or settings.mineru_model_source,
            },
            "enrich": {
                "enable_vlm": overrides.get("enable_vlm", base_config.enrich.enable_vlm),
                "vlm_enrich_forms": overrides.get(
                    "vlm_enrich_forms", base_config.enrich.vlm_enrich_forms
                ),
                "vlm_enrich_figures": overrides.get(
                    "vlm_enrich_figures", base_config.enrich.vlm_enrich_figures
                ),
                "vlm_enrich_tables": overrides.get(
                    "vlm_enrich_tables", base_config.enrich.vlm_enrich_tables
                ),
                "table_vlm_budget": overrides.get(
                    "table_vlm_budget", base_config.enrich.table_vlm_budget
                ),
                "table_min_cells": overrides.get(
                    "table_min_cells", base_config.enrich.table_min_cells
                ),
                "table_max_cells": overrides.get(
                    "table_max_cells", base_config.enrich.table_max_cells
                ),
            },
            "package": {
                "generate_dataset_md": base_config.package.generate_dataset_md,
                "generate_rag_md": base_config.package.generate_rag_md,
                "chunk_max_tokens": overrides.get(
                    "chunk_max_tokens", base_config.package.chunk_max_tokens
                ),
                "chunk_overlap_tokens": overrides.get(
                    "chunk_overlap_tokens", base_config.package.chunk_overlap_tokens
                ),
            },
        },
        "overrides": overrides,
    }


@router.get("/profiles/{profile_name}")
async def get_profile(
    profile_name: str,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Get a specific profile with user overrides merged.

    Returns profile configuration with:
    - Base profile settings
    - User overrides (if any)
    - Profile description and features
    """
    return await get_profile_with_overrides(profile_name, db)


@router.put("/profiles/{profile_name}")
async def update_profile_overrides(
    profile_name: str,
    update: ProfileOverrideUpdate,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Update profile-specific overrides. Only provided fields will be updated.

    These overrides are applied when creating runs with this profile.
    """
    # Validate profile name
    try:
        ProfileName(profile_name)
    except ValueError:
        raise HTTPException(404, f"Profile not found: {profile_name}")

    repo = SettingsRepository(db)
    key = f"profile_override:{profile_name}"

    # Get current overrides
    current = await repo.get(key) or {}

    # Update only provided fields with validation
    if update.method is not None:
        if update.method not in [m.value for m in MinerUMethod]:
            raise HTTPException(400, f"Invalid method: {update.method}")
        current["method"] = update.method

    if update.formula is not None:
        current["formula"] = update.formula

    if update.enable_vlm is not None:
        current["enable_vlm"] = update.enable_vlm

    if update.vlm_enrich_forms is not None:
        current["vlm_enrich_forms"] = update.vlm_enrich_forms

    if update.vlm_enrich_figures is not None:
        current["vlm_enrich_figures"] = update.vlm_enrich_figures

    if update.vlm_enrich_tables is not None:
        current["vlm_enrich_tables"] = update.vlm_enrich_tables

    if update.table_vlm_budget is not None:
        if update.table_vlm_budget < 0:
            raise HTTPException(400, "table_vlm_budget must be >= 0")
        current["table_vlm_budget"] = update.table_vlm_budget

    if update.table_min_cells is not None:
        if update.table_min_cells < 1:
            raise HTTPException(400, "table_min_cells must be >= 1")
        current["table_min_cells"] = update.table_min_cells

    if update.table_max_cells is not None:
        if update.table_max_cells < 1:
            raise HTTPException(400, "table_max_cells must be >= 1")
        current["table_max_cells"] = update.table_max_cells

    if update.chunk_max_tokens is not None:
        if not 64 <= update.chunk_max_tokens <= 8192:
            raise HTTPException(400, "chunk_max_tokens must be between 64 and 8192")
        current["chunk_max_tokens"] = update.chunk_max_tokens

    if update.chunk_overlap_tokens is not None:
        if not 0 <= update.chunk_overlap_tokens <= 1024:
            raise HTTPException(400, "chunk_overlap_tokens must be between 0 and 1024")
        current["chunk_overlap_tokens"] = update.chunk_overlap_tokens

    # Save
    await repo.set(key, current)

    return {
        "message": f"Profile {profile_name} overrides updated",
        "profile": profile_name,
        "overrides": current,
    }


@router.delete("/profiles/{profile_name}")
async def reset_profile_overrides(
    profile_name: str,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """
    Reset profile overrides to defaults (remove all user customizations).
    """
    # Validate profile name
    try:
        ProfileName(profile_name)
    except ValueError:
        raise HTTPException(404, f"Profile not found: {profile_name}")

    repo = SettingsRepository(db)
    key = f"profile_override:{profile_name}"
    await repo.delete(key)

    return {"message": f"Profile {profile_name} overrides reset to defaults"}
