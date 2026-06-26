"""
Database entity models (Pydantic schemas).
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class StageName(StrEnum):
    INGEST = "ingest"
    PARSE = "parse"
    NORMALIZE = "normalize"
    ENRICH = "enrich"
    PACKAGE = "package"
    CHUNK = "chunk"


# Document models
class DocCreate(BaseModel):
    """Input for creating a document."""

    source_path: str
    sha256: str
    ext: str
    size_bytes: int
    meta: dict[str, Any] | None = None


class Doc(BaseModel):
    """Document entity."""

    doc_id: str
    source_path: str
    sha256: str
    ext: str
    size_bytes: int
    created_at: datetime
    meta: dict[str, Any] | None = None


# Run models
class RunCreate(BaseModel):
    """Input for creating a run."""

    doc_id: str
    profile: str
    config: dict[str, Any]
    config_hash: str
    use_cache: bool = True
    force_stages: list[StageName] | None = None


class Run(BaseModel):
    """Run entity."""

    run_id: str
    doc_id: str
    profile: str
    config: dict[str, Any]
    config_hash: str
    status: RunStatus
    use_cache: bool
    force_stages: list[StageName] | None
    created_at: datetime
    updated_at: datetime


class RunWithStages(Run):
    """Run with its stages."""

    stages: list["RunStage"] = Field(default_factory=list)


# Run stage models
class RunStageCreate(BaseModel):
    """Input for creating a run stage."""

    run_id: str
    stage: StageName


class RunStage(BaseModel):
    """Run stage entity."""

    id: int
    run_id: str
    stage: StageName
    status: StageStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None


# Cache models
class CacheEntry(BaseModel):
    """Cache entry entity."""

    cache_key: str
    doc_id: str
    stage: StageName
    config_hash: str
    path: str
    created_at: datetime


class CacheEntryCreate(BaseModel):
    """Input for creating a cache entry."""

    cache_key: str
    doc_id: str
    stage: StageName
    config_hash: str
    path: str


# Enrich models
class EnrichEntry(BaseModel):
    """Enrichment cache entry."""

    id: int
    doc_id: str
    block_id: str
    vlm_config_hash: str
    prompt_version: str
    output: dict[str, Any]
    created_at: datetime


class EnrichEntryCreate(BaseModel):
    """Input for creating an enrichment entry."""

    doc_id: str
    block_id: str
    vlm_config_hash: str
    prompt_version: str
    output: dict[str, Any]
