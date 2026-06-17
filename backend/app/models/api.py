"""
API request/response models.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.config import ProfileName
from app.models.entities import RunStatus, StageName, StageStatus


# Ingest
class IngestRequest(BaseModel):
    """Request to ingest a document."""

    path: str | None = Field(default=None, description="Local file or folder path")
    url: str | None = Field(default=None, description="URL to fetch (HTML)")


class IngestResponse(BaseModel):
    """Response from ingest."""

    doc_id: str
    source_path: str
    ext: str
    size_bytes: int
    already_exists: bool = False


# Runs
class RunCreateRequest(BaseModel):
    """Request to create a new run."""

    doc_id: str
    profile: ProfileName = ProfileName.ACCURATE
    config_overrides: dict[str, Any] | None = None
    use_cache: bool = True
    force_stages: list[StageName] | None = None


class RunResponse(BaseModel):
    """Run information response."""

    run_id: str
    doc_id: str
    profile: str
    status: RunStatus
    use_cache: bool
    created_at: datetime
    updated_at: datetime
    current_stage: StageName | None = None
    stage_progress: dict[str, Any] | None = None


class StageResponse(BaseModel):
    """Run stage information."""

    stage: StageName
    status: StageStatus
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None = None
    error: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None


class RunDetailResponse(RunResponse):
    """Detailed run response with stages."""

    config: dict[str, Any]
    stages: list[StageResponse]


class RunListResponse(BaseModel):
    """List of runs."""

    runs: list[RunResponse]
    total: int


# Documents
class DocResponse(BaseModel):
    """Document information response."""

    doc_id: str
    source_path: str
    ext: str
    size_bytes: int
    created_at: datetime
    run_count: int = 0


class DocListResponse(BaseModel):
    """List of documents."""

    docs: list[DocResponse]
    total: int


# Cache
class CacheInvalidateRequest(BaseModel):
    """Request to invalidate cache."""

    stages: list[StageName] | None = Field(
        default=None, description="Stages to invalidate, or all if not specified"
    )


class CacheInvalidateResponse(BaseModel):
    """Cache invalidation result."""

    invalidated_count: int


# Assets
class AssetInfo(BaseModel):
    """Asset information."""

    asset_id: str
    type: str
    doc_id: str
    run_id: str
    title: str | None
    page_idx: int
    asset_path: str
    triggers: list[str] = Field(default_factory=list)


class AssetsIndexResponse(BaseModel):
    """Assets index response."""

    assets: list[AssetInfo]
    total: int


# Quality
class QualityResponse(BaseModel):
    """Quality report response."""

    doc_id: str
    run_id: str
    blocks: dict[str, int]  # type -> count
    pages: list[dict[str, Any]]
    enrich_coverage: dict[str, Any]
    warnings: list[str]


# Health
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    database: str
