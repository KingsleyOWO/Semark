"""
Runs API routes.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.api.routes.settings import get_mineru_config, get_review_vlm_config, get_vlm_config
from app.config import PROFILES, PipelineConfig, settings
from app.core.task_queue import get_task_queue
from app.db.database import Database, get_db
from app.db.repositories import DocRepository, RunRepository, RunStageRepository, SettingsRepository
from app.models.api import (
    RunCreateRequest,
    RunDetailResponse,
    RunListResponse,
    RunResponse,
    StageResponse,
)
from app.models.entities import RunCreate, RunStatus, StageName, StageStatus

router = APIRouter(prefix="/runs", tags=["runs"])


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute hash of configuration for cache key."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def merge_config(base: PipelineConfig | dict, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge base config with overrides."""
    base_dict = base.model_dump() if hasattr(base, "model_dump") else base
    if not overrides:
        return base_dict

    def deep_merge(d1: dict, d2: dict) -> dict:
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(base_dict, overrides)


def _convert_profile_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert flat profile overrides to nested config structure."""
    if not overrides:
        return {}

    result: dict[str, Any] = {"mineru": {}, "enrich": {}, "package": {}}

    # MinerU overrides
    if "method" in overrides:
        result["mineru"]["method"] = overrides["method"]
    if "formula" in overrides:
        result["mineru"]["formula"] = overrides["formula"]

    # Enrich overrides
    if "enable_vlm" in overrides:
        result["enrich"]["enable_vlm"] = overrides["enable_vlm"]
    if "vlm_enrich_forms" in overrides:
        result["enrich"]["vlm_enrich_forms"] = overrides["vlm_enrich_forms"]
    if "vlm_enrich_figures" in overrides:
        result["enrich"]["vlm_enrich_figures"] = overrides["vlm_enrich_figures"]
    if "vlm_enrich_tables" in overrides:
        result["enrich"]["vlm_enrich_tables"] = overrides["vlm_enrich_tables"]
    if "table_vlm_budget" in overrides:
        result["enrich"]["table_vlm_budget"] = overrides["table_vlm_budget"]
    if "table_min_cells" in overrides:
        result["enrich"]["table_min_cells"] = overrides["table_min_cells"]
    if "table_max_cells" in overrides:
        result["enrich"]["table_max_cells"] = overrides["table_max_cells"]

    # Package overrides
    if "chunk_max_tokens" in overrides:
        result["package"]["chunk_max_tokens"] = overrides["chunk_max_tokens"]
    if "chunk_overlap_tokens" in overrides:
        result["package"]["chunk_overlap_tokens"] = overrides["chunk_overlap_tokens"]
    if "semantic_output_language" in overrides:
        result["package"]["semantic_output_language"] = overrides["semantic_output_language"]

    # Remove empty sections
    return {k: v for k, v in result.items() if v}


def _apply_runtime_settings(
    merged_config: dict[str, Any],
    mineru_settings: dict[str, Any],
    vlm_settings: dict[str, Any],
    review_vlm_settings: dict[str, Any],
) -> None:
    """Apply DB/env settings that are shared across profiles."""
    merged_config["mineru"]["backend"] = mineru_settings["backend"]
    merged_config["mineru"]["lang"] = mineru_settings["lang"]
    merged_config["mineru"]["table"] = mineru_settings["table"]
    merged_config["mineru"]["api_url"] = mineru_settings["api_url"]
    merged_config["mineru"]["vlm_url"] = mineru_settings["vlm_url"]
    merged_config["mineru"]["vlm_model_name"] = mineru_settings["vlm_model_name"]
    merged_config["mineru"]["model_source"] = mineru_settings["model_source"]
    merged_config["mineru"]["pdf_render_timeout"] = mineru_settings["pdf_render_timeout"]
    merged_config["mineru"]["pdf_render_threads"] = mineru_settings["pdf_render_threads"]
    merged_config["mineru"]["table_merge_enable"] = mineru_settings["table_merge_enable"]
    merged_config["mineru"]["processing_window_size"] = mineru_settings["processing_window_size"]
    merged_config["mineru"]["api_max_concurrent_requests"] = mineru_settings[
        "api_max_concurrent_requests"
    ]

    merged_config["vlm"]["base_url"] = vlm_settings["base_url"]
    merged_config["vlm"]["api_key"] = vlm_settings["api_key"]
    merged_config["vlm"]["model"] = vlm_settings["model"]
    merged_config["vlm"]["api_mode"] = vlm_settings["api_mode"]
    merged_config["vlm"]["request_timeout_seconds"] = vlm_settings["request_timeout_seconds"]
    merged_config["vlm"]["decode_params"]["temperature"] = vlm_settings["decode_params"][
        "temperature"
    ]
    merged_config["vlm"]["decode_params"]["top_p"] = vlm_settings["decode_params"]["top_p"]
    merged_config["vlm"]["decode_params"]["top_k"] = vlm_settings["decode_params"]["top_k"]
    merged_config["vlm"]["decode_params"]["max_tokens"] = vlm_settings["decode_params"][
        "max_tokens"
    ]
    merged_config["vlm"]["decode_params"]["repetition_penalty"] = vlm_settings["decode_params"][
        "repetition_penalty"
    ]
    # Force data_uri mode for reliable image transfer (no need for Ollama to access backend).
    merged_config["vlm"]["image_mode"] = "data_uri"

    merged_config["review_vlm"]["base_url"] = review_vlm_settings["base_url"]
    merged_config["review_vlm"]["api_key"] = review_vlm_settings["api_key"]
    merged_config["review_vlm"]["model"] = review_vlm_settings["model"]
    merged_config["review_vlm"]["api_mode"] = review_vlm_settings["api_mode"]
    merged_config["review_vlm"]["request_timeout_seconds"] = review_vlm_settings[
        "request_timeout_seconds"
    ]
    merged_config["review_vlm"]["decode_params"]["temperature"] = review_vlm_settings[
        "decode_params"
    ]["temperature"]
    merged_config["review_vlm"]["decode_params"]["top_p"] = review_vlm_settings["decode_params"][
        "top_p"
    ]
    merged_config["review_vlm"]["decode_params"]["top_k"] = review_vlm_settings["decode_params"][
        "top_k"
    ]
    merged_config["review_vlm"]["decode_params"]["max_tokens"] = review_vlm_settings[
        "decode_params"
    ]["max_tokens"]
    merged_config["review_vlm"]["decode_params"]["repetition_penalty"] = review_vlm_settings[
        "decode_params"
    ]["repetition_penalty"]
    merged_config["review_vlm"]["image_mode"] = "data_uri"


async def _cancel_running_stages(db: Database, run_id: str) -> None:
    """Mark any in-flight stages canceled when a run is canceled externally."""
    stage_repo = RunStageRepository(db)
    stages = await stage_repo.list_by_run(run_id)
    for stage in stages:
        if stage.status == StageStatus.RUNNING:
            await stage_repo.update_status(
                run_id,
                stage.stage,
                StageStatus.CANCELED,
                error={"message": "Run was canceled"},
            )




HIDDEN_PIPELINE_RUNS_KEY = "hidden_pipeline_run_ids"


async def _get_hidden_pipeline_run_ids(db: Database) -> set[str]:
    settings_repo = SettingsRepository(db)
    data = await settings_repo.get(HIDDEN_PIPELINE_RUNS_KEY) or {}
    run_ids = data.get("run_ids", [])
    if not isinstance(run_ids, list):
        return set()
    return {str(run_id) for run_id in run_ids}


async def _hide_pipeline_runs(db: Database, run_ids: list[str]) -> None:
    settings_repo = SettingsRepository(db)
    hidden = await _get_hidden_pipeline_run_ids(db)
    hidden.update(run_ids)
    await settings_repo.set(HIDDEN_PIPELINE_RUNS_KEY, {"run_ids": sorted(hidden)})

def _stage_progress_from_stages(stages) -> tuple[StageName | None, dict[str, Any] | None]:
    """Extract compact progress/status context for run list rows."""
    running_stage = next((stage for stage in stages if stage.status == StageStatus.RUNNING), None)
    if running_stage:
        progress = (running_stage.stats or {}).get("progress")
        return running_stage.stage, progress if isinstance(progress, dict) else None

    terminal_stage = next(
        (
            stage
            for stage in reversed(stages)
            if stage.status in (StageStatus.FAILED, StageStatus.CANCELED)
        ),
        None,
    )
    if terminal_stage:
        return terminal_stage.stage, None

    latest_stage = next(
        (stage for stage in reversed(stages) if stage.status == StageStatus.SUCCEEDED),
        None,
    )
    return latest_stage.stage if latest_stage else None, None


def _load_split_document_summary(outputs_dir: Path) -> dict[str, Any]:
    """Read split-document counts and lightweight metadata for a run."""
    index_path = outputs_dir / "documents_index.json"
    if not index_path.exists():
        return {
            "documents_total": 0,
            "main_document_count": 0,
            "extracted_document_count": 0,
            "documents": [],
        }

    try:
        raw_documents = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw_documents = []

    if not isinstance(raw_documents, list):
        raw_documents = []

    documents: list[dict[str, Any]] = []
    for document in raw_documents:
        if not isinstance(document, dict):
            continue
        item = dict(document)
        file_path = Path(str(item.get("file") or ""))
        item["filename"] = file_path.name if file_path.name else item.get("filename")
        item.pop("file", None)
        documents.append(item)

    main_count = sum(1 for item in documents if item.get("kind") == "main")
    return {
        "documents_total": len(documents),
        "main_document_count": main_count,
        "extracted_document_count": max(len(documents) - main_count, 0),
        "documents": documents,
    }


def _load_quality_gate_summary(outputs_dir: Path) -> dict[str, Any]:
    """Read compact quality gate status for management lists."""
    quality_gate_path = outputs_dir / "quality_gate.json"
    if not quality_gate_path.exists():
        return {"quality_gate_status": "unknown", "quality_score": None, "quality_issue_count": 0}

    try:
        report = json.loads(quality_gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"quality_gate_status": "unknown", "quality_score": None, "quality_issue_count": 0}

    issues = report.get("issues") if isinstance(report, dict) else []
    return {
        "quality_gate_status": report.get("status", "unknown") if isinstance(report, dict) else "unknown",
        "quality_score": report.get("score") if isinstance(report, dict) else None,
        "quality_issue_count": len(issues) if isinstance(issues, list) else 0,
    }


@router.post("", response_model=RunResponse)
async def create_run(
    request: RunCreateRequest,
    db: Database = Depends(get_db),
) -> RunResponse:
    """
    Create a new pipeline run for a document.
    """
    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)
    stage_repo = RunStageRepository(db)

    # Verify document exists
    doc = await doc_repo.get(request.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {request.doc_id}")

    # Build configuration
    base_config = PROFILES.get(request.profile)
    if not base_config:
        raise HTTPException(status_code=400, detail=f"Unknown profile: {request.profile}")

    # Get profile overrides from database (user's saved profile settings)
    settings_repo = SettingsRepository(db)
    profile_overrides = await settings_repo.get(f"profile_override:{request.profile.value}") or {}

    # Convert profile overrides to nested config structure
    profile_config_overrides = _convert_profile_overrides(profile_overrides)

    # Merge: base_config + profile_overrides + request.config_overrides
    merged_config = merge_config(base_config, profile_config_overrides)
    if request.config_overrides:
        merged_config = merge_config(
            PipelineConfig(**merged_config) if isinstance(merged_config, dict) else merged_config,
            request.config_overrides,
        )

    # Apply runtime settings from database/env (user's saved settings)
    mineru_settings = await get_mineru_config(db)
    vlm_settings = await get_vlm_config(db)
    review_vlm_settings = await get_review_vlm_config(db)
    _apply_runtime_settings(merged_config, mineru_settings, vlm_settings, review_vlm_settings)

    config_hash = compute_config_hash(merged_config)

    # Create run
    run = await run_repo.create(
        RunCreate(
            doc_id=request.doc_id,
            profile=request.profile.value,
            config=merged_config,
            config_hash=config_hash,
            use_cache=request.use_cache,
            force_stages=request.force_stages,
        )
    )

    # Create all stages
    await stage_repo.create_all_stages(run.run_id)

    return RunResponse(
        run_id=run.run_id,
        doc_id=run.doc_id,
        profile=run.profile,
        status=run.status,
        use_cache=run.use_cache,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.post("/{run_id}/execute")
async def execute_run(
    run_id: str,
    background: bool = Query(default=True, description="Run in background"),
    db: Database = Depends(get_db),
) -> dict:
    """
    Execute a pipeline run.

    If background=True (default), submits to task queue and returns immediately.
    If background=False, waits for completion and returns result.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if run.status not in (RunStatus.PENDING, RunStatus.FAILED, RunStatus.CANCELED):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot execute run with status: {run.status}. "
                "Only pending, failed, or canceled runs can be executed."
            ),
        )

    task_queue = await get_task_queue(db)

    if background:
        # Submit to background queue
        task_info = await task_queue.submit(run_id)
        return {
            "message": "Run submitted for execution",
            "run_id": run_id,
            "task_status": task_info.status.value,
        }
    else:
        # Execute synchronously
        from app.core.orchestrator import get_orchestrator
        orchestrator = get_orchestrator(db)
        result = await orchestrator.execute(run_id)

        return {
            "message": "Run completed",
            "run_id": run_id,
            "success": result.success,
            "status": (
                result.final_status.value
                if hasattr(result.final_status, "value")
                else str(result.final_status)
            ),
            "error": result.error,
            "stats": result.stats,
        }


@router.get("/{run_id}/task_status")
async def get_task_status(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    Get the task queue status for a run.
    """
    task_queue = await get_task_queue(db)
    task_info = task_queue.get_status(run_id)

    if not task_info:
        return {
            "run_id": run_id,
            "in_queue": False,
            "message": "Run is not in the task queue",
        }

    return {
        "run_id": run_id,
        "in_queue": True,
        "status": task_info.status.value,
        "current_stage": task_info.current_stage.value if task_info.current_stage else None,
        "created_at": task_info.created_at.isoformat(),
        "started_at": task_info.started_at.isoformat() if task_info.started_at else None,
        "finished_at": task_info.finished_at.isoformat() if task_info.finished_at else None,
    }


@router.get("", response_model=RunListResponse)
async def list_runs(
    status: RunStatus | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    include_hidden: bool = Query(
        default=False,
        description="Include runs hidden from the Pipeline Runs view.",
    ),
    db: Database = Depends(get_db),
) -> RunListResponse:
    """
    List all runs with optional status filter.
    """
    run_repo = RunRepository(db)

    hidden_run_ids = set() if include_hidden else await _get_hidden_pipeline_run_ids(db)

    # 獲取總數和當前頁數據
    total = await run_repo.count(status=status, exclude_run_ids=hidden_run_ids)
    runs = await run_repo.list_all(
        status=status,
        limit=limit,
        offset=offset,
        exclude_run_ids=hidden_run_ids,
    )

    stage_repo = RunStageRepository(db)
    responses: list[RunResponse] = []
    for r in runs:
        stages = await stage_repo.list_by_run(r.run_id)
        current_stage, stage_progress = _stage_progress_from_stages(stages)
        responses.append(
            RunResponse(
                run_id=r.run_id,
                doc_id=r.doc_id,
                profile=r.profile,
                status=r.status,
                use_cache=r.use_cache,
                created_at=r.created_at,
                updated_at=r.updated_at,
                current_stage=current_stage,
                stage_progress=stage_progress,
            )
        )

    return RunListResponse(
        runs=responses,
        total=total,
    )


@router.get("/stats")
async def get_runs_stats(
    db: Database = Depends(get_db),
) -> dict[str, int]:
    """
    Get run statistics by status.
    Returns count of runs for each status.
    """
    run_repo = RunRepository(db)

    hidden_run_ids = await _get_hidden_pipeline_run_ids(db)

    # Get counts for each visible Pipeline run status
    stats = {
        "total": await run_repo.count(exclude_run_ids=hidden_run_ids),
        "pending": await run_repo.count(
            status=RunStatus.PENDING,
            exclude_run_ids=hidden_run_ids,
        ),
        "running": await run_repo.count(
            status=RunStatus.RUNNING,
            exclude_run_ids=hidden_run_ids,
        ),
        "succeeded": await run_repo.count(
            status=RunStatus.SUCCEEDED,
            exclude_run_ids=hidden_run_ids,
        ),
        "failed": await run_repo.count(
            status=RunStatus.FAILED,
            exclude_run_ids=hidden_run_ids,
        ),
        "canceled": await run_repo.count(
            status=RunStatus.CANCELED,
            exclude_run_ids=hidden_run_ids,
        ),
    }

    return stats


@router.get("/outputs-summary")
async def list_outputs_summary(
    status: RunStatus | None = Query(default=RunStatus.SUCCEEDED),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    include_hidden: bool = Query(default=True),
    has_documents_only: bool = Query(default=True),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """List generated document outputs without N+1 split-document requests."""
    run_repo = RunRepository(db)
    doc_repo = DocRepository(db)

    hidden_run_ids = set() if include_hidden else await _get_hidden_pipeline_run_ids(db)
    candidate_limit = min(max(limit * 3, limit), 500)
    candidate_offset = offset if not has_documents_only else 0
    runs = await run_repo.list_all(
        status=status,
        limit=candidate_limit,
        offset=candidate_offset,
        exclude_run_ids=hidden_run_ids,
    )

    items: list[dict[str, Any]] = []
    for run in runs:
        outputs_dir = settings.get_run_path(run.doc_id, run.run_id) / "outputs"
        document_summary = _load_split_document_summary(outputs_dir)
        if has_documents_only and document_summary["documents_total"] == 0:
            continue

        doc = await doc_repo.get(run.doc_id)
        source_path = doc.source_path if doc else ""
        source_name = Path(source_path).stem if source_path else run.doc_id
        quality_summary = _load_quality_gate_summary(outputs_dir)
        items.append(
            {
                "run_id": run.run_id,
                "doc_id": run.doc_id,
                "profile": run.profile,
                "status": run.status.value,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
                "source_path": source_path,
                "source_name": source_name,
                **document_summary,
                **quality_summary,
            }
        )

    if has_documents_only:
        total = len(items)
        items = items[offset:offset + limit]
    else:
        total = await run_repo.count(status=status, exclude_run_ids=hidden_run_ids)

    return {"runs": items, "total": total}


@router.get("/{run_id}", response_model=RunDetailResponse)
async def get_run(
    run_id: str,
    db: Database = Depends(get_db),
) -> RunDetailResponse:
    """
    Get detailed run information including stages.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get_with_stages(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    stages = []
    for s in run.stages:
        duration = None
        if s.started_at and s.finished_at:
            duration = (s.finished_at - s.started_at).total_seconds()

        stages.append(
            StageResponse(
                stage=s.stage,
                status=s.status,
                started_at=s.started_at,
                finished_at=s.finished_at,
                duration_seconds=duration,
                error=s.error,
                stats=s.stats,
            )
        )

    current_stage, stage_progress = _stage_progress_from_stages(run.stages)

    return RunDetailResponse(
        run_id=run.run_id,
        doc_id=run.doc_id,
        profile=run.profile,
        status=run.status,
        use_cache=run.use_cache,
        created_at=run.created_at,
        updated_at=run.updated_at,
        current_stage=current_stage,
        stage_progress=stage_progress,
        config=run.config,
        stages=stages,
    )


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """
    Cancel a running pipeline.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run with status: {run.status}",
        )

    # Cancel in task queue
    task_queue = await get_task_queue(db)
    await task_queue.cancel(run_id)

    await _cancel_running_stages(db, run_id)
    await run_repo.update_status(run_id, RunStatus.CANCELED)

    return {"message": "Run canceled", "run_id": run_id}


@router.delete("/{run_id}")
async def delete_run(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    """
    Hide a run from Pipeline Runs while keeping generated documents available.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    await _hide_pipeline_runs(db, [run_id])

    return {"message": "Run hidden from Pipeline Runs", "run_id": run_id}


@router.post("/batch-create")
async def batch_create_runs(
    doc_ids: list[str],
    profile: str = Query(default="accurate", pattern="^(fast|accurate)$"),
    use_cache: bool = Query(default=False),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Batch create and execute pipeline runs for multiple documents.

    All runs are submitted to the task queue and executed with concurrency control.
    """
    from app.config import ProfileName

    doc_repo = DocRepository(db)
    run_repo = RunRepository(db)
    stage_repo = RunStageRepository(db)
    settings_repo = SettingsRepository(db)

    profile_enum = ProfileName(profile)
    base_config = PROFILES.get(profile_enum)
    if not base_config:
        raise HTTPException(status_code=400, detail=f"Unknown profile: {profile}")

    # Get profile overrides
    profile_overrides = await settings_repo.get(f"profile_override:{profile}") or {}
    profile_config_overrides = _convert_profile_overrides(profile_overrides)

    # Apply runtime settings
    mineru_settings = await get_mineru_config(db)
    vlm_settings = await get_vlm_config(db)
    review_vlm_settings = await get_review_vlm_config(db)

    created: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    task_queue = await get_task_queue(db)

    for doc_id in doc_ids:
        try:
            # Verify document exists
            doc = await doc_repo.get(doc_id)
            if not doc:
                errors.append({"doc_id": doc_id, "error": "Document not found"})
                continue

            # Merge configs
            merged_config = merge_config(base_config, profile_config_overrides)
            _apply_runtime_settings(merged_config, mineru_settings, vlm_settings, review_vlm_settings)

            config_hash = compute_config_hash(merged_config)

            # Create run
            run = await run_repo.create(
                RunCreate(
                    doc_id=doc_id,
                    profile=profile,
                    config=merged_config,
                    config_hash=config_hash,
                    use_cache=use_cache,
                    force_stages=None,
                )
            )

            # Create stages
            await stage_repo.create_all_stages(run.run_id)

            # Submit to task queue
            await task_queue.submit(run.run_id)

            created.append({"doc_id": doc_id, "run_id": run.run_id})

        except Exception as e:
            errors.append({"doc_id": doc_id, "error": str(e)})

    return {
        "message": f"Created {len(created)} runs",
        "profile": profile,
        "use_cache": use_cache,
        "created": created,
        "errors": errors,
    }


@router.post("/batch-delete")
async def batch_delete_runs(
    run_ids: list[str],
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Hide multiple runs from Pipeline Runs while keeping generated documents available.
    """
    run_repo = RunRepository(db)

    hidden: list[str] = []
    errors: list[dict[str, str]] = []

    for run_id in run_ids:
        try:
            run = await run_repo.get(run_id)
            if not run:
                errors.append({"run_id": run_id, "error": "Run not found"})
                continue
            hidden.append(run_id)
        except Exception as e:
            errors.append({"run_id": run_id, "error": str(e)})

    if hidden:
        await _hide_pipeline_runs(db, hidden)

    return {
        "message": f"Hidden {len(hidden)} runs from Pipeline Runs",
        "deleted": hidden,
        "errors": errors,
    }


@router.post("/batch-cancel")
async def batch_cancel_runs(
    run_ids: list[str],
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Batch cancel multiple pending or running runs.
    """
    run_repo = RunRepository(db)
    task_queue = await get_task_queue(db)

    canceled: list[str] = []
    skipped: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for run_id in run_ids:
        try:
            run = await run_repo.get(run_id)
            if not run:
                errors.append({"run_id": run_id, "error": "Run not found"})
                continue

            # Only cancel pending or running runs
            if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                skipped.append({"run_id": run_id, "status": run.status.value})
                continue

            # Cancel in task queue
            await task_queue.cancel(run_id)

            # Update status to canceled
            await _cancel_running_stages(db, run_id)
            await run_repo.update_status(run_id, RunStatus.CANCELED)

            canceled.append(run_id)
        except Exception as e:
            errors.append({"run_id": run_id, "error": str(e)})

    return {
        "message": f"Canceled {len(canceled)} runs, skipped {len(skipped)}",
        "canceled": canceled,
        "skipped": skipped,
        "errors": errors,
    }


@router.get("/{run_id}/document_ir")
async def get_document_ir(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    Get the document IR for a run.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    ir_path = settings.get_run_path(run.doc_id, run_id) / "document_ir.json"
    if not ir_path.exists():
        raise HTTPException(status_code=404, detail="Document IR not found")

    import orjson
    return orjson.loads(ir_path.read_bytes())


@router.get("/{run_id}/output")
async def get_output(
    run_id: str,
    view: str = Query(default="source", pattern="^(source|dataset|rag)$"),
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    Get the output markdown file for a run.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    filename_by_view = {
        "source": "source.md",
        # Backward compatibility for already-created runs.
        "dataset": "dataset.md",
        "rag": "rag.md",
    }
    output_path = settings.get_run_path(run.doc_id, run_id) / "outputs" / filename_by_view[view]
    if not output_path.exists() and view == "source":
        output_path = settings.get_run_path(run.doc_id, run_id) / "outputs" / "rag.md"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"Output not found: {view}")

    return FileResponse(output_path, media_type="text/markdown")


@router.get("/{run_id}/documents")
async def list_split_documents(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    List split markdown documents generated for a run.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    outputs_dir = settings.get_run_path(run.doc_id, run_id) / "outputs"
    index_path = outputs_dir / "documents_index.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Split documents index not found")

    documents = json.loads(index_path.read_text(encoding="utf-8"))
    for document in documents:
        file_path = Path(document.get("file", ""))
        document["filename"] = file_path.name
        document.pop("file", None)

    return {
        "run_id": run_id,
        "doc_id": run.doc_id,
        "documents": documents,
        "total": len(documents),
    }


@router.get("/{run_id}/documents/{document_id}")
async def get_split_document(
    run_id: str,
    document_id: str,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Get one split markdown document and its source-page metadata.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    outputs_dir = settings.get_run_path(run.doc_id, run_id) / "outputs"
    index_path = outputs_dir / "documents_index.json"
    documents_dir = outputs_dir / "documents"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Split documents index not found")

    documents = json.loads(index_path.read_text(encoding="utf-8"))
    document = next((item for item in documents if item.get("document_id") == document_id), None)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Split document not found: {document_id}")

    filename = Path(document.get("file", "")).name
    if not filename:
        raise HTTPException(status_code=404, detail=f"Split document file not found: {document_id}")

    document_path = (documents_dir / filename).resolve()
    documents_root = documents_dir.resolve()
    if not document_path.is_file() or documents_root not in document_path.parents:
        raise HTTPException(status_code=404, detail=f"Split document file not found: {document_id}")

    metadata = dict(document)
    metadata["filename"] = filename
    metadata.pop("file", None)
    return {
        "run_id": run_id,
        "doc_id": run.doc_id,
        "document": metadata,
        "content": document_path.read_text(encoding="utf-8"),
    }


@router.delete("/{run_id}/documents")
async def delete_split_documents(
    run_id: str,
    document_ids: list[str] = Body(..., min_length=1),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Delete selected generated split documents for a run.

    This is a document-management operation: it removes exported main/child
    markdown files and updates documents_index.json, but keeps the pipeline run
    record and other run artifacts intact.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    outputs_dir = settings.get_run_path(run.doc_id, run_id) / "outputs"
    index_path = outputs_dir / "documents_index.json"
    documents_dir = outputs_dir / "documents"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Split documents index not found")

    requested_ids = {str(document_id) for document_id in document_ids}
    documents = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise HTTPException(status_code=400, detail="Invalid split documents index")

    documents_root = documents_dir.resolve()
    kept: list[dict[str, Any]] = []
    deleted: list[str] = []
    errors: list[dict[str, str]] = []

    for document in documents:
        document_id = str(document.get("document_id") or "")
        if document_id not in requested_ids:
            kept.append(document)
            continue

        filename = Path(str(document.get("file") or "")).name
        if not filename:
            errors.append({"document_id": document_id, "error": "Document filename missing"})
            kept.append(document)
            continue

        document_path = (documents_dir / filename).resolve()
        if not document_path.is_relative_to(documents_root):
            errors.append({"document_id": document_id, "error": "Invalid document path"})
            kept.append(document)
            continue

        try:
            if document_path.exists():
                document_path.unlink()
            deleted.append(document_id)
        except OSError as exc:
            errors.append({"document_id": document_id, "error": str(exc)})
            kept.append(document)

    missing_ids = requested_ids - {str(item.get("document_id") or "") for item in documents}
    for document_id in sorted(missing_ids):
        errors.append({"document_id": document_id, "error": "Split document not found"})

    if deleted:
        index_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "message": f"Deleted {len(deleted)} split documents",
        "run_id": run_id,
        "deleted": deleted,
        "errors": errors,
        "remaining": len(kept),
    }


@router.get("/{run_id}/source_map")
async def get_source_map(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    Get the source map for a run (MD anchor to block mappings).
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    sm_path = settings.get_run_path(run.doc_id, run_id) / "source_map.json"
    if not sm_path.exists():
        raise HTTPException(status_code=404, detail="Source map not found")

    import orjson
    return orjson.loads(sm_path.read_bytes())


@router.get("/{run_id}/quality")
async def get_quality(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    Get the quality report for a run.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    quality_path = settings.get_run_path(run.doc_id, run_id) / "outputs" / "quality.json"
    if not quality_path.exists():
        raise HTTPException(status_code=404, detail="Quality report not found")

    import orjson
    return orjson.loads(quality_path.read_bytes())


@router.get("/{run_id}/quality_gate")
async def get_quality_gate(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    Get the RAG readiness quality gate report for a run.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    quality_gate_path = settings.get_run_path(run.doc_id, run_id) / "outputs" / "quality_gate.json"
    if not quality_gate_path.exists():
        return {
            "status": "unknown",
            "score": 0,
            "issues": [],
            "vlm_audit_candidates": [],
            "vlm_audits": [],
            "stats": {"message": "Quality gate report not found for this run."},
        }

    import orjson
    return orjson.loads(quality_gate_path.read_bytes())


@router.get("/{run_id}/assets_index")
async def get_assets_index(
    run_id: str,
    db: Database = Depends(get_db),
) -> list[dict]:
    """
    Get the assets index for a run.
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    index_path = settings.get_run_path(run.doc_id, run_id) / "outputs" / "assets_index.jsonl"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Assets index not found")

    import orjson
    assets = []
    for line in index_path.read_text().strip().split("\n"):
        if line:
            assets.append(orjson.loads(line))

    return assets


@router.get("/{run_id}/enrichments")
async def get_enrichments(
    run_id: str,
    block_id: str | None = Query(default=None, description="Filter by block_id"),
    kind: str | None = Query(default=None, description="Filter by enrichment kind"),
    needs_review: bool | None = Query(default=None, description="Filter by needs_review flag"),
    db: Database = Depends(get_db),
) -> dict:
    """
    Get the enrichments for a run.

    Optional filters:
    - block_id: Filter by specific block
    - kind: Filter by enrichment type (form_asset, figure_caption, table_summary)
    - needs_review: Filter by review flag
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    enrichments_path = settings.get_run_path(run.doc_id, run_id) / "outputs" / "enrichments.jsonl"
    if not enrichments_path.exists():
        return {"enrichments": [], "total": 0}

    import orjson
    enrichments = []
    for line in enrichments_path.read_text().strip().split("\n"):
        if not line:
            continue
        entry = orjson.loads(line)

        # Apply filters
        if block_id and entry.get("block_id") != block_id:
            continue
        if kind and entry.get("kind") != kind:
            continue
        if needs_review is not None:
            entry_needs_review = entry.get("quality", {}).get("needs_review", False)
            if entry_needs_review != needs_review:
                continue

        enrichments.append(entry)

    return {"enrichments": enrichments, "total": len(enrichments)}


@router.get("/{run_id}/manifest")
async def get_manifest(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    Get the manifest for a run (engine versions and config).
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    manifest_path = settings.get_run_path(run.doc_id, run_id) / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")

    import orjson
    return orjson.loads(manifest_path.read_bytes())


@router.post("/{run_id}/invalidate")
async def invalidate_cache(
    run_id: str,
    stages: list[str] = Query(default=["parse", "enrich"], description="Stages to invalidate"),
    db: Database = Depends(get_db),
) -> dict:
    """
    Invalidate cache for specific stages of a run's document.

    Stages:
    - parse: Clear filesystem parse cache
    - enrich: Clear SQLite enrich cache

    This allows re-running the pipeline without cached results.
    """
    from app.core.cache import CacheManager

    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Validate stages
    valid_stages = {"parse", "enrich"}
    invalid = set(stages) - valid_stages
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stages: {invalid}. Valid stages: {valid_stages}",
        )

    # Invalidate cache
    cache_manager = CacheManager(db)
    result = await cache_manager.invalidate_stages(run.doc_id, stages)

    return {
        "message": "Cache invalidated",
        "doc_id": run.doc_id,
        "run_id": run_id,
        "invalidated": result,
    }


@router.get("/{run_id}/cache_stats")
async def get_cache_stats(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    Get cache statistics for a run's document.
    """
    from app.core.cache import CacheManager

    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    cache_manager = CacheManager(db)
    stats = await cache_manager.get_cache_stats(run.doc_id)

    return {
        "doc_id": run.doc_id,
        "run_id": run_id,
        "cache": stats,
    }


# =============================================================================
# D4: Org Chart Debug API
# =============================================================================


@router.get("/{run_id}/org-chart")
async def get_org_chart(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    D4: 取得組織圖處理結果。

    回傳：
    - graph: canonical graph (nodes, edges, groups)
    - render_md: 最終渲染輸出
    - warnings: 處理過程中的警告
    - decision_trace: 決策分支追蹤
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    debug_dir = settings.get_run_path(run.doc_id, run_id) / "outputs" / "org_debug"

    if not debug_dir.exists():
        return {
            "found": False,
            "doc_id": run.doc_id,
            "run_id": run_id,
            "page_idx": 0,
            "graph": None,
            "render_md": "",
            "warnings": [],
            "decision_trace": {},
            "message": "此 run 沒有組織圖處理結果",
        }

    import orjson

    result: dict[str, Any] = {
        "found": True,
        "doc_id": run.doc_id,
        "run_id": run_id,
    }

    # 讀取 canonical graph
    canonical_path = debug_dir / "org_graph.canonical.json"
    if canonical_path.exists():
        data = orjson.loads(canonical_path.read_bytes())
        graph_data = data.get("graph", {})
        result["graph"] = graph_data
        result["page_idx"] = graph_data.get("page_idx", 0)
        result["warnings"] = data.get("warnings", [])
        result["decision_trace"] = data.get("decision_trace", {})
    else:
        result["graph"] = None
        result["page_idx"] = 0
        result["warnings"] = []
        result["decision_trace"] = {}

    # 讀取 render.md
    render_path = debug_dir / "org_render.md"
    if render_path.exists():
        result["render_md"] = render_path.read_text(encoding="utf-8")
    else:
        result["render_md"] = ""

    return result


@router.get("/{run_id}/org-chart/debug/index")
async def get_org_chart_debug_index(
    run_id: str,
    db: Database = Depends(get_db),
) -> dict:
    """
    D4: 取得組織圖 debug 檔案索引。

    回傳 org_debug/ 資料夾中所有檔案的名稱、大小、修改時間。
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    debug_dir = settings.get_run_path(run.doc_id, run_id) / "outputs" / "org_debug"

    if not debug_dir.exists():
        return {
            "found": False,
            "files": [],
        }

    from datetime import datetime

    files = []
    for f in debug_dir.iterdir():
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    # 按名稱排序
    files.sort(key=lambda x: x["name"])

    return {
        "found": True,
        "files": files,
        "path": str(debug_dir),
    }


@router.get("/{run_id}/org-chart/debug/file")
async def get_org_chart_debug_file(
    run_id: str,
    name: str = Query(..., description="檔案名稱，例如 org_nodes_candidates.json"),
    db: Database = Depends(get_db),
) -> FileResponse:
    """
    D4: 讀取組織圖 debug 檔案內容。

    支援的檔案：
    - org_input.json: 策略分支決策
    - org_nodes_candidates.json: MinerU blocks → nodes
    - org_edge_candidates.json: heuristics 候選
    - org_vlm1_units.raw.json: VLM#1 原始輸出
    - org_vlm1_units.validated.json: VLM#1 驗證後
    - org_vlm2_edges.raw.json: VLM#2 原始輸出
    - org_vlm2_edges.validated.json: VLM#2 驗證後
    - org_graph.canonical.json: 最終 graph + warnings
    - org_render.md: 最終輸出
    """
    run_repo = RunRepository(db)
    run = await run_repo.get(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    debug_dir = settings.get_run_path(run.doc_id, run_id) / "outputs" / "org_debug"

    if not debug_dir.exists():
        raise HTTPException(status_code=404, detail="Debug 資料夾不存在")

    # 安全性檢查：防止路徑遍歷攻擊
    file_path = debug_dir / name
    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(debug_dir.resolve())):
            raise HTTPException(status_code=400, detail="無效的檔案名稱")
    except Exception:
        raise HTTPException(status_code=400, detail="無效的檔案名稱")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"檔案不存在: {name}")

    # 根據副檔名決定 media type
    if name.endswith(".json"):
        media_type = "application/json"
    elif name.endswith(".md"):
        media_type = "text/markdown"
    else:
        media_type = "text/plain"

    return FileResponse(file_path, media_type=media_type)
