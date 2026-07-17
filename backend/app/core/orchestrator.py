"""
Pipeline Orchestrator - manages pipeline execution.

Stages: Ingest → Parse → Normalize → Enrich → Package → Chunk
"""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import PipelineConfig, settings
from app.db.database import Database
from app.db.repositories import DocRepository, RunRepository, RunStageRepository
from app.models.document_ir import DocumentIR, SourceInfo
from app.models.entities import Run, RunStatus, StageName, StageStatus
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.enrich import EnrichStage
from app.pipeline.stages.normalize import NormalizeStage, save_document_ir
from app.pipeline.stages.package import PackageStage
from app.pipeline.stages.parse import ParseStage

# Process-wide cap on concurrently-running *whole pipeline* executions
# (parse -> normalize -> enrich -> package -> chunk). Mirrors TaskQueue's
# "parse" worker-pool size (app.core.task_queue.py's pool_config["parse"]) so
# that background queue workers AND the direct/synchronous execute() path
# (POST /runs/{id}/execute?background=false in routes/runs.py) can never
# combine to run more full MinerU+VLM pipelines at once than the host can
# take. See _get_pipeline_semaphore.
DEFAULT_MAX_CONCURRENT_PIPELINES = 2

_pipeline_semaphore: asyncio.Semaphore | None = None


def _get_pipeline_semaphore(limit: int) -> asyncio.Semaphore:
    """Get (or lazily create) the process-wide pipeline concurrency semaphore.

    This is a module-level singleton rather than a plain instance attribute:
    TaskQueue and routes/runs.py's direct-execution path each construct
    their own PipelineOrchestrator instance (see TaskQueue.__init__'s
    self.orchestrator vs. get_orchestrator()'s module-level singleton), and
    both must still share one cap. Once created, the semaphore keeps its
    size for the life of the process -- call reset_pipeline_semaphore()
    (test-only) to force a resize.
    """
    global _pipeline_semaphore
    if _pipeline_semaphore is None:
        _pipeline_semaphore = asyncio.Semaphore(limit)
    return _pipeline_semaphore


def reset_pipeline_semaphore() -> None:
    """Drop the cached process-wide semaphore so the next orchestrator
    construction resizes it.

    Test-only seam, mirroring app.pipeline.corpus_rules.reset_rules_cache().
    Also necessary because asyncio.Semaphore binds to the event loop of its
    first acquire()/wait() (Python 3.10+): without a reset, a semaphore
    exercised by one pytest-asyncio test (its own event loop) would raise
    "bound to a different event loop" if reused by a later test.
    """
    global _pipeline_semaphore
    _pipeline_semaphore = None


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""

    doc_id: str
    run_id: str
    config: PipelineConfig
    run_path: Path

    # Stage outputs
    parse_cache_path: Path | None = None
    content_list_path: Path | None = None
    document_ir: DocumentIR | None = None

    # Stats
    stage_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_hits: dict[str, bool] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    success: bool
    run_id: str
    doc_id: str
    final_status: RunStatus
    document_ir: DocumentIR | None = None
    error: str | None = None
    failed_stage: StageName | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class PipelineOrchestrator:
    """
    Orchestrates the document processing pipeline.

    Manages stage execution, state tracking, and error handling.
    """

    def __init__(self, db: Database, max_concurrent_pipelines: int | None = None):
        self.db = db
        self.doc_repo = DocRepository(db)
        self.run_repo = RunRepository(db)
        self.stage_repo = RunStageRepository(db)

        # Active tasks for cancellation
        self._active_tasks: dict[str, asyncio.Task] = {}

        # Process-wide cap shared with any other PipelineOrchestrator
        # instance -- see _get_pipeline_semaphore.
        self._pipeline_semaphore = _get_pipeline_semaphore(
            max_concurrent_pipelines or DEFAULT_MAX_CONCURRENT_PIPELINES
        )

    async def execute(
        self,
        run_id: str,
        on_stage_complete: Callable[[StageName, dict], Coroutine] | None = None,
        on_stage_start: Callable[[StageName], Coroutine] | None = None,
    ) -> PipelineResult:
        """
        Execute the pipeline for a run.

        Args:
            run_id: Run ID to execute
            on_stage_complete: Optional callback when a stage completes
            on_stage_start: Optional callback when a stage starts

        Returns:
            PipelineResult with final status
        """
        # Load run
        run = await self.run_repo.get(run_id)
        if not run:
            return PipelineResult(
                success=False,
                run_id=run_id,
                doc_id="",
                final_status=RunStatus.FAILED,
                error=f"Run not found: {run_id}",
            )

        # Load document
        doc = await self.doc_repo.get(run.doc_id)
        if not doc:
            return PipelineResult(
                success=False,
                run_id=run_id,
                doc_id=run.doc_id,
                final_status=RunStatus.FAILED,
                error=f"Document not found: {run.doc_id}",
            )

        # Build config
        config = PipelineConfig.model_validate(run.config)

        # Create context
        run_path = settings.get_run_path(run.doc_id, run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        ctx = PipelineContext(
            doc_id=run.doc_id,
            run_id=run_id,
            config=config,
            run_path=run_path,
        )

        # Cap whole-pipeline concurrency across BOTH the task-queue workers
        # and the direct (background=false) execution path -- see
        # _get_pipeline_semaphore. `async with` guarantees the permit is
        # released on the CancelledError/exception paths below too.
        async with self._pipeline_semaphore:
            # Update run status
            await self.run_repo.update_status(run_id, RunStatus.RUNNING)

            try:
                return await self._run_stages(run, ctx, on_stage_complete, on_stage_start)
            except asyncio.CancelledError:
                await self.run_repo.update_status(run_id, RunStatus.CANCELED)
                return PipelineResult(
                    success=False,
                    run_id=run_id,
                    doc_id=run.doc_id,
                    final_status=RunStatus.CANCELED,
                    error="Pipeline was canceled",
                    stats=ctx.stage_stats,
                )

    async def _run_stages(
        self,
        run: Run,
        ctx: PipelineContext,
        on_stage_complete: Callable[[StageName, dict], Coroutine] | None,
        on_stage_start: Callable[[StageName], Coroutine] | None,
    ) -> PipelineResult:
        """Run all pipeline stages in order for an already-validated run/ctx.

        Extracted from execute() so it is a) a monkeypatchable seam for
        tests (e.g. replacing real stage work with a plain sleep) and b)
        entirely inside execute()'s concurrency-semaphore / cancellation
        handling, unchanged in behavior from before the extraction.
        """
        run_id = ctx.run_id

        # Execute stages in order
        stages = [
            (StageName.PARSE, self._run_parse),
            (StageName.NORMALIZE, self._run_normalize),
            (StageName.ENRICH, self._run_enrich),
            (StageName.PACKAGE, self._run_package),
            (StageName.CHUNK, self._run_chunk),
        ]

        for stage_name, stage_func in stages:
            # Check if stage should be forced
            force = (
                run.force_stages is not None
                and stage_name in run.force_stages
            )

            # Update stage status
            await self.stage_repo.update_status(
                run_id, stage_name, StageStatus.RUNNING
            )
            if on_stage_start:
                await on_stage_start(stage_name)

            try:
                # Execute stage
                stats = await stage_func(
                    ctx=ctx,
                    use_cache=run.use_cache and not force,
                )

                ctx.stage_stats[stage_name.value] = stats

                # Update stage status
                await self.stage_repo.update_status(
                    run_id, stage_name, StageStatus.SUCCEEDED, stats=stats
                )

                # Callback
                if on_stage_complete:
                    await on_stage_complete(stage_name, stats)

            except Exception as e:
                # Stage failed
                error_info = {"error": str(e), "type": type(e).__name__}
                await self.stage_repo.update_status(
                    run_id, stage_name, StageStatus.FAILED, error=error_info
                )
                await self.run_repo.update_status(run_id, RunStatus.FAILED)

                return PipelineResult(
                    success=False,
                    run_id=run_id,
                    doc_id=run.doc_id,
                    final_status=RunStatus.FAILED,
                    error=str(e),
                    failed_stage=stage_name,
                    stats=ctx.stage_stats,
                )

        # All stages completed
        await self.run_repo.update_status(run_id, RunStatus.SUCCEEDED)

        return PipelineResult(
            success=True,
            run_id=run_id,
            doc_id=run.doc_id,
            final_status=RunStatus.SUCCEEDED,
            document_ir=ctx.document_ir,
            stats=ctx.stage_stats,
        )

    async def _run_parse(
        self,
        ctx: PipelineContext,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Run parse stage."""
        stage = ParseStage(
            db=self.db,
            config=ctx.config.mineru,
        )

        result = await stage.run(
            doc_id=ctx.doc_id,
            run_id=ctx.run_id,
            use_cache=use_cache,
        )

        if not result.success:
            raise RuntimeError(f"Parse failed: {result.error}")

        ctx.parse_cache_path = result.cache_path
        ctx.content_list_path = result.content_list_path
        ctx.cache_hits["parse"] = result.cache_hit

        return result.stats

    async def _run_normalize(
        self,
        ctx: PipelineContext,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Run normalize stage."""
        if not ctx.content_list_path:
            # Try to find content_list from cache
            if ctx.parse_cache_path:
                for f in ctx.parse_cache_path.rglob("*_content_list.json"):
                    ctx.content_list_path = f
                    break

        if not ctx.content_list_path:
            raise RuntimeError("content_list.json not found")

        # Get source info
        doc = await self.doc_repo.get(ctx.doc_id)
        if not doc:
            raise RuntimeError(f"Document not found: {ctx.doc_id}")

        source_info = SourceInfo(
            path=doc.source_path,
            ext=doc.ext,
            sha256=doc.sha256,
            size_bytes=doc.size_bytes,
        )

        stage = NormalizeStage(config=ctx.config)

        result = await stage.run(
            doc_id=ctx.doc_id,
            run_id=ctx.run_id,
            content_list_path=ctx.content_list_path,
            source_info=source_info,
        )

        if not result.success:
            raise RuntimeError(f"Normalize failed: {result.error}")

        ctx.document_ir = result.document_ir

        # Save DocumentIR
        if ctx.document_ir:
            save_document_ir(ctx.document_ir, ctx.run_path)

        return result.stats

    async def _run_enrich(
        self,
        ctx: PipelineContext,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Run enrich stage (VLM enrichment)."""
        if not ctx.document_ir:
            raise RuntimeError("DocumentIR not available for enrich stage")

        stage = EnrichStage(
            db=self.db,
            config=ctx.config,
        )

        async def update_enrich_progress(stats: dict[str, Any]) -> None:
            await self.stage_repo.update_running_stats(
                ctx.run_id,
                StageName.ENRICH,
                stats,
            )

        result = await stage.run(
            doc_id=ctx.doc_id,
            run_id=ctx.run_id,
            document_ir=ctx.document_ir,
            run_path=ctx.run_path,
            parse_cache_path=ctx.parse_cache_path,
            use_cache=use_cache,
            progress_callback=update_enrich_progress,
            mineru_version=ctx.stage_stats.get("parse", {}).get("mineru_version"),
        )

        if not result.success:
            raise RuntimeError(f"Enrich failed: {result.error}")

        return result.stats

    async def _run_package(
        self,
        ctx: PipelineContext,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Run package stage."""
        if not ctx.document_ir:
            raise RuntimeError("DocumentIR not available for package stage")

        stage = PackageStage(config=ctx.config)

        result = await stage.run(
            doc_id=ctx.doc_id,
            run_id=ctx.run_id,
            document_ir=ctx.document_ir,
            run_path=ctx.run_path,
            parse_cache_path=ctx.parse_cache_path,
        )

        if not result.success:
            raise RuntimeError(f"Package failed: {result.error}")

        return result.stats

    async def _run_chunk(
        self,
        ctx: PipelineContext,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Run chunk stage."""
        if not ctx.document_ir:
            raise RuntimeError("DocumentIR not available for chunk stage")

        stage = ChunkStage(config=ctx.config)

        result = await stage.run(
            doc_id=ctx.doc_id,
            run_id=ctx.run_id,
            document_ir=ctx.document_ir,
            run_path=ctx.run_path,
        )

        if not result.success:
            raise RuntimeError(f"Chunk failed: {result.error}")

        return result.stats

    async def cancel(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        task = self._active_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def submit(
        self,
        run_id: str,
        on_stage_complete: Callable[[StageName, dict], Coroutine] | None = None,
    ) -> asyncio.Task:
        """
        Submit a pipeline run as a background task.

        Returns the task for tracking.
        """
        task = asyncio.create_task(
            self.execute(run_id, on_stage_complete)
        )
        self._active_tasks[run_id] = task

        # Clean up when done
        def cleanup(t):
            self._active_tasks.pop(run_id, None)

        task.add_done_callback(cleanup)

        return task

    def get_active_runs(self) -> list[str]:
        """Get list of currently running pipeline run IDs."""
        return [
            run_id
            for run_id, task in self._active_tasks.items()
            if not task.done()
        ]


# Global orchestrator instance
_orchestrator: PipelineOrchestrator | None = None


def get_orchestrator(db: Database) -> PipelineOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator(db)
    return _orchestrator
