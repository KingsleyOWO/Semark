"""
FastAPI application entry point.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import assets, docs, download, ingest, runs
from app.api.routes import settings as settings_routes
from app.config import settings
from app.core.task_queue import TaskQueue, get_task_queue
from app.db.database import db
from app.db.repositories import RunRepository, RunStageRepository
from app.models.entities import RunStatus, StageStatus


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler."""
    # Startup
    await db.connect()

    # Ensure workspace directories exist
    settings.workspace_path.mkdir(parents=True, exist_ok=True)
    settings.docs_path.mkdir(parents=True, exist_ok=True)

    await _cancel_orphan_runs()

    # Start queue workers now (rather than lazily on first request) so
    # there is always a live TaskQueue instance to stop() on shutdown.
    task_queue = await get_task_queue(db)

    yield

    # Shutdown
    await _shutdown_task_queue_and_db(task_queue)



async def _cancel_orphan_running_runs() -> int:
    """Cancel DB runs left running after a backend restart.

    Pipeline tasks live in the in-process task queue. After a process restart,
    any DB row still marked running cannot be resumed by that new queue, so it
    must not remain visible as active work in the UI.
    """

    run_repo = RunRepository(db)
    stage_repo = RunStageRepository(db)
    running_runs = await run_repo.list_all(status=RunStatus.RUNNING, limit=1000)
    for run in running_runs:
        stages = await stage_repo.list_by_run(run.run_id)
        for stage in stages:
            if stage.status == StageStatus.RUNNING:
                await stage_repo.update_status(
                    run.run_id,
                    stage.stage,
                    StageStatus.CANCELED,
                    error={"message": "Backend restarted before this stage completed"},
                )
        await run_repo.update_status(run.run_id, RunStatus.CANCELED)
    return len(running_runs)


async def _cancel_orphan_pending_runs() -> int:
    """Cancel DB runs left pending after a backend restart.

    Constraint: the task queue (app.core.task_queue) is a purely in-memory
    asyncio.Queue, so a run_id that was submitted (POST /runs/{id}/execute)
    but not yet dequeued when the process died has no worker left to ever
    pick it up after restart -- left alone it sits in PENDING forever with
    no indication anything is wrong (e.g. batch-create+submit 20 docs,
    restart after 5 have started -> 15 permanent "pending" zombies).

    Tradeoff: creating a run (POST /runs) does NOT itself submit it to the
    queue -- execute() is a separate call -- and the DB has no column
    recording "was queued" vs. "created but execution never requested"; the
    only place that distinction lived was the in-memory TaskQueue, which is
    exactly what a restart destroys. So this sweep cannot tell a truly
    orphaned (queued) PENDING run apart from one a user simply hasn't
    executed yet, and conservatively cancels both rather than leaving
    potential zombies invisible. Cost is low either way: re-executing a run
    is the same one action a user would take regardless of which case it
    was.
    """

    run_repo = RunRepository(db)
    stage_repo = RunStageRepository(db)
    pending_runs = await run_repo.list_all(status=RunStatus.PENDING, limit=1000)
    for run in pending_runs:
        stages = await stage_repo.list_by_run(run.run_id)
        for stage in stages:
            if stage.status == StageStatus.PENDING:
                await stage_repo.update_status(
                    run.run_id,
                    stage.stage,
                    StageStatus.CANCELED,
                    error={"message": "Backend restarted before this run was started"},
                )
        await run_repo.update_status(run.run_id, RunStatus.CANCELED)
    return len(pending_runs)


async def _cancel_orphan_runs() -> int:
    """Boot-time reconciliation sweep for runs stranded by a backend restart.

    Combines both blind spots created by the in-process, in-memory task
    queue: runs a prior process was actively executing (RUNNING) and runs
    it had merely accepted but not yet started (PENDING). Neither can be
    resumed by a freshly-started queue, so both are surfaced as CANCELED
    instead of appearing permanently "stuck" in the UI.
    """

    running = await _cancel_orphan_running_runs()
    pending = await _cancel_orphan_pending_runs()
    return running + pending


async def _shutdown_task_queue_and_db(task_queue: TaskQueue) -> None:
    """Stop queue workers before closing the DB connection.

    Ordering matters: worker cancellation is handled by
    PipelineOrchestrator.execute()'s `except asyncio.CancelledError` branch,
    which writes RunStatus.CANCELED through the DB. If db.disconnect() ran
    first, that write would raise "Database not connected" and the run
    would stay wrongly marked RUNNING until the next boot's sweep corrected
    it. Stopping the queue first lets in-flight cancellation writes land
    while the connection is still open. Extracted to a standalone function
    (called by lifespan) so shutdown ordering is directly unit-testable.
    """

    await task_queue.stop()
    await db.disconnect()


app = FastAPI(
    title="Semark API",
    description="Semantic Markdown for RAG from PDF, Office, HTML, and images",
    version="0.3.1",
    lifespan=lifespan,
)

# CORS middleware for frontend
private_lan_origin_regex = (
    r"^http://(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+):\d+$"
    if settings.cors_allow_private_lan
    else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=private_lan_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(ingest.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(docs.router, prefix="/api")
app.include_router(assets.router, prefix="/api")
app.include_router(download.router, prefix="/api")
app.include_router(settings_routes.router, prefix="/api")


@app.get("/api/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": app.version,
        "database": "connected" if db._connection else "disconnected",
    }


@app.get("/api/profiles")
async def list_profiles() -> dict:
    """List available pipeline profiles."""
    from app.config import ProfileName

    return {
        "profiles": [
            {
                "name": name.value,
                "description": _get_profile_description(name),
            }
            for name in ProfileName
        ],
        "default": settings.default_profile.value,
    }


def _get_profile_description(name) -> str:
    """Get profile description."""
    from app.config import ProfileName

    descriptions = {
        ProfileName.FAST: "Quick processing, minimal VLM enrichment",
        ProfileName.ACCURATE: "High quality with full VLM enrichment",
    }
    return descriptions.get(name, "")


@app.get("/api/settings")
async def get_settings() -> dict:
    """Get current settings (non-sensitive)."""
    return {
        "workspace_path": str(settings.workspace_path),
        "default_profile": settings.default_profile.value,
        "vlm": {
            "base_url": settings.vlm_base_url,
            "model": settings.vlm_model,
        },
    }


# For development: serve static files if frontend is built
# In production, use nginx or similar
if (settings.workspace_path.parent / "frontend" / "dist").exists():
    app.mount(
        "/",
        StaticFiles(
            directory=settings.workspace_path.parent / "frontend" / "dist",
            html=True,
        ),
        name="frontend",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
