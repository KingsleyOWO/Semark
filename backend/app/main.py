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

    await _cancel_orphan_running_runs()

    yield

    # Shutdown
    await db.disconnect()



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

app = FastAPI(
    title="Doc Parser API",
    description="Document Parser for RAG - Convert PDF/Word/Image/HTML to structured markdown",
    version="0.1.0",
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
        "version": "0.1.0",
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
