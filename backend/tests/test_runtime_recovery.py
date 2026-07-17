"""
Runtime-robustness fixes for the in-process pipeline runner.

1. Boot sweep only canceled orphaned RUNNING runs
   (main._cancel_orphan_running_runs). Because the task queue
   (app.core.task_queue) is purely in-memory (asyncio.Queue), a run that was
   submitted but not yet dequeued before a restart sits in PENDING forever
   with no worker left to ever pick it up (batch-create 20 docs, restart
   after 5 start -> 15 permanent "pending" zombies). Fixed by adding a
   parallel PENDING sweep (main._cancel_orphan_pending_runs), combined with
   the existing RUNNING sweep in main._cancel_orphan_runs().

2. Shutdown closed the DB (db.disconnect()) before ever stopping the task
   queue, and never called TaskQueue.stop() at all. A worker cancelled
   during shutdown hits orchestrator.execute()'s
   `except asyncio.CancelledError: await self.run_repo.update_status(...)`
   AFTER the DB connection is gone -> "Database not connected", leaving the
   run's status only corrected by the next boot's sweep. Fixed by starting
   the queue eagerly in lifespan and calling the extracted
   main._shutdown_task_queue_and_db() (stop() before disconnect()) on
   shutdown. TaskQueue.stop() itself already existed and is already
   graceful (cancels workers + in-flight runs, awaits both); it was simply
   never wired up.

3. The direct-execution path (POST /runs/{id}/execute?background=false,
   routes/runs.py -- not modified here) calls orchestrator.execute(run_id)
   with no concurrency limit, so N concurrent HTTP calls run N full
   MinerU+VLM pipelines and can OOM the host. Fixed with a process-wide
   asyncio.Semaphore in orchestrator.py sized from the same value that sizes
   TaskQueue's worker pool (DEFAULT_MAX_CONCURRENT_PIPELINES /
   pool_config["parse"]), acquired (async with) around the whole pipeline
   execution in PipelineOrchestrator.execute(), so queue workers and direct
   calls -- which construct *separate* PipelineOrchestrator instances --
   share one cap.

4. SEMARK_CORPUS_RULES_PATH (app.pipeline.corpus_rules) must never be set
   while running pytest -- see conftest.py's session-scoped autouse
   fixture. This file just pins that the fixture did its job.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main as main_module
from app.config import settings
from app.core import orchestrator as orchestrator_module
from app.core.orchestrator import PipelineOrchestrator, PipelineResult
from app.core.task_queue import TaskQueue, TaskStatus
from app.db.database import Database
from app.db.repositories import DocRepository, RunRepository, RunStageRepository
from app.models.entities import DocCreate, RunCreate, RunStatus, StageName, StageStatus
from app.pipeline.corpus_rules import RULES_PATH_ENV_VAR


@pytest.fixture(autouse=True)
def _reset_pipeline_semaphore():
    """asyncio.Semaphore lazily binds to the event loop of its first
    acquire()/wait(); pytest-asyncio gives each test function its own fresh
    loop, so the process-wide semaphore singleton MUST be reset between
    tests that exercise PipelineOrchestrator.execute() -- otherwise reusing
    it from a later test's loop raises "bound to a different event loop"."""
    orchestrator_module.reset_pipeline_semaphore()
    yield
    orchestrator_module.reset_pipeline_semaphore()


@asynccontextmanager
async def _open_db(tmp_path: Path):
    """Mirrors tests/test_enrich_cache_key.py's _open_db helper: a real
    temp-file SQLite DB (not hand-rolled fakes), so the boot sweep is
    exercised against the real RunRepository/RunStageRepository SQL."""
    db = Database(db_path=tmp_path / "runtime_recovery.db")
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()


# ---------------------------------------------------------------------------
# 1. Boot sweep: PENDING and RUNNING runs both get canceled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_sweep_cancels_orphaned_pending_and_running_runs(tmp_path, monkeypatch):
    async with _open_db(tmp_path) as db:
        monkeypatch.setattr("app.main.db", db)

        doc_repo = DocRepository(db)
        run_repo = RunRepository(db)
        stage_repo = RunStageRepository(db)

        await doc_repo.create(
            "doc-1", DocCreate(source_path="a.pdf", sha256="a", ext=".pdf", size_bytes=1)
        )

        running_run = await run_repo.create(
            RunCreate(doc_id="doc-1", profile="fast", config={}, config_hash="h1")
        )
        await stage_repo.create_all_stages(running_run.run_id)
        await run_repo.update_status(running_run.run_id, RunStatus.RUNNING)
        await stage_repo.update_status(running_run.run_id, StageName.PARSE, StageStatus.RUNNING)

        pending_run = await run_repo.create(
            RunCreate(doc_id="doc-1", profile="fast", config={}, config_hash="h2")
        )
        await stage_repo.create_all_stages(pending_run.run_id)
        # pending_run keeps its default PENDING status from create(); no
        # worker was ever dequeued for it (in-memory queue restarted empty).

        total_canceled = await main_module._cancel_orphan_runs()

        assert total_canceled == 2

        refreshed_running = await run_repo.get(running_run.run_id)
        refreshed_pending = await run_repo.get(pending_run.run_id)
        # Regression pin: pre-existing RUNNING -> CANCELED behavior
        # (already covered with fakes in test_startup_recovery.py).
        assert refreshed_running.status == RunStatus.CANCELED
        # New behavior: PENDING -> CANCELED.
        assert refreshed_pending.status == RunStatus.CANCELED

        running_stage_statuses = {
            s.stage: s.status for s in await stage_repo.list_by_run(running_run.run_id)
        }
        assert running_stage_statuses[StageName.PARSE] == StageStatus.CANCELED

        pending_stage_statuses = {
            s.stage: s.status for s in await stage_repo.list_by_run(pending_run.run_id)
        }
        assert all(
            status == StageStatus.CANCELED for status in pending_stage_statuses.values()
        )


# ---------------------------------------------------------------------------
# 2. Semaphore cap: separate orchestrator instances share one process-wide cap
# ---------------------------------------------------------------------------


class _StubRunRepo:
    """execute() only needs .get()/.update_status() before reaching the
    (stubbed-out) _run_stages seam -- no real DB required."""

    def __init__(self, run: SimpleNamespace):
        self._run = run
        self.status_updates: list[RunStatus] = []

    async def get(self, run_id: str):
        return self._run

    async def update_status(self, run_id: str, status: RunStatus) -> None:
        self.status_updates.append(status)


class _StubDocRepo:
    async def get(self, doc_id: str):
        return SimpleNamespace(
            doc_id=doc_id, source_path="x.pdf", ext=".pdf", sha256="x", size_bytes=1
        )


def _fake_run(run_id: str, doc_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id, doc_id=doc_id, config={}, force_stages=None, use_cache=True
    )


@pytest.mark.asyncio
async def test_execute_caps_concurrency_across_orchestrator_instances(tmp_path, monkeypatch):
    """Simulates queue workers + a direct call sharing one concurrency cap:
    4 separate PipelineOrchestrator instances (mirroring how TaskQueue and
    routes/runs.py's direct path each own a distinct instance -- see
    task_queue.py's self.orchestrator vs. orchestrator.get_orchestrator),
    all constructed with max_concurrent_pipelines=2, executed concurrently.
    Stubs out the real stage work via _run_stages, the seam execute()
    delegates to, with a sleep that records observed concurrency."""
    monkeypatch.setattr(settings, "workspace_path", tmp_path)

    concurrency = {"current": 0, "peak": 0}
    guard = asyncio.Lock()

    async def fake_run_stages(run, ctx, on_stage_complete, on_stage_start):
        async with guard:
            concurrency["current"] += 1
            concurrency["peak"] = max(concurrency["peak"], concurrency["current"])
        try:
            await asyncio.sleep(0.05)
        finally:
            async with guard:
                concurrency["current"] -= 1
        return PipelineResult(
            success=True,
            run_id=ctx.run_id,
            doc_id=ctx.doc_id,
            final_status=RunStatus.SUCCEEDED,
        )

    tasks = []
    for i in range(4):
        run = _fake_run(run_id=f"run-{i}", doc_id=f"doc-{i}")
        orch = PipelineOrchestrator(db=object(), max_concurrent_pipelines=2)
        orch.run_repo = _StubRunRepo(run)
        orch.doc_repo = _StubDocRepo()
        orch._run_stages = fake_run_stages
        tasks.append(asyncio.create_task(orch.execute(run.run_id)))

    results = await asyncio.gather(*tasks)

    assert concurrency["peak"] == 2
    assert len(results) == 4
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_execute_releases_semaphore_on_cancellation(tmp_path, monkeypatch):
    """Cancel/exception paths must release the slot (async with) -- prove it
    by canceling one execute() call mid-flight and confirming a second call
    against the same cap-1 semaphore isn't starved by a stuck permit."""
    monkeypatch.setattr(settings, "workspace_path", tmp_path)

    orch1 = PipelineOrchestrator(db=object(), max_concurrent_pipelines=1)
    run1 = _fake_run(run_id="run-cancel-me", doc_id="doc-1")
    orch1.run_repo = _StubRunRepo(run1)
    orch1.doc_repo = _StubDocRepo()

    started = asyncio.Event()

    async def hang_forever(run, ctx, on_stage_complete, on_stage_start):
        started.set()
        await asyncio.sleep(999)

    orch1._run_stages = hang_forever

    task = asyncio.create_task(orch1.execute(run1.run_id))
    await started.wait()
    task.cancel()
    result1 = await task
    assert result1.final_status == RunStatus.CANCELED

    # If the semaphore permit leaked on cancellation, this second call
    # (same shared semaphore, cap 1) would hang forever.
    orch2 = PipelineOrchestrator(db=object(), max_concurrent_pipelines=1)
    run2 = _fake_run(run_id="run-after", doc_id="doc-2")
    orch2.run_repo = _StubRunRepo(run2)
    orch2.doc_repo = _StubDocRepo()

    async def quick_stages(run, ctx, on_stage_complete, on_stage_start):
        return PipelineResult(
            success=True, run_id=ctx.run_id, doc_id=ctx.doc_id, final_status=RunStatus.SUCCEEDED
        )

    orch2._run_stages = quick_stages

    result2 = await asyncio.wait_for(orch2.execute(run2.run_id), timeout=1.0)
    assert result2.success is True


# ---------------------------------------------------------------------------
# 3. TaskQueue.stop(): cancels workers + running tasks; shutdown ordering
# ---------------------------------------------------------------------------


class _BlockingOrchestrator:
    """Mirrors test_task_queue.py's FakeOrchestrator: blocks until released
    so a test can observe the queue mid-flight before stopping it."""

    def __init__(self):
        self.started: list[str] = []
        self.canceled: list[str] = []
        self.release = asyncio.Event()

    async def execute(self, run_id, on_stage_complete=None, on_stage_start=None):
        self.started.append(run_id)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.canceled.append(run_id)
            raise
        return PipelineResult(
            success=True, run_id=run_id, doc_id=f"doc-{run_id}", final_status=RunStatus.SUCCEEDED
        )


@pytest.mark.asyncio
async def test_task_queue_stop_cancels_workers_and_awaits_running_tasks():
    queue = TaskQueue(db=object(), max_parse_concurrent=1)
    fake = _BlockingOrchestrator()
    queue.orchestrator = fake

    await queue.start()
    await queue.submit("run-1")
    await asyncio.sleep(0.05)  # let the worker dequeue it and start executing

    assert queue.get_active_count() == 1
    worker_tasks = list(queue._worker_tasks)
    running_tasks = list(queue._running_tasks.values())
    assert worker_tasks and all(not t.done() for t in worker_tasks)
    assert running_tasks and all(not t.done() for t in running_tasks)

    await queue.stop()

    assert all(t.done() for t in worker_tasks)
    assert all(t.done() for t in running_tasks)
    assert queue._running_tasks == {}
    assert queue._worker_tasks == []
    assert fake.canceled == ["run-1"]
    assert queue.get_status("run-1").status == TaskStatus.CANCELED


@pytest.mark.asyncio
async def test_shutdown_stops_task_queue_before_disconnecting_db(monkeypatch):
    order: list[str] = []

    class _FakeTaskQueue:
        async def stop(self):
            order.append("stop")

    class _FakeDb:
        async def disconnect(self):
            order.append("disconnect")

    monkeypatch.setattr("app.main.db", _FakeDb())

    await main_module._shutdown_task_queue_and_db(_FakeTaskQueue())

    assert order == ["stop", "disconnect"]


# ---------------------------------------------------------------------------
# 4. conftest.py: SEMARK_CORPUS_RULES_PATH must never be set under pytest
# ---------------------------------------------------------------------------


def test_corpus_rules_override_env_var_is_not_set_during_tests():
    assert RULES_PATH_ENV_VAR not in os.environ
