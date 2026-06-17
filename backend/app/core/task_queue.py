"""
Async task queue for pipeline execution.

Provides background task management with support for:
- Submitting pipeline runs
- Cancellation
- Status tracking
- Resume from checkpoint (future)
"""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from app.core.orchestrator import PipelineOrchestrator, PipelineResult
from app.db.database import Database
from app.models.entities import RunStatus, StageName


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class TaskInfo:
    """Information about a queued task."""

    run_id: str
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: PipelineResult | None = None
    current_stage: StageName | None = None


class TaskQueue:
    """
    Async FIFO task queue for pipeline execution.

    submit() only enqueues run IDs. A fixed worker pool consumes queued runs and
    executes them, which keeps pending/running status aligned with real work.
    """

    def __init__(
        self,
        db: Database,
        max_parse_concurrent: int = 2,
        max_enrich_gpu_concurrent: int = 1,
        max_enrich_http_concurrent: int = 4,
    ):
        self.db = db
        self.orchestrator = PipelineOrchestrator(db)

        self._task_info: dict[str, TaskInfo] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._callbacks: dict[str, Callable[[PipelineResult], Coroutine] | None] = {}

        self.pool_config = {
            "parse": max_parse_concurrent,
            "enrich_gpu": max_enrich_gpu_concurrent,
            "enrich_http": max_enrich_http_concurrent,
        }

        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._shutdown = False

    async def start(self) -> None:
        """Start queue workers."""
        if self._worker_tasks and any(not task.done() for task in self._worker_tasks):
            return

        self._shutdown = False
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(worker_id))
            for worker_id in range(self.pool_config["parse"])
        ]

    async def stop(self) -> None:
        """Stop queue workers and cancel in-flight work."""
        self._shutdown = True

        for task in list(self._running_tasks.values()):
            if not task.done():
                task.cancel()

        for task in self._worker_tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._running_tasks.clear()
        self._worker_tasks.clear()

    async def submit(
        self,
        run_id: str,
        on_complete: Callable[[PipelineResult], Coroutine] | None = None,
    ) -> TaskInfo:
        """Submit a pipeline run for FIFO background execution."""
        existing = self._task_info.get(run_id)
        if existing and existing.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return existing

        info = TaskInfo(
            run_id=run_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
        )
        self._task_info[run_id] = info
        self._callbacks[run_id] = on_complete
        await self._queue.put(run_id)
        return info

    async def _worker_loop(self, worker_id: int) -> None:
        """Consume queued run IDs until shutdown."""
        while not self._shutdown:
            try:
                run_id = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                info = self._task_info.get(run_id)
                if not info or info.status == TaskStatus.CANCELED:
                    continue

                run_task = asyncio.create_task(self._execute_run(run_id))
                self._running_tasks[run_id] = run_task
                await run_task
            except asyncio.CancelledError:
                task = self._running_tasks.get(run_id)
                if task and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise
            finally:
                self._running_tasks.pop(run_id, None)
                self._callbacks.pop(run_id, None)
                self._queue.task_done()

    async def _execute_run(self, run_id: str) -> None:
        """Execute one pipeline run and keep in-memory task status current."""
        info = self._task_info.get(run_id)
        if not info:
            return

        info.status = TaskStatus.RUNNING
        info.started_at = datetime.now()
        info.finished_at = None

        try:
            async def on_stage_start(stage: StageName):
                info.current_stage = stage

            async def on_stage_complete(stage: StageName, stats: dict):
                info.current_stage = stage

            result = await self.orchestrator.execute(
                run_id=run_id,
                on_stage_complete=on_stage_complete,
                on_stage_start=on_stage_start,
            )

            info.result = result
            info.finished_at = datetime.now()
            info.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED

            on_complete = self._callbacks.get(run_id)
            if on_complete:
                await on_complete(result)

        except asyncio.CancelledError:
            info.status = TaskStatus.CANCELED
            info.finished_at = datetime.now()
            raise

        except Exception as exc:
            info.status = TaskStatus.FAILED
            info.finished_at = datetime.now()
            info.result = PipelineResult(
                success=False,
                run_id=run_id,
                doc_id="",
                final_status=RunStatus.FAILED,
                error=str(exc),
            )

    async def cancel(self, run_id: str) -> bool:
        """Cancel a pending or running task."""
        info = self._task_info.get(run_id)
        if not info or info.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return False

        info.status = TaskStatus.CANCELED
        info.finished_at = datetime.now()

        task = self._running_tasks.get(run_id)
        if task and not task.done():
            task.cancel()

        return True

    def get_status(self, run_id: str) -> TaskInfo | None:
        """Get task status."""
        return self._task_info.get(run_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all task info."""
        return list(self._task_info.values())

    def get_active_count(self) -> int:
        """Get count of active running tasks."""
        return sum(
            1 for info in self._task_info.values()
            if info.status == TaskStatus.RUNNING
        )

    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        return sum(
            1 for info in self._task_info.values()
            if info.status == TaskStatus.PENDING
        )


_task_queue: TaskQueue | None = None


async def get_task_queue(db: Database) -> TaskQueue:
    """Get or create the global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(db)
        await _task_queue.start()
    return _task_queue
