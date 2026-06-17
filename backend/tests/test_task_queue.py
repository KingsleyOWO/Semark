import asyncio

import pytest

from app.core.orchestrator import PipelineResult
from app.core.task_queue import TaskQueue, TaskStatus
from app.models.entities import RunStatus, StageName


class FakeOrchestrator:
    def __init__(self):
        self.started: list[str] = []
        self.release = asyncio.Event()

    async def execute(self, run_id, on_stage_complete=None, on_stage_start=None):
        self.started.append(run_id)
        if on_stage_start:
            await on_stage_start(StageName.PARSE)
        await self.release.wait()
        return PipelineResult(
            success=True,
            run_id=run_id,
            doc_id=f"doc-{run_id}",
            final_status=RunStatus.SUCCEEDED,
        )


@pytest.mark.asyncio
async def test_task_queue_uses_fifo_workers_and_keeps_backlog_pending():
    queue = TaskQueue(db=object(), max_parse_concurrent=1)  # type: ignore[arg-type]
    fake = FakeOrchestrator()
    queue.orchestrator = fake  # type: ignore[assignment]

    await queue.start()
    try:
        first = await queue.submit("run-1")
        second = await queue.submit("run-2")
        await asyncio.sleep(0.05)

        assert fake.started == ["run-1"]
        assert first.status == TaskStatus.RUNNING
        assert second.status == TaskStatus.PENDING
        assert queue.get_active_count() == 1
        assert queue.get_pending_count() == 1

        fake.release.set()
        await asyncio.sleep(0.05)

        assert fake.started == ["run-1", "run-2"]
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_task_queue_cancel_pending_task_skips_execution():
    queue = TaskQueue(db=object(), max_parse_concurrent=1)  # type: ignore[arg-type]
    fake = FakeOrchestrator()
    queue.orchestrator = fake  # type: ignore[assignment]

    await queue.start()
    try:
        await queue.submit("run-1")
        pending = await queue.submit("run-2")
        await asyncio.sleep(0.05)

        assert await queue.cancel("run-2") is True
        assert pending.status == TaskStatus.CANCELED

        fake.release.set()
        await asyncio.sleep(0.05)

        assert fake.started == ["run-1"]
    finally:
        await queue.stop()
