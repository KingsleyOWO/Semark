"""One time limit per document, earlier stages included (decided 2026-10-06).

The review gets the six hours minus what parsing, drafting and packaging already
took, so each stage records its duration; a resumed run counts the stages it
completed before.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import PipelineConfig
from app.core.orchestrator import PipelineContext, PipelineOrchestrator
from app.models.entities import StageName


@pytest.mark.asyncio
async def test_each_stage_records_its_duration_and_the_review_sees_the_time_already_spent(tmp_path):
    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.run_repo, orchestrator.stage_repo = AsyncMock(), AsyncMock()
    ctx = PipelineContext(doc_id="doc", run_id="run", config=PipelineConfig(), run_path=tmp_path,
                          checkpoint={"completed": ["parse"]}, stage_stats={"parse": {"duration_seconds": 600.0}})
    seen = {}

    async def stage(ctx, use_cache=True):
        return {"done": True}

    async def chunk(ctx, use_cache=True):
        seen["elapsed"] = orchestrator._elapsed_before_review(ctx)
        return {"done": True}
    orchestrator._run_parse = orchestrator._run_normalize = orchestrator._run_enrich = stage
    orchestrator._run_package = stage
    orchestrator._run_chunk = chunk

    result = await orchestrator._run_stages(SimpleNamespace(force_stages=None, use_cache=True, doc_id="doc"),
                                            ctx, None, None)

    assert result.success
    assert all("duration_seconds" in ctx.stage_stats[s.value]
               for s in (StageName.NORMALIZE, StageName.ENRICH, StageName.PACKAGE, StageName.CHUNK))
    assert 600.0 <= seen["elapsed"] < 610


@pytest.mark.asyncio
async def test_the_chunk_stage_starts_the_review_with_the_time_already_spent(tmp_path, monkeypatch):
    import time

    from test_document_review import fixture

    from app.core import orchestrator as module

    seen = {}

    async def chunked(self, **kwargs):
        return SimpleNamespace(success=True, stats={})

    async def review(self, ir, run_path, **kwargs):
        seen.update(kwargs)
        return ir, {"summary": {}, "usage": {}}
    monkeypatch.setattr(module.ChunkStage, "run", chunked)
    monkeypatch.setattr(module.DocumentReview, "run", review)
    monkeypatch.setattr(module, "save_document_ir", lambda ir, path: None)
    monkeypatch.setattr(module, "finalize_delivery", lambda ir, path, config: {})
    ctx = PipelineContext(doc_id="doc", run_id="run", config=PipelineConfig(), run_path=tmp_path,
                          stage_stats={"enrich": {"duration_seconds": 3000.0}})
    ctx.document_ir, _ = fixture(tmp_path)
    ctx.stage_started = time.monotonic()

    await object.__new__(PipelineOrchestrator)._run_chunk(ctx)

    assert 3000.0 <= seen["elapsed_seconds"] < 3010
