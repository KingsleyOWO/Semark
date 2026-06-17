from types import SimpleNamespace

import pytest

from app.main import _cancel_orphan_running_runs
from app.models.entities import RunStatus, StageName, StageStatus


@pytest.mark.asyncio
async def test_cancel_orphan_running_runs_marks_runs_and_stages_canceled(monkeypatch):
    run = SimpleNamespace(run_id="run-orphan")
    running_stage = SimpleNamespace(status=StageStatus.RUNNING, stage=StageName.ENRICH)
    done_stage = SimpleNamespace(status=StageStatus.SUCCEEDED, stage=StageName.PARSE)
    calls = {"stage_updates": [], "run_updates": []}

    class FakeRunRepository:
        def __init__(self, db):
            pass

        async def list_all(self, status=None, limit=100, offset=0, exclude_run_ids=None):
            assert status == RunStatus.RUNNING
            return [run]

        async def update_status(self, run_id, status):
            calls["run_updates"].append((run_id, status))

    class FakeRunStageRepository:
        def __init__(self, db):
            pass

        async def list_by_run(self, run_id):
            assert run_id == "run-orphan"
            return [done_stage, running_stage]

        async def update_status(self, run_id, stage, status, error=None, stats=None):
            calls["stage_updates"].append((run_id, stage, status, error))

    monkeypatch.setattr("app.main.RunRepository", FakeRunRepository)
    monkeypatch.setattr("app.main.RunStageRepository", FakeRunStageRepository)

    assert await _cancel_orphan_running_runs() == 1
    assert calls["run_updates"] == [("run-orphan", RunStatus.CANCELED)]
    assert calls["stage_updates"] == [
        (
            "run-orphan",
            StageName.ENRICH,
            StageStatus.CANCELED,
            {"message": "Backend restarted before this stage completed"},
        )
    ]


def test_apply_runtime_settings_includes_vlm_timeout_and_decode_params():
    from app.api.routes.runs import _apply_runtime_settings

    merged = {
        "mineru": {},
        "vlm": {"decode_params": {}},
        "review_vlm": {"decode_params": {}},
    }
    mineru = {
        "backend": "pipeline",
        "lang": "ch",
        "table": True,
        "api_url": None,
        "vlm_url": None,
        "vlm_model_name": None,
        "model_source": "huggingface",
        "pdf_render_timeout": 300,
        "pdf_render_threads": 2,
        "table_merge_enable": True,
        "processing_window_size": None,
        "api_max_concurrent_requests": None,
    }
    vlm = {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen3-vl",
        "api_mode": "ollama",
        "request_timeout_seconds": 123.0,
        "decode_params": {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 2048,
            "repetition_penalty": 1.05,
        },
    }
    review_vlm = {
        **vlm,
        "model": "qwen3-vl-review",
        "request_timeout_seconds": 456.0,
        "decode_params": {
            **vlm["decode_params"],
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    }

    _apply_runtime_settings(merged, mineru, vlm, review_vlm)

    assert merged["vlm"]["request_timeout_seconds"] == 123.0
    assert merged["vlm"]["decode_params"]["top_k"] == 20
    assert merged["vlm"]["decode_params"]["repetition_penalty"] == 1.05
    assert merged["vlm"]["image_mode"] == "data_uri"
    assert merged["review_vlm"]["model"] == "qwen3-vl-review"
    assert merged["review_vlm"]["request_timeout_seconds"] == 456.0
    assert merged["review_vlm"]["decode_params"]["temperature"] == 0.0
    assert merged["review_vlm"]["decode_params"]["max_tokens"] == 4096
    assert merged["review_vlm"]["image_mode"] == "data_uri"
