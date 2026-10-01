"""Control-flow tests; model accuracy is measured separately on real documents."""
import asyncio
import json

import pytest
from PIL import Image

from app.config import AgenticConfig, PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.agentic import AgenticRepair, visual_issues


def schedule(tmp_path):
    path = tmp_path / "assets/pages/p0000.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (600, 500), "white").save(path)
    body = "<table><tr><td>工作</td>" + "".join(f"<td>{m}月</td>" for m in range(1, 7)) + "</tr>"
    body += "<tr><td>市場調查</td>" + "<td></td>" * 6 + "</tr></table>"
    return DocumentIR(doc_id="doc", run_id="run", page_kind="physical",
                      source=SourceInfo(path="source.pdf", ext="pdf", sha256="abc", size_bytes=1),
                      engine=EngineInfo(backend="pipeline", method="auto"),
                      pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
                      blocks=[Block(block_id="table", type=BlockType.TABLE, page_idx=0,
                                    bbox_norm=[0, 0, 1000, 1000], payload={"table_body": body})])


class Model:
    def __init__(self, supported=True, invalid=False, cancel=False):
        self.calls = []
        self.supported = supported
        self.invalid = invalid
        self.cancel = cancel

    async def query(self, task, payload, image, schema):
        self.calls.append(task)
        if self.cancel:
            self.cancel = False
            raise asyncio.CancelledError()
        if task == "decide":
            return {"action": "shell" if self.invalid else "reconstruct_table",
                    "block_id": payload["block_id"], "reason": "月份格有視覺資訊。"}, 10
        if task == "repair":
            return {"kind": "schedule", "facts": [{
                "subject": "市場調查", "relation": "執行月份", "object": "1月至3月",
                "source_cells": payload["required_source_cells"],
            }], "needs_review": False}, 20
        return {"observations": [{
                    "fact_indices": list(range(len(payload["candidate"]["facts"]))) if payload["candidate"] else [],
                    "source_cells": payload["required_source_cells"],
                    "observation": "第一列色塊對齊1月至3月。",
                }] if self.supported else [], "complete": self.supported,
                "missing_source_cells": [] if self.supported else payload["required_source_cells"],
                "needs_review": not self.supported, "existing_output_matches": payload["mode"] == "keep"}, 10


def config(mode="apply", **kwargs):
    return PipelineConfig(agentic=AgenticConfig(mode=mode, **kwargs))


def test_visual_risk_detects_gantt_that_has_normal_grid(tmp_path):
    assert visual_issues(schedule(tmp_path))[0]["risk"] == "schedule_cells_missing_visual_relations"


@pytest.mark.asyncio
async def test_verified_repair_preserves_source_and_resumes_without_model_calls(tmp_path):
    ir = schedule(tmp_path)
    original = ir.to_dict()
    fake = Model()
    output, report = await AgenticRepair(config(), fake, fake).run(ir, tmp_path)
    assert report["accepted"] == 1
    assert output.blocks[0].payload["table_body"] == ir.blocks[0].payload["table_body"]
    assert "1月至3月" in output.blocks[0].payload["verified_semantics"]
    assert ir.to_dict() == original
    count = len(fake.calls)
    replay, report = await AgenticRepair(config(), fake, fake).run(ir, tmp_path)
    assert len(fake.calls) == count
    assert replay.blocks[0].payload["verified_semantics"] == output.blocks[0].payload["verified_semantics"]


@pytest.mark.asyncio
async def test_shadow_mode_never_modifies_delivered_ir(tmp_path):
    ir = schedule(tmp_path)
    fake = Model()
    output, report = await AgenticRepair(config("shadow"), fake, fake).run(ir, tmp_path)
    assert report["accepted"] == 1
    assert output is ir
    assert "verified_semantics" not in ir.blocks[0].payload
    assert (tmp_path / "agentic/candidate_ir.json").exists()


@pytest.mark.asyncio
async def test_failed_verification_keeps_original_and_stops_at_attempt_limit(tmp_path):
    ir = schedule(tmp_path)
    fake = Model(supported=False)
    output, report = await AgenticRepair(config(max_attempts_per_block=2), fake, fake).run(ir, tmp_path)
    assert report["accepted"] == 0
    assert report["usage"]["actions"] == 2
    assert len(fake.calls) == 6
    assert output.blocks[0].payload["agentic_needs_review"]
    assert "verified_semantics" not in output.blocks[0].payload


@pytest.mark.asyncio
async def test_invalid_tool_cannot_execute_and_calls_are_bounded(tmp_path):
    fake = Model(invalid=True)
    output, report = await AgenticRepair(config(max_model_calls=2), fake, fake).run(schedule(tmp_path), tmp_path)
    assert fake.calls == ["repair", "decide"]
    assert report["accepted"] == 0
    assert report["usage"]["model_calls"] == 2
    assert output.blocks[0].payload["agentic_needs_review"]


@pytest.mark.asyncio
async def test_zero_budget_calls_no_model(tmp_path):
    fake = Model()
    _, report = await AgenticRepair(config(max_actions=0), fake, fake).run(schedule(tmp_path), tmp_path)
    assert not fake.calls
    assert report["unresolved"] == 1


@pytest.mark.asyncio
async def test_interruption_retains_attempt_and_call_budget(tmp_path):
    ir = schedule(tmp_path)
    fake = Model(cancel=True)
    with pytest.raises(asyncio.CancelledError):
        await AgenticRepair(config(), fake, fake).run(ir, tmp_path)
    state = json.loads((tmp_path / "agentic/state.json").read_text())
    assert state["usage"]["model_calls"] == 1
    assert state["usage"]["actions"] == 1
    _, report = await AgenticRepair(config(), fake, fake).run(ir, tmp_path)
    assert report["accepted"] == 1
    assert report["usage"]["actions"] == 2
    assert report["usage"]["model_calls"] == 4


@pytest.mark.asyncio
async def test_changed_image_invalidates_accepted_attempt(tmp_path):
    ir = schedule(tmp_path)
    fake = Model()
    await AgenticRepair(config(), fake, fake).run(ir, tmp_path)
    Image.new("RGB", (600, 500), "blue").save(tmp_path / "assets/pages/p0000.png")
    await AgenticRepair(config(), fake, fake).run(ir, tmp_path)
    assert len(fake.calls) == 6
