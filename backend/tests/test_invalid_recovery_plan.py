"""A plan outside the issue's scope is feedback for the next plan, not the end of the issue.

Spot check 2026-10-07: nine page issues of one report ended as planner_failed because
the planner chose a picture that had already used both full re-reads. The rejected
plan is recorded with what was allowed and counts as an attempt, so the planner can
choose again but cannot loop.
"""
import pytest
from PIL import Image
from test_document_recovery import RecoveryModel, setup

from app.models.document_ir import Block, BlockType


@pytest.mark.asyncio
async def test_an_out_of_scope_plan_is_rejected_and_the_planner_plans_again(tmp_path):
    model = RecoveryModel(actions=["resolve_source", "read_regions"])
    ir, owner, runner = await setup(tmp_path, model)

    result = await runner.run(ir)

    steps = runner.state["steps"]
    assert [(s["action"], s["status"]) for s in steps] == [("invalid_plan", "rejected"), ("read_regions", "applied")]
    assert steps[0]["planned_action"] == "resolve_source" and steps[0]["result"]
    assert runner.state["issues"]["page:0"]["status"] != "planner_failed"
    assert "Revenue 100" in (tmp_path / "outputs/documents/main.md").read_text()
    assert result is not ir


@pytest.mark.asyncio
async def test_repeated_invalid_plans_stop_at_the_attempt_limit(tmp_path):
    model = RecoveryModel(action="resolve_source")
    ir, owner, runner = await setup(tmp_path, model)

    result = await runner.run(ir)

    assert result is ir
    limit = owner.limits.max_recovery_attempts_per_issue
    assert [s["action"] for s in runner.state["steps"]] == ["invalid_plan"] * limit
    assert model.tasks.count("recovery_plan") == limit
    assert runner.state["issues"]["page:0"]["status"] == "attempt_limit"


@pytest.mark.asyncio
async def test_a_page_issue_only_offers_the_pictures_its_review_did_not_accept(tmp_path):
    # Spot check 2026-10-07: the misread picture had no re-read left, so the planner was
    # told to pick a listed target and re-read the accepted picture beside it: 6.5 minutes,
    # and the misread word stayed.
    class Planner(RecoveryModel):
        def __init__(self):
            super().__init__()
            self.plans = []

        async def query(self, task, payload, image, schema):
            if task != "recovery_plan":
                return await super().query(task, payload, image, schema)
            self.tasks.append(task)
            self.plans.append(payload)
            if len(self.plans) == 1:
                return {"action": "repair_image", "block_id": "accepted", "reason": "Repair the listed picture."}, 10
            return {"action": "defer", "reason": "The misread picture has no re-read left."}, 10

    model = Planner()
    ir, owner, runner = await setup(tmp_path, model)
    for name in ("accepted", "misread"):
        picture = runner.inventory.directory / f"{name}.png"
        Image.new("RGB", (100, 80), "green").save(picture)
        ir.blocks.append(Block(block_id=name, type=BlockType.IMAGE, page_idx=0, payload={"img_path": str(picture)}))
        runner.inventory.word_images[name] = picture
        runner.inventory.mapping[name] = [0]
    for role in ("primary", "review"):
        runner.state["steps"].append({"key": role, "issue_id": "image:misread", "action": "repair_image", "block_id": "misread",
                                      "model_role": role, "page_idx": 0, "status": "rejected"})
    owner.state["pages"]["0"]["review"]["blocks"] = [
        {"block_id": "accepted", "status": "verified", "context_complete": True},
        {"block_id": "misread", "status": "needs_review", "context_complete": False}]

    result = await runner.run(ir, issue_ids={"page:0"})

    assert result is ir
    assert [plan["image_targets"] for plan in model.plans] == [[], []]
    assert "repair_image" not in model.plans[0]["tools"]
    steps = [s for s in runner.state["steps"] if s["issue_id"] == "page:0"]
    assert [(s["action"], s["planned_action"], s["block_id"]) for s in steps] == [("invalid_plan", "repair_image", "accepted")]
    assert runner.state["issues"]["page:0"]["status"] == "deferred"
