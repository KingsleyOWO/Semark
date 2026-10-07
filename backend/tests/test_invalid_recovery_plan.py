"""A plan outside the issue's scope is feedback for the next plan, not the end of the issue.

Spot check 2026-10-07: nine page issues of one report ended as planner_failed because
the planner chose a picture that had already used both full re-reads. The rejected
plan is recorded with what was allowed and counts as an attempt, so the planner can
choose again but cannot loop.
"""
import pytest
from test_document_recovery import RecoveryModel, setup


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
