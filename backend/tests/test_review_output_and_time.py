"""The review writes less and leaves its repairs time (decided 2026-10-15).

5 reports rerun on v20.22 spent 15.6 h in review calls, 81% of it on the first audit of
every page, about half of each call writing the answer. Layout relations no finding can
fail, boxes nothing reads and a sentence per verified block were a third of that writing;
the first audit used almost all the time, so 325 of 338 issues were never tried, and four
calls waited ten minutes for nothing.
"""
import asyncio

import pytest
from pydantic import ValidationError
from test_document_review import AuditModel, config, fixture

from app.pipeline import agentic
from app.pipeline.document_review import (
    UNEXPLAINED_FINDING,
    BlockReview,
    DocumentReview,
    PageReview,
)
from app.pipeline.image_source_review import ImageRelationshipCheck, WordPictureItem
from app.pipeline.review_policy import picture_content_confirmed


def _reading(image_type, relationships):
    return WordPictureItem.model_validate({
        "source_id": "image0", "text": "晴諾米", "complete": True, "observation": "Picture.",
        "image_type": image_type, "relationships": relationships, "relationships_complete": True})


def test_layout_relations_are_not_kept_and_boxes_are_not_needed():
    reading = _reading("diagram", [
        {"relation_id": "r0", "kind": "other", "source_fact": "標誌位於左上角", "certain": True},
        {"relation_id": "r1", "kind": "value_pair", "source_fact": "品牌活躍度為 1.67", "certain": True},
        {"relation_id": "r2", "kind": "arrow", "source_fact": "箭頭由定位指向策略", "direction": "right",
         "bbox": [100, 100, 600, 300], "certain": True}])
    assert [r.relation_id for r in reading.relationships] == ["r1", "r2"]
    # A side on a photo is layout as well.
    photo = _reading("photo", [{"relation_id": "r0", "kind": "spatial_position", "source_fact": "講者站在左側",
                                "horizontal": "left", "certain": True}])
    assert photo.relationships == []
    with pytest.raises(ValidationError, match="bbox"):
        _reading("chart", [{"relation_id": "r0", "kind": "value_pair", "source_fact": "2024 年 50%",
                            "bbox": [600, 100, 100, 300], "certain": True}])


def test_a_matched_check_is_its_quote_and_a_verified_block_needs_no_sentence():
    check = ImageRelationshipCheck(relation_id="r1", status="matched", delivered_quote="品牌活躍度 1.67")
    finding = BlockReview(block_id="b000412", status="verified", image_relationship_checks=[check])
    assert (check.observation, finding.observation) == ("", "")
    PageReview(blocks=[finding], source_complete=True, reading_order_correct=True,
               observation="Page matches.", issues=[])


def test_an_unexplained_failure_still_holds_the_picture_back():
    finding = BlockReview(block_id="b000412", status="needs_review")
    assert finding.observation == UNEXPLAINED_FINDING
    page = {"images": [{"path": "p.png"}], "review": {"blocks": [finding.model_dump()], "issues": []}}
    assert not picture_content_confirmed(page, "b000412", {"b000412"}, "晴諾米")


@pytest.mark.asyncio
async def test_a_review_call_stops_at_five_minutes_whatever_the_settings(tmp_path, monkeypatch):
    assert agentic.AGENTIC_CALL_TIMEOUT_SECONDS == 300
    _, source = fixture(tmp_path)

    class Stalled(AuditModel):
        async def query(self, *args):
            await asyncio.sleep(5)

    settings = config()
    settings.review_vlm.request_timeout_seconds = 600
    owner = DocumentReview(settings, Stalled(), Stalled())
    owner.state_path = tmp_path / "state.json"
    owner.state = {"usage": {"seconds": 0, "model_calls": 0, "tokens": 0}, "calls": []}
    owner.limits = settings.agentic
    monkeypatch.setattr(agentic, "AGENTIC_CALL_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(TimeoutError):
        await owner._call("audit_page", {"block_id": "p0"}, source, PageReview)
    assert owner.state["calls"][0]["timeout_seconds"] == 0.05


@pytest.mark.asyncio
async def test_the_first_audit_leaves_a_share_of_the_time_to_repairs(tmp_path):
    ir, source = fixture(tmp_path, pages=8)

    class Slow(AuditModel):
        async def query(self, *args):
            await asyncio.sleep(0.2)
            return await super().query(*args)

    async def rebuild(_):
        pass

    review = DocumentReview(config(max_seconds=2.4, recovery_reserve_share=0.5), Slow(), Slow())
    await review.run(ir, tmp_path, source_path=source, rebuild=rebuild)
    calls = review.state["calls"]
    first = [c for c in calls if c["phase"] == "audit"]
    # Eight pages at 0.2 s need 1.6 s: the first audit stops at its 1.2 s, and the pages it
    # left are audited once the repairs found nothing to do.
    assert len(first) < 8 and sum(c["duration_seconds"] for c in first) <= 1.2 + 0.1
    assert any(c["phase"] == "final_review" for c in calls)
    assert {p["status"] for p in review.state["pages"].values()} == {"verified"}
