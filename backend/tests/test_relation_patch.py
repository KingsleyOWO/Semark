"""A picture that fails only because its delivery omits a relationship gets that sentence, not a re-read.

Lumio v20.1: a product page's delivery listed the three size options but never
said 30cm is the selected one; a social-media slide lacked the value paired with
Average Likes. A full repair re-reads every pixel (8-20 calls) and fails the
same way; adding the relationship the audit saw takes one per-image audit. A
rejected patch leaves the delivery untouched, and the next attempt is a full
repair, never the same patch again.
"""
import json

import pytest
from PIL import Image
from test_document_recovery import RecoveryModel, cfg, setup

from app.models.document_ir import Block, BlockType
from app.pipeline.image_recovery import relation_patch_statements
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.package import PackageStage

FACT = "The 30cm (167 pieces) option is selected with a black border."
READING = {"source_id": "image0", "text": "30cm (167 pieces)\n40cm (388 pieces)", "complete": True,
           "observation": "Product page.", "kind": "transcription", "image_type": "document",
           "relationships": [{"relation_id": "r0", "kind": "table_relation", "source_fact": FACT,
                              "bbox": [0, 0, 1000, 500], "certain": True}],
           "relationships_complete": True}


def _record(status="missing", kind="image_relationship_mismatch"):
    return {"page_idx": 0, "status": "needs_review", "word_image_readings": {"picture": READING},
            "review": {"blocks": [{"block_id": "picture", "status": "needs_review", "observation": "Missing relation.",
                                   "image_relationship_checks": [{"relation_id": "r0", "status": status,
                                                                  "delivered_quote": "", "observation": "Not stated."}]}]},
            "image_issues": [{"block_id": "picture", "failure_kind": kind, "detail": "Image relationship r0 remains missing."}]}


def test_only_a_picture_failing_on_omitted_relationships_gets_a_patch():
    assert relation_patch_statements({"pages": {"0": _record()}}, 0, "picture") == ["圖片關係：" + FACT]
    assert relation_patch_statements({"pages": {"0": _record(status="contradicted")}}, 0, "picture") == []
    assert relation_patch_statements({"pages": {"0": _record(kind="reading_disagreement")}}, 0, "picture") == []


class PatchModel(RecoveryModel):
    async def query(self, task, payload, image, schema):
        self.tasks.append(task)
        if task == "recovery_plan":
            return {"action": "repair_image", "reason": "The delivery omits a relationship."}, 10
        if task in {"audit_image_parts", "audit_page"}:
            blocks = []
            for bid in payload["required_block_ids"]:
                checks = ([{"relation_id": "r0", "status": "matched", "delivered_quote": "圖片關係：" + FACT,
                            "observation": "Selected option stated."}] if bid.endswith("_image_r0") else [])
                blocks.append({"block_id": bid, "status": "verified", "context_complete": True,
                               "observation": "Matches the source.", "image_relationship_checks": checks})
            return {"blocks": blocks, "source_complete": True, "reading_order_correct": True,
                    "observation": "Source and delivery compared.", "issues": []}, 10
        raise AssertionError("a relation patch reads no pixels: " + task)


@pytest.mark.asyncio
async def test_a_relation_patch_keeps_the_delivery_and_adds_the_missing_sentence(tmp_path):
    model = PatchModel()
    config = cfg(max_recovery_actions=1)
    config.agentic.max_seconds = 2000
    ir, owner, runner = await setup(tmp_path, model, config)
    picture = tmp_path / "picture.png"
    Image.new("RGB", (400, 300), "white").save(picture)
    ir.blocks.append(Block(block_id="picture", type=BlockType.IMAGE, page_idx=0, reading_order=1,
                           payload={"img_path": str(picture)}))
    runner.inventory.mapping["picture"] = [0]
    original = runner.inventory.directory / "picture.png"
    original.write_bytes(picture.read_bytes())
    runner.inventory.word_images["picture"] = original
    runner.inventory.image_positions["picture"] = [{"page_idx": 0, "bbox": [0, 200, 1000, 900]}]
    owner.state["pages"]["0"] = _record()
    from app.pipeline.document_recovery import render_candidate
    runner.renderer = render_candidate

    async def rebuild(candidate):
        settings = owner.config.model_copy(deep=True)
        settings.enrich.enable_vlm = False
        settings.package.enable_semantic_repair = False
        assert (await PackageStage(settings).run(candidate.doc_id, candidate.run_id, candidate, tmp_path,
                                                 parse_cache_path=tmp_path)).success
        assert (await ChunkStage(settings).run(candidate.doc_id, candidate.run_id, candidate, tmp_path)).success
    runner.rebuild = rebuild
    (tmp_path / "outputs").mkdir(exist_ok=True)
    (tmp_path / "outputs/enrichments.jsonl").write_text(json.dumps({
        "block_id": "picture", "kind": "figure_description", "output": {
            "semantic_caption": "此圖為拼圖商品頁，列出三種尺寸選項與價格。",
            "facts": ["商品提供 30cm、40cm、50cm 三種尺寸。"], "all_text": ["30cm (167 pieces)", "40cm (388 pieces)"]}},
        ensure_ascii=False))
    await rebuild(ir)

    result = await runner.run(ir)

    step = runner.state["steps"][0]
    assert (step["action"], step.get("method"), step["status"]) == ("repair_image", "relation_patch", "applied")
    kept = next(b for b in result.blocks if b.block_id.endswith("_image_k0"))
    added = next(b for b in result.blocks if b.block_id.endswith("_image_r0"))
    assert "商品提供 30cm、40cm、50cm 三種尺寸。" in kept.payload["text"]
    assert added.payload["text"] == "圖片關係：" + FACT
    assert not {"read_image_parts", "verify_image_parts", "resolve_image_evidence"} & set(model.tasks)


def test_after_a_patch_the_next_attempt_is_a_full_repair_not_the_patch_again(tmp_path):
    from types import SimpleNamespace

    from app.pipeline.document_recovery import RecoveryController
    from app.pipeline.evidence import EvidenceStore
    from app.pipeline.recovery_models import RecoveryDecision

    runner = object.__new__(RecoveryController)
    runner.owner = SimpleNamespace(config=cfg(), state={"pages": {"0": _record()}})
    runner.inventory = SimpleNamespace(mapping={"picture": [0]})
    runner.evidence = EvidenceStore(tmp_path, "source")
    runner.state = {"steps": []}
    decision = RecoveryDecision(action="repair_image", block_id="picture", reason="Repair picture")
    assert runner.repair_method("picture") == "relation_patch"

    patch = {"action": "repair_image", "model_role": "primary", "block_id": "picture", "method": "relation_patch"}
    runner.state["steps"].append(patch)
    assert runner.repair_method("picture") == "full"
    assert runner.select_strategy(decision, [patch], ["repair_image", "defer"]).action == "repair_image"

    full = {"action": "repair_image", "model_role": "primary", "block_id": "picture"}
    assert runner.select_strategy(decision, [patch, full], ["repair_image", "defer"]).action == "defer"
