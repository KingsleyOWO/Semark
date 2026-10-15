"""A picture the audits fault gets those exact spans fixed, with words an independent reader saw.

Validation v20.6: a delivery patch added the missing footer relation correctly, then
the re-audit failed it on older faults in the kept text (an invented 「橡皮」, page
66 drafted as 06, a 「最佳賣家」 gloss), and the next method never ran: the step key
ignored the method, so it was "already attempted". A full re-read failed on
text-heavy pictures while merging evidence. A targeted edit names quotes; the
system applies them only when they delete description wording, or bring in
wording PaddleOCR or the blind second reading contain. A transcription line is
never deleted, and changes only into a line OCR reads.
"""
import json

import pytest
from PIL import Image
from test_delivery_patch import FACT, READING, PatchModel, _record
from test_document_recovery import cfg, setup

from app.models.document_ir import Block, BlockType
from app.pipeline.image_edits import ImageEdit, apply_edits
from app.pipeline.image_recovery import next_image_method
from app.pipeline.image_source_review import WordPictureItem
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.package import PackageStage

DELIVERED = ("此為晴諾米品牌指南第 06 頁，展示名片、信封、筆與橡皮等品牌物件。\n"
             "40cm 標示為 BEST SELLER（最佳賣家/暢銷）。\n"
             "06\nApplication\nLUMIO Brand Guidelines | 87")
TRANSCRIPTION = {"06", "Application", "LUMIO Brand Guidelines | 87"}
OCR = ["06", "Application", "BEST SELLER", "LUMIO Brand Guidelines | 07", "66"]


def _reading(text="06\nApplication\nLUMIO Brand Guidelines | 07", relationships=()):
    return WordPictureItem.model_validate({
        "source_id": "image0", "text": text, "complete": True, "observation": "Slide.", "image_type": "document",
        "relationships": list(relationships), "relationships_complete": True})


def _apply(*edits, ocr=OCR, reading=None):
    return apply_edits(DELIVERED, [ImageEdit.model_validate({"reason": "Audit finding.", **e}) for e in edits],
                       transcription=TRANSCRIPTION, ocr_lines=ocr, reading=reading or _reading())


def test_description_wording_the_picture_does_not_show_is_deleted():
    text, statements, applied, rejected = _apply({"op": "delete", "quote": "與橡皮"},
                                                 {"op": "delete", "quote": "（最佳賣家/暢銷）"})

    assert "筆等品牌物件" in text and "BEST SELLER。" in text
    assert (len(applied), rejected, statements) == (2, [], [])


def test_a_description_correction_uses_only_wording_an_independent_reader_saw():
    text, _, applied, rejected = _apply({"op": "replace", "quote": "第 06 頁，", "replacement": "第 99 頁，"},
                                        {"op": "replace", "quote": "名片、信封", "replacement": "名片、明信片"},
                                        {"op": "replace", "quote": "第 06 頁", "replacement": "第 66 頁"})

    assert "第 66 頁" in text and "名片、信封" in text
    assert len(applied) == 1 and len(rejected) == 2


def test_a_transcription_line_changes_only_into_the_line_ocr_reads():
    text, _, applied, _ = _apply({"op": "replace", "quote": "LUMIO Brand Guidelines | 87",
                                  "replacement": "LUMIO Brand Guidelines | 07"})
    assert text.endswith("LUMIO Brand Guidelines | 07") and len(applied) == 1

    # The auditor's own reading, which OCR does not back, never rewrites printed text.
    text, _, applied, rejected = _apply({"op": "replace", "quote": "| 87", "replacement": "| 67"})
    assert text == DELIVERED and not applied and len(rejected) == 1


def test_a_transcription_line_is_never_deleted():
    text, _, applied, rejected = _apply({"op": "delete", "quote": "Application"}, {"op": "delete", "quote": "| 87"})

    assert text == DELIVERED and not applied and len(rejected) == 2


def test_a_quote_must_name_one_exact_delivered_span():
    text, _, applied, rejected = _apply({"op": "delete", "quote": "06"}, {"op": "delete", "quote": "與鉛筆"})

    assert text == DELIVERED and not applied and len(rejected) == 2


def test_a_misreading_the_draft_repeats_is_fixed_everywhere():
    # 晴嵐 on v20.23: 12 of 13 refused corrections quoted a span the draft wrote 2-7 times.
    delivered = "包裝標示「花生世家」字樣。\n花生世家\n品牌名稱為花生世家，搭配花生插圖。"
    edit = ImageEdit(op="replace", quote="花生世家", replacement="花生田家", reason="Image shows 花生田家.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"花生世家"}, ocr_lines=["花生田家"],
                                             reading=_reading("花生田家"))

    assert "花生世家" not in text and text.count("花生田家") == 3 and len(applied) == 1 and not rejected


def test_a_repeated_span_the_picture_shows_somewhere_is_not_changed_everywhere():
    # 「茶」 misread for 「大」 once, while the picture also prints 「茶飲」: the other 茶 are right.
    delivered = "招牌寫「茶」字。\n茶飲\n店內提供茶飲與甜點。"
    edit = ImageEdit(op="replace", quote="茶", replacement="大", reason="Image shows 大.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"茶飲"}, ocr_lines=["大", "茶飲"],
                                             reading=_reading("大\n茶飲"))

    assert text == delivered and not applied and "shows it" in rejected[0]["refused"]


def test_one_occurrence_the_rules_refuse_stops_the_whole_edit():
    # The transcription line would become wording OCR does not read, so no occurrence changes.
    delivered = "標語「Peanut House」。\nPeanut House 花生"
    edit = ImageEdit(op="replace", quote="Peanut House", replacement="PeanuTend", reason="Image shows PeanuTend.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"Peanut House 花生"},
                                             ocr_lines=["PeanuTend"], reading=_reading("PeanuTend"))

    assert text == delivered and not applied and "transcription line" in rejected[0]["refused"]


def test_a_repeated_span_changes_everywhere_only_into_wording_ocr_reads():
    # Replayed on 814 stored proposals: the blind reading alone renamed a company everywhere.
    delivered = "看板印有「qingya 晴諾米」字樣。\n品牌名稱 qingya 晴諾米。"
    edit = ImageEdit(op="replace", quote="qingya 晴諾米", replacement="qingva 晴諾來", reason="Image shows qingva.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription=set(), ocr_lines=["看板"],
                                             reading=_reading("qingva 晴諾來"))

    assert text == delivered and not applied and "PaddleOCR does not read" in rejected[0]["refused"]


def test_only_a_certain_relation_of_the_blind_reading_is_added():
    relation = {"relation_id": "r0", "kind": "value_pair", "source_fact": "頁碼為 07", "certain": True}
    _, statements, applied, rejected = _apply({"op": "add_relation", "relation_id": "r0"},
                                              {"op": "add_relation", "relation_id": "r7"},
                                              reading=_reading(relationships=[relation]))

    assert statements == ["圖片關係：頁碼為 07"] and len(applied) == 1 and len(rejected) == 1


def test_methods_run_cheapest_first_and_each_once():
    state = {"pages": {"0": _record()}}
    assert next_image_method(state, {"steps": []}, 0, "picture") == "delivery_patch"
    patch = {"action": "repair_image", "block_id": "picture", "method": "delivery_patch"}
    assert next_image_method(state, {"steps": [patch]}, 0, "picture") == "targeted_edit"
    edit = {"action": "repair_image", "block_id": "picture", "method": "targeted_edit"}
    assert next_image_method(state, {"steps": [patch, edit]}, 0, "picture") == "full"

    content = {**_record(kind="image_content_mismatch"), "review": {"blocks": [
        {"block_id": "picture", "status": "needs_review", "observation": "Footer reads 07, delivery says 87."}]}}
    assert next_image_method({"pages": {"0": content}}, {"steps": []}, 0, "picture") == "targeted_edit"
    blurry = _record(kind="unreadable_text")
    assert next_image_method({"pages": {"0": blurry}}, {"steps": []}, 0, "picture") == "full"


class EditModel(PatchModel):
    """Fails any audit that still sees the invented eraser; proposes deleting it."""

    def __init__(self):
        super().__init__()
        self.proposals = []

    async def query(self, task, payload, image, schema):
        if task == "propose_image_edits":
            self.tasks.append(task)
            self.proposals.append(payload)
            return {"edits": [{"op": "delete", "quote": "與橡皮", "reason": "No eraser in the photo."},
                              {"op": "add_relation", "relation_id": "r0", "reason": "Relation omitted."}],
                    "observation": "One invented object, one omitted relation."}, 10
        if task in {"audit_image_parts", "audit_page"} and "橡皮" in json.dumps(payload.get("blocks"), ensure_ascii=False):
            self.tasks.append(task)
            return {"blocks": [{"block_id": bid, "status": "needs_review", "context_complete": False,
                                "observation": "Lists 橡皮, which the photo does not show.",
                                "image_relationship_checks": []} for bid in payload["required_block_ids"]],
                    "source_complete": False, "reading_order_correct": True,
                    "observation": "Invented object.", "issues": ["k0 lists 橡皮 not visible in the image"]}, 10
        return await super().query(task, payload, image, schema)


@pytest.mark.asyncio
async def test_after_a_rejected_patch_a_targeted_edit_fixes_what_the_audit_named(tmp_path):
    model = EditModel()
    config = cfg(max_recovery_actions=2)
    config.agentic.max_seconds = 3000
    config.agentic.max_model_calls = 60
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
            "semantic_caption": "此圖為拼圖商品頁，展示筆與橡皮等周邊。",
            "facts": ["商品提供 30cm、40cm、50cm 三種尺寸。"], "all_text": READING["text"].splitlines()}},
        ensure_ascii=False))
    await rebuild(ir)

    result = await runner.run(ir)

    steps = [(s.get("method"), s["status"]) for s in runner.state["steps"] if s["issue_id"] == "image:picture"]
    assert steps == [("delivery_patch", "rejected"), ("targeted_edit", "applied")]
    # This fixture's page plan names no picture; it is rejected with feedback until the attempt limit.
    page_steps = [s for s in runner.state["steps"] if s["issue_id"] == "page:0"]
    assert [s["action"] for s in page_steps] == ["invalid_plan"] * owner.limits.max_recovery_attempts_per_issue
    assert any("橡皮" in issue for issue in model.proposals[0]["known_issues"])
    delivered = "\n".join(b.payload["text"] for b in result.blocks if b.payload.get("agentic_image_origin") == "picture")
    assert "橡皮" not in delivered and "圖片關係：" + FACT in delivered
    assert "30cm (167 pieces)" in delivered
    assert not {"read_image_parts", "verify_image_parts", "resolve_image_evidence"} & set(model.tasks)


@pytest.mark.asyncio
async def test_every_picture_gets_its_first_method_before_any_picture_its_second(tmp_path):
    # v20.6 ran each picture's methods back to back: two pictures could spend the six
    # actions while eight others never got their cheap attempt.
    model = PatchModel()
    config = cfg(max_recovery_actions=2)
    config.agentic.max_seconds = 3000
    ir, owner, runner = await setup(tmp_path, model, config)
    for bid in ("first", "second"):
        picture = runner.inventory.directory / f"{bid}.png"
        Image.new("RGB", (400, 300), "white").save(picture)
        ir.blocks.append(Block(block_id=bid, type=BlockType.IMAGE, page_idx=0, payload={"img_path": str(picture)}))
        runner.inventory.mapping[bid] = [0]
        runner.inventory.word_images[bid] = picture
    owner.state["pages"]["0"] = {"page_idx": 0, "status": "needs_review", "image_only_issues": True, "image_issues": [
        {"block_id": bid, "failure_kind": "image_content_mismatch", "detail": "Footer digit differs."}
        for bid in ("first", "second")]}

    await runner.run(ir)

    assert [(s["block_id"], s.get("method")) for s in runner.state["steps"]] == [
        ("first", "targeted_edit"), ("second", "targeted_edit")]


def test_a_description_edit_never_breaks_wording_ocr_reads():
    # Spot check B: the review model misread the card name 「晴空光年」 as 「晴空年」, and an
    # edit deleted 「光」 from the description, which the deletion rule allowed. OCR read 光年.
    delivered = "卡片1角色名為「露米光年」，品牌角色為天真者。\n露米光年"
    edit = ImageEdit.model_validate({"op": "replace", "quote": "「露米光年」", "replacement": "「露米年」",
                                     "reason": "Audit reads 露米年."})
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"露米光年"},
                                             ocr_lines=["露米光年"], reading=_reading("露米年"))

    assert text == delivered and not applied
    assert "OCR" in rejected[0]["refused"]
