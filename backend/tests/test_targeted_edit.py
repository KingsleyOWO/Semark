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
    # 「茶飲」 misread once for 「茶點」, while the picture also prints 「茶飲」: the other one is right.
    delivered = "招牌寫「茶飲」。\n茶飲\n店內提供茶飲與甜點。"
    edit = ImageEdit(op="replace", quote="茶飲", replacement="茶點", reason="Image shows 茶點.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"茶飲"}, ocr_lines=["茶點", "茶飲"],
                                             reading=_reading("茶點\n茶飲"))

    assert text == delivered and not applied and "shows it" in rejected[0]["refused"]


def test_one_occurrence_the_rules_refuse_stops_the_whole_edit():
    # Deleting the word would cut the transcription line, so the description keeps it too.
    delivered = "標語「Peanut Houze」。\nPeanut Houze 花生"
    edit = ImageEdit(op="delete", quote="Houze", reason="Image shows no Houze.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"Peanut Houze 花生"},
                                             ocr_lines=["Peanut House"], reading=_reading("Peanut House"))

    assert text == delivered and not applied and "transcription line" in rejected[0]["refused"]


def test_a_repeated_span_changes_everywhere_only_into_wording_ocr_reads():
    # Replayed on 814 stored proposals: the blind reading alone renamed a company everywhere.
    delivered = "看板印有「qingya 晴諾米」字樣。\n品牌名稱 qingya 晴諾米。"
    edit = ImageEdit(op="replace", quote="qingya 晴諾米", replacement="qingva 晴諾來", reason="Image shows qingva.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription=set(), ocr_lines=["看板"],
                                             reading=_reading("qingva 晴諾來"))

    assert text == delivered and not applied and "PaddleOCR does not read" in rejected[0]["refused"]


def test_a_repeated_span_is_not_rewritten_into_something_else():
    # Replayed: OCR read only "T" of a certificate's small "Technical Manager", and a right-to-left
    # couplet in its left-to-right order; neither is a misreading to fix everywhere.
    delivered = "簽名下方為 Technical Manager。\nTechnical Manager"
    edit = ImageEdit(op="replace", quote="Technical Manager", replacement="T", reason="OCR reads T.")
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"Technical Manager"},
                                             ocr_lines=["T"], reading=_reading("SGS"))
    assert text == delivered and not applied and "rewrites" in rejected[0]["refused"]

    couplet = "橫批「時來運轉萬事順」。\n時來運轉萬事順"
    edit = ImageEdit(op="replace", quote="時來運轉萬事順", replacement="成事萬轉運來時", reason="OCR.")
    assert not apply_edits(couplet, [edit], transcription={"時來運轉萬事順"}, ocr_lines=["成事萬轉運来時"],
                           reading=_reading("成事萬轉運來時"))[2]


def test_new_wording_keeps_the_deliverys_script_and_a_script_change_is_no_edit():
    # PaddleOCR answers 片装 for printed 片裝.
    delivered = "包裝標示 8片入。\n8片入"
    edit = ImageEdit(op="replace", quote="8片入", replacement="8片装", reason="OCR reads 8片装.")
    text, _, applied, _ = apply_edits(delivered, [edit], transcription={"8片入"}, ocr_lines=["8片装"],
                                      reading=_reading("8片裝"))
    assert text == "包裝標示 8片裝。\n8片裝" and applied[0]["replacement"] == "8片裝"

    same = ImageEdit(op="replace", quote="8片裝", replacement="8片装", reason="OCR reads 8片装.")
    text, _, applied, rejected = apply_edits("8片裝", [same], transcription={"8片裝"}, ocr_lines=["8片装"],
                                             reading=None)
    assert text == "8片裝" and not applied and "nothing" in rejected[0]["refused"]


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


def test_a_line_changes_in_the_number_both_readers_read_though_ocr_garbles_its_label():
    # A page-sized chart: OCR reads 「品度」 for the label but the score 3.33 cleanly.
    delivered = "雷達圖分值如下。\n品牌辨識度 3.23\n品牌辨識度最弱。"
    edit = ImageEdit(op="replace", quote="品牌辨識度 3.23", replacement="品牌辨識度 3.33", reason="Image shows 3.33.")
    text, _, applied, _ = apply_edits(delivered, [edit], transcription={"品牌辨識度 3.23"},
                                      ocr_lines=["品度", "3.33"], reading=_reading("品牌辨識度 3.33"))
    assert "品牌辨識度 3.33" in text and applied

    # The blind reading keeps the old number: OCR alone does not change a printed line.
    delivered = "包裝標示「木耳 290g」。\n木耳 290g\n淨重木耳 290g。"
    edit = ImageEdit(op="replace", quote="木耳 290g", replacement="木耳 200g", reason="OCR reads 200g.")
    text, _, applied, _ = apply_edits(delivered, [edit], transcription={"木耳 290g"},
                                      ocr_lines=["木", "200g"], reading=_reading("木耳 290g"))
    assert text == delivered and not applied


def test_a_latin_word_is_read_only_whole():
    delivered = "簽名為 Daniel Liu。\nDaniel Liu\n證書由 Daniel Liu 簽署。"
    edit = ImageEdit(op="replace", quote="Daniel Liu", replacement="Daniel Leu", reason="Signature reads Leu.")
    reading = _reading("Daniel Leu")
    assert apply_edits(delivered, [edit], transcription={"Daniel Liu"}, ocr_lines=["Daniel", "Leu"],
                       reading=reading)[2]
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"Daniel Liu"},
                                             ocr_lines=["Daniel", "Fleur"], reading=reading)
    assert text == delivered and not applied and "PaddleOCR does not read" in rejected[0]["refused"]


def test_wording_another_picture_prints_replaces_a_name_the_draft_invented():
    # A tiny chart OCR garbles 「品牌统力」; a sharper form in the same report prints the eight names.
    delivered = "雷達圖八項指標。\n品牌決策力 1.4\n品牌決策力 在企業營運層面統一品牌發展目標。"
    edit = ImageEdit(op="replace", quote="品牌決策力", replacement="品牌統御力", reason="Image shows 品牌統御力.")
    form = ["品牌辨識度", "品牌重視度", "品牌統御力", "品牌創新度"]
    arguments = dict(transcription={"品牌決策力 1.4"}, ocr_lines=["品牌统力", "1.4"],
                     reading=_reading("品牌統御力 1.4"))
    text, _, applied, _ = apply_edits(delivered, [edit], document_ocr=[form, ["品牌统力", "1.4"]], **arguments)
    assert "品牌決策力" not in text and text.count("品牌統御力") == 2 and applied
    # Without another picture's reading the garbled OCR backs nothing.
    assert not apply_edits(delivered, [edit], **arguments)[2]
    # Nor when this picture's own OCR is nearer the quote.
    near = dict(arguments, ocr_lines=["品牌決策", "1.4"])
    assert not apply_edits(delivered, [edit], document_ocr=[form], **near)[2]


def test_a_relation_quotes_only_lettering_the_delivery_or_ocr_has():
    invented = {"relation_id": "r1", "kind": "value_pair", "certain": True,
                "source_fact": "「願景」對應「我們為誰解決什麼問題」。"}
    blank = {"relation_id": "r2", "kind": "value_pair", "certain": True,
             "source_fact": "「__ Application __」位於「06」下方。"}
    _, statements, applied, rejected = _apply({"op": "add_relation", "relation_id": "r1"},
                                              {"op": "add_relation", "relation_id": "r2"},
                                              reading=_reading(relationships=[invented, blank]))
    assert statements == ["圖片關係：「__ Application __」位於「06」下方。"]
    assert "neither delivered nor read" in rejected[0]["refused"]


def test_an_unreadable_spot_is_filled_only_with_what_ocr_reads():
    # The blind reading named the same signature three ways on three forms.
    delivered = "訪談人員為手寫簽名（字跡模糊難辨）。\n訪談人員：[?]（簽名）"
    guesses = [ImageEdit(op="replace", quote="（字跡模糊難辨）", replacement="「晴諾米」", reason="Signature."),
               ImageEdit(op="replace", quote="訪談人員：[?]（簽名）", replacement="訪談人員：晴諾米（簽名）",
                         reason="Signature.")]
    text, _, applied, rejected = apply_edits(delivered, guesses, transcription=set(), ocr_lines=["訪談人員"],
                                             reading=_reading("訪談人員：晴諾米（簽名）"))
    assert text == delivered and not applied and all("unreadable spot" in r["refused"] for r in rejected)

    text, _, applied, _ = apply_edits(delivered, guesses[1:], transcription=set(), ocr_lines=["晴諾米"],
                                      reading=_reading("訪談人員：晴諾米（簽名）"))
    assert "訪談人員：晴諾米（簽名）" in text and applied
    # Dropping the remark brings in no wording.
    drop = ImageEdit(op="replace", quote="手寫簽名（字跡模糊難辨）", replacement="手寫簽名", reason="No remark.")
    assert apply_edits(delivered, [drop], transcription=set(), ocr_lines=["訪談人員"], reading=None)[2]


def test_a_replacement_spanning_lines_never_drops_a_transcription_line():
    """2026-10-16: 「晴諾米\\nBright Forever\\n晴川銀行」 became 「晴諾米\\n晴川銀行」; the badge reads "Bright Together"."""
    delivered = "合作夥伴標誌牆。\n晴諾米\nBright Forever\n晴川銀行"
    edit = ImageEdit.model_validate({"op": "replace", "quote": "晴諾米\nBright Forever\n晴川銀行",
                                     "replacement": "晴諾米\n晴川銀行", "reason": "Not a separate brand."})
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription={"晴諾米", "Bright Forever", "晴川銀行"},
                                             ocr_lines=["晴諾米", "晴川銀行", "We", "Jamily"], reading=_reading("晴諾米"))
    assert text == delivered and not applied
    assert rejected[0]["refused"] == "a transcription line is never deleted"


def test_a_hedge_is_removed_only_when_ocr_reads_what_it_held_back():
    """2026-10-16: 「NORIA」（或相近字樣） became 「NORIA」; the screen shows "?ORIA"."""
    delivered = "會議室螢幕顯示「NORIA」（或相近字樣）的標誌。"
    edit = ImageEdit.model_validate({"op": "replace", "quote": "「NORIA」（或相近字樣）", "replacement": "「NORIA」",
                                     "reason": "The screen shows NORIA."})
    for ocr, removed in ((["ORIA"], False), (["NORIA"], True)):
        text, _, applied, rejected = apply_edits(delivered, [edit], transcription=set(), ocr_lines=ocr,
                                                 reading=_reading("NORIA"))
        assert bool(applied) == removed and ("或相近字樣" not in text) == removed


def test_a_description_respells_a_latin_word_only_into_what_ocr_reads():
    """Rerun 2026-10-16: the blind reading alone turned a label's Lumora (OCR 「Lumara」) into Lumosa."""
    delivered = "包裝標籤印有 Lumora 字樣。"
    edit = ImageEdit.model_validate({"op": "replace", "quote": "Lumora", "replacement": "Lumosa", "reason": "Label."})
    text, _, applied, rejected = apply_edits(delivered, [edit], transcription=set(), ocr_lines=["Lumara"],
                                             reading=_reading("Lumosa"))
    assert text == delivered and not applied
    assert rejected[0]["refused"] == "a Latin word changes only into one this picture's PaddleOCR reads"
