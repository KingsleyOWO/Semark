"""Repair time goes to repairs that can succeed (rerun 2026-10-15, one 82-page report).

Of 3.6 h of repair, 57 of 74 page audits re-audited pages a commit had not changed,
59 plans plus OCR runs preceded repairs that read OCR themselves, pictures with only
"illegible" remarks were re-read for remarks on other blocks, unmapped pictures were
sent to a tool that refuses pictures, and 55 tried issues were reported as never tried.
"""
import json

import pytest
from PIL import Image
from test_document_recovery import RecoveryModel, setup

from app.adapters import ocr
from app.models.document_ir import Block, BlockType
from app.pipeline.document_recovery import (
    RecoveryController,
    edit_disputes_issue,
    ocr_scale,
)
from app.pipeline.review_sources import DeliveryExcerpts, SourceInventory
from app.pipeline.stages.chunk import ChunkStage


def _picture(runner, ir, name, page=0):
    path = runner.inventory.directory / f"{name}.png"
    Image.new("RGB", (120, 90), "white").save(path)
    ir.blocks.append(Block(block_id=name, type=BlockType.IMAGE, page_idx=page, payload={"img_path": str(path)}))
    runner.inventory.word_images[name] = path
    runner.inventory.mapping[name] = [page]
    return path


@pytest.mark.asyncio
async def test_only_a_picture_a_hard_remark_names_is_re_read_for_its_page(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    photo, chart = "b000101", "b000102"
    for name in (photo, chart):
        _picture(runner, ir, name)
    ir.blocks.append(Block(block_id="b000103", type=BlockType.TEXT, page_idx=0, payload={"text": "營業額成長。"}))
    record = {"page_idx": 0, "status": "needs_review", "review": {
        "blocks": [{"block_id": photo, "status": "needs_review",
                    "observation": "Small print on the shelf is too small to read."},
                   {"block_id": chart, "status": "needs_review",
                    "observation": "The delivery reads 2023 where the chart shows 2024."},
                   {"block_id": "b000103", "status": "needs_review", "observation": "誤植「營收」為「營業」。"}],
        "issues": ["b000103 誤植「營收」為「營業」。"], "reading_order_correct": True}}
    assert runner.unaccepted_pictures(ir, record) == {chart}
    assert not runner.unattributed_page_problem(record)
    # A page issue naming a picture is a hard remark about it even when its own finding is soft.
    record["review"]["issues"].append(f"{photo} omits the sign 「晴諾米拼圖」 the original shows.")
    assert runner.unaccepted_pictures(ir, record) == {chart, photo}
    # An unexplained finding says nothing is right.
    record["review"]["blocks"][0]["observation"] = ""
    record["review"]["issues"].pop()
    assert runner.unaccepted_pictures(ir, record) == {chart, photo}


@pytest.mark.asyncio
async def test_a_page_whose_remaining_faults_are_pictures_without_a_re_read_closes(tmp_path):
    class Planner(RecoveryModel):
        async def query(self, task, payload, image, schema):
            if task == "recovery_plan":
                raise AssertionError("no plan is needed")
            return await super().query(task, payload, image, schema)

    ir, owner, runner = await setup(tmp_path, Planner())
    _picture(runner, ir, "b000201")
    runner.state["steps"].append({"key": "done", "issue_id": "image:b000201", "action": "repair_image",
                                  "block_id": "b000201", "model_role": "primary", "page_idx": 0,
                                  "status": "rejected"})
    owner.state["pages"]["0"]["review"] = {
        "blocks": [{"block_id": "b0", "status": "verified", "context_complete": True},
                   {"block_id": "b000201", "status": "needs_review", "observation": "The logo names are wrong."}],
        "issues": ["b000201 lists 晴諾米 where the original shows 晴諾來."], "reading_order_correct": True}

    await runner.run(ir, issue_ids={"page:0"})

    issue = runner.state["issues"]["page:0"]
    assert issue["status"] == "deferred" and "re-read again" in issue["reason"]


@pytest.mark.asyncio
async def test_an_unmapped_picture_is_not_sent_to_a_tool_that_refuses_pictures(tmp_path):
    class Planner(RecoveryModel):
        async def query(self, task, payload, image, schema):
            if task == "recovery_plan":
                raise AssertionError("no plan is needed")
            return await super().query(task, payload, image, schema)

    ir, owner, runner = await setup(tmp_path, Planner())
    ir.blocks.append(Block(block_id="floating", type=BlockType.IMAGE, page_idx=0, payload={"img_path": "x.png"}))
    runner.inventory.mapping["floating"] = []

    await runner.run(ir, issue_ids={"block:floating"})

    issue = runner.state["issues"]["block:floating"]
    assert issue["status"] == "deferred" and issue["reason"].startswith("unmapped_picture")
    assert not runner.state["steps"]


def test_a_picture_in_a_one_page_table_takes_the_table_page(tmp_path):
    def picture(bid, parent, sha):
        return Block(block_id=bid, type=BlockType.IMAGE, page_idx=0, payload={
            "img_path": f"{bid}.png", "word_source": {"parent_table_block_id": parent, "image_sha256": sha}})

    inventory = SourceInventory.__new__(SourceInventory)
    inventory.directory = tmp_path
    inventory.word_images = {}
    inventory.mapping = {"table": [6], "header-table": [2, 3], "in-cell": [], "in-header": [], "loose": []}
    inventory.mapping_details = {}
    Image.new("RGB", (40, 40), "white").save(tmp_path / "word-image-aaa.png")
    blocks = [picture("in-cell", "table", "aaa"), picture("in-header", "header-table", "bbb"),
              picture("loose", None, "ccc")]

    inventory.place_pictures_in_tables(blocks)

    assert inventory.mapping["in-cell"] == [6]
    assert inventory.mapping_details["in-cell"]["method"] == "parent_table_page"
    assert inventory.word_images["in-cell"] == tmp_path / "word-image-aaa.png"
    # A repeated header row is on several pages; a picture outside a table has no anchor.
    assert inventory.mapping["in-header"] == [] and inventory.mapping["loose"] == []


@pytest.mark.asyncio
async def test_a_whole_picture_ocr_plan_runs_as_the_repair_that_reads_ocr_itself(tmp_path, monkeypatch):
    model = RecoveryModel(actions=["ocr_image"])
    ir, owner, runner = await setup(tmp_path, model)
    _picture(runner, ir, "sign")
    owner.state["pages"]["0"].update(image_issues=[{"block_id": "sign", "failure_kind": "image_content_mismatch",
                                                     "detail": "The sign reads 晴諾米, not 晴諾來."}])
    called = []

    async def repair(ir, problem, decision, folder, skill):
        called.append(decision.action)
        raise ValueError("stop here")

    monkeypatch.setattr(runner, "repair_image", repair)
    await runner.run(ir, issue_ids={"image:sign"}, max_new_actions=1)

    assert called == ["repair_image"]
    assert [s["action"] for s in runner.state["steps"]] == ["repair_image"]


@pytest.mark.asyncio
async def test_a_full_re_read_gets_whole_picture_ocr_once(tmp_path, monkeypatch):
    ir, owner, runner = await setup(tmp_path)
    _picture(runner, ir, "sign")
    observed = []

    async def fake_ocr(ir, problem, decision, folder, skill):
        observed.append((decision.block_id, decision.region, decision.scale))
        runner.evidence.add(provider="paddleocr", kind="transcription", page=0, block_id="sign",
                            source=runner.inventory.word_images["sign"], bbox=[0, 0, 1000, 1000],
                            content={"region": [0, 0, 1000, 1000], "scale": decision.scale, "lines": [],
                                     "status": "ok", "source_sha256": "x"})

    monkeypatch.setattr(ocr, "ocr_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(runner, "ocr_image", fake_ocr)
    await runner.picture_ocr(ir, {"page_idx": 0}, "sign", tmp_path / "step")
    await runner.picture_ocr(ir, {"page_idx": 0}, "sign", tmp_path / "step")
    assert observed == [("sign", None, 2)]


def test_ocr_scale_halves_until_the_region_fits_the_pixel_limit(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (3446, 1714), "white").save(path)
    assert ocr_scale(path, [0, 0, 1000, 1000], 4) == 1
    assert ocr_scale(path, [0, 0, 500, 500], 4) == 2
    assert ocr_scale(path, [0, 0, 200, 200], 4) == 4


def test_a_tried_issue_out_of_time_stays_unresolved_and_an_untried_one_is_budget_exhausted():
    tried, untried = {"status": "unresolved", "attempts": 2}, {"status": "pending"}
    RecoveryController.out_of_budget(tried, reason="Insufficient budget")
    RecoveryController.out_of_budget(untried, reason="Insufficient budget")
    assert (tried["status"], untried["status"]) == ("unresolved", "budget_exhausted")
    assert tried["stopped_by_budget"] and untried["stopped_by_budget"]


@pytest.mark.asyncio
async def test_an_issue_the_edit_model_disputes_waits_for_the_others(tmp_path):
    folder = tmp_path / "edit"
    folder.mkdir()
    (folder / "targeted-edit.json").write_text(json.dumps({"proposal": {"edits": [], "observation": "Matches."}}))
    assert edit_disputes_issue(folder)
    (folder / "targeted-edit.json").write_text(json.dumps({"proposal": {"edits": [{"op": "delete"}]}}))
    assert not edit_disputes_issue(folder)
    assert not edit_disputes_issue(tmp_path / "missing")

    ir, owner, runner = await setup(tmp_path)
    for name in ("first", "second"):
        _picture(runner, ir, name)
    owner.state["pages"]["0"].update(image_issues=[
        {"block_id": name, "failure_kind": "image_content_mismatch", "detail": f"{name} is wrong."}
        for name in ("first", "second")])
    order = [p["id"] for p in runner.problems(ir) if p.get("kind") == "image"]
    assert order == ["image:first", "image:second"]
    runner.state["issues"]["image:first"] = {"id": "image:first", "status": "unresolved", "disputed": True}
    assert [p["id"] for p in runner.problems(ir) if p.get("kind") == "image"] == ["image:second", "image:first"]


@pytest.mark.asyncio
async def test_a_repair_lacking_quoted_wording_ocr_reads_does_not_resolve_its_issue(tmp_path, monkeypatch):
    ir, owner, runner = await setup(tmp_path)
    _picture(runner, ir, "panel")
    ir.blocks.append(Block(block_id="recovery_x_image_k0", type=BlockType.TEXT, page_idx=0,
                           payload={"text": "左欄：品牌定位", "agentic_image_origin": "panel"}))

    async def lines(source, inventory):
        return ["品牌定位建構", "品牌溝通體驗"]

    monkeypatch.setattr(ocr, "ocr_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(owner, "_ocr_lines", lines)
    problem = {"id": "image:panel", "block_id": "panel", "page_idx": 0,
               "issues": ["panel omits the headings '品牌定位建構' and 「品牌溝通體驗」 and the note 「晴諾米」."]}
    assert await runner.quoted_wording_missing(problem, "panel", ir) == ["品牌定位建構", "品牌溝通體驗"]
    ir.blocks[-1].payload["text"] = "品牌定位建構\n品牌溝通體驗"
    assert await runner.quoted_wording_missing(problem, "panel", ir) == []


def test_a_line_repeated_elsewhere_names_only_the_owners_in_scope(tmp_path):
    from test_document_review import fixture

    def block(bid, text, order):
        return Block(block_id=bid, type=BlockType.TEXT, page_idx=0, reading_order=order, payload={"text": text})

    blocks = [block("a", "No.1", 0), block("b", "晴諾米拼圖工作室", 1), block("c", "No.1", 2),
              block("d", "No.1", 3), block("e", "No.1", 4)]
    fixture(tmp_path, blocks=blocks)
    main = tmp_path / "outputs/documents/main.md"
    renderer = ChunkStage()

    def scope(blocks):
        documents, _ = DeliveryExcerpts(tmp_path, blocks).for_blocks([blocks[0]], renderer)
        return [{k: v for k, v in d.items() if k != "paragraph"} for d in documents]

    # Two delivered "No.1" lines for four source blocks: ownership is ambiguous.
    main.write_text("晴諾米拼圖工作室\nNo.1\nNo.1\n")
    before = scope(blocks)
    assert before and all(d["block_ids"] == ["a"] and d["also_elsewhere"] for d in before)
    # Another picture's "No.1" repaired away: this scope's excerpt does not change.
    assert scope(blocks[:4]) == before
