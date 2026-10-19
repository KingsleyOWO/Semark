"""Repairs fix what an audit found wrong and are kept when they add no new error (decided 2026-10-13).

Batch of 44 reports, 2026-10-08: 3127 issues were queued for repair and 83 resolved.
647 of them were relationships not spelled out (9 resolved), and a repair passed only
when its audit found nothing at all: 582 rejections named older faults or an unstated
relationship. Now such remarks are not queued, a repair fails only on a problem in
what it delivered, and a correction an audit spells out is applied when PaddleOCR
reads it, without a model writing text.
"""
import json

import pytest
from PIL import Image
from test_document_recovery import RecoveryModel, cfg, setup

from app.adapters import ocr
from app.models.document_ir import Block, BlockType
from app.pipeline.image_edits import review_corrections, spelled_out_corrections
from app.pipeline.image_recovery import next_image_method
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.package import PackageStage

LAYOUT = "Image relationship r0 remains missing: 頁尾位於右下角。"
MISREAD = "Footer 'LUMIO Brand Guidelines | 87' should be 'LUMIO Brand Guidelines | 07'."
INVENTED = "k0 lists 橡皮, which the photo does not show."
DESCRIPTION = "頁面展示名片、信封、筆與橡皮等品牌物件。"
READ_LINES = ["Application", "LUMIO Brand Guidelines | 87"]
OCR_LINES = ["Application", "LUMIO Brand Guidelines | 07"]


def _record(*remarks, kind="image_content_mismatch"):
    return {"page_idx": 0, "status": "needs_review", "reason": "source_delivery_mismatch",
            "images": [{"path": "page.png"}],
            "review": {"blocks": [{"block_id": "picture", "status": "needs_review", "observation": remarks[0]}],
                       "issues": [f"picture: {r}" for r in remarks], "source_complete": False,
                       "reading_order_correct": True},
            "image_issues": [{"block_id": "picture", "failure_kind": kind, "detail": remarks[0]}]}


async def _picture_run(tmp_path, model, record, monkeypatch, *, actions=2, description=DESCRIPTION):
    config = cfg(max_recovery_actions=actions)
    config.agentic.max_seconds = 3000
    config.agentic.max_model_calls = 60
    ir, owner, runner = await setup(tmp_path, model, config)
    picture = runner.inventory.directory / "picture.png"
    Image.new("RGB", (400, 300), "white").save(picture)
    ir.blocks.append(Block(block_id="picture", type=BlockType.IMAGE, page_idx=0, reading_order=1,
                           payload={"img_path": str(picture)}))
    runner.inventory.mapping["picture"] = [0]
    runner.inventory.word_images["picture"] = picture
    runner.inventory.image_positions["picture"] = [{"page_idx": 0, "bbox": [0, 200, 1000, 900]}]
    owner.state["pages"]["0"] = record
    from app.pipeline.document_recovery import render_candidate
    runner.renderer = render_candidate
    monkeypatch.setattr(ocr, "ocr_python", lambda: "/usr/bin/python3")

    async def ocr_lines(*_):
        return OCR_LINES
    owner._ocr_lines = ocr_lines

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
            "semantic_caption": "晴諾米品牌指南。", "facts": [description], "all_text": READ_LINES}},
        ensure_ascii=False))
    await rebuild(ir)
    return ir, owner, runner


class AuditingModel(RecoveryModel):
    """Plans repair_image; audits find the invented eraser while it is delivered, and nothing else."""

    def __init__(self, edits=()):
        super().__init__(action="repair_image")
        self.edits = list(edits)

    async def query(self, task, payload, image, schema):
        if task == "propose_image_edits":
            self.tasks.append(task)
            return {"edits": self.edits, "observation": "Edits for the named faults."}, 10
        if task in {"audit_image_parts", "audit_page"}:
            self.tasks.append(task)
            text = json.dumps(payload.get("blocks"), ensure_ascii=False)
            faults = ["k0 lists 橡皮 not visible in the image"] if "橡皮" in text else []
            return {"blocks": [{"block_id": bid, "status": "needs_review" if faults else "verified",
                                "context_complete": not faults,
                                "observation": INVENTED if faults else "Matches the source.",
                                "image_relationship_checks": []} for bid in payload["required_block_ids"]],
                    "source_complete": not faults, "reading_order_correct": True,
                    "observation": "Source and delivery compared.", "issues": faults}, 10
        return await super().query(task, payload, image, schema)


def test_a_picture_or_page_whose_review_only_misses_a_relationship_is_not_queued(tmp_path):
    import asyncio

    async def problems(record):
        ir, owner, runner = await setup(tmp_path)
        ir.blocks.append(Block(block_id="picture", type=BlockType.IMAGE, page_idx=0, reading_order=1,
                               payload={"img_path": "picture.png"}))
        runner.inventory.mapping["picture"] = [0]
        owner.state["pages"]["0"] = record
        return [p["id"] for p in runner.problems(ir)]

    assert asyncio.run(problems(_record(LAYOUT, kind="image_relationship_mismatch"))) == []
    assert asyncio.run(problems(_record(MISREAD))) == ["image:picture", "page:0"]


def test_a_correction_the_review_spells_out_is_applied_only_where_ocr_reads_it():
    delivered = "\n".join([DESCRIPTION, *READ_LINES])
    assert review_corrections([MISREAD], delivered, OCR_LINES) == {
        "LUMIO Brand Guidelines | 87": "LUMIO Brand Guidelines | 07"}
    # OCR reading neither side, a remark without a correction, or a layout remark: nothing.
    assert review_corrections([MISREAD], delivered, ["Application"]) == {}
    assert review_corrections(["Footer 'LUMIO Brand Guidelines | 87' and 'Application' differ in size."],
                              delivered, OCR_LINES) == {}
    assert spelled_out_corrections([LAYOUT]) == []


def test_a_short_misreading_written_several_times_is_corrected_everywhere_but_never_inside_a_word():
    # No length limit: a three-letter brand misread in every line is a misreading like any other.
    remark = "The logo reads 'CNL', not 'CML' as delivered."
    delivered = "標誌為 CML。\nCML 晴諾米\nCML"
    assert review_corrections([remark], delivered, ["CNL 晴諾米"]) == {"CML": "CNL"}
    # A span that also sits inside a longer word is not replaced everywhere (「tt」 in Matt).
    remark = "The label 'tt' should be 'tf'."
    assert review_corrections([remark], "顏色為 tt。\ntt", ["tf"]) == {"tt": "tf"}
    assert review_corrections([remark], "顏色為 tt。\nMatt tt", ["tf"]) == {}


def test_the_spelled_out_correction_is_tried_before_any_other_method():
    state = {"pages": {"0": _record(MISREAD)}}
    assert next_image_method(state, {"steps": []}, 0, "picture") == "review_correction"
    tried = {"action": "repair_image", "block_id": "picture", "method": "review_correction"}
    assert next_image_method(state, {"steps": [tried]}, 0, "picture") == "targeted_edit"


@pytest.mark.asyncio
async def test_a_spelled_out_correction_is_applied_without_a_model_writing_text(tmp_path, monkeypatch):
    model = AuditingModel()
    ir, _, runner = await _picture_run(tmp_path, model, _record(MISREAD), monkeypatch, actions=1,
                                       description="頁面展示名片、信封與筆等品牌物件。")
    result = await runner.run(ir)

    step = next(s for s in runner.state["steps"] if s["issue_id"] == "image:picture")
    assert (step.get("method"), step["status"]) == ("review_correction", "applied")
    assert runner.state["issues"]["image:picture"]["status"] == "resolved"
    delivered = "\n".join(b.payload["text"] for b in result.blocks if b.payload.get("agentic_image_origin") == "picture")
    assert "LUMIO Brand Guidelines | 07" in delivered and "| 87" not in delivered
    assert not {"propose_image_edits", "read_image_parts", "verify_image_parts"} & set(model.tasks)
    # Only the lines read off the picture survive if the picture is later left unverified.
    read = [line for b in result.blocks if b.payload.get("agentic_image_origin") == "picture"
            for line in b.payload["agentic_read_lines"]]
    assert read == ["Application", "LUMIO Brand Guidelines | 07"]


@pytest.mark.asyncio
async def test_a_correction_blocked_by_an_older_fault_is_carried_into_the_targeted_edit(tmp_path, monkeypatch):
    # The correction alone leaves the invented eraser, so it is not committed; the targeted
    # edit applies it again with its own deletion and is kept once its audit finds nothing new.
    model = AuditingModel(edits=[{"op": "delete", "quote": "與橡皮", "reason": "No eraser in the photo."}])
    ir, _, runner = await _picture_run(tmp_path, model, _record(MISREAD, INVENTED), monkeypatch)
    result = await runner.run(ir)

    steps = [(s.get("method"), s["status"]) for s in runner.state["steps"] if s["issue_id"] == "image:picture"]
    assert steps == [("review_correction", "rejected"), ("targeted_edit", "applied")]
    delivered = "\n".join(b.payload["text"] for b in result.blocks if b.payload.get("agentic_image_origin") == "picture")
    assert "LUMIO Brand Guidelines | 07" in delivered and "橡皮" not in delivered


@pytest.mark.asyncio
async def test_a_targeted_edit_is_kept_when_only_an_older_fault_remains(tmp_path, monkeypatch):
    # The edit fixes the footer; the eraser it did not touch stays and no longer undoes the fix.
    model = AuditingModel(edits=[{"op": "replace", "quote": "LUMIO Brand Guidelines | 87",
                                  "replacement": "LUMIO Brand Guidelines | 07", "reason": "OCR reads 07."}])
    record = _record("Footer reads 07, delivery says 87.", INVENTED)
    ir, _, runner = await _picture_run(tmp_path, model, record, monkeypatch)
    result = await runner.run(ir)

    steps = [(s.get("method"), s["status"]) for s in runner.state["steps"] if s["issue_id"] == "image:picture"]
    # The picture stays repairable (2026-10-16): the eraser is still faulted, so its next method, a full
    # re-read, is tried; this fixture's model cannot read, and the kept edit stays as it was.
    assert steps == [("targeted_edit", "applied"), (None, "rejected")]
    assert runner.state["issues"]["image:picture"]["status"] == "improved"
    delivered = "\n".join(b.payload["text"] for b in result.blocks if b.payload.get("agentic_image_origin") == "picture")
    assert "LUMIO Brand Guidelines | 07" in delivered and "橡皮" in delivered


@pytest.mark.asyncio
async def test_a_corrected_line_is_still_a_line_read_off_the_picture(tmp_path, monkeypatch):
    """2026-10-16: an unverified radar chart delivered neither its fixed label nor its fixed score: the
    corrected lines matched no drafted line and OCR, garbling the label, read no line whole."""
    model = AuditingModel(edits=[{"op": "replace", "quote": "| 87", "replacement": "| 07", "reason": "OCR reads 07."}])
    record = _record("Footer reads 07, delivery says 87.")
    record["word_image_readings"] = {"picture": {
        "source_id": "image0", "text": "Application\nLUMIO Brand Guidelines | 07", "complete": True,
        "observation": "Slide.", "image_type": "document", "relationships": [], "relationships_complete": True}}
    ir, owner, runner = await _picture_run(tmp_path, model, record, monkeypatch, description="頁面展示名片與信封。")

    async def garbled(*_):
        return ["Application", "LUMI0 Brand Gu1de | 07"]
    owner._ocr_lines = garbled
    result = await runner.run(ir)

    steps = [(s.get("method"), s["status"]) for s in runner.state["steps"] if s["issue_id"] == "image:picture"]
    assert steps == [("targeted_edit", "applied")]
    read = [line for b in result.blocks if b.payload.get("agentic_image_origin") == "picture"
            for line in b.payload["agentic_read_lines"]]
    assert "圖片文字：LUMIO Brand Guidelines | 07" in read or "LUMIO Brand Guidelines | 07" in read


def test_a_quote_inside_the_other_is_no_misreading():
    """Rerun 2026-10-16: 「社群策略 建議」, which the picture shows, became 「建議」 because OCR reads the shorter."""
    remark = "來源圖片顯示「社群策略 建議」，交付文字誤寫為「建議」。"
    assert review_corrections([remark], "標籤：社群策略 建議\n品牌活動 建議", ["社群策略", "晴諾米", "建議"]) == {}


class SecondRepairModel(AuditingModel):
    """Audits fault the footer while it reads 87, the heading while it is misspelt, and always the eraser."""

    async def query(self, task, payload, image, schema):
        if task in {"audit_image_parts", "audit_page"}:
            self.tasks.append(task)
            text = json.dumps(payload.get("blocks"), ensure_ascii=False)
            faults = [INVENTED] + (["Footer reads 07, delivery says 87."] if "| 87" in text else []) + (
                ["The heading 'Applicatoin' should be 'Application'."] if "Applicatoin" in text else [])
            return {"blocks": [{"block_id": bid, "status": "needs_review", "context_complete": False,
                                "observation": " ".join(faults), "image_relationship_checks": []}
                               for bid in payload["required_block_ids"]],
                    "source_complete": False, "reading_order_correct": True,
                    "observation": "Source and delivery compared.", "issues": faults}, 10
        return await super().query(task, payload, image, schema)


@pytest.mark.asyncio
async def test_a_repaired_picture_is_repaired_again_by_its_next_method(tmp_path, monkeypatch):
    """Rerun 2026-10-16: 13 of 30 repaired pictures kept faults their next audit spelled out, because a
    committed repair turned the picture into text no method could reach. A radar chart kept a wrong
    completion date; a certificate a company name one character off."""
    model = SecondRepairModel(edits=[{"op": "replace", "quote": "LUMIO Brand Guidelines | 87",
                                      "replacement": "LUMIO Brand Guidelines | 07", "reason": "OCR reads 07."}])
    ir, owner, runner = await _picture_run(tmp_path, model, _record("Footer reads 07, delivery says 87."),
                                           monkeypatch, actions=3)
    (tmp_path / "outputs/enrichments.jsonl").write_text(json.dumps({
        "block_id": "picture", "kind": "figure_description", "output": {
            "semantic_caption": "晴諾米品牌指南。", "facts": [DESCRIPTION],
            "all_text": ["Applicatoin", "LUMIO Brand Guidelines | 87"]}}, ensure_ascii=False))
    await runner.rebuild(ir)
    result = await runner.run(ir)

    steps = [(s.get("method"), s["status"]) for s in runner.state["steps"] if s["issue_id"] == "image:picture"]
    # The edit fixes the footer; the heading's correction, spelled out only by the audit after it, is
    # applied to the repaired picture; the eraser no method removes faults neither. The re-read that
    # comes next cannot read in this fixture and leaves both repairs as they were.
    assert steps[:2] == [("targeted_edit", "applied"), ("review_correction", "applied")]
    assert runner.state["issues"]["image:picture"]["status"] == "improved"
    delivered = "\n".join(b.payload["text"] for b in result.blocks if b.payload.get("agentic_image_origin") == "picture")
    assert "LUMIO Brand Guidelines | 07" in delivered and "Application" in delivered
    assert "| 87" not in delivered and "Applicatoin" not in delivered
    # Nothing read off the picture is lost on the way, and the picture keeps its file and page.
    read = [line for b in result.blocks if b.payload.get("agentic_image_origin") == "picture"
            for line in b.payload["agentic_read_lines"]]
    assert {"Application", "LUMIO Brand Guidelines | 07"} <= set(read)
    assert runner.state["repaired_pictures"]["picture"]["type"] == "image"
    assert runner.inventory.mapping["picture"] == [0]
    assert result.get_block("picture") is None


def test_an_edit_of_a_repaired_picture_keeps_its_gap_marker_and_a_reread_replaces_it():
    from app.models.document_ir import DocumentIR, EngineInfo, SourceInfo
    from app.pipeline.image_recovery import replace_picture

    def text(bid, origin="picture", **payload):
        return Block(block_id=bid, type=BlockType.TEXT, page_idx=0,
                     payload={"text": bid, "agentic_image_origin": origin, **payload})
    ir = DocumentIR(doc_id="d", run_id="r", source=SourceInfo(path="d.docx", ext="docx", sha256="abc", size_bytes=1),
                    engine=EngineInfo(backend="pipeline", method="auto"), blocks=[
        text("before", origin=None), text("k0"),
        text("g0", agentic_gap={"reason": "image_text_unreadable"}, agentic_depends_on_block_ids=["k0"]),
        text("k1"), text("after", origin=None)])
    new = [text("n0"), text("n1")]
    edited = replace_picture(ir, "picture", new, keep_gaps=True)
    assert [b.block_id for b in edited.blocks] == ["before", "n0", "n1", "g0", "after"]
    assert edited.get_block("g0").payload["agentic_depends_on_block_ids"] == ["n0"]
    reread = replace_picture(ir, "picture", new)
    assert [b.block_id for b in reread.blocks] == ["before", "n0", "n1", "after"]
    assert [b.reading_order for b in reread.blocks] == [0, 1, 2, 3]
    assert ir.get_block("k0") is not None  # the IR itself is left as it was


def test_a_remark_saying_mis_recorded_or_mislabelled_spells_out_a_correction():
    """Rerun 2026-10-16: reviews wrote 誤記為 ten times and 誤標為 once; a legend's correction went unseen."""
    assert spelled_out_corrections(["圖例「晴諾米診斷分數」誤記為「晴諾米部分分數」。"]) == [("晴諾米診斷分數", "晴諾米部分分數")]
    assert spelled_out_corrections(["標籤「晴諾米」被誤標為「晴諾來」。"]) == [("晴諾米", "晴諾來")]


class ReadingModel(SecondRepairModel):
    """SecondRepairModel whose two readers read the picture alike, with a line the draft missed."""

    async def query(self, task, payload, image, schema):
        if task in {"read_image_parts", "verify_image_parts"}:
            self.tasks.append(task)
            return {"image_type": "document", "visual_facts": [], "visual_independent_of_text": False,
                    "text_regions": [{"region_id": "t0", "bbox": [0, 0, 1000, 300], "readable": True,
                                      "text": "Application"},
                                     {"region_id": "t1", "bbox": [0, 700, 1000, 1000], "readable": True,
                                      "text": "LUMIO Brand Guidelines | 07"},
                                     {"region_id": "t2", "bbox": [0, 400, 1000, 500], "readable": True,
                                      "text": "Stationery Set"}],
                    "complete": True, "observation": "Slide heading and footer."}, 10
        if task == "resolve_image_evidence":
            self.tasks.append(task)
            return {"selections": [{"source_ref": ref, "decision": "supported", "reason": "Printed as read."}
                                   for ref in payload["required_source_refs"]],
                    "accounted_for": True, "observation": "OCR and the readers agree."}, 10
        return await super().query(task, payload, image, schema)


@pytest.mark.asyncio
async def test_a_reread_of_a_repaired_picture_keeps_its_relationship_provenance(tmp_path, monkeypatch):
    """Test run 2026-10-19: an edit attaches the blind reading even when it lists no relationship; the
    re-read after it did not, and was refused as image_relationship_provenance_lost."""
    model = ReadingModel(edits=[{"op": "replace", "quote": "LUMIO Brand Guidelines | 87",
                                 "replacement": "LUMIO Brand Guidelines | 07", "reason": "OCR reads 07."}])
    record = _record("Footer reads 07, delivery says 87.")
    record["word_image_readings"] = {"picture": {
        "source_id": "image0", "text": "Application\nLUMIO Brand Guidelines | 07", "complete": True,
        "observation": "Slide.", "image_type": "document", "relationships": [], "relationships_complete": True}}
    ir, owner, runner = await _picture_run(tmp_path, model, record, monkeypatch, actions=3)
    (tmp_path / "outputs/enrichments.jsonl").write_text(json.dumps({
        "block_id": "picture", "kind": "figure_description", "output": {
            "semantic_caption": "晴諾米品牌指南。", "facts": [DESCRIPTION],
            "all_text": ["Applicatoin", "LUMIO Brand Guidelines | 87"]}}, ensure_ascii=False))
    await runner.rebuild(ir)
    await runner.run(ir)

    steps = [(s.get("method"), s["status"], str(s.get("result", ""))) for s in runner.state["steps"]
             if s["issue_id"] == "image:picture" and s["action"] == "repair_image"]
    assert steps[0][:2] == ("targeted_edit", "applied")
    assert all("provenance_lost" not in result for _, _, result in steps), steps
    assert any(method is None for method, _, _ in steps)  # the re-read ran
