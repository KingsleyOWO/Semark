"""Source-review regressions: no heuristic bypass, no unreviewed success."""
import asyncio
import json
from unittest.mock import AsyncMock

import fitz
import pytest

from app.config import AgenticConfig, PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.agentic import visual_issues
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.review_sources import SourceInventory, word_block_pages


def fixture(tmp_path, pages=1, blocks=None):
    source = tmp_path / "original.pdf"
    with fitz.open() as pdf:
        for index in range(pages):
            page = pdf.new_page()
            page.insert_text((60, 60), f"Source page {index + 1}: Revenue 100, cost 80.")
        pdf.save(source)
    ir = DocumentIR(
        doc_id="doc", run_id="run", page_kind="physical",
        source=SourceInfo(path=str(source), ext="pdf", sha256="source", size_bytes=source.stat().st_size),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=i) for i in range(pages)],
        blocks=blocks if blocks is not None else [
            Block(block_id=f"b{i}", type=BlockType.TEXT, page_idx=i, reading_order=i,
                  payload={"text": f"Source page {i + 1}: Revenue 100, cost 80."}) for i in range(pages)
        ],
    )
    write_delivery(tmp_path, ir)
    return ir, source


def write_delivery(root, ir):
    output = root / "outputs"
    (output / "documents").mkdir(parents=True, exist_ok=True)
    from app.pipeline.stages.chunk import ChunkStage
    renderer = ChunkStage()
    def text(b):
        return renderer._block_to_text(b) if b.type == BlockType.TABLE else b.get_text()
    (output / "documents/main.md").write_text("\n\n".join(text(b) for b in ir.blocks))
    (output / "chunks.jsonl").write_text("\n".join(json.dumps({
        "chunk_id": b.block_id, "block_ids": [b.block_id], "page_indices": [b.page_idx],
        "content": text(b),
    }) for b in ir.blocks))
    (output / "quality_gate.json").write_text('{"status":"pass","issues":[],"stats":{}}')


class AuditModel:
    def __init__(self, callback=None, cancel=False):
        self.calls = []
        self.callback = callback
        self.cancel = cancel

    async def query(self, task, payload, image, schema):
        self.calls.append((task, payload, image))
        assert task == "audit_page"
        if self.cancel:
            self.cancel = False
            raise asyncio.CancelledError()
        value = {
            "blocks": [{"block_id": b, "status": "verified", "observation": "Source values 100 and 80 match the delivered row."}
                       for b in payload["required_block_ids"]],
            "source_complete": True, "reading_order_correct": True,
            "observation": "The entire target page matches documents and chunks.", "issues": [],
        }
        if self.callback:
            value = self.callback(payload, value)
        return value, 100


def config(**values):
    return PipelineConfig(agentic=AgenticConfig(**values))


@pytest.mark.asyncio
async def test_plain_text_without_risk_is_reviewed_and_final_delivery_passes(tmp_path):
    ir, source = fixture(tmp_path)
    assert not visual_issues(ir)
    model = AuditModel()
    output, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 1
    assert report["summary"]["verified_pages"] == 1
    assert report["summary"]["verified_blocks"] == 1
    assert finalize_delivery(output, tmp_path, config())["auto_rag_ready"]


@pytest.mark.asyncio
async def test_source_page_omitted_entirely_by_parser_still_gets_reviewed(tmp_path):
    ir, source = fixture(tmp_path, pages=2)
    ir.pages.pop()
    ir.blocks.pop()
    write_delivery(tmp_path, ir)

    def compare(payload, value):
        if payload["page_idx"] == 1:
            assert not payload["required_block_ids"]
            value.update(source_complete=False, issues=["Source page 2 is absent from the delivery."])
        return value

    model = AuditModel(compare)
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 2
    assert report["summary"]["total_pages"] == 2
    assert report["summary"]["needs_review_pages"] == 1
    assert not finalize_delivery(ir, tmp_path, config())["auto_rag_ready"]


@pytest.mark.asyncio
async def test_normal_grid_numeric_swap_is_not_filtered_out(tmp_path):
    ir, source = fixture(tmp_path)
    block = ir.blocks[0]
    block.type = BlockType.TABLE
    block.payload = {"table_body": "<table><tr><th>Revenue</th><th>Cost</th></tr><tr><td>80</td><td>100</td></tr></table>"}
    write_delivery(tmp_path, ir)
    assert not visual_issues(ir)

    def compare(payload, value):
        assert "80" in payload["delivered_documents"][0]["text"]
        value["blocks"][0].update(status="needs_review", observation="Revenue should be 100; cost should be 80.")
        return value

    model = AuditModel(compare)
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert report["blocks"][0]["status"] == "needs_review"
    assert not report["summary"]["complete"]


@pytest.mark.asyncio
async def test_budget_limit_leaves_all_other_pages_and_blocks_unreviewed(tmp_path):
    ir, source = fixture(tmp_path, pages=3)
    model = AuditModel()
    output, report = await DocumentReview(config(max_model_calls=1), model, model).run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 1
    assert report["summary"]["verified_pages"] == 1
    assert report["summary"]["not_reviewed_pages"] == 2
    assert report["summary"]["not_reviewed_blocks"] == 2
    assert not finalize_delivery(output, tmp_path, config(max_model_calls=1))["auto_rag_ready"]


@pytest.mark.asyncio
async def test_default_is_enabled_but_explicit_off_makes_no_calls(tmp_path):
    assert PipelineConfig().agentic.mode == "apply"
    ir, source = fixture(tmp_path)
    model = AuditModel()
    _, report = await DocumentReview(config(mode="off"), model, model).run(ir, tmp_path, source_path=source)
    assert not model.calls
    assert not report["summary"]["complete"]
    assert not finalize_delivery(ir, tmp_path, config(mode="off"))["auto_rag_ready"]


@pytest.mark.asyncio
async def test_resume_reuses_only_matching_source_and_delivery(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel()
    await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 1 and report["usage"]["model_calls"] == 1
    path = tmp_path / "outputs/chunks.jsonl"
    path.write_text(path.read_text().replace("100", "999"))
    await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_cancellation_keeps_reserved_call_cost(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel(cancel=True)
    with pytest.raises(asyncio.CancelledError):
        await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    state = json.loads((tmp_path / "agentic/review_state.json").read_text())
    assert state["usage"]["model_calls"] == 1
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert report["usage"]["model_calls"] == 2


@pytest.mark.asyncio
async def test_model_omission_of_a_block_cannot_pass_page(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel(lambda _, v: {**v, "blocks": []})
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 2  # one bounded final retry; omission still cannot pass
    assert report["summary"]["needs_review_pages"] == 1


@pytest.mark.asyncio
async def test_ir_text_cannot_substitute_for_missing_actual_output(tmp_path):
    ir, source = fixture(tmp_path)
    (tmp_path / "outputs/documents/main.md").write_text("Unrelated text.")
    (tmp_path / "outputs/chunks.jsonl").write_text("")

    def compare(payload, value):
        assert payload["blocks"][0]["untrusted_extracted_text"]
        assert not payload["delivered_documents"] and not payload["delivered_chunks"]
        return {**value, "source_complete": False, "issues": ["No delivered source content."]}

    model = AuditModel(compare)
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert report["summary"]["needs_review_pages"] == 1


@pytest.mark.asyncio
async def test_final_validation_rejects_changed_document_or_source_image(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel()
    await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    main = tmp_path / "outputs/documents/main.md"
    original = main.read_text()
    main.write_text(original.replace("100", "999"))
    assert not finalize_delivery(ir, tmp_path, config())["final_validation_passed"]
    main.write_text(original)
    report = json.loads((tmp_path / "outputs/review_coverage.json").read_text())
    evidence = tmp_path / report["pages"][0]["images"][0]["path"]
    evidence.write_bytes(b"changed")
    assert not finalize_delivery(ir, tmp_path, config())["final_validation_passed"]


@pytest.mark.asyncio
async def test_source_inventory_missing_is_never_certified_by_existing_ir(tmp_path):
    ir, source = fixture(tmp_path)
    source.unlink()
    model = AuditModel()
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path)
    assert not report["inventory_complete"]
    assert not report["summary"]["complete"]
    assert report["blocks"][0]["status"] == "not_reviewed"


@pytest.mark.asyncio
async def test_word_text_only_is_converted_and_mapped_to_physical_pages(tmp_path, monkeypatch):
    ir, pdf = fixture(tmp_path, pages=2)
    word = tmp_path / "original.docx"
    word.write_bytes(b"word conversion fixture")
    ir.page_kind = "logical"
    for block in ir.blocks:
        block.page_idx = 0
    from app.pipeline.stages.parse import ParseStage
    convert = AsyncMock(return_value=(pdf, None))
    monkeypatch.setattr(ParseStage, "_convert_office_to_pdf", convert)
    inventory = SourceInventory(ir, tmp_path, word)
    await inventory.prepare()
    assert inventory.complete and inventory.pages == [0, 1]
    assert inventory.mapping == {"b0": [0], "b1": [1]}
    convert.assert_awaited_once()


def test_repeated_word_labels_are_not_assigned_a_guessed_page():
    block = Block(block_id="text", type=BlockType.TEXT, page_idx=0, payload={"text": "Repeated heading"})
    assert word_block_pages(block, ["repeatedheading", "repeatedheading"]) == []


@pytest.mark.asyncio
async def test_successful_local_repair_requires_fresh_final_page_review(tmp_path):
    from test_agentic_repair import Model
    ir, source = fixture(tmp_path)
    ir.blocks[0].type = BlockType.TABLE
    ir.blocks[0].payload = {"table_body": "<table><tr><td>工作</td><td>時間</td></tr><tr><td>市場調查</td><td>1月至3月</td></tr></table>"}
    write_delivery(tmp_path, ir)

    class MixedModel(Model):
        def __init__(self):
            super().__init__()
            self.audits = 0

        async def query(self, task, payload, image, schema):
            if task == "audit_page":
                self.audits += 1
                result, tokens = await AuditModel().query(task, payload, image, schema)
                if self.audits == 1:
                    result["blocks"][0].update(status="needs_review", observation="Output relationship requires reconstruction.")
                return result, tokens
            return await super().query(task, payload, image, schema)

    model = MixedModel()
    rebuilds = []

    async def rebuild(repaired):
        from app.pipeline.stages.chunk import ChunkStage
        rebuilds.append(True)
        text = ChunkStage(config())._block_to_text(repaired.blocks[0])
        (tmp_path / "outputs/documents/main.md").write_text(text)
        (tmp_path / "outputs/chunks.jsonl").write_text(json.dumps({
            "chunk_id": "fixed", "block_ids": ["b0"], "page_indices": [0], "content": text,
        }))

    _, report = await DocumentReview(config(max_model_calls=5, recovery_enabled=False), model, model).run(
        ir, tmp_path, source_path=source, rebuild=rebuild)
    assert model.audits == 2 and rebuilds
    assert model.calls == ["repair", "decide", "verify"]
    assert report["usage"]["model_calls"] == 5
    assert report["summary"]["verified_pages"] == 1


@pytest.mark.asyncio
async def test_shadow_never_changes_original_even_when_review_fails(tmp_path):
    ir, source = fixture(tmp_path)
    original = ir.to_dict()
    model = AuditModel(lambda _, value: {**value, "source_complete": False, "issues": ["Missing paragraph"]})
    output, report = await DocumentReview(config(mode="shadow"), model, model).run(ir, tmp_path, source_path=source)
    assert output is ir and ir.to_dict() == original
    assert not report["summary"]["complete"]



@pytest.mark.asyncio
async def test_continuation_delivery_assigned_to_previous_page_is_available(tmp_path):
    ir, source = fixture(tmp_path, pages=2)
    ir.blocks.pop()
    ir.blocks[0].type = BlockType.TABLE
    ir.blocks[0].payload = {"table_body": "<table><tr><td>First page</td><td>100</td></tr><tr><td>Continuation</td><td>80</td></tr></table>"}
    write_delivery(tmp_path, ir)

    def compare(payload, value):
        if payload["page_idx"] == 1:
            assert not payload["required_block_ids"]
            assert payload["context_blocks"][0]["block_id"] == "b0"
            assert any("Continuation" in p["text"] for p in payload["delivered_documents"])
            assert any("Continuation" in c["content"] for c in payload["delivered_chunks"])
        return value

    model = AuditModel(compare)
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert report["summary"]["verified_pages"] == 2


@pytest.mark.asyncio
async def test_invalid_review_remains_needs_review_when_budget_is_exhausted(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel(lambda _, v: {**v, "blocks": []})
    _, report = await DocumentReview(config(max_model_calls=1), model, model).run(ir, tmp_path, source_path=source)
    assert report["summary"]["needs_review_pages"] == 1
    assert report["summary"]["not_reviewed_pages"] == 0


@pytest.mark.asyncio
async def test_budget_does_not_render_every_unreviewed_page(tmp_path, monkeypatch):
    ir, source = fixture(tmp_path, pages=12)
    original = SourceInventory.image
    requested = []
    def render(inventory, index):
        requested.append(index)
        return original(inventory, index)
    monkeypatch.setattr(SourceInventory, "image", render)
    model = AuditModel()
    _, report = await DocumentReview(config(max_model_calls=1), model, model).run(ir, tmp_path, source_path=source)
    assert set(requested) <= {0, 1}
    assert report["summary"]["not_reviewed_pages"] == 11



def test_unseparated_large_document_does_not_become_one_page_prompt(tmp_path):
    from app.pipeline.review_sources import DeliveryExcerpts
    from app.pipeline.stages.chunk import ChunkStage
    ir, _ = fixture(tmp_path)
    main = tmp_path / "outputs/documents/main.md"
    main.write_text("Source page 1: Revenue 100, cost 80.\n" +
                    "\n".join(f"Unrelated section {i}: company description and background." for i in range(3000)))
    docs, _ = DeliveryExcerpts(tmp_path).for_blocks(ir.blocks, ChunkStage(config()))
    assert any("Revenue 100, cost 80" in d["text"] for d in docs)
    assert sum(len(d["text"]) for d in docs) < 10000
    assert max(len(d["text"]) for d in docs) < 2400



@pytest.mark.asyncio
async def test_repair_resume_preserves_per_block_attempt_limit_and_shared_actions(tmp_path):
    from test_agentic_repair import Model
    ir, source = fixture(tmp_path)
    ir.blocks[0].type = BlockType.TABLE
    ir.blocks[0].payload = {"table_body": "<table><tr><td>工作</td><td>時間</td></tr><tr><td>市場調查</td><td>1月至3月</td></tr></table>"}
    write_delivery(tmp_path, ir)

    class CancelRepair(Model):
        async def query(self, task, payload, image, schema):
            if task == "audit_page":
                value, _ = await AuditModel().query(task, payload, image, schema)
                value["blocks"][0].update(status="needs_review", observation="Table needs review.")
                return value, 10
            if task == "repair":
                self.calls.append(task)
                raise asyncio.CancelledError()
            return await super().query(task, payload, image, schema)

    model = CancelRepair()
    rebuild = AsyncMock()
    for _ in range(2):
        with pytest.raises(asyncio.CancelledError):
            await DocumentReview(config(max_actions=4, recovery_enabled=False), model, model).run(
                ir, tmp_path, source_path=source, rebuild=rebuild)
    # Both attempts were interrupted, and neither may be silently reset.
    _, report = await DocumentReview(config(max_actions=4, recovery_enabled=False), model, model).run(
        ir, tmp_path, source_path=source, rebuild=rebuild)
    assert model.calls == ["repair", "repair"]
    assert report["usage"]["actions"] == 2
    assert not report["summary"]["complete"]



@pytest.mark.asyncio
async def test_page_issue_cannot_coexist_with_certified_block_labels(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel(lambda _, v: {**v, "issues": ["Revenue column is misaligned."]})
    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert report["pages"][0]["status"] == "needs_review"
    assert report["blocks"][0]["status"] == "needs_review"
    assert report["summary"]["verified_blocks"] == 0



@pytest.mark.asyncio
async def test_exhausted_resume_keeps_source_pages_absent_from_ir(tmp_path):
    ir, source = fixture(tmp_path, pages=2)
    ir.pages.pop()
    ir.blocks.pop()
    write_delivery(tmp_path, ir)
    model = AuditModel()
    cfg = config(max_model_calls=1, max_seconds=1)
    _, report = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    state_path = tmp_path / "agentic/review_state.json"
    state = json.loads(state_path.read_text())
    state["usage"]["seconds"] = 2
    state_path.write_text(json.dumps(state))
    _, resumed = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    assert resumed["summary"]["total_pages"] == 2
    assert resumed["inventory_complete"]
    assert resumed["summary"]["not_reviewed_pages"] == 1
    assert len(model.calls) == 1



@pytest.mark.asyncio
@pytest.mark.parametrize("blind_text,complete,verified", [
    ("Source page 1: Revenue 100, cost 80.", True, True),
    ("Source page 1: Revenue 999, cost 80.", True, False),
    ("", False, False),
])
async def test_review_includes_original_word_picture_beyond_the_page_thumbnail(tmp_path, monkeypatch, blind_text, complete, verified):
    from PIL import Image
    ir, source = fixture(tmp_path)
    original = tmp_path / 'agentic/evidence/original-picture.png'
    original.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (240, 120), 'green').save(original)
    prepare = SourceInventory.prepare

    async def with_original(self, **kwargs):
        await prepare(self, **kwargs)
        self.word_images = {ir.blocks[0].block_id: original}

    monkeypatch.setattr(SourceInventory, 'prepare', with_original)
    class BlindAuditModel(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == 'verify_native_reading':
                self.calls.append((task, payload, image))
                # Blind: no caption, prose or prior output; only the document language code.
                assert set(payload) == {'sources', 'image_order', 'source_relation_contract', 'relationship_language'}
                return {'items': [{'source_id': sid, 'text': blind_text, 'complete': complete,
                                   'kind': 'transcription', 'observation': 'Read original lettering.',
                                   'image_type': 'other', 'relationships': [], 'relationships_complete': True}
                                  for sid in payload['image_order']]}, 100
            return await super().query(task, payload, image, schema)

    model = BlindAuditModel()
    reviewer = DocumentReview(config(), model, model)
    _, report = await reviewer.run(ir, tmp_path, source_path=source)
    assert report['summary']['verified_pages'] == int(verified)
    assert report['summary']['needs_review_pages'] == int(not verified)
    _, payload, images = model.calls[-1]
    assert original in images
    label = payload['evidence_images'][images.index(original)]
    assert 'Original Word embedded image for b0' in label
    assert 'untrusted delivered description' in label
    assert len(model.calls) == 2
    final = finalize_delivery(ir, tmp_path, config())
    assert final['auto_rag_ready'] is verified
    if not verified:
        assert not (tmp_path / 'outputs/rag_chunks.jsonl').read_text().strip()
        assert 'Revenue 100' not in (tmp_path / 'outputs/delivery.md').read_text()
    # An unrelated rebuild can re-audit outputs without repeating the same blind source read.
    inventory = SourceInventory(ir, tmp_path, source)
    await inventory.prepare()
    await reviewer._word_picture_readings(ir.blocks, inventory)
    assert len(model.calls) == 2
    Image.new('RGB', (240, 120), 'blue').save(original)
    await reviewer._word_picture_readings(ir.blocks, inventory)
    assert len(model.calls) == 3
