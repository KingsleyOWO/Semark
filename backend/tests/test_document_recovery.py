"""Source-grounded recovery tests: routing, rollback, budgets, resume and four real tool paths."""
import asyncio
import json
from unittest.mock import AsyncMock

import fitz
import pytest
from test_document_review import AuditModel, fixture, write_delivery

from app.config import AgenticConfig, PipelineConfig
from app.models.document_ir import Block, BlockType
from app.pipeline.document_recovery import RecoveryController, restore_recovery
from app.pipeline.document_review import DocumentReview
from app.pipeline.review_sources import SourceInventory


@pytest.fixture(autouse=True)
def isolate_pdf_parser(monkeypatch, tmp_path):
    # Preserve the real adapter for cancellation tests, but never use the installed CLI.
    # Tests for successful extraction supply their own parser or executable.
    from app.config import settings
    monkeypatch.setattr(settings, "mineru_cli_path", str(tmp_path / "unavailable-mineru-cli"))


def cfg(**changes):
    return PipelineConfig(agentic=AgenticConfig(max_model_calls=40, max_seconds=600, **changes))


class RecoveryModel(AuditModel):
    def __init__(self, action="read_regions", reject=False, actions=None):
        super().__init__()
        self.action, self.reject, self.actions = action, reject, list(actions or [])
        self.tasks = []

    async def query(self, task, payload, image, schema):
        self.tasks.append(task)
        if task == "recovery_plan":
            action = self.actions.pop(0) if self.actions else self.action
            return {"action": action, "layout": "horizontal2", "reason": "Source page is missing substantive content."}, 10
        if task == "read_region":
            return {"text": "Revenue 100, cost 80." if payload["core_bbox"][1] == 0 else "End of report.",
                    "complete": True, "observation": "Source crop supports this content."}, 20
        if task == "merge_regions":
            return {"text": "\n".join(r["text"] for r in payload["readings"]),
                    "complete": True, "observation": "All source content retained without overlap."}, 10
        if task == "match_source":
            return {"page_idx": payload["candidate_pages"][0], "ambiguous": False,
                    "evidence": "Source page 1: Revenue 100, cost 80."}, 10
        if task == "verify_match":
            return {"supported": not self.reject, "ambiguous": self.reject,
                    "observation": "Matched distinctive figures to source image."}, 10
        if task == "repair_delivery":
            blocks = payload["blocks"]
            return {"complete": True, "edits": [{"block_id": blocks[1]["block_id"],
                    "duplicate_of": blocks[0]["block_id"], "evidence_ids": payload["source_evidence_ids"], "reason": "The source contains this text once."}]}, 10
        if task == "audit_page":
            data, tokens = await super().query(task, payload, image, schema)
            if self.reject:
                data.update(source_complete=False, issues=["Source content is missing."])
            return data, tokens
        raise AssertionError(task)


async def setup(tmp_path, model=None, config=None):
    ir, source = fixture(tmp_path)
    controller = DocumentReview(config or cfg(), model or RecoveryModel(), model or RecoveryModel())
    controller.state_path = tmp_path / "agentic/review_state.json"
    controller.state = {"usage": {"model_calls": 0, "seconds": 0, "actions": 0, "tokens": 0},
                        "calls": [], "pages": {"0": {"page_idx": 0, "status": "needs_review",
                        "reason": "source_delivery_mismatch", "review": {"issues": ["Missing content."]}}}}
    controller._phase = "recovery"
    inventory = SourceInventory(ir, tmp_path, source)
    await inventory.prepare()

    async def rebuild(candidate):
        write_delivery(tmp_path, candidate)

    async def renderer(config, candidate, folder, original, cache):
        write_delivery(folder, candidate)

    runner = RecoveryController(controller, inventory, tmp_path, rebuild, renderer=renderer)
    return ir, controller, runner


@pytest.mark.asyncio
async def test_regions_recover_omitted_page_and_verify_actual_output(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    original = ir.to_dict()
    result = await runner.run(ir)
    assert result.to_dict() != original and ir.to_dict() == original
    assert len(result.blocks) == 1
    assert runner.state["steps"][0]["status"] == "applied"
    assert "Revenue 100" in (tmp_path / "outputs/documents/main.md").read_text()
    restored, changed = restore_recovery(owner, ir, tmp_path)
    assert changed and restored.to_dict() == result.to_dict()
    snapshot = tmp_path / runner.state["accepted"]["file"]
    snapshot.write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        restore_recovery(owner, ir, tmp_path)


@pytest.mark.asyncio
async def test_failed_candidate_rolls_back_and_planner_can_choose_another_tool(tmp_path, monkeypatch):
    model = RecoveryModel(reject=True, actions=["read_regions", "reparse_pdf", "defer"])
    ir, owner, runner = await setup(tmp_path, model)
    before = (tmp_path / "outputs/documents/main.md").read_bytes()
    parse = AsyncMock(side_effect=ValueError("parser unavailable"))
    monkeypatch.setattr(runner, "reparse_pdf", parse)
    result = await runner.run(ir)
    assert result is ir and (tmp_path / "outputs/documents/main.md").read_bytes() == before
    assert [s["action"] for s in runner.state["steps"]] == ["read_regions", "reparse_pdf"]
    assert all(s["status"] == "rejected" for s in runner.state["steps"])
    assert runner.state["issues"]["page:0"]["status"] == "deferred"
    assert "accepted" not in runner.state
    assert owner.state["pages"]["0"]["status"] == "needs_review"


@pytest.mark.asyncio
async def test_same_evidence_and_tool_cannot_loop_on_failure(tmp_path):
    ir, owner, runner = await setup(tmp_path, RecoveryModel(reject=True))
    await runner.run(ir)
    assert runner.state["actions"] == 3
    assert [s["action"] for s in runner.state["steps"]] == ["read_regions", "read_source", "reparse_pdf"]
    assert runner.state["issues"]["page:0"]["status"] == "attempt_limit"
    before = runner.state["actions"]
    await runner.run(ir)
    assert runner.state["actions"] == before


@pytest.mark.asyncio
async def test_verified_source_mapping_does_not_certify_delivery(tmp_path):
    ir, owner, runner = await setup(tmp_path, RecoveryModel(action="resolve_source"))
    owner.state["pages"] = {}
    ir.page_kind = "logical"
    runner.inventory.mapping = {"b0": []}
    result = await runner.run(ir)
    assert result.blocks[0].payload["agentic_source_pages"] == [0]
    assert runner.state["issues"]["block:b0"]["status"] == "located"
    # Locating a block is followed by a separate audit of the real rebuilt delivery.
    assert owner.model.tasks[-1] == "audit_page"
    assert runner.state["steps"][0]["result"]["source_complete"] is False
    fresh = SourceInventory(result, tmp_path, runner.inventory.pdf)
    await fresh.prepare()
    assert fresh.mapping["b0"] == [0]


@pytest.mark.asyncio
async def test_ambiguous_mapping_is_not_adopted(tmp_path):
    ir, owner, runner = await setup(tmp_path, RecoveryModel(action="resolve_source", reject=True))
    owner.state["pages"] = {}
    runner.inventory.mapping = {"b0": []}
    result = await runner.run(ir)
    assert result is ir
    assert "agentic_source_pages" not in ir.blocks[0].payload


@pytest.mark.asyncio
async def test_pdf_reparse_extracts_only_target_source_page(tmp_path, monkeypatch):
    from app.adapters.mineru import MinerUAdapter, MinerUResult
    ir, owner, runner = await setup(tmp_path, RecoveryModel(action="reparse_pdf"))
    calls = []

    async def parse(self, source, output, config):
        with fitz.open(source) as pdf:
            calls.append(len(pdf))
            assert "Revenue 100" in pdf[0].get_text()
        output.mkdir(parents=True)
        path = output / "content_list.json"
        path.write_text(json.dumps([{"type": "text", "text": "Revenue 100, cost 80.",
                                     "bbox": [0, 0, 1000, 1000], "page_idx": 0}]))
        return MinerUResult(success=True, output_dir=output, content_list_path=path)

    monkeypatch.setattr(MinerUAdapter, "parse", parse)
    result = await runner.run(ir)
    assert calls == [1]
    assert result.blocks[0].payload["agentic_recovery_tool"] == "reparse_pdf"
    assert runner.state["steps"][0]["status"] == "applied"


@pytest.mark.asyncio
async def test_duplicate_rebuild_retains_one_identical_block(tmp_path):
    ir, owner, runner = await setup(tmp_path, RecoveryModel(action="rebuild_delivery"))
    duplicate = ir.blocks[0].model_copy(deep=True)
    duplicate.block_id = "duplicate"
    ir.blocks.append(duplicate)
    runner.inventory.mapping["duplicate"] = [0]
    write_delivery(tmp_path, ir)
    result = await runner.run(ir)
    assert [b.block_id for b in result.blocks] == ["b0"]
    assert len(ir.blocks) == 2


@pytest.mark.asyncio
async def test_candidate_cannot_pass_using_ir_instead_of_real_delivery(tmp_path):
    ir, owner, runner = await setup(tmp_path)

    async def broken(config, candidate, folder, original, cache):
        write_delivery(folder, candidate)
        (folder / "outputs/chunks.jsonl").write_text("")

    runner.renderer = broken
    result = await runner.run(ir)
    assert result is ir
    assert runner.state["steps"][0]["status"] == "rejected"
    assert "audit_page" not in owner.model.tasks


@pytest.mark.asyncio
async def test_shadow_saves_candidate_without_changing_delivery(tmp_path):
    ir, owner, runner = await setup(tmp_path, config=cfg(mode="shadow"))
    before = (tmp_path / "outputs/documents/main.md").read_bytes()
    result = await runner.run(ir)
    assert result is ir
    assert (tmp_path / "outputs/documents/main.md").read_bytes() == before
    assert runner.state["steps"][0]["status"] == "candidate_only"
    assert "accepted" not in runner.state


@pytest.mark.asyncio
async def test_scope_validation_rejects_cross_page_replacement(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    runner.inventory.mapping["b0"] = [0, 1]
    result = await runner.run(ir)
    assert result is ir
    assert runner.state["actions"] == 0
    assert not runner.state["steps"]
    assert "read_region" not in owner.model.tasks


@pytest.mark.asyncio
async def test_budget_preflight_prevents_unverifiable_tool_execution(tmp_path):
    config = cfg()
    config.agentic.max_model_calls = 3
    model = RecoveryModel()
    ir, owner, runner = await setup(tmp_path, model, config)
    await runner.run(ir)
    assert model.tasks == ["recovery_plan"]
    assert runner.state["actions"] == 0
    assert runner.state["issues"]["page:0"]["status"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_cancel_preserves_action_and_does_not_publish_candidate(tmp_path, monkeypatch):
    ir, owner, runner = await setup(tmp_path)
    monkeypatch.setattr(runner, "read_regions", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await runner.run(ir)
    assert runner.state["actions"] == 1
    assert runner.state["steps"][0]["status"] == "interrupted"
    assert "accepted" not in runner.state


@pytest.mark.asyncio
async def test_pipeline_routes_text_problem_to_model_chosen_tool(tmp_path, monkeypatch):
    import app.pipeline.document_recovery as module
    ir, source = fixture(tmp_path)

    class Integrated(RecoveryModel):
        async def query(self, task, payload, image, schema):
            data, tokens = await super().query(task, payload, image, schema)
            if task == "audit_page" and payload["required_block_ids"] == ["b0"]:
                data.update(source_complete=False, issues=["Missing end of report."])
            return data, tokens

    async def renderer(config, candidate, folder, original, cache):
        write_delivery(folder, candidate)
    monkeypatch.setattr(module, "render_candidate", renderer)
    model = Integrated()
    result, report = await DocumentReview(cfg(), model, model).run(
        ir, tmp_path, source_path=source,
        rebuild=lambda candidate: asyncio.to_thread(write_delivery, tmp_path, candidate))
    assert report["summary"]["verified_pages"] == 1
    assert report["recovery"]["steps"][0]["action"] == "read_regions"
    assert all(b.block_id.startswith("recovery_") for b in result.blocks)


@pytest.mark.asyncio
async def test_repair_acceptance_survives_unrelated_page_pending_status(tmp_path):
    from app.pipeline.agentic_content import replacement_text
    block = Block(block_id="image", type=BlockType.IMAGE, page_idx=0, payload={
        "agentic_needs_review": True, "verified_semantics": "Chart value: 100.",
        "agentic_evidence": {"tool_version": "visual-repair-v2", "status": "complete"},
    })
    assert replacement_text(block) == "Chart value: 100."


@pytest.mark.asyncio
async def test_real_package_and_chunk_stages_render_candidate(tmp_path, monkeypatch):
    from app.pipeline.document_recovery import render_candidate
    ir, owner, runner = await setup(tmp_path)
    from app.pipeline.stages.package import PackageStage
    original_package = PackageStage.run

    async def guarded_package(stage, *args, **kwargs):
        assert not stage.config.enrich.enable_vlm
        assert not stage.config.package.enable_semantic_repair
        return await original_package(stage, *args, **kwargs)

    monkeypatch.setattr(PackageStage, "run", guarded_package)
    runner.renderer = render_candidate
    async def consistent_rebuild(candidate):
        # Production commit must reproduce exactly the delivery audited in isolation.
        from app.pipeline.stages.chunk import ChunkStage
        from app.pipeline.stages.package import PackageStage
        package_config = owner.config.model_copy(deep=True)
        package_config.enrich.enable_vlm = False
        package_config.package.enable_semantic_repair = False
        await PackageStage(package_config).run(candidate.doc_id, candidate.run_id, candidate, tmp_path)
        await ChunkStage(owner.config).run(candidate.doc_id, candidate.run_id, candidate, tmp_path)
    runner.rebuild = consistent_rebuild
    result = await runner.run(ir)
    assert result is not ir, runner.state
    candidate = next(runner.folder.glob("*/candidate_ir.json")).parent
    assert (candidate / "outputs/documents/main.md").is_file()
    assert (candidate / "outputs/chunks.jsonl").is_file()
    assert "100" in (candidate / "outputs/chunks.jsonl").read_text()


@pytest.mark.asyncio
async def test_reserved_calls_return_to_final_review_when_no_recovery_needed(tmp_path):
    ir, source = fixture(tmp_path, pages=12)
    model = AuditModel()
    config = PipelineConfig(agentic=AgenticConfig(max_model_calls=18, recovery_reserve_calls=8))
    _, report = await DocumentReview(config, model, model).run(
        ir, tmp_path, source_path=source, rebuild=AsyncMock())
    assert report["summary"]["verified_pages"] == 12
    assert report["usage"]["model_calls"] == 12


@pytest.mark.asyncio
async def test_dense_page_requires_all_regions_and_global_order_verification(tmp_path):
    from app.pipeline.review_sources import validate_review_coverage

    class DenseModel(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == "verify_layout":
                self.calls.append((task, payload, image))
                return {"reading_order_correct": True, "observation": "Global order matches source.", "issues": []}, 10
            return await super().query(task, payload, image, schema)

    blocks = [Block(block_id=f"b{i}", type=BlockType.TEXT, page_idx=0, reading_order=i,
                    bbox_norm=[0, i * 1000 // 60, 1000, (i + 1) * 1000 // 60],
                    payload={"text": f"Source row {i}: revenue {i + 100}, cost {i + 80}."}) for i in range(60)]
    ir, source = fixture(tmp_path, blocks=blocks)
    model = DenseModel()
    _, report = await DocumentReview(cfg(), model, model).run(ir, tmp_path, source_path=source)
    assert report["summary"]["verified_pages"] == 1
    assert len(report["pages"][0]["regions"]) == 4
    assert model.calls[-1][0] == "verify_layout"
    assert validate_review_coverage(ir, tmp_path, cfg())["complete"]
    path = tmp_path / "outputs/review_coverage.json"
    report["pages"][0]["regions"].pop()
    path.write_text(json.dumps(report))
    assert not validate_review_coverage(ir, tmp_path, cfg())["complete"]


@pytest.mark.asyncio
async def test_commit_failure_aborts_and_resume_keeps_previous_publication(tmp_path):
    from app.pipeline.document_recovery import RecoveryCommitError
    ir, owner, runner = await setup(tmp_path)
    runner.rebuild = AsyncMock(side_effect=RuntimeError("disk error"))
    with pytest.raises(RecoveryCommitError):
        await runner.run(ir)
    restored, changed = restore_recovery(owner, ir, tmp_path)
    assert not changed and restored is ir
    assert "accepted" not in runner.state and "pending_commit" not in runner.state
    assert runner.state["steps"][0]["status"] == "commit_rolled_back"


@pytest.mark.asyncio
async def test_tool_timeout_is_charged_and_stops_further_work(tmp_path, monkeypatch):
    config = cfg()
    config.agentic.max_seconds = 0.03
    # Force an underestimated operation to exercise the actual deadline, independently of preflight.
    import app.pipeline.review_budget as budgets
    monkeypatch.setattr(budgets, "tool_cost", lambda *a, **kw: {"model_calls": 2, "seconds": 0.001})
    monkeypatch.setattr(budgets, "call_seconds", lambda *a: 0.001)
    ir, owner, runner = await setup(tmp_path, config=config)

    async def slow(*args):
        await asyncio.sleep(2)
    monkeypatch.setattr(runner, "read_regions", slow)
    result = await runner.run(ir)
    assert result is ir and "accepted" not in runner.state
    assert runner.state["actions"] == 1
    assert owner.state["usage"]["seconds"] >= config.agentic.max_seconds


@pytest.mark.asyncio
async def test_cancel_pdf_tool_terminates_parser_process(tmp_path, monkeypatch):
    import sys

    from app.adapters.mineru import MinerUAdapter
    adapter = MinerUAdapter()
    adapter.cli_path = sys.executable
    monkeypatch.setattr(adapter, "_build_args", lambda *args: ["-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(adapter, "_build_env", lambda *args: {})
    created = []
    spawn = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    task = asyncio.create_task(adapter.parse(tmp_path / "source.pdf", tmp_path / "parse"))
    for _ in range(100):
        if created:
            break
        await asyncio.sleep(0.01)
    assert created
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].returncode is not None


@pytest.mark.asyncio
async def test_region_assembly_cannot_drop_values(tmp_path):
    class DroppingModel(RecoveryModel):
        async def query(self, task, payload, image, schema):
            if task == "merge_regions":
                return {"text": "Revenue 100.", "complete": True, "observation": "Merged."}, 10
            return await super().query(task, payload, image, schema)
    ir, owner, runner = await setup(tmp_path, DroppingModel(), cfg(max_recovery_actions=1))
    result = await runner.run(ir)
    assert result is not ir
    assert "Revenue 100, cost 80." in result.blocks[0].get_text()
    assert "End of report." in result.blocks[0].get_text()
    assert "merge_regions" not in owner.model.tasks


@pytest.mark.asyncio
async def test_new_recovery_disabled_keeps_legacy_review_path(tmp_path):
    ir, source = fixture(tmp_path)
    model = RecoveryModel(reject=True)
    _, report = await DocumentReview(cfg(recovery_enabled=False), model, model).run(
        ir, tmp_path, source_path=source, rebuild=AsyncMock())
    assert "recovery_plan" not in model.tasks
    assert not report["recovery"]


@pytest.mark.asyncio
async def test_generic_image_marker_can_be_replaced_by_full_visual_description(tmp_path):
    class VisualModel(RecoveryModel):
        async def query(self, task, payload, image, schema):
            if task == "read_region":
                text = ("Revenue 100, cost 80. [Image: Product display]" if payload["core_bbox"][1] == 0
                        else "Wooden cabinet containing puzzle boxes.")
                return {"text": text, "complete": True, "observation": "Visible source content."}, 10
            if task == "merge_regions":
                return {"text": "Revenue 100, cost 80. Wooden cabinet containing puzzle boxes.",
                        "complete": True, "observation": "Detailed visual description replaces generic marker."}, 10
            return await super().query(task, payload, image, schema)
    ir, owner, runner = await setup(tmp_path, VisualModel())
    result = await runner.run(ir)
    assert result is not ir
    assert "Wooden cabinet" in result.blocks[0].get_text()


@pytest.mark.asyncio
async def test_word_replacement_preserves_logical_page_and_physical_evidence(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    ir.page_kind = "logical"
    runner.inventory.mapping["b0"] = [7]
    block = Block(block_id="new", type=BlockType.TEXT, page_idx=7,
                  payload={"text": "Revenue 100, cost 80.", **runner.source_payload(7)})
    candidate = runner.replace_page(ir, 7, [block])
    assert candidate.blocks[0].page_idx == 0
    assert candidate.blocks[0].payload["agentic_source_pages"] == [7]


@pytest.mark.asyncio
async def test_omitted_pdf_page_recovery_adds_page_to_inventory(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    block = Block(block_id="new", type=BlockType.TEXT, page_idx=1,
                  payload={"text": "Previously omitted source page.", **runner.source_payload(1)})
    candidate = runner.replace_page(ir, 1, [block])
    assert [p.page_idx for p in candidate.pages] == [0, 1]
    assert [b.block_id for b in candidate.blocks] == ["b0", "new"]


def test_candidate_and_committed_packaging_filter_superseded_enrichment_identically(tmp_path):
    from app.pipeline.agentic_content import filter_recovery_enrichments
    ir, _ = fixture(tmp_path)
    ir.blocks[0].payload.update(agentic_recovery_tool="read_regions", agentic_source_pages=[0])
    entries = {"b0": {"kind": "figure"}, "b_removed": {"kind": "figure"},
               "form0": {"kind": "form_asset", "input": {"page_idx": 0}},
               "form1": {"kind": "form_asset", "input": {"page_idx": 1}}}
    assert set(filter_recovery_enrichments(ir, entries)) == {"b0", "form1"}
    assert set(entries) == {"b0", "b_removed", "form0", "form1"}


class NativeModel(RecoveryModel):
    async def query(self, task, payload, image, schema):
        if task in {'read_image_parts', 'read_image_parts_alternate', 'verify_image_parts'}:
            self.tasks.append(task)
            return {'image_type': 'other', 'visual_facts': [], 'visual_independent_of_text': False,
                    'text_regions': [], 'complete': False, 'observation': 'Image content is unreadable.'}, 10
        if task in {'read_source', 'read_source_alternate', 'verify_native_reading'}:
            self.tasks.append(task)
            return {'items': [{'source_id': s['id'], 'text': s.get('text', ''), 'complete': bool(s.get('text')),
                               'observation': 'Native text is legible.' if s.get('text') else 'Image text is unreadable.',
                               'kind': 'transcription'} for s in payload['sources']]}, 10
        if task == 'audit_page':
            data, tokens = await super().query(task, payload, image, schema)
            gaps = {g['block_id'] for g in payload.get('declared_gaps', [])}
            if gaps:
                data.update(source_complete=False, declared_gaps_only=True, issues=['Declared unreadable image.'])
            for b in data['blocks']:
                b.update(status='needs_review' if b['block_id'] in gaps else 'verified',
                         context_complete=b['block_id'] not in gaps)
            return data, tokens
        return await super().query(task, payload, image, schema)


@pytest.mark.asyncio
async def test_native_page_reading_uses_original_text_and_real_candidate_output(tmp_path):
    model = NativeModel(action='read_source')
    ir, owner, runner = await setup(tmp_path, model)
    result = await runner.run(ir)
    assert result is not ir
    assert result.blocks[0].payload['agentic_recovery_tool'] == 'read_source'
    assert 'Source page 1: Revenue 100, cost 80.' in result.blocks[0].get_text()
    assert runner.state['steps'][0]['status'] == 'applied'
    assert model.tasks.count('read_source') == 0  # Native text needs no generative transcription.


@pytest.mark.asyncio
async def test_native_partial_candidate_retains_gap_and_original_image_without_guessed_text(tmp_path):
    from PIL import Image

    from app.pipeline.recovery_evidence import native_materials
    model = NativeModel(action='read_source')
    ir, owner, runner = await setup(tmp_path, model)
    image = tmp_path / 'unreadable.png'
    Image.new('RGB', (480, 240), 'white').save(image)
    pdf_path = tmp_path / 'with-image.pdf'
    with fitz.open(runner.inventory.pdf) as doc:
        doc[0].insert_image(fitz.Rect(50, 120, 450, 320), filename=image)
        doc.save(pdf_path)
    runner.inventory.pdf = pdf_path
    import hashlib
    runner.source_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    result = await runner.run(ir)
    assert result is not ir
    gaps = [b for b in result.blocks if b.payload.get('agentic_gap')]
    assert len(gaps) == 1 and '無法可靠辨識' in gaps[0].get_text()
    assert (tmp_path / gaps[0].payload['agentic_source_image']).is_file()
    assert runner.state['steps'][0]['status'] == 'applied'
    assert len(native_materials(runner.inventory, 0, tmp_path / 'native')) == 2
    assert owner.state['pages'].get('0', {}).get('status') != 'verified'
    assert runner.state['issues']['page:0']['status'] == 'applied_partial'
    actions = runner.state['actions']
    await runner.run(result)
    assert runner.state['actions'] == actions


@pytest.mark.asyncio
async def test_native_image_reading_cannot_invent_another_source(tmp_path, monkeypatch):
    class Missing(NativeModel):
        async def query(self, task, payload, image, schema):
            value, tokens = await super().query(task, payload, image, schema)
            if task == 'read_image_parts':
                value['source_id'] = 'invented'
            return value, tokens
    ir, owner, runner = await setup(tmp_path, Missing(action='read_source'), cfg(max_recovery_actions=1))
    from app.pipeline import recovery_evidence
    monkeypatch.setattr(recovery_evidence, 'native_materials', lambda *args: [
        {'id': 'image-0', 'text': '', 'bbox': [0, 0, 1000, 1000], 'kind': 'pdf_embedded_image',
         'image': runner.inventory.image(0)}])
    result = await runner.run(ir)
    assert result is ir
    assert runner.state['steps'][0]['status'] == 'response_format_error'
    assert 'extra_forbidden' in runner.state['steps'][0]['result']


@pytest.mark.asyncio
async def test_native_materials_ignore_unused_pdf_image_resources(tmp_path):
    from PIL import Image

    from app.pipeline.recovery_evidence import native_materials
    ir, owner, runner = await setup(tmp_path)
    image = tmp_path / 'used.png'
    Image.new('RGB', (400, 250), 'green').save(image)
    path = tmp_path / 'resources.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_image(fitz.Rect(20, 20, 420, 270), filename=image)
        for xref in page.get_contents():
            pdf.update_stream(xref, b'')
        assert page.get_images() and not page.get_image_info(xrefs=True)
        pdf.save(path)
    runner.inventory.pdf = path
    assert native_materials(runner.inventory, 0, tmp_path / 'extracted') == []


@pytest.mark.asyncio
async def test_alternate_reader_uses_only_configured_review_endpoint_and_shared_budget(tmp_path, monkeypatch):
    from app.pipeline.recovery_models import RecoveryDecision
    primary, alternate = NativeModel(action='read_source'), NativeModel(action='read_source')
    ir, owner, runner = await setup(tmp_path, primary)
    owner.reviewer = alternate
    owner.config.review_vlm.model = 'configured-alternate'
    from app.pipeline import recovery_evidence
    monkeypatch.setattr(recovery_evidence, 'native_materials', lambda *args: [
        {'id': 'image-0', 'text': '', 'bbox': [0, 0, 1000, 1000], 'kind': 'pdf_embedded_image',
         'image': runner.inventory.image(0)}])
    decision = runner.select_strategy(RecoveryDecision(action='read_source', model_role='review', reason='Try other reader.'), [], ['read_source'])
    await runner.read_source(ir, {'page_idx': 0}, decision, tmp_path / 'alternate', 'Read source.')
    assert primary.tasks == [] and alternate.tasks == ['read_image_parts_alternate', 'verify_image_parts']
    assert owner.state['usage']['model_calls'] == 2
    owner.config.review_vlm.model = owner.config.vlm.model
    decision = runner.select_strategy(decision, [], ['read_source'])
    assert decision.model_role == 'primary'


@pytest.mark.asyncio
async def test_native_reading_reserves_final_verification_budget(tmp_path):
    from app.pipeline.agentic import BudgetExhausted
    from app.pipeline.recovery_models import RecoveryDecision
    model = NativeModel(action='read_source')
    ir, owner, runner = await setup(tmp_path, model)
    owner.limits.max_model_calls = 1
    with pytest.raises(BudgetExhausted):
        await runner.read_source(ir, {'page_idx': 0}, RecoveryDecision(action='read_source', reason='Read.'),
                                 tmp_path / 'native', 'Read source.')
    assert model.tasks == []


@pytest.mark.asyncio
async def test_full_pipeline_adopts_verified_partial_native_result_and_finishes(tmp_path, monkeypatch):
    from PIL import Image

    from app.pipeline.final_delivery import finalize_delivery
    class EndToEnd(NativeModel):
        async def query(self, task, payload, image, schema):
            data, tokens = await super().query(task, payload, image, schema)
            if task == 'audit_page' and not any(b.startswith('recovery_') for b in payload['required_block_ids']):
                data.update(source_complete=False, issues=['Image was omitted by native parsing.'])
            return data, tokens
    ir, source = fixture(tmp_path)
    image = tmp_path / 'unreadable.png'
    Image.new('RGB', (400, 250), 'white').save(image)
    with fitz.open(source) as pdf:
        pdf[0].insert_image(fitz.Rect(50, 120, 450, 370), filename=image)
        pdf.saveIncr()
    model = EndToEnd(action='read_source')
    configuration = cfg()
    async def renderer(config, candidate, folder, original, cache):
        write_delivery(folder, candidate)
    monkeypatch.setattr("app.pipeline.document_recovery.render_candidate", renderer)
    async def rebuild(candidate):
        write_delivery(tmp_path, candidate)
    result, report = await DocumentReview(configuration, model, model).run(
        ir, tmp_path, source_path=source, rebuild=rebuild)
    final = finalize_delivery(result, tmp_path, configuration)
    assert final['processing_status'] == 'partial' and final['partial_rag_ready']
    assert not final['requires_human_review']
    assert report['recovery']['issues']['page:0']['status'] == 'applied_partial'
    assert report['summary']['needs_review_pages'] == 1
    safe = (tmp_path / 'outputs/rag_chunks.jsonl').read_text()
    assert 'Revenue 100' in safe and '無法可靠辨識' not in safe
    assert '無法可靠辨識' in (tmp_path / 'outputs/delivery.md').read_text()


@pytest.mark.asyncio
async def test_blind_reading_disagreement_becomes_gap_and_is_never_published_as_fact(tmp_path, monkeypatch):
    from app.pipeline import recovery_evidence
    class Disagree(NativeModel):
        async def query(self, task, payload, image, schema):
            if task in {'read_image_parts', 'verify_image_parts'}:
                if task == 'verify_image_parts':
                    assert '999' not in json.dumps(payload)
                    assert 'skill' not in payload
                return {'image_type': 'document', 'visual_facts': [], 'visual_independent_of_text': False,
                        'text_regions': [{'region_id': 'value', 'bbox': [0, 0, 1000, 1000], 'readable': True,
                                          'text': 'Revenue 999.' if task == 'read_image_parts' else 'Revenue 100.'}],
                        'complete': True, 'observation': 'Independent image reading.'}, 10
            return await super().query(task, payload, image, schema)
    ir, owner, runner = await setup(tmp_path, Disagree(action='read_source'))
    monkeypatch.setattr(recovery_evidence, 'native_materials', lambda *args: [
        {'id': 'text-0', 'text': 'Source page 1: Revenue 100, cost 80.', 'bbox': [0, 0, 1000, 200],
         'kind': 'native_text', 'image': None},
        {'id': 'image-0', 'text': '', 'bbox': [0, 200, 1000, 1000], 'kind': 'pdf_embedded_image',
         'image': runner.inventory.image(0)}])
    result = await runner.run(ir)
    assert result is not ir
    assert '999' not in (tmp_path / 'outputs/documents/main.md').read_text()
    assert any(b.payload.get('agentic_gap') for b in result.blocks)
    assert '無法可靠辨識' in (tmp_path / 'outputs/documents/main.md').read_text()


def test_native_wrapped_heading_is_not_exported_as_an_orphan_character(tmp_path):
    from types import SimpleNamespace

    from app.pipeline.recovery_evidence import native_materials
    path = tmp_path / 'source.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 100), 'Heading part')
        page.insert_text((50, 130), 'X')
        pdf.save(path)
    materials = native_materials(SimpleNamespace(pdf=path, source_path=path), 0, tmp_path / 'materials')
    assert len(materials) == 1 and materials[0]['text'] == 'Heading part\nX'


@pytest.mark.asyncio
async def test_candidate_audit_keeps_word_original_picture_checks(tmp_path):
    from PIL import Image

    class WordModel(RecoveryModel):
        async def query(self, task, payload, image, schema):
            if task == 'verify_native_reading':
                self.tasks.append(task)
                return {'items': [{'source_id': 'image0', 'text': 'Actual original sign',
                                   'complete': True, 'kind': 'transcription',
                                   'observation': 'Read the original image.', 'image_type': 'photo',
                                   'relationships': [], 'relationships_complete': True}]}, 10
            return await super().query(task, payload, image, schema)

    model = WordModel()
    ir, owner, runner = await setup(tmp_path, model)
    block = ir.blocks[0]
    block.type = BlockType.IMAGE
    block.payload = {'caption': 'An invented shop name', 'word_source': {'image_sha256': 'original'}}
    original = tmp_path / 'agentic/evidence/word-picture.png'
    Image.new('RGB', (40, 40), 'green').save(original)
    runner.inventory.word_images[block.block_id] = original
    folder = tmp_path / 'candidate'
    folder.mkdir()
    async def picture_renderer(config, candidate, folder, original, cache):
        from app.pipeline.stages.chunk import ChunkStage
        text = ChunkStage(config)._block_to_text(candidate.blocks[0])
        write_delivery(folder, candidate)
        (folder / 'outputs/documents/main.md').write_text(text)
        (folder / 'outputs/chunks.jsonl').write_text(json.dumps({
            'chunk_id': 'image', 'block_ids': [block.block_id], 'content': text}))

    runner.renderer = picture_renderer
    assert not await runner.validate(ir.model_copy(deep=True), [0], folder)
    assert 'verify_native_reading' in model.tasks
    evidence = folder / original.relative_to(tmp_path)
    assert evidence.read_bytes() == original.read_bytes()
    record = json.loads((folder / 'validation-page-0.json').read_text())
    assert record['review']['blocks'][0]['status'] == 'needs_review'


@pytest.mark.asyncio
async def test_dense_band_and_long_order_check_are_reviewed_in_parts(tmp_path):
    """2026-10-08: seven image-dense pages in four reports were never reviewed: a band had
    over 45 blocks, or the order check's delivered boundaries ran 55-109k characters."""
    from app.pipeline.document_review import LAYOUT_PAYLOAD_LIMIT
    from app.pipeline.review_sources import validate_review_coverage

    class PartsModel(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == "verify_layout":
                self.calls.append((task, payload, image))
                return {"reading_order_correct": True, "observation": "Order matches source.", "issues": []}, 10
            return await super().query(task, payload, image, schema)

    # Sixty long paragraphs, all in the top quarter of the page.
    blocks = [Block(block_id=f"b{i}", type=BlockType.TEXT, page_idx=0, reading_order=i,
                    bbox_norm=[0, i * 4, 1000, i * 4 + 3],
                    payload={"text": f"Source row {i}: " + " ".join(f"item{i}-{j}" for j in range(60))})
              for i in range(60)]
    ir, source = fixture(tmp_path, blocks=blocks)
    model = PartsModel()
    _, report = await DocumentReview(cfg(), model, model).run(ir, tmp_path, source_path=source)

    page = report["pages"][0]
    assert page["status"] == "verified", page.get("reason")
    assert [r["core"] for r in page["regions"]] == [[0, 0, 1000, 250], [0, 250, 1000, 500],
                                                    [0, 500, 1000, 750], [0, 750, 1000, 1000]]
    assert [p["core"] for p in page["regions"][0]["parts"]] == [[0, 0, 1000, 125], [0, 125, 1000, 250]]
    layouts = [payload for task, payload, _ in model.calls if task == "verify_layout"]
    assert len(layouts) > 1 and page["layout_review"]["parts"] == len(layouts)
    assert all(len(str(payload)) <= LAYOUT_PAYLOAD_LIMIT for payload in layouts)
    for before, after in zip(layouts, layouts[1:]):
        last = (before["document_boundaries"] + before["chunk_boundaries"])[-1]
        assert (after["document_boundaries"] + after["chunk_boundaries"])[0] == last
    assert validate_review_coverage(ir, tmp_path, cfg())["complete"]


@pytest.mark.asyncio
async def test_one_failed_part_fails_the_dense_page(tmp_path):
    class OneWrongPart(AuditModel):
        async def query(self, task, payload, image, schema):
            self.calls.append((task, payload, image))
            if task == "verify_layout":
                wrong = payload.get("boundary_part", {}).get("part") == 2
                return {"reading_order_correct": not wrong, "observation": "Checked the supplied boundaries.",
                        "issues": ["Rows 30 and 31 swapped."] if wrong else []}, 10
            return await super().query(task, payload, image, schema)

    blocks = [Block(block_id=f"b{i}", type=BlockType.TEXT, page_idx=0, reading_order=i,
                    bbox_norm=[0, i * 4, 1000, i * 4 + 3],
                    payload={"text": f"Source row {i}: " + " ".join(f"item{i}-{j}" for j in range(60))})
              for i in range(60)]
    ir, source = fixture(tmp_path, blocks=blocks)
    _, report = await DocumentReview(cfg(), OneWrongPart(), OneWrongPart()).run(ir, tmp_path, source_path=source)

    assert report["pages"][0]["status"] == "needs_review"
    assert "Rows 30 and 31 swapped." in report["pages"][0]["review"]["issues"]


@pytest.mark.asyncio
async def test_a_thin_strip_still_too_dense_is_split_left_and_right(tmp_path):
    """2026-10-08: a logo wall put ~120 logos within one band; halving top/bottom alone
    reached the thinnest strip with the request still too large."""
    from app.pipeline.review_sources import validate_review_coverage

    class Layout(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == "verify_layout":
                self.calls.append((task, payload, image))
                return {"reading_order_correct": True, "observation": "Order matches source.", "issues": []}, 10
            return await super().query(task, payload, image, schema)

    # Sixty logos side by side, each as tall as the strip they sit in.
    blocks = [Block(block_id=f"logo{i}", type=BlockType.TEXT, page_idx=0, reading_order=i,
                    bbox_norm=[i * 16, 5, i * 16 + 15, 45], payload={"text": f"Brand {i}"}) for i in range(60)]
    ir, source = fixture(tmp_path, blocks=blocks)
    model = Layout()
    _, report = await DocumentReview(cfg(), model, model).run(ir, tmp_path, source_path=source)

    page = report["pages"][0]
    assert page["status"] == "verified", page.get("reason")
    cores = [part["core"] for part in page["regions"][0]["parts"]]
    assert any(core[0] > 0 or core[2] < 1000 for core in cores)
    audits = [payload for task, payload, _ in model.calls if task == "audit_page"]
    assert all(len(payload["required_block_ids"]) <= 45 for payload in audits)
    assert validate_review_coverage(ir, tmp_path, cfg())["complete"]


@pytest.mark.asyncio
async def test_a_crop_whose_answer_is_cut_off_is_reviewed_in_halves(tmp_path):
    """2026-10-08: on the logo wall a crop of ~40 logos fit the request, but the review of
    all of them overran the model's answer length (output_truncated)."""

    class ShortAnswers(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == "verify_layout":
                self.calls.append((task, payload, image))
                return {"reading_order_correct": True, "observation": "Order matches source.", "issues": []}, 10
            if len(payload["required_block_ids"]) > 20:
                self.calls.append((task, payload, image))
                raise ValueError("output_truncated: response exceeded the configured token limit")
            return await super().query(task, payload, image, schema)

    blocks = [Block(block_id=f"logo{i}", type=BlockType.TEXT, page_idx=0, reading_order=i,
                    bbox_norm=[(i % 10) * 100, 10 + (i // 10) * 38, (i % 10) * 100 + 90, 40 + (i // 10) * 38],
                    payload={"text": f"Brand {i}"}) for i in range(60)]
    ir, source = fixture(tmp_path, blocks=blocks)
    model = ShortAnswers()
    _, report = await DocumentReview(cfg(), model, model).run(ir, tmp_path, source_path=source)

    page = report["pages"][0]
    assert page["status"] == "verified", page.get("reason")
    assert any(task == "audit_page" and len(payload["required_block_ids"]) > 20 for task, payload, _ in model.calls)
    assert len(page["regions"][0]["parts"]) > 2


def test_a_picture_document_is_one_boundary_in_the_order_check():
    from app.pipeline.document_review import layout_boundaries

    documents = [{"file": "outputs/documents/main.md", "paragraph": 7, "text": "Clients", "block_ids": ["h"], "binding": "source_text"},
                 *[{"file": "outputs/documents/fig-a.md", "paragraph": i, "text": f"Logo A line {i}",
                    "block_ids": ["a"], "binding": "asset_document"} for i in range(5)],
                 {"file": "outputs/documents/fig-b.md", "paragraph": 0, "text": "Logo B", "block_ids": ["b"], "binding": "asset_document"},
                 {"file": "outputs/documents/main.md", "paragraph": 8, "text": "Row 1", "block_ids": ["t"], "binding": "source_text"},
                 {"file": "outputs/documents/main.md", "paragraph": 9, "text": "Row 2", "block_ids": ["t"], "binding": "source_text"}]
    boundaries = layout_boundaries(documents, [{"chunk_id": "c0", "content": "Clients Logo A", "block_ids": ["h", "a"]}])

    texts = [item.get("text", item.get("content")) for _, item in boundaries]
    assert [kind for kind, _ in boundaries] == ["document"] * 5 + ["chunk"]
    assert texts[1].startswith("Logo A line 0") and texts[1].endswith("Logo A line 4")
    assert boundaries[1][1]["paragraph"] == 0 and boundaries[1][1]["last_paragraph"] == 4
    # Lines of the main document keep their own boundaries: rows stay in sequence.
    assert texts[3:5] == ["Row 1 … Row 1", "Row 2 … Row 2"]


@pytest.mark.asyncio
async def test_page_tools_never_replace_word_text(tmp_path):
    """Rerun 2026-10-16: none of 8 page reads and rebuilds on two Word reports was right."""
    ir, _, runner = await setup(tmp_path)
    block = ir.blocks[0]
    problem = {"page_idx": 0}
    assert runner.tool_constraints(problem, [block])["read_source"]["available"]
    block.payload["word_source"] = {"source": "native_docx", "block_index": 3}
    constraints = runner.tool_constraints(problem, [block])
    assert not any(constraints[a]["available"] for a in ("read_source", "read_regions", "reparse_pdf", "rebuild_delivery"))
    assert constraints["repair_image"]["available"]
