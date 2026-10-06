"""Image-level partial delivery must retain facts without laundering guessed text."""
import json

import pytest
from PIL import Image
from pydantic import ValidationError
from test_document_recovery import RecoveryModel, cfg, setup

from app.models.document_ir import Block, BlockType
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import delivered_documents, finalize_delivery
from app.pipeline.image_reading import (
    ImageParts,
    ReadingFormatError,
    read_contract,
    reconcile_image_parts,
)
from app.pipeline.recovery_models import NativeReading
from app.pipeline.review_sources import needs_content_recovery
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.package import PackageStage


def parts(text='Shop A', kind='photo', **changes):
    return ImageParts.model_validate({
        'image_type': kind, 'visual_facts': ['店內有陳列貨架與展示桌。'],
        'visual_independent_of_text': kind == 'photo',
        'text_regions': [{'region_id': 'sign', 'bbox': [0, 0, 1000, 200], 'readable': bool(text), 'text': text}],
        'complete': True, 'observation': 'Visible photograph and sign.', **changes,
    })


def test_photo_scene_can_survive_a_disputed_sign_without_guess_leakage():
    first = parts(visual_facts=['店內有陳列貨架。', 'This is Shop A.'])
    result = reconcile_image_parts(first, parts('Shop B'))
    assert result['visual_facts'] == ['店內有陳列貨架。']
    assert not result['text_regions'] and result['partial']
    assert 'Shop A' not in json.dumps(result) and 'Shop B' not in json.dumps(result)
    assert result['gaps'][0]['bbox'] == [0, 0, 1000, 200]


@pytest.mark.parametrize('kind', ['logo', 'diagram', 'chart', 'table', 'document'])
def test_essential_labels_cannot_be_replaced_with_generic_visuals(kind):
    result = reconcile_image_parts(parts('Label A', kind), parts('Label B', kind))
    assert not result['visual_facts'] and not result['text_regions'] and result['gaps']


def test_different_image_regions_cannot_agree_on_the_wrong_occurrence():
    first = parts()
    other = parts()
    other.text_regions[0].bbox = [0, 800, 1000, 1000]
    result = reconcile_image_parts(first, other)
    assert not result['text_regions'] and len(result['gaps']) == 2
    good = reconcile_image_parts(parts(), parts())
    assert good['text_regions'][0]['text'] == 'Shop A' and not good['gaps']


def test_empty_scene_and_claimed_complete_is_a_contract_error_not_unreadable_source():
    with pytest.raises(ValidationError):
        NativeReading.model_validate({'items': [{'source_id': 's', 'text': '', 'complete': True,
                                                'kind': 'visual_summary', 'observation': 'A shop interior.'}]})
    with pytest.raises(ValidationError):
        parts(visual_facts=[], text_regions=[])
    assert not needs_content_recovery({'status': 'needs_review', 'reason': 'reading_format_error: empty field'})


@pytest.mark.asyncio
async def test_one_contract_repair_and_cache_preserve_valid_visual_content(tmp_path):
    class FormatModel(RecoveryModel):
        async def query(self, task, payload, image, schema):
            self.tasks.append(task)
            assert task == 'verify_native_reading'
            repaired = len(self.tasks) > 1
            if repaired:
                assert 'response_contract_error' in payload
            return {'items': [{'source_id': 's', 'text': '店內有貨架。' if repaired else '',
                               'complete': True, 'kind': 'visual_summary', 'observation': 'A shop.'}]}, 10
    model = FormatModel()
    _, owner, runner = await setup(tmp_path, model)
    image = runner.inventory.image(0)
    value = await read_contract(owner, 'verify_native_reading', {'sources': [{'source_id': 's'}]},
                                image, NativeReading, source_ids=['s'])
    assert value.items[0].text == '店內有貨架。'
    assert owner.state['usage']['model_calls'] == 2
    assert len(owner.state['source_reading_cache']) == 1
    value = await read_contract(owner, 'verify_native_reading', {'sources': [{'source_id': 's'}]},
                                image, NativeReading, source_ids=['s'])
    assert value.items[0].complete and owner.state['usage']['model_calls'] == 2


@pytest.mark.asyncio
async def test_persisted_contract_exhaustion_does_not_repeat_or_cache_invalid_content(tmp_path):
    class Invalid(RecoveryModel):
        async def query(self, task, payload, image, schema):
            self.tasks.append(task)
            return {'items': [{'source_id': 'invented', 'text': 'A shop', 'complete': True,
                               'kind': 'visual_summary', 'observation': 'A shop.'}]}, 10
    model = Invalid()
    _, owner, runner = await setup(tmp_path, model)
    image = runner.inventory.image(0)
    for _ in range(2):
        with pytest.raises(ReadingFormatError):
            await read_contract(owner, 'verify_native_reading', {}, image, NativeReading, source_ids=['s'])
        owner.state = json.loads(owner.state_path.read_text())
    assert owner.state['usage']['model_calls'] == 2 and not owner.state['source_reading_cache']


@pytest.mark.asyncio
async def test_contract_repair_cannot_exceed_shared_model_budget(tmp_path):
    class Invalid(RecoveryModel):
        async def query(self, task, payload, image, schema):
            return {'items': []}, 10
    _, owner, runner = await setup(tmp_path, Invalid())
    owner.limits.max_model_calls = 1
    with pytest.raises(ReadingFormatError, match="retry_budget_exhausted; source readability remains unknown"):
        await read_contract(owner, 'verify_native_reading', {}, runner.inventory.image(0), NativeReading)
    assert owner.state['usage']['model_calls'] == 1


class ImageModel(RecoveryModel):
    async def query(self, task, payload, image, schema):
        if task == 'propose_image_edits':
            self.tasks.append(task)
            return {'edits': [], 'observation': 'No span can be fixed with independent wording.'}, 10
        if task == 'revise_image_regions':
            self.tasks.append(task)
            return {'regions': [{'block_id': t['block_id'], 'region_kind': 'uncertain',
                                 'evidence_ids': [t['source_evidence_id']], 'reason': 'No supported local reading.'}
                                for t in payload['targets']]}, 10
        if task == 'resolve_image_evidence':
            self.tasks.append(task)
            return {'selections': [{'source_ref': ref, 'decision': 'uncertain', 'reason': 'Synthetic image has no lettering.'}
                                   for ref in payload['required_source_refs']],
                    'accounted_for': True, 'observation': 'Keep disputed regions as gaps.'}, 10
        if task in {'read_image_parts', 'verify_image_parts'}:
            self.tasks.append(task)
            assert 'Shop A' not in json.dumps(payload) and 'Shop B' not in json.dumps(payload)
            return parts('Shop A' if task == 'read_image_parts' else 'Shop B').model_dump(), 10
        if task == 'recovery_plan':
            self.tasks.append(task)
            assert 'repair_image' in payload['tools'] and 'rebuild_delivery' not in payload['tools']
            assert payload['issue']['failure_kind'] == 'reading_disagreement'
            return {'action': 'repair_image', 'reason': 'Read the disputed picture separately.'}, 10
        if task in {'audit_image_parts', 'audit_page'}:
            self.tasks.append(task)
            gaps = {b['block_id'] for b in payload.get('declared_gaps', [])}
            return {'blocks': [{'block_id': bid, 'status': 'needs_review' if bid in gaps or bid == 'unrelated' else 'verified',
                                'context_complete': bid not in gaps and bid != 'unrelated',
                                'observation': 'Source supports scene facts; sign text is uncertain.'}
                               for bid in payload['required_block_ids']],
                    'source_complete': not gaps and 'unrelated' not in payload['required_block_ids'],
                    'reading_order_correct': True, 'declared_gaps_only': bool(gaps) and 'unrelated' not in payload['required_block_ids'],
                    'observation': 'The source and actual delivery were compared.',
                    'issues': ['Uncertain lettering.'] if gaps else []}, 10
        return await super().query(task, payload, image, schema)


@pytest.mark.asyncio
async def test_image_only_recovery_and_real_delivery_retain_scene_and_exclude_wrong_sign(tmp_path):
    model = ImageModel()
    # A targeted edit with nothing it may change is rejected first; the full re-read comes next.
    ir, owner, runner = await setup(tmp_path, model, cfg(max_recovery_actions=2).model_copy(update={"agentic": cfg(max_recovery_actions=2).agentic.model_copy(update={"max_seconds": 3000})}))
    picture = tmp_path / 'picture.png'
    Image.new('RGB', (200, 140), 'green').save(picture)
    image_block = Block(block_id='picture', type=BlockType.IMAGE, page_idx=0, reading_order=1,
                        payload={'img_path': str(picture), 'caption': 'Guessed PhantomBrand999'})
    unrelated = Block(block_id='unrelated', type=BlockType.TEXT, page_idx=0, reading_order=2,
                      payload={'text': 'Unrelated unverified content.'})
    ir.blocks.extend([image_block, unrelated])
    runner.inventory.mapping.update(picture=[0], unrelated=[0])
    original = runner.inventory.directory / 'picture.png'
    original.write_bytes(picture.read_bytes())
    runner.inventory.word_images['picture'] = original
    runner.inventory.image_positions['picture'] = [{'page_idx': 0, 'bbox': [0, 200, 1000, 900]}]
    owner.state['pages']['0'].update(image_only_issues=True, image_issues=[
        {'block_id': 'picture', 'failure_kind': 'reading_disagreement', 'detail': 'Disputed sign.'}])
    from app.pipeline.document_recovery import render_candidate
    runner.renderer = render_candidate
    async def rebuild(candidate):
        config = owner.config.model_copy(deep=True)
        config.enrich.enable_vlm = False
        config.package.enable_semantic_repair = False
        result = await PackageStage(config).run(candidate.doc_id, candidate.run_id, candidate, tmp_path)
        assert result.success, result.error
        result = await ChunkStage(config).run(candidate.doc_id, candidate.run_id, candidate, tmp_path)
        assert result.success, result.error
    runner.rebuild = rebuild
    await rebuild(ir)
    (tmp_path / 'outputs/enrichments.jsonl').write_text(json.dumps({
        'block_id': 'picture', 'kind': 'figure_description', 'output': {
            'semantic_caption': 'PhantomBrand999', 'facts': ['PhantomBrand999'], 'keywords': ['PhantomBrand999']}}))
    result = await runner.run(ir)
    assert result is not ir
    assert result.get_block('unrelated').payload == unrelated.payload
    assert [(s['action'], s.get('method'), s['status']) for s in runner.state['steps']] == [
        ('repair_image', 'targeted_edit', 'rejected'), ('repair_image', None, 'applied')]
    assert runner.state['issues']['image:picture']['status'] == 'applied_partial'
    from app.pipeline.review_sources import digest
    folder = runner.folder / runner.state['steps'][1]['key'][:16]
    provenance = json.loads((folder / 'candidate-provenance.json').read_text())
    assert provenance['status'] == 'candidate_source_checked'
    for change in provenance['changes']:
        block = result.get_block(change['block_id'])
        assert change['after_sha256'] == (digest(block.model_dump(exclude={'reading_order'})) if block else None)
    assert 'read_regions' not in model.tasks and 'read_source' not in model.tasks
    assert any(b.payload.get('agentic_gap') for b in result.blocks)
    for path in delivered_documents(tmp_path / 'outputs'):
        assert 'PhantomBrand999' not in path.read_text()
    assert 'PhantomBrand999' not in (tmp_path / 'outputs/assets_index.jsonl').read_text()
    assert 'PhantomBrand999' not in (tmp_path / 'outputs/chunks.jsonl').read_text()
    # The scoped fixture predates the full review's checkpoint initialization.
    owner.state['fingerprint'] = 'isolated-scope-fixture'
    owner._save()
    # Use the ordinary review and final export path, including the unrelated unresolved block.
    reviewed, _ = await DocumentReview(owner.config, model, model).run(result, tmp_path, source_path=runner.inventory.pdf)
    final = finalize_delivery(reviewed, tmp_path, owner.config)
    assert final['processing_status'] == 'partial' and final['partial_rag_ready']
    safe = (tmp_path / 'outputs/rag_chunks.jsonl').read_text()
    assert '貨架' in safe and 'Shop A' not in safe and 'Shop B' not in safe and 'PhantomBrand999' not in safe
    assert 'Unrelated unverified' not in safe
    assert '此區域未納入文字解讀' in (tmp_path / 'outputs/delivery.md').read_text()


def test_verified_explicit_gaps_do_not_trigger_page_rewrites_without_new_evidence():
    record = {'status': 'needs_review', 'reason': 'source_delivery_mismatch', 'declared_gap_ids': ['gap'],
              'review': {'source_complete': False, 'reading_order_correct': True, 'declared_gaps_only': True,
                         'blocks': [{'block_id': 'scene', 'status': 'verified', 'context_complete': True}, {'block_id': 'gap', 'status': 'needs_review'}]}}
    assert not needs_content_recovery(record)
    record['review']['blocks'][0]['status'] = 'needs_review'
    assert needs_content_recovery(record)  # An additional unexplained problem is still actionable.


def test_image_tool_estimate_reserves_both_contract_repairs_and_audits():
    from app.pipeline.review_budget import tool_cost
    assert tool_cost({}, 'repair_image')['model_calls'] == 18


def test_image_audit_uses_actual_scoped_spans_without_neighbor_caption_contamination(tmp_path):
    from app.pipeline.image_recovery import image_delivery_excerpts
    block = Block(block_id='gap', type=BlockType.TEXT, page_idx=0, payload={'text': 'Unresolved logo lettering.'})
    out = tmp_path / 'outputs'
    (out / 'documents').mkdir(parents=True)
    (out / 'documents/main.md').write_text('A neighboring retail scene.\nUnresolved logo lettering.\nOther picture.')
    (out / 'chunks.jsonl').write_text(json.dumps({'chunk_id': 'mixed', 'block_ids': ['gap', 'neighbor'],
                                               'content': 'A neighboring retail scene.\nUnresolved logo lettering.'}))
    docs, chunks = image_delivery_excerpts([block], tmp_path, ChunkStage(), chunks_required=True)
    assert docs[0]['text'] == chunks[0]['content'] == block.get_text()
    assert 'neighboring' not in json.dumps([docs, chunks])
    (out / 'documents/main.md').write_text('A neighboring retail scene only.')
    with pytest.raises(ValueError, match='actual documents/chunks'):
        image_delivery_excerpts([block], tmp_path, ChunkStage(), chunks_required=True)


@pytest.mark.asyncio
async def test_unreadable_logo_can_finish_as_one_explicit_gap_without_neighbor_contamination(tmp_path):
    from app.pipeline.document_recovery import render_candidate
    from app.pipeline.recovery_models import RecoveryDecision

    class Logo(ImageModel):
        async def query(self, task, payload, image, schema):
            if task in {'read_image_parts', 'verify_image_parts'}:
                return parts('Name A' if task == 'read_image_parts' else 'Name B', 'logo').model_dump(), 10
            if task == 'audit_image_parts':
                assert all('Revenue' not in doc['text'] for doc in payload['delivered_documents'])
                assert all('Revenue' not in chunk['content'] for chunk in payload['delivered_chunks'])
            return await super().query(task, payload, image, schema)
    ir, owner, runner = await setup(tmp_path, Logo())
    picture = runner.inventory.directory / 'logo.png'
    Image.new('RGB', (200, 140), 'yellow').save(picture)
    ir.blocks.append(Block(block_id='logo', type=BlockType.IMAGE, page_idx=0, reading_order=1,
                           payload={'caption': 'Old uncertain name', 'img_path': str(picture)}))
    runner.inventory.mapping['logo'] = [0]
    runner.inventory.word_images['logo'] = picture
    runner.renderer = render_candidate
    folder = tmp_path / 'candidate'
    folder.mkdir()
    result, pages = await runner.repair_image(ir, {'block_id': 'logo', 'page_idx': 0},
        RecoveryDecision(action='repair_image', reason='Read original image.'), folder, 'Read original image.')
    replacements = [b for b in result.blocks if b.payload.get('agentic_image_origin') == 'logo']
    assert len(replacements) == 1 and replacements[0].payload.get('agentic_gap')
    assert await runner.validate(result, pages, folder, image_origin='logo')
    assert 'Name A' not in '\n'.join(p.read_text() for p in delivered_documents(folder / 'outputs'))


@pytest.mark.parametrize('kind', ['photo', 'logo'])
def test_repeated_model_agreement_does_not_erase_conflicting_prior_blind_reading(kind):
    result = reconcile_image_parts(parts('Wrong A', kind), parts('Wrong A', kind), prior_transcription='Wrong B')
    assert not result['text_regions'] and result['gaps']
    assert bool(result['visual_facts']) == (kind == 'photo')
    assert 'Wrong A' not in json.dumps(result) and 'Wrong B' not in json.dumps(result)
    good = reconcile_image_parts(parts('Correct', kind), parts('Correct', kind), prior_transcription='Correct')
    assert good['text_regions'] and not good['gaps']


@pytest.mark.asyncio
async def test_rejected_photo_detail_is_pruned_once_and_actual_rebuilt_delivery_is_reaudited(tmp_path):
    from app.pipeline.document_recovery import render_candidate
    from app.pipeline.recovery_models import RecoveryDecision
    class Partial(ImageModel):
        audits = 0
        async def query(self, task, payload, image, schema):
            if task in {'read_image_parts', 'verify_image_parts'}:
                return parts('BrandOne' if task == 'read_image_parts' else 'BrandTwo',
                             visual_facts=['店內有貨架。', 'The sign is brightly lit.']).model_dump(), 10
            result, tokens = await super().query(task, payload, image, schema)
            if task == 'audit_image_parts':
                self.audits += 1
                if self.audits == 1:
                    detail = next(b for b in result['blocks'] if b['block_id'].endswith('_v1'))
                    detail['context_complete'] = False
                else:
                    assert self.audits == 2
                    assert 'The sign is brightly lit.' not in json.dumps(payload['delivered_documents'])
                    assert any(b['block_id'].endswith('_v1') for b in payload['declared_gaps'])
            return result, tokens
    model = Partial()
    ir, owner, runner = await setup(tmp_path, model)
    picture = runner.inventory.directory / 'picture.png'
    Image.new('RGB', (200, 140), 'green').save(picture)
    ir.blocks.append(Block(block_id='picture', type=BlockType.IMAGE, page_idx=0,
                           payload={'caption': 'Unverified old wording', 'img_path': str(picture)}))
    runner.inventory.mapping['picture'] = [0]
    runner.inventory.word_images['picture'] = picture
    runner.renderer = render_candidate
    folder = tmp_path / 'candidate'
    folder.mkdir()
    result, pages = await runner.repair_image(ir, {'block_id': 'picture', 'page_idx': 0},
        RecoveryDecision(action='repair_image', reason='Read original image.'), folder, 'Read original image.')
    assert await runner.validate(result, pages, folder, image_origin='picture')
    assert model.audits == 2
    delivery = '\n'.join(p.read_text() for p in delivered_documents(folder / 'outputs'))
    assert '店內有貨架' in delivery and 'The sign is brightly lit.' not in delivery


@pytest.mark.asyncio
async def test_image_budget_reuses_exact_valid_reads_but_not_other_source_or_prompt(tmp_path):
    from app.pipeline.document_recovery import SKILLS
    from app.pipeline.image_reading import remaining_read_calls
    from app.pipeline.image_recovery import image_read_requests
    from app.pipeline.recovery_models import RecoveryDecision
    ir, owner, runner = await setup(tmp_path, ImageModel())
    picture = runner.inventory.directory / 'picture.png'
    Image.new('RGB', (200, 140), 'green').save(picture)
    ir.blocks.append(Block(block_id='picture', type=BlockType.IMAGE, page_idx=0, payload={'img_path': str(picture)}))
    runner.inventory.mapping['picture'] = [0]
    runner.inventory.word_images['picture'] = picture
    problem = {'block_id': 'picture', 'page_idx': 0}
    features = {'embedded_images': 1, 'word_images': 1}
    skill = (SKILLS / 'repair_image' / 'SKILL.md').read_text()
    assert runner.estimate_tool('repair_image', features, ir, problem)['model_calls'] == 18
    folder = tmp_path / 'candidate'
    folder.mkdir()
    await runner.repair_image(ir, problem, RecoveryDecision(action='repair_image', reason='Read original.'), folder, skill)
    calls = owner.state['usage']['model_calls']
    assert runner.estimate_tool('repair_image', features, ir, problem)['model_calls'] == 12
    await runner.repair_image(ir, problem, RecoveryDecision(action='repair_image', reason='Read original.'), folder, skill)
    assert owner.state['usage']['model_calls'] == calls
    task, payload = image_read_requests(owner.config, ir, skill)[0]
    assert remaining_read_calls(owner, task, {**payload, 'skill': 'Different instructions'}, picture, ImageParts) == 2
    Image.new('RGB', (200, 140), 'yellow').save(picture)
    assert runner.estimate_tool('repair_image', features, ir, problem)['model_calls'] == 18


@pytest.mark.asyncio
async def test_page_level_issue_retains_three_repairs_when_fourth_final_audit_times_out(tmp_path, monkeypatch):
    class ScopedImages(ImageModel):
        async def query(self, task, payload, image, schema):
            if task == 'recovery_plan':
                self.tasks.append(task)
                assert 'repair_image' in payload['tools']
                return {'action': 'repair_image', 'block_id': payload['image_targets'][0]['block_id'],
                        'reason': 'Patch this mapped picture only.'}, 10
            return await super().query(task, payload, image, schema)
    config = cfg(max_recovery_actions=4, max_recovery_attempts_per_issue=3)
    config.agentic.max_seconds = 4000
    config.agentic.max_model_calls = 100
    ir, owner, runner = await setup(tmp_path, ScopedImages(), config)
    ir.blocks[0].type = BlockType.TABLE
    ir.blocks[0].payload = {'table_body': '<table><tr><td>Revenue 100, cost 80.</td></tr></table>'}
    original = ir.blocks[0].model_dump()
    for index in range(4):
        name = f'picture{index}'
        image = runner.inventory.directory / f'{name}.png'
        Image.new('RGB', (100, 80), (index * 50, 100, 100)).save(image)
        ir.blocks.append(Block(block_id=name, type=BlockType.IMAGE, page_idx=0,
            payload={'img_path': str(image), 'word_source': {'parent_table_block_id': 'b0'}}))
        runner.inventory.word_images[name] = image
        runner.inventory.mapping[name] = [0]
        runner.inventory.image_positions[name] = [{'page_idx': 0, 'bbox': [0, 100, 1000, 900]}]
    from app.pipeline.document_recovery import render_candidate
    runner.renderer = render_candidate
    async def rebuild(candidate):
        package_config = owner.config.model_copy(deep=True)
        package_config.enrich.enable_vlm = False
        package_config.package.enable_semantic_repair = False
        for stage in [PackageStage(package_config), ChunkStage(package_config)]:
            result = await stage.run(candidate.doc_id, candidate.run_id, candidate, tmp_path)
            assert result.success, result.error
    runner.rebuild = rebuild
    await rebuild(ir)
    # Image candidate validation remains real; this isolates the whole-page status transition.
    async def inspect(page, candidate, inventory, root):
        remaining = [b for b in candidate.blocks if b.type == BlockType.IMAGE]
        record = {'page_idx': page, 'status': 'needs_review' if remaining else 'not_reviewed',
                  'reason': 'source_delivery_mismatch' if remaining else 'final_tool_review_timeout',
                  'review': ({'issues': ['Descriptions still missing.'], 'reading_order_correct': True,
                    'source_complete': False, 'blocks': [
                        {'block_id': b.block_id,
                         'status': 'needs_review' if b.payload.get('agentic_gap') or b.type == BlockType.IMAGE else 'verified',
                         'context_complete': not b.payload.get('agentic_gap') and b.type != BlockType.IMAGE}
                        for b in candidate.blocks]} if remaining else {})}
        owner.state['pages'][str(page)] = record
        return record
    monkeypatch.setattr(owner, '_inspect', inspect)
    result = await runner.run(ir, issue_ids={'page:0'})
    assert result.get_block('b0').model_dump() == original
    assert len(runner.state['steps']) == 4
    assert {s['block_id'] for s in runner.state['steps']} == {f'picture{i}' for i in range(4)}
    assert [s['status'] for s in runner.state['steps']] == ['applied', 'applied', 'applied', 'rejected']
    assert all(runner.state['issues'][f'image:picture{i}']['status'] == 'applied_partial' for i in range(3))
    assert [b.block_id for b in result.blocks if b.type == BlockType.IMAGE] == ['picture3']
    assert owner.state['pages']['0']['status'] == 'needs_review'
    assert owner.state['pages']['0']['review']['blocks']  # Last successful page certificate survives.


def test_failed_image_strategy_does_not_escalate_to_whole_page_rewrite(tmp_path):
    from types import SimpleNamespace

    from app.pipeline.document_recovery import RecoveryController
    from app.pipeline.recovery_models import RecoveryDecision
    owner = SimpleNamespace(config=cfg())
    runner = object.__new__(RecoveryController)
    runner.owner = owner
    from app.pipeline.evidence import EvidenceStore
    runner.evidence = EvidenceStore(tmp_path, "source")
    decision = RecoveryDecision(action='repair_image', block_id='picture', reason='Repair picture')
    recent = [{'action': 'repair_image', 'model_role': 'primary', 'block_id': 'picture'}]
    selected = runner.select_strategy(decision, recent, ['repair_image', 'read_source', 'read_regions', 'defer'])
    assert selected.action == 'defer'
    other = runner.select_strategy(decision.model_copy(update={'block_id': 'picture2'}), recent,
                                   ['repair_image', 'read_regions', 'defer'])
    assert other.action == 'repair_image' and other.block_id == 'picture2'


@pytest.mark.asyncio
async def test_later_page_audit_reopens_scoped_image_success(tmp_path):
    ir, _, runner = await setup(tmp_path)
    block = ir.blocks[0]
    block.payload.update(agentic_image_origin="picture")
    runner.state["issues"]["image:picture"] = {"page_idx": 0, "status": "resolved"}
    runner.reconcile_image_findings(ir, 0, {"review": {"blocks": [
        {"block_id": block.block_id, "status": "needs_review", "context_complete": False}]}})
    issue = runner.state["issues"]["image:picture"]
    assert issue["status"] == "needs_review" and issue["reason"] == "final_page_review_disagrees"
    assert issue["unverified_block_ids"] == [block.block_id]
    runner.reconcile_image_findings(ir, 0, {"status": "not_reviewed"})
    assert issue["status"] == "needs_review"
