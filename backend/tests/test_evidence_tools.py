"""Evidence identifiers bind candidate edits to source; only final audit certifies delivery."""
import json
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from test_document_recovery import RecoveryModel, setup
from test_document_review import AuditModel, config, fixture
from test_image_recovery import parts

from app.config import settings
from app.models.document_ir import Block, BlockType
from app.pipeline.document_review import DocumentReview
from app.pipeline.evidence import EvidenceStore
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.image_evidence import resolve_image_evidence
from app.pipeline.image_reading import (
    ReadingFormatError,
    read_image_parts_bounded,
    reconcile_image_parts,
)
from app.pipeline.recovery_models import RecoveryDecision


def test_evidence_rejects_foreign_scope_changed_record_and_changed_source(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (40, 40), 'white').save(source)
    store = EvidenceStore(tmp_path, 'document-sha')
    eid = store.add(provider='ocr', kind='transcription', page=0, block_id='picture', source=source,
                    content={'text': 'Revenue 100'})
    assert store.get(eid)['status'] == 'observation'
    for page, bid, reason in [(1, 'picture', 'page_mismatch'), (0, 'other', 'block_mismatch')]:
        with pytest.raises(ValueError, match=reason):
            store.get(eid, page=page, block_id=bid)
    with pytest.raises(ValueError, match='identity_changed'):
        EvidenceStore(tmp_path, 'other-document').get(eid)
    path = store.directory / (eid + '.json')
    original = path.read_bytes()
    value = json.loads(original)
    value['content']['text'] = 'Revenue 999'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='identity_changed'):
        store.get(eid)
    path.write_bytes(original)
    snapshot = tmp_path / store.get(eid)['source']['path']
    snapshot.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source_changed'):
        store.get(eid)


@pytest.mark.asyncio
async def test_candidate_cannot_modify_unrelated_content_or_cite_foreign_page(tmp_path):
    ir, _, runner = await setup(tmp_path)
    other = Block(block_id='other', type=BlockType.TEXT, page_idx=1, payload={'text': 'Untouched'})
    ir.blocks.append(other)
    runner.inventory.mapping['other'] = [1]
    candidate = ir.model_copy(deep=True)
    candidate.get_block('other').payload['text'] = 'Fabricated'
    with pytest.raises(ValueError, match='unrelated_or_cross_page'):
        runner.record_candidate(ir, candidate, [0], tmp_path, 'read_regions')
    eid = runner.evidence.add(provider='source', kind='page_image', page=1)
    candidate = ir.model_copy(deep=True)
    candidate.blocks[0].payload.update(agentic_source_pages=[0], agentic_evidence_ids=[eid])
    with pytest.raises(ValueError, match='candidate_evidence_page_mismatch'):
        runner.record_candidate(ir, candidate, [0], tmp_path, 'read_regions')


@pytest.mark.asyncio
async def test_patch_requires_source_citation_and_preserves_heading_role(tmp_path):
    class Patch(RecoveryModel):
        cite = False
        async def query(self, task, payload, image, schema):
            return {'complete': True, 'edits': [{'block_id': 'b0', 'text': 'Source corrected heading',
                    'reason': 'Visible heading', 'evidence_ids': payload['source_evidence_ids'] if self.cite else []}]}, 10
    model = Patch()
    ir, _, runner = await setup(tmp_path, model)
    ir.blocks[0].payload['text_level'] = 2
    decision = RecoveryDecision(action='rebuild_delivery', reason='Fix heading')
    with pytest.raises(ValueError, match='requires_in_scope_source_evidence'):
        await runner.rebuild_delivery(ir, {'page_idx': 0}, decision, tmp_path, '')
    model.cite = True
    result, pages = await runner.rebuild_delivery(ir, {'page_idx': 0}, decision, tmp_path, '')
    assert pages == [0] and result.blocks[0].payload['text_level'] == 2
    assert result.blocks[0].payload['agentic_evidence_ids']


@pytest.mark.asyncio
async def test_ocr_observation_is_cached_and_a_failed_full_repair_is_not_rerun_on_new_evidence(tmp_path, monkeypatch):
    # 2026-10-06 rule: each recovery attempt is a different method. A full repair rerun after
    # an OCR observation failed the same way every time in the validation runs.
    from app.adapters import ocr
    ir, owner, runner = await setup(tmp_path)
    image = runner.inventory.directory / 'picture.png'
    Image.new('RGB', (100, 100), 'white').save(image)
    block = Block(block_id='picture', type=BlockType.IMAGE, page_idx=0, payload={'img_path': str(image)})
    ir.blocks.append(block)
    runner.inventory.mapping['picture'] = [0]
    runner.inventory.word_images['picture'] = image
    monkeypatch.setattr(ocr, 'ocr_runtime_identity', lambda: 'runtime-v1')
    import hashlib
    result = {'source_sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'region': [0,0,1000,1000],
              'scale': 1, 'status': 'ok', 'lines': [{'line_id': 'line0', 'text': 'Observed', 'bbox': [0,0,500,500], 'score': .9}],
              'version': 'test', 'detector': 'test', 'recognizer': 'test', 'device': 'cpu'}
    reader = AsyncMock(return_value=result)
    monkeypatch.setattr(ocr, 'read_ocr', reader)
    repair = RecoveryDecision(action='repair_image', block_id='picture', reason='Resolve lettering')
    recent = [{'action': 'repair_image', 'block_id': 'picture', 'page_idx': 0, 'evidence_revision': ''}]
    assert runner.select_strategy(repair, recent, ['repair_image', 'defer']).action == 'defer'
    before = ir.to_dict()
    decision = RecoveryDecision(action='ocr_image', block_id='picture', reason='Read lettering')
    for _ in range(2):
        observed = await runner.ocr_image(ir, {'page_idx': 0}, decision, tmp_path / 'ocr', '')
    assert observed['cache_hit'] and reader.await_count == 1 and ir.to_dict() == before
    assert owner.state['usage']['model_calls'] == 0
    assert runner.select_strategy(repair, recent, ['repair_image', 'defer']).action == 'defer'
    monkeypatch.setattr(ocr, 'ocr_runtime_identity', lambda: 'runtime-v2')
    await runner.ocr_image(ir, {'page_idx': 0}, decision, tmp_path / 'ocr-new', '')
    assert reader.await_count == 2


@pytest.mark.asyncio
async def test_unavailable_ocr_is_not_offered_to_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'ocr_python_path', str(tmp_path / 'missing-python'))
    ir, _, runner = await setup(tmp_path)
    assert not runner.tool_constraints({'page_idx': 0}, ir.blocks)['ocr_image']['available']


@pytest.mark.asyncio
@pytest.mark.parametrize('invent', [False, True])
async def test_source_adjudication_can_replace_wrong_prior_but_cannot_invent_ids(tmp_path, invent):
    class Select(RecoveryModel):
        async def query(self, task, payload, image, schema):
            assert task == 'resolve_image_evidence'
            rows = [{'source_ref': c['source_ref'], 'decision': 'supported' if c['provider'] == 'paddleocr' else 'uncertain',
                     'reason': 'Check original pixels'} for c in payload['candidates']]
            if invent:
                rows[0]['source_ref'] = 'fabricated'
            return {'selections': rows, 'accounted_for': True, 'observation': 'Candidate still needs independent source audit'}, 10
    ir, _, runner = await setup(tmp_path, Select())
    block = ir.blocks[0]
    source = runner.inventory.image(0)
    runner.evidence.add(provider='paddleocr', kind='transcription', page=0, block_id=block.block_id,
        source=source, content={'lines': [{'line_id': 'line0', 'text': 'Correct', 'score': .7, 'bbox': [0,0,1000,200]}]})
    primary, independent = parts('Wrong'), parts('Wrong')
    # Agreement between two model calls must not suppress contradictory OCR evidence.
    reconciled = reconcile_image_parts(primary, independent)
    assert not reconciled['gaps']
    async def resolve():
        return await resolve_image_evidence(runner, block, 0, source, primary, independent,
                                            {'text': 'Wrong prior'}, reconciled, tmp_path)
    if invent:
        with pytest.raises(ReadingFormatError, match='invented_evidence'):
            await resolve()
    else:
        candidate, ids = await resolve()
        # What only OCR reads is not delivered in OCR's spelling (test run 2026-10-19); the region stays unread.
        assert not candidate['text_regions']
        assert [g['reason'] for g in candidate['gaps']] == ['image_text_unresolved_after_evidence_review']
        assert all(runner.evidence.get(eid)['status'] == 'observation' for eid in ids)
        assert json.loads((tmp_path / 'evidence-resolution.json').read_text())['status'] == 'candidate_requires_source_audit'


@pytest.mark.asyncio
async def test_dense_image_format_failure_splits_again_with_seven_calls_and_resume_cache(tmp_path):
    class Dense(RecoveryModel):
        async def query(self, task, payload, image, schema):
            self.tasks.append(task)
            core = payload.get('source_core', [0,0,1000,1000])
            count = 17 if core[3] - core[1] > 251 else 1
            return {'image_type': 'document', 'visual_facts': [], 'visual_independent_of_text': False,
                    'text_regions': [{'region_id': str(i), 'bbox': [0,0,1000,1000], 'readable': True,
                                      'text': str(core[1])} for i in range(count)],
                    'complete': True, 'observation': 'Read only owned source crop'}, 10
    model = Dense()
    _, owner, runner = await setup(tmp_path, model)
    for _ in range(2):
        result = await read_image_parts_bounded(owner, 'read_image_parts', {}, runner.inventory.image(0))
        assert len(result.text_regions) == 4
        assert result.text_regions[0].bbox[1] == 0 and result.text_regions[-1].bbox[3] == 1000
        assert all(a.bbox[3] == b.bbox[1] for a, b in zip(result.text_regions, result.text_regions[1:]))
    assert len(model.tasks) == 7


@pytest.mark.asyncio
async def test_unverified_navigation_does_not_exclude_explicitly_self_contained_content(tmp_path):
    blocks = [Block(block_id='title', type=BlockType.TEXT, page_idx=0,
                    payload={'text': 'Chapter overview', 'text_level': 1}),
              Block(block_id='body', type=BlockType.TEXT, page_idx=0,
                    payload={'text': 'Revenue in 2027 was USD 100.'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    def review(_, value):
        value.update(source_complete=False, issues=['Navigation title is uncertain'])
        by_id = {b['block_id']: b for b in value['blocks']}
        by_id['title'].update(status='needs_review', context_complete=False)
        by_id['body'].update(status='verified', context_complete=True, depends_on_block_ids=[])
        return value
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    text = (tmp_path / 'outputs/rag_chunks.jsonl').read_text()
    assert 'Revenue in 2027 was USD 100.' in text and 'Chapter overview' not in text


@pytest.mark.asyncio
async def test_ocr_timeout_kills_worker_group_and_uses_isolated_imports(tmp_path, monkeypatch):
    import asyncio

    from app.adapters import ocr
    picture = tmp_path / 'source.png'
    Image.new('RGB', (100, 100), 'white').save(picture)
    monkeypatch.setattr(ocr, 'ocr_python', lambda: '/operator/ocr/python')
    killed, arguments = [], []
    class Worker:
        pid = 12345
        returncode = None
        async def communicate(self, data):
            await asyncio.sleep(5)
        async def wait(self):
            self.returncode = -9
    worker = Worker()
    async def launch(*args, **kwargs):
        arguments.extend(args)
        assert kwargs['start_new_session']
        return worker
    monkeypatch.setattr(ocr.asyncio, 'create_subprocess_exec', launch)
    monkeypatch.setattr(ocr.os, 'killpg', lambda pid, signal: killed.append(pid))
    with pytest.raises(TimeoutError):
        await ocr.read_ocr(picture, [0,0,1000,1000], 1, tmp_path / 'ocr', timeout=.01)
    assert arguments[1] == '-I' and killed == [12345] and worker.returncode == -9


def test_table_group_rows_are_never_promoted_to_headers():
    from app.pipeline.package_utils import semantic_table_to_text
    from app.pipeline.stages.chunk import ChunkStage
    from app.pipeline.table_structure import canonical_table_html, table_structure
    table = Block(block_id='table', type=BlockType.TABLE, page_idx=0, payload={'table_body':
        '<table><tr><td>Category</td><td></td><td>Deliverable</td></tr>'
        '<tr><td rowspan="2">Group A</td><td>Research</td><td>Report 1</td></tr>'
        '<tr><td>Positioning</td><td>Strategy 1</td></tr>'
        '<tr><td>Group B</td><td>Website</td><td>Site 1</td></tr></table>'})
    before = table.model_dump()
    text = semantic_table_to_text(canonical_table_html(table))
    assert '欄位：' not in text
    assert '第 4 列：第 1 欄：Group B；第 2 欄：Website；第 3 欄：Site 1' in text
    assert '第 1 欄（跨 2 列）：Group A' in text
    assert '第 3 列：第 2 欄：Positioning；第 3 欄：Strategy 1' in text
    assert text.count('Group A') == 1
    assert ChunkStage()._block_to_text(table) == text
    assert len(table_structure(table).cells) == 11 and table.model_dump() == before


def test_detail_views_use_original_pixels_and_bounded_source_regions(tmp_path):
    from app.pipeline.image_evidence import image_detail_views
    source = tmp_path / 'original.png'
    Image.new('RGB', (200, 100), 'yellow').save(source)
    before = source.read_bytes()
    store = EvidenceStore(tmp_path, 'source-sha')
    eid = store.add(provider='paddleocr', kind='transcription', page=0, block_id='picture', source=source,
                    content={'lines':[{'text':'Sign', 'bbox':[300,100,700,300]}]})
    paths, regions = image_detail_views(store, source, [eid])
    assert paths[0] == source and len(paths) == 2 and regions == [[260,60,740,340]]
    with Image.open(paths[1]) as crop:
        assert crop.size == (384,112) and crop.getpixel((0,0)) == (255,255,0)
    assert source.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid_link', [False, True])
async def test_resolution_links_different_detector_boxes_without_duplicate_gap(tmp_path, invalid_link):
    class Link(RecoveryModel):
        async def query(self, task, payload, image, schema):
            supported = next(c['source_ref'] for c in payload['candidates'] if c['provider'] == 'paddleocr')
            return {'selections':[{'source_ref':c['source_ref'],
                'decision':'supported' if c['provider']=='paddleocr' else 'uncertain',
                'covered_by': None if c['provider']=='paddleocr' else ('invented' if invalid_link else supported),
                'reason':'Same source lettering, different detector rectangle'} for c in payload['candidates']],
                'accounted_for':True, 'observation':'Still requires source audit'}, 10
    ir, _, runner = await setup(tmp_path, Link())
    source = runner.inventory.image(0)
    block = ir.blocks[0]
    runner.evidence.add(provider='paddleocr',kind='transcription',page=0,block_id=block.block_id,source=source,
        content={'lines':[{'line_id':'line0','text':'Correct','score':.9,'bbox':[300,100,700,250]}]})
    a, b = parts('Wrong'), parts('Wrong')
    result = reconcile_image_parts(a,b,prior_transcription='Other wrong')
    async def resolve():
        return await resolve_image_evidence(runner,block,0,source,a,b,{},result,tmp_path)
    if invalid_link:
        with pytest.raises(ReadingFormatError,match='invalid_region_link'):
            await resolve()
    else:
        candidate,_ = await resolve()
        # One gap for the linked boxes; OCR's 'Correct', which no model reads, is not delivered.
        assert not candidate['text_regions']
        assert [g['bbox'] for g in candidate['gaps']] == [[300,100,700,250]]


def test_final_audit_rejection_returns_bounded_actionable_feedback(tmp_path):
    from app.pipeline.document_recovery import RecoveryController
    (tmp_path / 'validation-image.json').write_text(json.dumps({'review':{
        'source_complete':False, 'declared_gaps_only':False,
        'issues':['Gap is a graphic, not unreadable lettering.'],
        'blocks':[{'block_id':'gap','status':'needs_review','observation':'Logo was classified as text.'},
                  {'block_id':'text','status':'verified','observation':'Supported'}]}}))
    feedback = RecoveryController.validation_feedback(tmp_path)
    assert feedback[0]['blocks'] == [{'block_id':'gap','status':'needs_review','observation':'Logo was classified as text.'}]
    assert feedback[0]['issues'] == ['Gap is a graphic, not unreadable lettering.']


@pytest.mark.asyncio
async def test_ocr_capacity_failure_is_shared_across_issues_but_different_crop_can_retry(tmp_path, monkeypatch):
    from app.adapters import ocr
    ir, _, runner = await setup(tmp_path)
    image = runner.inventory.directory / 'capacity.png'
    Image.new('RGB', (100, 100), 'white').save(image)
    ir.blocks.append(Block(block_id='picture', type=BlockType.IMAGE, page_idx=0))
    runner.inventory.mapping['picture'] = [0]
    runner.inventory.word_images['picture'] = image
    monkeypatch.setattr(ocr, 'ocr_runtime_identity', lambda: 'same-runtime')
    reader = AsyncMock(side_effect=ValueError('ocr_output_capacity_exceeded: select a smaller region'))
    monkeypatch.setattr(ocr, 'read_ocr', reader)
    decision = RecoveryDecision(action='ocr_image', block_id='picture', reason='Read text')
    for issue in ['image:picture', 'page:0']:
        with pytest.raises(ValueError, match='ocr_output_capacity_exceeded'):
            await runner.ocr_image(ir, {'id': issue, 'page_idx': 0}, decision, tmp_path / issue.replace(':', '-'), '')
    assert reader.await_count == 1
    decision.region = [0, 0, 1000, 500]
    with pytest.raises(ValueError, match='ocr_output_capacity_exceeded'):
        await runner.ocr_image(ir, {'page_idx': 0}, decision, tmp_path / 'crop', '')
    assert reader.await_count == 2
    monkeypatch.setattr(ocr, 'ocr_runtime_identity', lambda: 'new-runtime')
    with pytest.raises(ValueError, match='ocr_output_capacity_exceeded'):
        await runner.ocr_image(ir, {'page_idx': 0}, decision, tmp_path / 'new-runtime', '')
    assert reader.await_count == 3
