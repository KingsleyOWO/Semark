"""Dense adjudication and corrected image dependencies keep explicit, source-bound scope."""
import json

import pytest
from test_document_recovery import RecoveryModel, setup
from test_image_information import RegionModel, local_candidate

from app.pipeline.agentic import ModelOutputError
from app.pipeline.image_evidence import resolve_image_evidence_bounded
from app.pipeline.image_recovery import replace_image_review_dependencies


def observations(count):
    return [{'source_ref': 'ev_' + 'a' * 64 + f':region{i}', 'evidence_id': 'ev_' + 'a' * 64,
             'text': f'Literal label {i}', 'bbox': [0, i * 7, 1000, i * 7 + 6],
             'provider': 'paddleocr', 'score': .999} for i in range(count)]


class SelectRegions(RecoveryModel):
    def __init__(self, max_rows=24, omit=False):
        super().__init__()
        self.max_rows, self.omit, self.requests = max_rows, omit, []

    async def query(self, task, payload, images, schema):
        assert task == 'resolve_image_evidence'
        self.requests.append(payload)
        assert len(payload['candidates']) <= 24
        assert all(len(ref) < 8 for ref in payload['required_source_refs'])
        if len(payload['candidates']) > self.max_rows:
            raise ModelOutputError('Synthetic truncated JSON response', 100, 'length', '{')
        refs = payload['required_source_refs']
        if self.omit and len(refs) > 1:
            refs = refs[:-1]
        return {'selections': [{'source_ref': ref, 'decision': 'supported', 'reason': 'Visible source label.'}
                               for ref in refs], 'accounted_for': True,
                'observation': 'Only listed regions compared to source pixels.'}, 10


@pytest.mark.asyncio
async def test_128_candidates_are_six_bounded_responses_with_exact_original_ids(tmp_path):
    model = SelectRegions()
    _, owner, runner = await setup(tmp_path, model)
    candidates = observations(128)
    result, gaps, records = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), candidates)
    assert len(model.requests) == 6 and len(result.selections) == 128 and not gaps
    assert {s.source_ref for s in result.selections} == {c['source_ref'] for c in candidates}
    assert all(r['status'] == 'candidate_requires_source_audit' for r in records)
    assert all(len(json.dumps(r['resolution'])) < 6500 for r in records)
    assert not any('verified' in r for r in records)
    # Valid cached observations are usable even when the call budget is exhausted.
    owner.limits.max_model_calls = owner.state['usage']['model_calls']
    again, gaps, _ = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), candidates)
    assert again == result and not gaps and len(model.requests) == 6


@pytest.mark.asyncio
async def test_truncated_output_splits_in_shared_budget_and_caches_exact_children(tmp_path):
    model = SelectRegions(max_rows=12)
    _, owner, runner = await setup(tmp_path, model)
    result, gaps, records = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), observations(48))
    assert [len(p['candidates']) for p in model.requests] == [24, 12, 12, 24, 12, 12]
    assert len(result.selections) == 48 and not gaps
    assert sum(r['status'] == 'response_contract_failed' for r in records) == 2
    assert owner.state['usage']['model_calls'] == 6
    again, gaps, _ = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), observations(48))
    assert again == result and not gaps and len(model.requests) == 6


@pytest.mark.asyncio
async def test_missing_ids_split_and_do_not_cache_unaccounted_candidate(tmp_path):
    model = SelectRegions(omit=True)
    _, owner, runner = await setup(tmp_path, model)
    result, gaps, records = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), observations(2))
    assert len(model.requests) == 3 and not gaps and len(result.selections) == 2
    assert records[0]['status'] == 'response_contract_failed'
    assert len(owner.state['source_reading_cache']) == 2


@pytest.mark.asyncio
async def test_resolution_reserves_final_audits_and_does_not_call_budget_gap_unreadable(tmp_path):
    model = SelectRegions()
    _, owner, runner = await setup(tmp_path, model)
    owner.limits.max_model_calls = 2
    result, gaps, _ = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), observations(20))
    assert not model.requests and not result.accounted_for
    assert len(result.selections) == 20 and all(s.decision == 'uncertain' for s in result.selections)
    assert gaps == [{'bbox': [0, 0, 1000, 139], 'reason': 'image_evidence_budget_deferred'}]


@pytest.mark.asyncio
async def test_revised_source_audit_replaces_old_review_edge_and_keeps_valid_scene(tmp_path):
    """Real b000014 failure: revision says context complete, old map->gap edge persists."""
    class Corrected(RegionModel):
        async def query(self, task, payload, images, schema):
            if task == 'revise_image_regions':
                self.proposals.append(payload)
                return {'regions': [{'block_id': t['block_id'],
                    'region_kind': 'text' if t['block_id'] == 'sign' else 'graphic',
                    'text': '晴諾米拼圖工作坊' if t['block_id'] == 'sign' else '',
                    'description': '黃色拼圖形狀的圖示。' if t['block_id'] == 'icon' else '',
                    'evidence_ids': [t['source_evidence_id']], 'reason': 'Independent source region content.'}
                    for t in payload['targets']]}, 10
            result, tokens = await super().query(task, payload, images, schema)
            if task == 'audit_image_parts':
                sign = next(b for b in result['blocks'] if b['block_id'] == 'sign')
                if len(self.audits) == 1:
                    sign.update(context_complete=False, depends_on_block_ids=['icon'])
            return result, tokens

    model = Corrected()
    _, _, runner, candidate, folder = await local_candidate(tmp_path, model)
    assert await runner.validate(candidate, [0], folder, image_origin='picture')
    assert len(model.audits) == 2
    sign = candidate.get_block('sign')
    assert sign.payload['agentic_depends_on_block_ids'] == []
    assert sign.payload['agentic_review_depends_on_block_ids'] == []
    assert '黃色拼圖' in candidate.get_block('icon').get_text()
    assert not (folder / 'verified-subset').exists()


def test_new_audit_cannot_remove_structural_label_dependency():
    from types import SimpleNamespace

    from app.models.document_ir import Block, BlockType

    block = Block(block_id='chart-value', type=BlockType.TEXT, page_idx=0,
                  payload={'text': '100', 'agentic_depends_on_block_ids': ['unit']})
    replace_image_review_dependencies(block, SimpleNamespace(depends_on_block_ids=['uncertain-title']))
    replace_image_review_dependencies(block, SimpleNamespace(depends_on_block_ids=[]))
    assert block.payload['agentic_depends_on_block_ids'] == ['unit']
    assert block.payload['agentic_source_depends_on_block_ids'] == ['unit']


@pytest.mark.asyncio
async def test_repeated_format_splits_stop_at_six_calls_and_resume_does_not_reset_cap(tmp_path):
    model = SelectRegions(max_rows=6)
    _, owner, runner = await setup(tmp_path, model)
    candidates = observations(128)
    result, gaps, _ = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), candidates)
    assert len(model.requests) == 6 and len(result.selections) == 128 and gaps
    assert any(s.decision == 'supported' for s in result.selections)
    assert all(g['reason'] == 'image_evidence_budget_deferred' for g in gaps)
    resumed, again, _ = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), candidates)
    assert resumed == result and again == gaps and len(model.requests) == 6


def test_four_local_regions_are_never_silently_dropped_from_source_views(tmp_path):
    from PIL import Image

    from app.pipeline.image_evidence import source_region_views

    source = tmp_path / 'source.png'
    Image.new('RGB', (100, 100), 'white').save(source)
    regions = [[i * 100, 0, i * 100 + 99, 100] for i in range(4)]
    paths, shown = source_region_views(source, regions, max_regions=4)
    assert shown == regions and len(paths) == 5


@pytest.mark.asyncio
async def test_cross_batch_same_line_has_one_scoped_selection_and_no_repeated_required_ids(tmp_path):
    class Covered(SelectRegions):
        async def query(self, task, payload, images, schema):
            self.requests.append(payload)
            context = payload['context_candidates']
            chosen = context[0]['source_ref'] if context else payload['required_source_refs'][0]
            return {'selections': [{'source_ref': ref,
                'decision': 'supported' if ref == chosen else 'uncertain',
                'covered_by': None if ref == chosen else chosen, 'reason': 'Same complete visible line.'}
                for ref in payload['required_source_refs']], 'accounted_for': True,
                'observation': 'Context is checked against source pixels, not an extra output row.'}, 10

    model = Covered()
    _, owner, runner = await setup(tmp_path, model)
    candidates = [{**c, 'bbox': [0, 0, 1000, 100], 'text': 'Same line'} for c in observations(25)]
    result, gaps, _ = await resolve_image_evidence_bounded(owner, 'image', runner.inventory.image(0), candidates)
    assert len(model.requests) == 2 and len(result.selections) == 25 and not gaps
    selected = [s for s in result.selections if s.decision == 'supported']
    assert len(selected) == 1
    assert all(s.covered_by == selected[0].source_ref for s in result.selections if s.decision == 'uncertain')
    assert len(model.requests[-1]['context_candidates']) == 1
    assert not set(model.requests[0]['required_source_refs']) & set(model.requests[1]['required_source_refs'])


def test_image_validation_does_not_reserve_the_transaction_final_page_twice():
    from types import SimpleNamespace

    from app.pipeline.image_reading import image_validation_tasks

    assert image_validation_tasks(SimpleNamespace()) == ['audit_image_parts', 'audit_page']
    assert image_validation_tasks(SimpleNamespace(_protected_budget=(2, 150))) == ['audit_image_parts']


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['image_evidence_budget_deferred', 'image_evidence_response_contract_failed'])
async def test_withheld_chart_preserves_pending_evidence_cause_in_actual_delivery(tmp_path, reason):
    from app.models.document_ir import BlockType, DocumentIR
    from app.pipeline.document_recovery import render_candidate
    from app.pipeline.image_recovery import build_image_parts_candidate

    ir, owner, runner = await setup(tmp_path)
    ir.blocks[0].type = BlockType.IMAGE
    parts = {'visual_facts': [], 'text_regions': [], 'text_dependent': True, 'information_status': 'informative',
             'gaps': [{'bbox': [0, 0, 200, 200], 'reason': 'image_text_unreadable_or_disputed'},
                      {'bbox': [200, 0, 600, 200], 'reason': reason}]}
    folder = tmp_path / 'chart-candidate'
    candidate = build_image_parts_candidate(runner, ir, ir.blocks[0], 0, [0, 0, 1000, 1000],
        runner.inventory.image(0), parts, [], folder)
    # Exercise saved IR and the production outputs, not just an intermediate dict.
    reloaded = DocumentIR.model_validate_json(candidate.model_dump_json())
    assert len(reloaded.blocks) == 1
    gap = reloaded.blocks[0].payload['agentic_gap']
    assert gap['reason'] == reason and gap['image_bbox'] == [0, 0, 1000, 1000]
    assert set(gap['source_reasons']) == {'image_text_unreadable_or_disputed', reason}
    await render_candidate(owner.config, reloaded, folder, runner.root, None)
    document = '\n'.join(p.read_text() for p in (folder / 'outputs/documents').glob('*.md'))
    chunks = (folder / 'outputs/chunks.jsonl').read_text()
    for actual in [document, chunks]:
        assert '尚未完成證據核對' in actual and '無法可靠辨識' not in actual


@pytest.mark.asyncio
async def test_image_audit_format_retry_uses_same_budget_and_persisted_resume_cache(tmp_path):
    class AuditFormat(RegionModel):
        async def query(self, task, payload, images, schema):
            result, tokens = await super().query(task, payload, images, schema)
            if task == 'audit_image_parts':
                assert '300 characters' in payload['response_instruction']
                if len(self.audits) == 1:
                    result['blocks'][0]['observation'] = 'Long observation. ' * 50
                    raise ModelOutputError('blocks.0.observation exceeds 700 characters', 123,
                                           'stop', json.dumps(result))
                assert 'response_contract_error' in payload
            return result, tokens

    model = AuditFormat()
    _, owner, runner, candidate, folder = await local_candidate(tmp_path, model)
    candidate.get_block('icon').payload.pop('agentic_gap')
    candidate.get_block('icon').payload['text'] = '圖片可確認內容：黃色拼圖形狀的圖示。'
    owner.state['usage']['model_calls'] = 6  # The real failed case has only two calls left.
    owner.limits.max_model_calls = 8
    assert await runner.validate(candidate, [0], folder, image_origin='picture', image_pruned=True)
    assert owner.state['usage']['model_calls'] == 8 and len(model.audits) == 2
    assert len(owner.state['source_reading_cache']) == 1
    # Resume the exact audited source/output scope without resetting spent attempts.
    owner.state = json.loads(owner.state_path.read_text())
    assert await runner.validate(candidate, [0], folder, image_origin='picture', image_pruned=True)
    assert owner.state['usage']['model_calls'] == 8 and len(model.audits) == 2
