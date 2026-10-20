"""Information is source-judged; local proposals cannot self-certify or erase dependencies."""
import asyncio
import json

import pytest
from PIL import Image
from pydantic import ValidationError
from test_document_recovery import setup
from test_document_review import AuditModel, config, fixture
from test_image_recovery import ImageModel

from app.models.document_ir import Block, BlockType
from app.pipeline.document_recovery import render_candidate
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.image_reading import ImageParts, reconcile_image_parts
from app.pipeline.recovery_models import ImageRegionRevision


class RegionModel(ImageModel):
    def __init__(self, mode='graphic'):
        super().__init__()
        self.mode = mode
        self.audits = []
        self.proposals = []

    async def query(self, task, payload, images, schema):
        if task == 'revise_image_regions':
            self.proposals.append(payload)
            assert len(images) == 2 and payload['source_detail_regions'] == [[100, 100, 300, 300]]
            assert 'graphic' in payload['targets'][0]['audit_observation']
            target = payload['targets'][0]
            if self.mode == 'cancel':
                raise asyncio.CancelledError()
            if self.mode == 'invalid':
                return {'regions': []}, 10
            return {'regions': [{'block_id': 'foreign' if self.mode == 'foreign' else target['block_id'],
                'region_kind': 'graphic', 'description': '黃色拼圖形狀的圖示。',
                'evidence_ids': ['ev_' + '0' * 64] if self.mode == 'evidence' else [target['source_evidence_id']],
                'reason': 'Visible shape is a graphic, not lettering.'}]}, 10
        if task == 'audit_image_parts':
            self.audits.append(payload)
            gaps = {b['block_id']: b for b in payload['declared_gaps']}
            neutral = bool(gaps) and all(g['reason'] == 'region_not_interpreted' for g in gaps.values())
            fixed = not gaps and self.mode not in {'reject', 'reject_all', 'dependent'}
            accepted = (fixed or neutral) and self.mode != 'reject_all'
            return {'blocks': [{'block_id': bid, 'status': 'verified' if bid == 'sign' or fixed else 'needs_review',
                'context_complete': bid == 'sign' or fixed,
                'depends_on_block_ids': ['icon'] if bid == 'sign' and self.mode == 'dependent' else [],
                'observation': 'A graphic, not an unreadable character.' if bid == 'icon' else 'Literal source sign supported.'}
                for bid in payload['required_block_ids']],
                'source_complete': fixed, 'reading_order_correct': True,
                'declared_gaps_only': neutral and accepted,
                'issues': [] if fixed else ['Graphic region needs a faithful description or an explicit omission.'],
                'observation': 'Compared source pixels with actual delivered excerpts.'}, 10
        return await super().query(task, payload, images, schema)


async def local_candidate(tmp_path, model):
    ir, owner, runner = await setup(tmp_path, model)
    original = runner.root / 'agentic/evidence/picture.png'
    original.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (189, 142), 'yellow').save(original)
    ir.blocks.append(Block(block_id='picture', type=BlockType.IMAGE, page_idx=0,
                           payload={'img_path': str(original), 'caption': 'Old wrong name'}))
    runner.inventory.mapping['picture'] = [0]
    runner.inventory.word_images['picture'] = original
    runner.renderer = render_candidate
    candidate = ir.model_copy(deep=True)
    common = {**runner.source_payload(0), 'agentic_image_origin': 'picture',
              'agentic_source_image': str(original.relative_to(runner.root)), 'agentic_content_kind': 'transcription'}
    candidate.blocks[-1:] = [
        Block(block_id='sign', type=BlockType.TEXT, page_idx=0, payload={**common, 'text': '圖片文字：晴諾米拼圖工作室',
                                  'agentic_image_region': [350, 150, 705, 240]}),
        Block(block_id='icon', type=BlockType.TEXT, page_idx=0, payload={**common, 'text': 'Wrong unreadable-text gap',
            'agentic_image_region': [100, 100, 300, 300], 'agentic_gap': {'reason': 'image_text_unresolved'}})]
    folder = tmp_path / 'candidate'
    folder.mkdir()
    return ir, owner, runner, candidate, folder


@pytest.mark.asyncio
async def test_local_graphic_revision_requires_new_actual_delivery_audit_and_preserves_neighbors(tmp_path):
    model = RegionModel()
    ir, owner, runner, candidate, folder = await local_candidate(tmp_path, model)
    assert await runner.validate(candidate, [0], folder, image_origin='picture')
    assert len(model.proposals) == 1 and len(model.audits) == 2
    assert [b.block_id for b in candidate.blocks[:-2]] == [b.block_id for b in ir.blocks[:-1]]
    for block in ir.blocks[:-1]:
        assert candidate.get_block(block.block_id).model_dump() == block.model_dump()
    assert candidate.get_block('sign').get_text() == '圖片文字：晴諾米拼圖工作室'
    assert '黃色拼圖' in candidate.get_block('icon').get_text()
    assert not candidate.get_block('icon').payload.get('agentic_gap')
    assert all('Wrong unreadable' not in d['text'] for d in model.audits[-1]['delivered_documents'])
    assert any('黃色拼圖' in c['content'] for c in model.audits[-1]['delivered_chunks'])
    runner.evidence.validate_candidate(candidate)
    patch = json.loads((folder / 'image-region-patch.json').read_text())
    assert patch['status'] == 'candidate_requires_source_audit'
    assert (folder / 'validation-image-before-region-revision.json').is_file()
    assert json.loads((folder / 'validation-image.json').read_text())['review']['source_complete']
    assert owner.state['usage']['model_calls'] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['reject', 'foreign', 'evidence', 'invalid'])
async def test_rejected_or_invalid_proposal_only_retains_reaudited_independent_subset(tmp_path, mode):
    model = RegionModel(mode)
    _, _, runner, candidate, folder = await local_candidate(tmp_path, model)
    assert await runner.validate(candidate, [0], folder, image_origin='picture')
    assert len(model.proposals) == (2 if mode == 'invalid' else 1)
    assert candidate.get_block('icon').payload['agentic_gap']['reason'] == 'region_not_interpreted'
    assert '黃色拼圖' not in candidate.get_block('icon').get_text()
    assert '晴諾米' in candidate.get_block('sign').get_text()
    assert len(model.audits) == (3 if mode == 'reject' else 2)
    assert json.loads((folder / 'validation-image.json').read_text())['review']['declared_gaps_only']


@pytest.mark.asyncio
async def test_failed_revision_and_subset_leave_original_delivery_intact(tmp_path):
    model = RegionModel('reject_all')
    _, _, runner, candidate, folder = await local_candidate(tmp_path, model)
    assert not await runner.validate(candidate, [0], folder, image_origin='picture')
    assert candidate.get_block('icon').get_text() == 'Wrong unreadable-text gap'
    assert len(model.proposals) == 1 and len(model.audits) == 3
    assert 'Wrong unreadable-text gap' in (folder / 'outputs/chunks.jsonl').read_text()
    assert '黃色拼圖' not in (folder / 'outputs/chunks.jsonl').read_text()


@pytest.mark.asyncio
async def test_missing_dependency_cannot_survive_as_independent_fact(tmp_path):
    model = RegionModel('dependent')
    _, owner, runner, candidate, folder = await local_candidate(tmp_path, model)
    owner.limits.max_model_calls = 2  # initial audit only; insufficient for revision + audit + final audit
    assert not await runner.validate(candidate, [0], folder, image_origin='picture')
    assert not model.proposals and len(model.audits) == 1
    assert candidate.get_block('sign').payload['agentic_depends_on_block_ids'] == ['icon']


@pytest.mark.asyncio
async def test_region_revision_cancellation_propagates_without_mutating_candidate(tmp_path):
    model = RegionModel('cancel')
    _, _, runner, candidate, folder = await local_candidate(tmp_path, model)
    with pytest.raises(asyncio.CancelledError):
        await runner.validate(candidate, [0], folder, image_origin='picture')
    assert candidate.get_block('icon').get_text() == 'Wrong unreadable-text gap'
    assert len(model.audits) == 1


def test_non_informative_requires_explicit_reason_and_cannot_hide_text_regions():
    base = dict(image_type='other', visual_facts=[], visual_independent_of_text=True,
                text_regions=[], complete=True, observation='Blank pixels.')
    with pytest.raises(ValidationError):
        ImageParts(**base)
    blank = ImageParts(**base, information_status='non_informative', information_reason='Only uniform background.')
    assert reconcile_image_parts(blank, blank)['information_status'] == 'non_informative'
    with pytest.raises(ValidationError):
        ImageParts(**{**base, 'text_regions': [{'region_id': 't', 'bbox': [0, 0, 1000, 1000],
                    'readable': False, 'text': ''}]}, information_status='non_informative', information_reason='Small.')
    with pytest.raises(ValidationError):
        ImageRegionRevision(block_id='x', region_kind='no_information', text='Hidden lettering',
                            evidence_ids=['ev_x'], reason='Decorative.')


@pytest.mark.asyncio
@pytest.mark.parametrize('verified', [True, False])
async def test_only_verified_non_informative_notes_leave_rag_but_keep_reading_provenance(tmp_path, verified):
    blocks = [Block(block_id='good', type=BlockType.TEXT, page_idx=0, payload={'text': 'Revenue 100.'}),
              Block(block_id='blank', type=BlockType.TEXT, page_idx=0,
                    payload={'text': '圖片區域：無可提取的實質資訊。', 'agentic_information_status': 'non_informative'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    def review(_, value):
        for b in value['blocks']:
            b['context_complete'] = True
            if b['block_id'] == 'blank' and not verified:
                b['status'] = 'needs_review'
                value.update(source_complete=False, issues=['Source contains real information.'])
        return value
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    rag = (tmp_path / 'outputs/rag_chunks.jsonl').read_text()
    assert 'Revenue' in rag and '無可提取' not in rag
    delivery = (tmp_path / 'outputs/delivery.md').read_text()
    assert ('無可提取' in delivery) == verified


@pytest.mark.asyncio
async def test_verified_number_cannot_survive_when_required_label_revision_fails(tmp_path):
    model = RegionModel('dependent')
    _, _, runner, candidate, folder = await local_candidate(tmp_path, model)
    assert not await runner.validate(candidate, [0], folder, image_origin='picture')
    assert len(model.proposals) == 1 and len(model.audits) == 2
    assert candidate.get_block('icon').get_text() == 'Wrong unreadable-text gap'
    assert not (folder / 'verified-subset').exists()


@pytest.mark.asyncio
async def test_more_than_four_failed_regions_skip_proposal_and_audit_subset_once(tmp_path):
    model = RegionModel('reject')
    _, _, runner, candidate, folder = await local_candidate(tmp_path, model)
    icon = candidate.get_block('icon')
    candidate.blocks.extend(icon.model_copy(update={'block_id': f'icon_{i}'}, deep=True) for i in range(4))
    assert await runner.validate(candidate, [0], folder, image_origin='picture')
    assert not model.proposals and len(model.audits) == 2
    gaps = [b for b in candidate.blocks if b.payload.get('agentic_gap')]
    # The five pruned regions are one marker that keeps each region (2026-10-16).
    assert len(gaps) == 1 and len(gaps[0].payload['agentic_gap']['regions']) == 5
    assert gaps[0].get_text() == '〔原文第 1 頁圖片：未納入文字解讀的區域（保留來源圖） 5 處〕'


@pytest.mark.asyncio
@pytest.mark.parametrize('link', ['sign', 'foreign', 'disjoint'])
async def test_already_covered_region_needs_owned_overlapping_retained_block_and_reaudit(tmp_path, link):
    class Covered(RegionModel):
        async def query(self, task, payload, images, schema):
            if task == 'revise_image_regions':
                self.proposals.append(payload)
                target = payload['targets'][0]
                return {'regions': [{'block_id': target['block_id'], 'region_kind': 'text',
                    'covered_by': 'foreign' if link == 'foreign' else 'sign',
                    'evidence_ids': [target['source_evidence_id']], 'reason': 'Same lettering already transcribed.'}]}, 10
            return await super().query(task, payload, images, schema)
    model = Covered()
    _, _, runner, candidate, folder = await local_candidate(tmp_path, model)
    candidate.get_block('icon').payload['agentic_image_region'] = (
        [10, 10, 20, 20] if link == 'disjoint' else [350, 125, 650, 175])
    assert await runner.validate(candidate, [0], folder, image_origin='picture')
    assert len(model.proposals) == 1 and len(model.audits) == 2
    if link == 'sign':
        assert candidate.get_block('icon') is None
        assert model.audits[-1]['required_block_ids'] == ['sign']
        assert candidate.get_block('sign').payload['agentic_evidence_ids']
    else:
        assert candidate.get_block('icon').payload['agentic_gap']['reason'] == 'region_not_interpreted'
    assert candidate.get_block('sign').get_text().count('晴諾米') == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['audit_rejection', 'timeout', 'budget'])
async def test_local_improvement_failure_preserves_already_verified_partial_delivery(tmp_path, failure):
    class Partial(RegionModel):
        async def query(self, task, payload, images, schema):
            if task == 'revise_image_regions' and failure == 'timeout':
                raise TimeoutError('Local source reading timed out')
            value, tokens = await super().query(task, payload, images, schema)
            if task == 'audit_image_parts' and len(self.audits) == 1:
                value['declared_gaps_only'] = True
            return value, tokens
    model = Partial('reject_all')
    _, owner, runner, candidate, folder = await local_candidate(tmp_path, model)
    if failure == 'budget':
        owner.limits.max_model_calls = 1
    assert await runner.validate(candidate, [0], folder, image_origin='picture')
    assert candidate.get_block('icon').get_text() == 'Wrong unreadable-text gap'
    assert '黃色拼圖' not in (folder / 'outputs/chunks.jsonl').read_text()
    assert len(model.audits) == (2 if failure == 'audit_rejection' else 1)
    assert not (folder / 'verified-subset').exists()
