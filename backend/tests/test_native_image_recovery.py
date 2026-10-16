"""Native page recovery keeps independently auditable scenes beside lettering gaps."""
import json

import pytest
from test_document_recovery import NativeModel, cfg, setup

from app.pipeline import recovery_evidence
from app.pipeline.agentic import BudgetExhausted
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.image_reading import ImageParts, reconcile_image_parts
from app.pipeline.recovery_models import RecoveryDecision


def photo_parts(*, complete=True, image_type='photo', scene='People sit around tables in a room.'):
    return {'information_status': 'informative', 'information_reason': 'Visible people and lettering.',
            'image_type': image_type, 'visual_facts': [scene],
            'visual_independent_of_text': image_type == 'photo',
            'text_regions': [{'region_id': 'sign', 'bbox': [100, 100, 900, 250],
                              'readable': False, 'text': ''}],
            'complete': complete, 'observation': 'Scene is visible; small lettering is unresolved.'}


class SceneModel(NativeModel):
    def __init__(self, *, reject_scene=False, incomplete=False, split=False):
        super().__init__(action='read_source', actions=['read_source', 'defer'])
        self.reject_scene, self.incomplete, self.split = reject_scene, incomplete, split
        self.inputs = []

    async def query(self, task, payload, image, schema):
        if task in {'read_image_parts', 'verify_image_parts'}:
            self.tasks.append(task)
            self.inputs.append((task, payload, image))
            assert 'full_page_context' not in json.dumps(payload)
            assert 'Revenue 100' not in json.dumps(payload)
            assert 'People sit' not in json.dumps(payload)
            if self.split and 'source_core' not in payload:
                return {'malformed_response': True}, 1
            return photo_parts(complete=not (self.incomplete and task == 'verify_image_parts')), 1
        data, tokens = await super().query(task, payload, image, schema)
        if task == 'audit_page' and self.reject_scene:
            for finding in data['blocks']:
                if finding['block_id'].endswith('_v0'):
                    finding.update(status='needs_review', context_complete=False,
                                   observation='Claimed scene is unsupported by original image.')
                    data.update(declared_gaps_only=False, source_complete=False,
                                issues=['Scene candidate is unsupported.'])
        return data, tokens


async def scene_setup(tmp_path, monkeypatch, model):
    ir, owner, runner = await setup(tmp_path, model, cfg(max_recovery_actions=1))
    source = runner.inventory.image(0)
    monkeypatch.setattr(recovery_evidence, 'native_materials', lambda *args: [
        {'id': 'text-0', 'text': 'Source page 1: Revenue 100, cost 80.', 'bbox': [0, 0, 1000, 200],
         'kind': 'native_text', 'image': None},
        {'id': 'image-1', 'text': '', 'bbox': [0, 200, 1000, 1000],
         'kind': 'pdf_embedded_image', 'image': source}])
    return ir, owner, runner


@pytest.mark.asyncio
@pytest.mark.parametrize('incomplete', [False, True])
async def test_native_photo_scene_survives_text_gap_only_after_real_candidate_audit(tmp_path, monkeypatch, incomplete):
    model = SceneModel(incomplete=incomplete)
    ir, owner, runner = await scene_setup(tmp_path, monkeypatch, model)
    result = await runner.run(ir)
    assert result is not ir and runner.state['steps'][0]['status'] == 'applied'
    scenes = [b for b in result.blocks if b.payload.get('agentic_content_kind') == 'visual_summary']
    gaps = [b for b in result.blocks if b.payload.get('agentic_gap')]
    assert len(scenes) == 1 and gaps
    assert 'People sit around tables' in scenes[0].get_text()
    assert not scenes[0].payload.get('agentic_depends_on_block_ids')
    assert all(b.payload['agentic_native_source_id'] == 'image-1' for b in scenes + gaps)
    assert all((tmp_path / b.payload['agentic_source_image']).is_file() for b in scenes + gaps)
    assert model.tasks.count('read_image_parts') == model.tasks.count('verify_image_parts') == 1
    assert 'audit_page' in model.tasks
    assert owner.state['pages']['0']['review']['declared_gaps_only']
    owner.state['fingerprint'] = 'isolated-native-source-fixture'
    owner._save()
    reviewed, _ = await DocumentReview(owner.config, model, model).run(
        result, tmp_path, source_path=runner.inventory.pdf)
    final = finalize_delivery(reviewed, tmp_path, owner.config)
    assert final['processing_status'] == 'partial'
    safe = (tmp_path / 'outputs/rag_chunks.jsonl').read_text()
    assert 'People sit around tables' in safe and 'Revenue 100' in safe
    assert '無法可靠辨識' not in safe
    checks = json.loads(next((tmp_path / 'agentic/recovery').glob('*/native-checks.json')).read_text())
    assert checks[0]['candidate']['visual_facts']
    assert checks[0]['independent']['complete'] is not incomplete


@pytest.mark.asyncio
async def test_agreeing_readers_cannot_publish_an_unsupported_scene(tmp_path, monkeypatch):
    model = SceneModel(reject_scene=True)
    ir, owner, runner = await scene_setup(tmp_path, monkeypatch, model)
    before_ir = ir.to_dict()
    before_document = (tmp_path / 'outputs/documents/main.md').read_bytes()
    result = await runner.run(ir)
    assert result.to_dict() == before_ir
    assert (tmp_path / 'outputs/documents/main.md').read_bytes() == before_document
    assert runner.state['steps'][0]['status'] == 'rejected'
    assert 'accepted' not in runner.state
    assert 'audit_page' in model.tasks  # Two matching readers alone never certify their claim.
    assert owner.state['usage']['model_calls'] >= 4


@pytest.mark.parametrize('kind', ['logo', 'diagram', 'chart', 'table', 'document'])
def test_incomplete_text_dependent_visual_cannot_use_photo_scene_preservation(kind):
    first = ImageParts.model_validate(photo_parts(image_type=kind))
    second = ImageParts.model_validate(photo_parts(image_type=kind, complete=False))
    candidate = reconcile_image_parts(first, second)
    assert not candidate['visual_facts'] and not candidate['text_regions']
    assert candidate['gaps'] and candidate['text_dependent']


@pytest.mark.asyncio
async def test_native_contract_failure_splits_once_per_reader_with_shared_six_call_limit(tmp_path, monkeypatch):
    model = SceneModel(split=True)
    ir, owner, runner = await scene_setup(tmp_path, monkeypatch, model)
    candidate, pages = await runner.read_source(ir, {'page_idx': 0},
        RecoveryDecision(action='read_source', reason='Read original materials.'), tmp_path / 'candidate', 'Read original image.')
    assert pages == [0]
    assert model.tasks.count('read_image_parts') == model.tasks.count('verify_image_parts') == 3
    assert owner.state['usage']['model_calls'] == 6
    assert any(b.payload.get('agentic_content_kind') == 'visual_summary' for b in candidate.blocks)
    assert len([p for _, p, _ in model.inputs if 'source_core' in p]) == 4


@pytest.mark.asyncio
async def test_native_reading_does_not_spend_final_audit_reservation(tmp_path, monkeypatch):
    model = SceneModel()
    ir, owner, runner = await scene_setup(tmp_path, monkeypatch, model)
    owner.limits.max_model_calls = 7  # Six possible image calls leave only one of the two audits.
    with pytest.raises(BudgetExhausted):
        await runner.read_source(ir, {'page_idx': 0}, RecoveryDecision(action='read_source', reason='Read.'),
                                 tmp_path / 'candidate', 'Read original image.')
    assert model.tasks == []
    owner.limits.max_model_calls = 8
    owner._protected_budget = (1, 75)
    await runner.read_source(ir, {'page_idx': 0}, RecoveryDecision(action='read_source', reason='Read.'),
                             tmp_path / 'candidate', 'Read original image.')
    assert owner.available_budget()[0] == 5  # Protected final audit is not charged twice.


@pytest.mark.asyncio
async def test_a_page_reread_names_the_picture_it_replaces_and_keeps_its_drafted_lines(tmp_path, monkeypatch):
    """A page re-read keeps the picture's link and its drafted lines (2026-10-16): a Word picture matched
    by its bytes keeps its own document, and a reading that leaves its sign a gap drops no read name."""
    from app.models.document_ir import Block, BlockType

    ir, owner, runner = await setup(tmp_path, SceneModel(), cfg(max_recovery_actions=1))
    source = runner.inventory.image(0)
    monkeypatch.setattr(recovery_evidence, 'native_materials', lambda *args: [
        {'id': 'image-0', 'text': '', 'bbox': [0, 200, 1000, 1000], 'kind': 'word_embedded_image',
         'image': source, 'word_sha256': 'a' * 64}])
    ir.blocks.append(Block(block_id='pic', type=BlockType.IMAGE, page_idx=0, reading_order=len(ir.blocks),
                           payload={'img_path': 'images/pic.png', 'word_source': {'image_sha256': 'a' * 64}}))
    runner.inventory.mapping['pic'] = [0]
    (tmp_path / 'outputs').mkdir(exist_ok=True)
    with open(tmp_path / 'outputs/enrichments.jsonl', 'a') as handle:
        handle.write(json.dumps({'block_id': 'pic', 'output': {'all_text': ['晴諾米拼圖工作室']}}, ensure_ascii=False) + '\n')
    candidate, _ = await runner.read_source(ir, {'page_idx': 0}, RecoveryDecision(action='read_source', reason='Read.'),
                                            tmp_path / 'candidate', 'Read original image.')
    pieces = [b for b in candidate.blocks if b.payload.get('agentic_native_source_id') == 'image-0']
    assert pieces and {b.payload.get('agentic_replaced_picture') for b in pieces} == {'pic'}
    assert {b.payload.get('agentic_image_path') for b in pieces} == {'images/pic.png'}
    assert any(b.get_text() == '圖片文字：晴諾米拼圖工作室' for b in pieces)
    assert not any(b.payload.get('agentic_image_origin') for b in pieces)
