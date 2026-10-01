"""A corrected, measured relationship can replace a fallible first observation."""
import copy

import pytest
from PIL import Image, ImageDraw
from test_document_recovery import setup
from test_document_review import AuditModel

from app.models.document_ir import BlockType
from app.pipeline.image_geometry import (
    LocatedReading,
    canonical_reading,
    geometry_eligible,
    join_located_labels,
    measured_axes,
)
from app.pipeline.image_source_review import inherited_image_readings
from app.pipeline.recovery_models import RecoveryDecision
from app.pipeline.review_budget import tool_cost


def chart(path, *, grid=False):
    picture = Image.new('RGB', (600, 400), 'white')
    draw = ImageDraw.Draw(picture)
    draw.line((45, 230, 570, 230), fill='navy', width=2)
    draw.line((180, 20, 180, 380), fill='navy', width=2)
    draw.text((255, 90), 'ALPHA', fill='black')
    if grid:
        draw.line((45, 120, 570, 120), fill='navy', width=2)
    picture.save(path)
    return path


def source_reading():
    return {'source_id': 'image0', 'text': 'LOW HIGH ALPHA', 'kind': 'transcription',
        'complete': True, 'image_type': 'chart', 'relationships_complete': True,
        'observation': 'Untrusted earlier position was left.', 'relationships': [
            {'relation_id': 'r0', 'kind': 'axis_mapping', 'source_fact': 'Horizontal axis goes from LOW to HIGH.',
             'bbox': [0, 550, 1000, 600], 'certain': True},
            {'relation_id': 'r1', 'kind': 'spatial_position', 'source_fact': 'ALPHA is left and above.',
             'bbox': [420, 210, 600, 320], 'horizontal': 'left', 'vertical': 'above', 'certain': True}]}


def located():
    prior = source_reading()
    prior['relationships'][1]['label'] = 'ALPHA'
    return LocatedReading.model_validate(prior)


def test_axes_are_measured_away_from_image_midpoint_and_grids_do_not_enter(tmp_path):
    axes = measured_axes(chart(tmp_path / 'axis.png'))
    assert axes['intersection'] == pytest.approx([301, 576], abs=2)
    assert measured_axes(chart(tmp_path / 'grid.png', grid=True)) is None
    Image.new('RGB', (600, 400), 'white').save(tmp_path / 'blank.png')
    assert measured_axes(tmp_path / 'blank.png') is None
    assert not geometry_eligible({'image_type': 'photo', 'relationships': []})


def test_bbox_geometry_corrects_first_model_but_retains_axis_crossings(tmp_path):
    axes = measured_axes(chart(tmp_path / 'axis.png'))
    prior = source_reading()
    result = canonical_reading(located(), axes, prior, language='en')
    assert result.relationships[1].horizontal == 'right'
    assert prior['relationships'][1]['horizontal'] == 'left'
    assert 'right relative to the vertical axis' in result.relationships[1].source_fact
    crossing = located()
    crossing.relationships[1].bbox = [290, 550, 400, 600]
    value = canonical_reading(crossing, axes, prior, language='zh-TW')
    assert value.relationships[1].horizontal == value.relationships[1].vertical == 'spans_axis'
    assert '橫跨垂直軸' in value.relationships[1].source_fact


@pytest.mark.parametrize('change', ['missing_relation', 'repurposed_id', 'incomplete', 'uncertain', 'wrong_source', 'not_diagram'])
def test_geometry_cannot_launder_omissions_or_wrong_sources(tmp_path, change):
    value = located()
    if change == 'missing_relation':
        value.relationships.pop()
    elif change == 'repurposed_id':
        value.relationships[1].kind = 'other'
    elif change == 'incomplete':
        value.relationships_complete = False
    elif change == 'uncertain':
        value.relationships[1].certain = False
    elif change == 'wrong_source':
        value.source_id = 'unrelated-image'
    elif change == 'not_diagram':
        value.image_type = 'photo'
    with pytest.raises(ValueError):
        canonical_reading(value, measured_axes(chart(tmp_path / 'axis.png')), source_reading(), language='en')


@pytest.mark.parametrize('reject', [False, True])
@pytest.mark.asyncio
async def test_corrected_prior_requires_actual_delivery_audits_and_keeps_old_evidence(tmp_path, reject):
    class Model(AuditModel):
        async def query(self, task, payload, image, schema):
            self.calls.append((task, payload, image))
            if task == 'read_image_relations':
                assert 'measured_axes' in payload
                return located().model_dump(), 10
            assert task in {'audit_image_parts', 'audit_page'}
            observations = payload['independent_image_readings']['b0']
            assert observations['relationships'][1]['horizontal'] == 'right'
            output = []
            for block in payload['blocks']:
                text = block['untrusted_extracted_text']
                checks = [{'relation_id': r['relation_id'], 'status': 'contradicted' if reject else 'matched',
                    'delivered_quote': r['source_fact'], 'observation': 'Independent pixel/delivery check.',
                    'horizontal': r['horizontal'], 'vertical': r['vertical'], 'direction': r['direction']}
                    for r in observations['relationships'] if r['source_fact'] in text]
                output.append({'block_id': block['block_id'], 'status': 'verified', 'context_complete': True,
                               'observation': 'Literal source and actual output compared.', 'image_relationship_checks': checks})
            return {'blocks': output, 'source_complete': True, 'reading_order_correct': True,
                    'observation': 'Full source checked.', 'issues': []}, 10

    model = Model()
    ir, owner, runner = await setup(tmp_path, model)
    source = chart(tmp_path / 'chart.png')
    original = ir.blocks[0]
    original.type = BlockType.IMAGE
    original.payload = {'caption': 'ALPHA is incorrectly described as left.'}
    runner.inventory.word_images = {'b0': source}
    prior = source_reading()
    owner.state['pages']['0']['word_image_readings'] = {'b0': copy.deepcopy(prior)}
    folder = tmp_path / 'candidate'
    folder.mkdir()
    candidate, pages = await runner.repair_image(ir, {'block_id': 'b0', 'page_idx': 0},
        RecoveryDecision(action='repair_image', block_id='b0', reason='Check measured geometry.'), folder, '')
    assert owner.state['pages']['0']['word_image_readings']['b0'] == prior
    blocks = [b for b in candidate.blocks if b.payload.get('agentic_image_origin') == 'b0']
    corrected = inherited_image_readings(blocks, tmp_path)['b0']
    assert corrected.relationships[1].horizontal == 'right'
    record = runner.evidence.get(blocks[0].payload['agentic_relationship_evidence_id'])
    parents = [runner.evidence.get(eid) for eid in record['parents']]
    assert any(p['content'] == prior for p in parents)
    assert any(p['kind'] == 'image_axis_measurement' for p in parents)
    assert await runner.validate(candidate, pages, folder, image_origin='b0') is not reject
    inventory = await runner.candidate_inventory(candidate, folder)
    review = await owner._inspect(0, candidate, inventory, folder)
    assert (review['status'] == 'verified') is not reject
    assert [c[0] for c in model.calls] == ['read_image_relations', 'audit_image_parts', 'audit_page']


def test_geometry_budget_is_separate_from_free_form_image_recovery():
    small = tool_cost({}, 'repair_image', image_geometry=True)
    full = tool_cost({}, 'repair_image')
    assert small['model_calls'] == 6
    assert small['seconds'] < full['seconds']


def test_located_label_wraps_preserve_all_source_characters_and_repeated_occurrences():
    source = 'PIECE\nTALES\nOther\nPIECE\nTALES\nBuffalo\nGAMES & PUZZLES\n\nAlpha\n999\nBeta'
    joined = join_located_labels(source, ['PIECE TALES', 'Buffalo GAMES & PUZZLES', 'Alpha 100 Beta'])
    assert joined.count('PIECE TALES') == 2
    assert 'Buffalo GAMES & PUZZLES' in joined
    assert 'Alpha\n999\nBeta' in joined
    assert ''.join(joined.split()) == ''.join(source.split())
    assert '\n\n' in joined
