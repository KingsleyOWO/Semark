"""Regression coverage for row boundaries, bounded context and lossless assembly."""
import json
from types import SimpleNamespace

import fitz
import pytest
from pydantic import ValidationError
from test_document_recovery import RecoveryModel, setup

from app.models.document_ir import Block, BlockType
from app.pipeline.recovery_models import RecoveryDecision, RegionReading
from app.pipeline.region_recovery import plan_regions


def table_source(tmp_path, *, merged=False):
    path = tmp_path / 'table.pdf'
    pdf = fitz.open()
    page = pdf.new_page(width=600, height=800)
    for y in (80, 240, 440, 600, 720):
        page.draw_line((40, y), (560, y))
    for x in (40, 300, 560):
        page.draw_line((x, 80), (x, 720))
    labels = [('Metric', 'Value'), ('Revenue', '100'), ('Profit', '80'), ('Growth', '20')]
    for y, (label, value) in zip((100, 260, 460, 620), labels):
        page.insert_text((50, y), label)
        page.insert_text((320, y), value)
    pdf.save(path)
    pdf.close()
    block = Block(block_id='native', type=BlockType.TABLE, page_idx=0, payload={'table_body':
        '<table>' + ''.join(f'<tr><td>{label}</td><td>{value}</td></tr>' for label, value in labels) + '</table>'})
    return SimpleNamespace(pdf=path, mapping={'native': [0]}, blocks=lambda *_: [block])


def test_source_row_boundary_replaces_midpoint_and_binds_native_cells(tmp_path):
    inventory = table_source(tmp_path)
    plan = plan_regions(inventory, None, 0, 'horizontal2')
    assert [p['core'] for p in plan] == [[0, 0, 1000, 550], [0, 550, 1000, 1000]]
    ids = [r['source_id'] for p in plan for r in p['owned_rows']]
    assert ids == ['native:r1', 'native:r2', 'native:r3', 'native:r4']
    assert plan[1]['context_rows'][0]['source_id'] == 'native:r1'
    assert plan[1]['owned_rows'][0]['native_cells'][1]['text'] == '80'


def test_table_avoids_left_right_quadrants_and_ambiguous_native_binding(tmp_path):
    inventory = table_source(tmp_path)
    source = inventory.blocks()[0]
    duplicate = source.model_copy(update={'block_id': 'other'})
    inventory.mapping['other'] = [0]
    inventory.blocks = lambda *_: [source, duplicate]
    plan = plan_regions(inventory, None, 0, 'grid4')
    assert all(p['core'][0] == 0 and p['core'][2] == 1000 for p in plan)
    assert not any(r['native_cells'] for p in plan for r in p['owned_rows'])


@pytest.mark.parametrize('value', [
    {'text': '', 'complete': True, 'status': 'complete'},
    {'text': '', 'complete': False, 'status': 'needs_context'},
    {'text': '', 'complete': True, 'status': 'needs_context', 'context_request': 'below'},
    {'text': '', 'complete': False, 'status': 'unreadable', 'context_request': 'below'},
])
def test_incomplete_or_context_requests_cannot_claim_completion(value):
    with pytest.raises(ValidationError):
        RegionReading(observation='source detail', **value)


@pytest.mark.asyncio
async def test_context_expands_without_changing_ownership_or_repeating_completed_reads(tmp_path):
    class ContextModel(RecoveryModel):
        def __init__(self):
            super().__init__()
            self.reads = []

        async def query(self, task, payload, image, schema):
            if task == 'read_region':
                self.reads.append(payload)
                if payload['core_bbox'][1] == 500 and payload['crop_bbox'][1] > 0:
                    return {'text': '', 'complete': False, 'status': 'needs_context',
                            'context_request': 'above', 'observation': 'Need the preceding header.'}, 10
            return await super().query(task, payload, image, schema)
    model = ContextModel()
    ir, owner, runner = await setup(tmp_path, model)
    decision = RecoveryDecision(action='read_regions', reason='Recover source content.')
    candidate, _ = await runner.read_regions(ir, {'page_idx': 0}, decision, tmp_path / 'first', '')
    assert len(model.reads) == 3
    assert model.reads[-1]['core_bbox'] == [0, 500, 1000, 1000]
    assert model.reads[-1]['crop_bbox'] == [0, 0, 1000, 1000]
    owner.state = json.loads(owner.state_path.read_text())
    resumed, _ = await runner.read_regions(ir, {'page_idx': 0}, decision, tmp_path / 'resumed', '')
    assert len(model.reads) == 3
    assert resumed.blocks[0].get_text() == candidate.blocks[0].get_text()
    assert candidate.blocks[0].get_text() == 'Revenue 100, cost 80.\n\nEnd of report.'
    assert 'merge_regions' not in model.tasks


@pytest.mark.asyncio
async def test_continuation_is_bounded_and_does_not_publish_incomplete_candidate(tmp_path):
    class NeedsContext(RecoveryModel):
        async def query(self, task, payload, image, schema):
            self.tasks.append(task)
            return {'text': '', 'complete': False, 'status': 'needs_context',
                    'context_request': 'below', 'observation': 'Need more source.'}, 10
    model = NeedsContext()
    ir, owner, runner = await setup(tmp_path, model)
    before = ir.model_dump()
    with pytest.raises(ValueError, match='region_context_exhausted'):
        await runner.read_regions(ir, {'page_idx': 0}, RecoveryDecision(action='read_regions', reason='Read'), tmp_path / 'candidate', '')
    assert model.tasks == ['read_region', 'read_region']
    assert ir.model_dump() == before and 'accepted' not in runner.state


@pytest.mark.asyncio
async def test_unreadable_is_not_retried_as_missing_context(tmp_path):
    class Unreadable(RecoveryModel):
        async def query(self, task, payload, image, schema):
            self.tasks.append(task)
            return {'text': '', 'complete': False, 'status': 'unreadable', 'observation': 'Pixels illegible.'}, 10
    model = Unreadable()
    ir, _, runner = await setup(tmp_path, model)
    with pytest.raises(ValueError, match='region_unreadable'):
        await runner.read_regions(ir, {'page_idx': 0}, RecoveryDecision(action='read_regions', reason='Read'), tmp_path / 'candidate', '')
    assert model.tasks == ['read_region']


@pytest.mark.asyncio
async def test_native_cell_omission_is_rejected_before_whole_page_audit(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    inventory = table_source(tmp_path)
    runner.inventory.pdf = inventory.pdf
    runner.inventory.blocks = inventory.blocks
    runner.inventory.mapping = inventory.mapping
    with pytest.raises(ValueError, match='region_omitted_native_cell'):
        await runner.read_regions(ir, {'page_idx': 0}, RecoveryDecision(action='read_regions', reason='Read'), tmp_path / 'candidate', '')
    assert 'audit_page' not in owner.model.tasks
