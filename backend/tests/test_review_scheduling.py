"""Whole-operation budgets, early tool routing, and bounded review evidence."""
import asyncio
import hashlib
import json

import pytest
from PIL import Image
from test_document_recovery import RecoveryModel, setup
from test_document_review import AuditModel, fixture, write_delivery

from app.config import AgenticConfig, PipelineConfig
from app.models.document_ir import BlockType
from app.pipeline.document_review import DocumentReview, PageReview
from app.pipeline.review_budget import document_budget, elapsed_before_review, tool_cost


def test_a_review_is_bounded_only_by_six_hours_whatever_its_earlier_stages_took():
    # Decided 2026-10-06: no limit on model calls or repair actions. Each problem's planned
    # methods (at most three, each once) decide the work; six hours caps it. Since 2026-10-15
    # the drafts are not counted in them: they took up to 4.3 of the six, leaving repairs none.
    limits = AgenticConfig()
    budget = document_budget(limits, 47, elapsed_seconds=3600)
    assert budget['policy'] == 'review_time_limit'
    assert budget['max_seconds'] == 6 * 3600 and budget['elapsed_before_review_seconds'] == 3600
    assert document_budget(limits, 3, elapsed_seconds=7 * 3600)['max_seconds'] == 6 * 3600
    assert limits.max_model_calls >= 10 ** 6 and limits.max_recovery_actions >= 10 ** 6 and limits.max_actions >= 10 ** 6
    assert limits.max_recovery_attempts_per_issue == 3
    assert limits.max_seconds == 6 * 3600  # planning must not mutate shared profiles


def test_a_stored_run_configuration_keeps_the_budget_it_was_created_with():
    stored = AgenticConfig(max_model_calls=120, max_seconds=7200, max_recovery_actions=6, adaptive_budget=True)
    budget = document_budget(stored, 47, elapsed_seconds=3600)
    assert (budget['max_seconds'], budget['max_model_calls']) == (6225, 120)
    explicit = document_budget(AgenticConfig(max_seconds=600, max_model_calls=18), 47)
    assert explicit['max_seconds'] == 600 and explicit['max_model_calls'] == 18


def test_time_spent_before_the_review_is_recorded():
    stats = {'parse': {'duration_seconds': 120.0}, 'enrich': {'duration_seconds': 3000.0},
             'normalize': {}, 'package': {'duration_seconds': 30.5}}
    assert elapsed_before_review(stats, current_stage_seconds=4.5) == 3155.0


def test_tool_estimate_covers_both_reads_and_both_delivery_audits():
    cost = tool_cost({}, 'read_source', pictures=4)
    assert cost['model_calls'] == 26  # Four images, two blind readers with <=3 calls each, two audits.
    assert cost['seconds'] > 68
    slow = {'calls': [{'task': 'read_image_parts', 'duration_seconds': 100}]}
    assert tool_cost(slow, 'read_source', pictures=4)['seconds'] > cost['seconds']
    assert tool_cost({}, 'read_regions', layout='horizontal4')['model_calls'] == 8
    assert tool_cost({}, 'rebuild_delivery', word_pictures=5)['model_calls'] == 9
    assert tool_cost({}, 'resolve_source', word_pictures=4)['model_calls'] == 8
    censored = {'calls': [{'task': 'recovery_plan', 'duration_seconds': 120, 'error': 'TimeoutError: '}]}
    from app.pipeline.review_budget import call_seconds
    assert call_seconds(censored, 'recovery_plan') >= 150
    slow_parse = {'recovery': {'steps': [{'action': 'reparse_pdf', 'duration_seconds': 500}]}}
    assert tool_cost(slow_parse, 'reparse_pdf')['seconds'] >= 625


@pytest.mark.asyncio
async def test_insufficient_seconds_skips_planning_and_never_starts_unfinishable_tool(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    owner.limits.max_seconds = 68
    original = ir.to_dict()
    output = await runner.run(ir)
    assert output.to_dict() == original
    assert not owner.model.tasks
    assert runner.state['actions'] == 0
    assert runner.state['issues']['page:0']['status'] == 'budget_exhausted'
    assert runner.state['issues']['page:0']['estimates']['read_source']['seconds'] > 68


@pytest.mark.asyncio
async def test_audit_cannot_spend_recovery_reserve_during_a_running_call(tmp_path):
    ir, source = fixture(tmp_path)

    class Slow(AuditModel):
        async def query(self, *args):
            await asyncio.sleep(1)
            return await super().query(*args)

    owner = DocumentReview(PipelineConfig(), Slow(), Slow())
    owner.state_path = tmp_path / 'state.json'
    owner.state = {'usage': {'seconds': 0, 'model_calls': 0, 'tokens': 0}, 'calls': []}
    owner.limits = AgenticConfig(max_seconds=0.3)
    owner._phase = 'audit'
    owner._reserve = (0, 0.27)
    with pytest.raises(TimeoutError):
        await owner._call('audit_page', {'block_id': 'p0'}, source, PageReview)
    assert owner.state['usage']['seconds'] < 0.15
    entry = owner.state['calls'][0]
    assert 0 < entry['timeout_seconds'] <= 0.031
    assert entry['duration_seconds'] >= 0.02
    assert entry['phase'] == 'audit' and entry['error'].startswith('TimeoutError')


@pytest.mark.asyncio
async def test_table_issue_waits_for_page_coverage_and_never_uses_fixed_legacy_repair(tmp_path, monkeypatch):
    import app.pipeline.document_recovery as recovery
    ir, source = fixture(tmp_path, pages=2)
    ir.blocks[0].type = BlockType.TABLE
    ir.blocks[0].payload = {'table_body': '<table><tr><td>Revenue 100, cost 80.</td></tr></table>'}
    write_delivery(tmp_path, ir)

    class Routed(RecoveryModel):
        def __init__(self):
            super().__init__()
            self.trace = []

        async def query(self, task, payload, image, schema):
            self.trace.append((task, payload.get('block_id'), payload))
            if task == 'recovery_plan':
                assert image is None  # route from evidence summaries, not another large image inference
                assert payload['source_features']['table_blocks'] == 1
            value, tokens = await super().query(task, payload, image, schema)
            if task == 'audit_page' and 'b0' in payload['required_block_ids']:
                value.update(source_complete=False, issues=['Missing illustration in summary table.'])
                value['blocks'][0]['status'] = 'needs_review'
            return value, tokens

    async def renderer(config, candidate, folder, original, cache):
        write_delivery(folder, candidate)

    async def rebuild(candidate):
        write_delivery(tmp_path, candidate)

    monkeypatch.setattr(recovery, 'render_candidate', renderer)
    model = Routed()
    owner = DocumentReview(PipelineConfig(), model, model)
    _, report = await owner.run(ir, tmp_path, source_path=source, rebuild=rebuild)
    tasks = [item[0] for item in model.trace]
    later_page = next(i for i, item in enumerate(model.trace) if item[:2] == ('audit_page', 'page:1'))
    assert tasks.index('recovery_plan') > later_page
    assert 'repair' not in tasks
    assert report['recovery']['steps'][0]['status'] == 'applied'
    assert report['summary']['verified_pages'] == 2
    assert report['budget']['hard_max_seconds'] == 6 * 3600  # one six-hour limit per document since 2026-10-06
    assert report['timing']['recovery_plan']['calls'] == 1
    assert owner.config.agentic.max_seconds == 6 * 3600


@pytest.mark.asyncio
async def test_context_overviews_are_smaller_but_cross_page_blocks_keep_full_detail(tmp_path):
    ir, source = fixture(tmp_path, pages=3)
    model = AuditModel()
    owner = DocumentReview(PipelineConfig(), model, model)
    await owner.run(ir, tmp_path, source_path=source)
    middle = next(c for c in model.calls if c[1]['page_idx'] == 1)
    sizes = [Image.open(p).size for p in middle[2]]
    assert sizes[0][0] > 480
    assert all(w <= 480 and h <= 680 for w, h in sizes[1:])
    # Explicit cross-page provenance requires full-size neighboring evidence.
    ir.blocks[1].payload.update(agentic_source_pages=[1, 2],
                               agentic_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    model = AuditModel()
    await DocumentReview(PipelineConfig(), model, model).run(ir, tmp_path, source_path=source)
    middle = next(c for c in model.calls if c[1]['page_idx'] == 1)
    assert all(Image.open(p).size[0] > 480 for p in middle[2])
    report = json.loads((tmp_path / 'outputs/review_coverage.json').read_text())
    assert report['pages'][1]['status'] == 'verified'


@pytest.mark.asyncio
async def test_neighbor_observations_do_not_trigger_rewrite_or_certify_neighbor(tmp_path):
    ir, source = fixture(tmp_path, pages=3)

    def report_context(payload, value):
        if payload['page_idx'] == 1:
            value['blocks'] += [{'block_id': b['block_id'], 'status': 'needs_review',
                                 'observation': 'Neighbor was supplied only for context.'}
                                for b in payload['context_blocks']]
        return value

    model = AuditModel(report_context)
    _, report = await DocumentReview(PipelineConfig(), model, model).run(ir, tmp_path, source_path=source)
    assert report['summary']['verified_pages'] == 3
    assert [b['block_id'] for b in report['pages'][1]['review']['blocks']] == ['b1']
    assert len(model.calls) == 3


@pytest.mark.parametrize('ids', [['context'], ['target', 'invented'], ['target', 'target']])
def test_context_observations_never_excuse_missing_invented_or_duplicate_targets(ids):
    from app.pipeline.document_review import target_block_reviews
    review = PageReview(blocks=[{'block_id': i, 'status': 'verified', 'observation': 'Source matches.'} for i in ids],
                        source_complete=True, reading_order_correct=True, observation='Checked.', issues=[])
    with pytest.raises(ValueError, match='omitted, duplicated, or invented'):
        target_block_reviews(review, ['target'], ['context'])


@pytest.mark.asyncio
async def test_failed_audit_response_retries_only_failed_page(tmp_path):
    ir, source = fixture(tmp_path, pages=2)
    config = PipelineConfig(agentic=AgenticConfig(max_model_calls=4, adaptive_budget=False))
    count = 0

    def temporary_failure(payload, value):
        nonlocal count
        if payload['page_idx'] == 1:
            count += 1
            if count == 1:
                value['blocks'] = []
        return value

    async def must_not_rewrite(candidate):
        raise AssertionError('Reviewer protocol failure must not rewrite source content')

    model = AuditModel(temporary_failure)
    _, report = await DocumentReview(config, model, model).run(ir, tmp_path, source_path=source, rebuild=must_not_rewrite)
    assert report['summary']['verified_pages'] == 2
    assert [c[1]['page_idx'] for c in model.calls] == [0, 1, 1]


@pytest.mark.asyncio
async def test_unrelated_cover_rebuild_reuses_scoped_review_but_changed_chunk_reaudits(tmp_path):
    from app.models.document_ir import Block
    from app.pipeline.review_sources import SourceInventory
    ir, source = fixture(tmp_path, pages=4)
    model = AuditModel()
    owner = DocumentReview(PipelineConfig(), model, model)
    await owner.run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 4
    reviewed_bundle = owner.state['pages']['3']['delivery_sha256']
    ir.blocks.insert(0, Block(block_id='new-cover', type=BlockType.TEXT, page_idx=0,
                              payload={'text': 'Additional cover material only: Acme registration.'}))
    for index, block in enumerate(ir.blocks):
        block.reading_order = index
    write_delivery(tmp_path, ir)
    chunk_file = tmp_path / 'outputs/chunks.jsonl'
    chunks = [json.loads(line) for line in chunk_file.read_text().splitlines()]
    for index, chunk in enumerate(chunks):
        chunk['chunk_id'] = f'renumbered-{index}'
    chunk_file.write_text('\n'.join(json.dumps(c) for c in chunks))
    inventory = SourceInventory(ir, tmp_path, source)
    await inventory.prepare()
    page = await owner._inspect(3, ir, inventory, tmp_path)
    assert len(model.calls) == 4
    assert page['status'] == 'verified' and page['reviewed_delivery_sha256'] == reviewed_bundle
    assert page['delivery_sha256'] != reviewed_bundle and page['reuse_reason']
    chunks[-1]['content'] = chunks[-1]['content'].replace('100', '999')
    chunk_file.write_text('\n'.join(json.dumps(c) for c in chunks))
    await owner._inspect(3, ir, inventory, tmp_path)
    assert len(model.calls) == 5
    assert '999' in model.calls[-1][1]['delivered_chunks'][-1]['content']


@pytest.mark.parametrize('changed', ['text', 'order', 'adjacency', 'membership', 'evidence'])
def test_scoped_cache_still_binds_content_sequence_adjacency_membership_and_images(changed):
    import copy

    from app.pipeline.review_sources import review_scope_digest
    payload = {'blocks': [{'block_id': 'a', 'reading_order': 10}, {'block_id': 'b', 'reading_order': 11}],
               'delivered_documents': [{'file': 'main.md', 'paragraph': 100, 'text': 'Revenue 100'},
                                       {'file': 'main.md', 'paragraph': 101, 'text': 'Cost 80'}],
               'delivered_chunks': [{'chunk_id': 'c10', 'content': 'Revenue 100. Cost 80.', 'block_ids': ['a', 'b']}]}
    evidence = [{'path': 'source.png', 'sha256': 'original'}]
    original = review_scope_digest('v', payload, evidence)
    other, images = copy.deepcopy(payload), copy.deepcopy(evidence)
    if changed == 'text':
        other['delivered_documents'][0]['text'] = 'Revenue 999'
    elif changed == 'order':
        other['delivered_documents'].reverse()
    elif changed == 'adjacency':
        other['delivered_documents'][1]['paragraph'] += 1
    elif changed == 'membership':
        other['delivered_chunks'][0]['block_ids'] = ['a']
    else:
        images[0]['sha256'] = 'different'
    assert review_scope_digest('v', other, images) != original
    assert payload['delivered_documents'][0]['paragraph'] == 100


@pytest.mark.asyncio
async def test_planner_format_error_retries_under_shared_budget_without_spending_action(tmp_path):
    from app.pipeline.recovery_models import RecoveryDecision
    class InvalidScale(RecoveryModel):
        async def query(self, task, payload, image, schema):
            self.tasks.append(task)
            if len(self.tasks) == 1:
                return {'action': 'read_regions', 'reason': 'Read table', 'scale': 2}, 10
            assert 'Only OCR accepts a scale' in payload['format_feedback']
            return {'action': 'read_regions', 'reason': 'Read source rows'}, 10
    model = InvalidScale()
    _, owner, runner = await setup(tmp_path, model)
    result = await runner.call('recovery_plan', {'tools': ['read_regions']}, None, RecoveryDecision)
    assert result.action == 'read_regions' and result.scale == 1
    assert owner.state['usage']['model_calls'] == 2 and runner.state['actions'] == 0


@pytest.mark.asyncio
async def test_dense_source_page_is_split_during_initial_audit(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from app.models.document_ir import Block
    ir, owner, runner = await setup(tmp_path)
    ir.blocks = [Block(block_id=f'b{i}', type=BlockType.TEXT, page_idx=0,
                       bbox_norm=[0, i * 10, 500, i * 10 + 5],
                       payload={'text': f'Source text {i}'}) for i in range(46)]
    runner.inventory.mapping = {b.block_id: [0] for b in ir.blocks}
    write_delivery(tmp_path, ir)
    owner._phase = 'audit'
    split = AsyncMock(return_value={'status': 'needs_review'})
    monkeypatch.setattr(owner, '_inspect_segments', split)
    await owner._inspect_page(0, ir, runner.inventory, tmp_path)
    split.assert_awaited_once()


@pytest.mark.asyncio
async def test_known_missing_schedule_relations_get_recovery_before_minor_earlier_page(tmp_path):
    from app.models.document_ir import Block
    ir, owner, runner = await setup(tmp_path)
    block = Block(block_id='schedule', type=BlockType.TABLE, page_idx=1, payload={'table_body':
        '<table><tr><td>Work</td><td>6月</td><td>7月</td><td>8月</td><td>9月</td></tr>'
        '<tr><td>Task</td><td></td><td></td><td></td><td></td></tr></table>'})
    ir.blocks.append(block)
    runner.inventory.pages.append(1)
    runner.inventory.mapping[block.block_id] = [1]
    owner.state['pages']['1'] = {'page_idx': 1, 'status': 'needs_review', 'reason': 'source_delivery_mismatch',
                                 'review': {'issues': ['Missing shaded monthly relations.']}}
    assert runner.problems(ir)[0]['id'] == 'page:1'


@pytest.mark.asyncio
async def test_planner_retry_feedback_cannot_exceed_context_limit(tmp_path):
    from app.pipeline.agentic import ModelOutputError
    from app.pipeline.recovery_models import RecoveryDecision

    class Broken(RecoveryModel):
        async def query(self, task, payload, image, schema):
            self.tasks.append(task)
            raise ModelOutputError('Invalid tool arguments ' + 'x' * 600, 10, 'stop', '{}')

    model = Broken()
    _, owner, runner = await setup(tmp_path, model)
    payload = {'context': 'x' * 41600}
    with pytest.raises(ValueError, match='recovery_context_limit'):
        await runner.call('recovery_plan', payload, None, RecoveryDecision)
    assert model.tasks == ['recovery_plan']
    assert owner.state['usage']['model_calls'] == 1
    assert runner.state['actions'] == 0
