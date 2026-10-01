"""Layout observations must crop the right source without certifying or losing content."""
import asyncio
import json
from unittest.mock import AsyncMock

import fitz
import pytest
from pydantic import ValidationError
from test_document_recovery import RecoveryModel, setup

from app.adapters.mineru import MinerUAdapter, MinerUResult
from app.pipeline.document_recovery import crop_region
from app.pipeline.layout_evidence import MAX_REGIONS, read_layout
from app.pipeline.recovery_models import RecoveryDecision
from app.pipeline.review_budget import tool_cost


def put(root, suffix, data):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ('sample_' + suffix + '.json')
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize('kind,data', [
    ('model', [{'page_info': {'page_no': 0, 'width': 1200, 'height': 1600},
                'layout_dets': [{'category_id': 3, 'poly': [120, 320, 600, 320, 600, 960, 120, 960], 'score': .9}]}]),
    ('model', [{'page_info': {'page_no': 0, 'width': 1200, 'height': 1600},
                'layout_dets': [{'label': 'image', 'bbox': [120, 320, 600, 960], 'score': .9}]}]),
    ('model', [[{'type': 'image', 'bbox': [.1, .2, .5, .6]}]]),
    ('middle', {'pdf_info': [{'page_idx': 0, 'page_size': [600, 800],
                             'preproc_blocks': [{'type': 'image', 'bbox': [60, 160, 300, 480]}]}]}),
    ('content_list', [{'type': 'image', 'page_idx': 0, 'bbox': [100, 200, 500, 600]}]),
])
def test_backend_units_refer_to_the_same_physical_region(tmp_path, kind, data):
    put(tmp_path, kind, data)
    result = read_layout(tmp_path, 0, [600, 800], 'source')
    assert result['regions'][0]['bbox'] == [100, 200, 500, 600]
    assert result['regions'][0]['type'] == 'image'
    assert result['exhaustive'] is False
    assert 'text' not in result['regions'][0]


def test_units_are_declared_not_guessed_from_small_coordinates(tmp_path):
    put(tmp_path, 'content_list', [{'type': 'image', 'page_idx': 0, 'bbox': [0, 0, 1, 1]}])
    assert read_layout(tmp_path, 0, [600, 800], 's')['regions'][0]['bbox'] == [0, 0, 1, 1]


def test_invalid_boxes_and_scores_cannot_become_source_proof(tmp_path):
    put(tmp_path, 'model', [[{'type': 'image', 'bbox': b, 'score': float('nan')} for b in
                           [[.1, .2, .5, .6], [.5, .2, .1, .6], [-.1, 0, 1, 1],
                            [0, 0, 1000, 1000], [0, 0, float('nan'), .5], [True, 0, 1, 1]]]])
    result = read_layout(tmp_path, 0, [600, 800], 's')
    assert len(result['regions']) == 1 and result['rejected_regions'] == 5
    assert result['regions'][0]['score'] is None


def test_middle_preferred_no_duplicate_observations_and_truncation_explicit(tmp_path):
    put(tmp_path, 'middle', {'pdf_info': [{'page_idx': 0, 'page_size': [600, 800],
        'preproc_blocks': [{'type': 'text', 'bbox': [10, 10+i, 100, 11+i]} for i in range(70)]
                          + [{'type': 'image', 'bbox': [200, 600, 500, 750]}]}]})
    put(tmp_path, 'model', [[{'type': 'table', 'bbox': [.1, .2, .5, .6]}]])
    result = read_layout(tmp_path, 0, [600, 800], 's')
    assert result['artifact_kind'] == 'middle' and result['total_regions'] == 71
    assert len(result['regions']) == MAX_REGIONS and result['omitted_regions'] == 7
    assert result['regions'][0]['type'] == 'image'


def test_metadata_dimensions_and_external_symlink_are_not_trusted(tmp_path):
    root = tmp_path / 'parse'
    path = put(root, 'middle', {'pdf_info': [{'page_idx': 0, 'page_size': [800, 600],
                                           'preproc_blocks': [{'type': 'image', 'bbox': [1, 2, 100, 200]}]}]})
    assert read_layout(root, 0, [600, 800], 's') is None
    other = put(tmp_path, 'content_list', [{'type': 'image', 'page_idx': 0, 'bbox': [0, 0, 100, 100]}])
    path.unlink()
    (root / other.name).symlink_to(other)
    assert read_layout(root, 0, [600, 800], 's') is None


async def with_cache(tmp_path, model=None):
    ir, owner, runner = await setup(tmp_path, model=model)
    cache = tmp_path / 'parse-cache'
    put(cache, 'content_list', [{'type': 'image', 'page_idx': 0, 'bbox': [100, 100, 300, 300]},
                              {'type': 'table', 'page_idx': 0, 'bbox': [100, 550, 900, 950]}])
    runner.layout_store.parse_cache = cache
    return ir, owner, runner, cache


@pytest.mark.asyncio
async def test_selection_is_bound_to_page_source_artifact_and_observed_ids(tmp_path):
    _, _, runner, cache = await with_cache(tmp_path)
    store = runner.layout_store
    observed = await store.inspect(0)
    ids = [observed['regions'][0]['region_id']]
    assert store.selected(0, ids, observed)[0]['bbox'] == [100, 100, 300, 300]
    for invalid in [['../outside'], ids * 2, ['layout_invented']]:
        with pytest.raises(ValueError):
            store.selected(0, invalid, observed)
    for wrong in [{**observed, 'source_sha256': 'other'}, {**observed, 'page_idx': 1}, None]:
        with pytest.raises(ValueError):
            store.selected(0, ids, wrong)
    with pytest.raises(ValueError):
        store.selected(1, ids, observed)
    path = cache / 'sample_content_list.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='source-bound'):
        store.selected(0, ids, observed)


class LayoutModel(RecoveryModel):
    def __init__(self, *, defer=False, repeat=False):
        super().__init__()
        self.defer, self.repeat = defer, repeat
        self.plans, self.details = [], []

    async def query(self, task, payload, image, schema):
        if task == 'recovery_plan':
            self.tasks.append(task)
            self.plans.append(payload)
            if not payload.get('layout_observation') or self.repeat:
                return {'action': 'inspect_layout', 'reason': 'Locate the omitted visual.'}, 10
            if self.defer:
                return {'action': 'defer', 'reason': 'No remaining reliable source evidence.'}, 10
            return {'action': 'read_regions', 'reason': 'Magnify the source image without dropping the page.',
                    'region_ids': [payload['layout_observation']['regions'][0]['region_id']]}, 10
        if task == 'read_region':
            self.details.append((payload, image))
        return await super().query(task, payload, image, schema)


@pytest.mark.asyncio
async def test_query_only_does_not_rebuild_accept_or_certify_delivery(tmp_path, monkeypatch):
    model = LayoutModel(defer=True)
    ir, owner, runner, _ = await with_cache(tmp_path, model)
    parse = AsyncMock(side_effect=AssertionError('cache must be reused'))
    monkeypatch.setattr(MinerUAdapter, 'parse', parse)
    before = (tmp_path / 'outputs/documents/main.md').read_bytes()
    runner.rebuild = AsyncMock(side_effect=AssertionError('query cannot publish'))
    out = await runner.run(ir)
    assert out is ir and (tmp_path / 'outputs/documents/main.md').read_bytes() == before
    assert owner.state['pages']['0']['status'] == 'needs_review'
    assert runner.state['actions'] == 0 and runner.state['queries'] == 1
    assert runner.state['steps'][0]['status'] == 'observed'
    assert 'accepted' not in runner.state and not parse.called
    assert 'inspect_layout' not in model.plans[1]['tools']


@pytest.mark.asyncio
async def test_query_and_selected_detail_fit_one_repair_action_and_keep_full_coverage(tmp_path):
    model = LayoutModel()
    ir, owner, runner, _ = await with_cache(tmp_path, model)
    out = await runner.run(ir, max_new_actions=1)
    assert runner.state['queries'] == 1 and runner.state['actions'] == 1
    assert [s['status'] for s in runner.state['steps']] == ['observed', 'applied']
    assert model.tasks.count('audit_page') == 1  # Actual rebuilt excerpts are identical; exact audit cache is reusable.
    assert 'Revenue 100' in out.blocks[0].get_text() and 'End of report.' in out.blocks[0].get_text()
    assert [p['core_bbox'] for p, _ in model.details] == [[0, 0, 1000, 500], [0, 500, 1000, 1000]]
    assert len(model.details[0][1]) == 3 and len(model.details[1][1]) == 2
    assert model.details[0][0]['detail_images'][0]['image_number'] == 3
    assert out.blocks[0].payload['agentic_layout_region_ids']


@pytest.mark.asyncio
async def test_failed_query_is_bounded_and_does_not_use_repair_attempt(tmp_path, monkeypatch):
    model = LayoutModel(repeat=True)
    ir, _, runner = await setup(tmp_path, model=model)
    runner.owner.limits.max_seconds = 900  # Query reserves bounded context reads plus both audits.
    query = AsyncMock(side_effect=ValueError('layout unavailable'))
    monkeypatch.setattr(runner, 'inspect_layout', query)
    await runner.run(ir, max_new_actions=1)
    assert query.await_count == 1 and runner.state['queries'] == 1
    assert runner.state['actions'] == 1
    assert runner.state['steps'][0]['status'] == 'rejected'


@pytest.mark.asyncio
async def test_layout_inspection_and_reparse_share_one_physical_page_parse(tmp_path, monkeypatch):
    ir, _, runner = await setup(tmp_path)
    calls = []

    async def parse(self, source, output, config):
        with fitz.open(source) as pdf:
            assert len(pdf) == 1 and 'Revenue 100' in pdf[0].get_text()
        calls.append(1)
        path = put(output, 'content_list', [{'type': 'text', 'page_idx': 0,
                    'bbox': [0, 0, 1000, 1000], 'text': 'Revenue 100, cost 80.'}])
        return MinerUResult(success=True, output_dir=output, content_list_path=path)

    monkeypatch.setattr(MinerUAdapter, 'parse', parse)
    first = await runner.layout_store.inspect(0)
    second = await runner.layout_store.inspect(0)
    assert first['cache_hit'] is False and second['cache_hit'] is True
    candidate, pages = await runner.reparse_pdf(ir, {'page_idx': 0},
        RecoveryDecision(action='reparse_pdf', reason='Compare parsed content.'), tmp_path / 'candidate', '')
    assert calls == [1] and pages == [0] and candidate.blocks[0].get_text() == 'Revenue 100, cost 80.'
    content = runner.layout_store.parsed_page(0)
    content.write_text('[]')
    assert runner.layout_store.parsed_page(0) is None


@pytest.mark.asyncio
async def test_native_word_logical_cache_is_not_treated_as_physical_geometry(tmp_path, monkeypatch):
    ir, _, runner, _ = await with_cache(tmp_path)
    ir.page_kind = 'logical'
    assert runner.layout_store.cached(0) is None
    ir.page_kind = 'physical'
    runner.layout_store.config.start_page = 4
    assert runner.layout_store.cached(0) is None


@pytest.mark.asyncio
async def test_query_timeout_preserves_existing_content_and_budget(tmp_path, monkeypatch):
    model = LayoutModel()
    ir, owner, runner = await setup(tmp_path, model=model)
    before = ir.to_dict()

    async def stop(*args):
        raise asyncio.CancelledError()

    owner.limits.max_seconds = 900
    monkeypatch.setattr(runner, 'inspect_layout', stop)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(ir)
    assert ir.to_dict() == before and 'accepted' not in runner.state
    assert runner.state['steps'][0]['status'] == 'interrupted'
    assert owner.state['usage']['model_calls'] == 1


def test_query_reserves_followup_repair_and_details_cost_more_than_plain_bands():
    cold = tool_cost({}, 'inspect_layout')
    warm = tool_cost({}, 'inspect_layout', layout_cached=True)
    plain = tool_cost({}, 'read_regions')
    detailed = tool_cost({}, 'read_regions', detail_regions=3)
    assert cold['seconds'] > warm['seconds'] > detailed['seconds'] > plain['seconds']
    assert warm['model_calls'] == detailed['model_calls'] + 1
    with pytest.raises(ValidationError):
        RecoveryDecision(action='inspect_layout', reason='bad', region_ids=['foreign'])
    with pytest.raises(ValidationError):
        RecoveryDecision(action='read_regions', reason='bad', region_ids=['same', 'same'])


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_normalized_crop_selects_the_marked_area_after_page_rotation(tmp_path, rotation):
    from types import SimpleNamespace

    from PIL import Image
    pdf_path = tmp_path / 'rotation.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.draw_rect(fitz.Rect(120, 240, 240, 400), color=(1, 0, 0), fill=(1, 0, 0))
        page.set_rotation(rotation)
        rect = fitz.Rect(120, 240, 240, 400) * page.rotation_matrix
        box = [round(rect.x0 / page.rect.width * 1000), round(rect.y0 / page.rect.height * 1000),
               round(rect.x1 / page.rect.width * 1000), round(rect.y1 / page.rect.height * 1000)]
        pdf.save(pdf_path)
    image, _ = crop_region(SimpleNamespace(pdf=pdf_path), 0, box, tmp_path, 'target')
    with Image.open(image) as picture:
        assert picture.getpixel((picture.width // 2, picture.height // 2))[:3] == (255, 0, 0)


@pytest.mark.asyncio
async def test_single_page_output_zero_maps_back_to_requested_physical_page(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from test_document_review import fixture

    from app.config import MinerUConfig
    from app.pipeline.layout_evidence import LayoutEvidence

    ir, pdf = fixture(tmp_path, pages=2)
    inventory = SimpleNamespace(pdf=pdf, pages=[0, 1], ir=ir, word_images={}, image_positions={})
    store = LayoutEvidence(inventory, MinerUConfig(), tmp_path, hashlib.sha256(pdf.read_bytes()).hexdigest())

    async def parse(self, source, output, config):
        with fitz.open(source) as single:
            assert len(single) == 1 and 'Source page 2' in single[0].get_text()
        path = put(output, 'content_list', [{'type': 'image', 'page_idx': 0, 'bbox': [100, 200, 400, 500]}])
        return MinerUResult(success=True, output_dir=output, content_list_path=path)

    monkeypatch.setattr(MinerUAdapter, 'parse', parse)
    result = await store.inspect(1)
    assert result['page_idx'] == 1 and result['regions'][0]['bbox'] == [100, 200, 400, 500]
    ids = [result['regions'][0]['region_id']]
    with pytest.raises(ValueError):
        store.selected(0, ids, result)
    assert store.selected(1, ids, result)
    pdf.write_bytes(pdf.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='source PDF changed'):
        store.selected(1, ids, result)
    with pytest.raises(ValueError, match='source PDF changed'):
        await store.parse_page(1)


@pytest.mark.asyncio
async def test_inspection_will_not_spend_budget_needed_for_repair(tmp_path, monkeypatch):
    model = LayoutModel()
    ir, owner, runner, _ = await with_cache(tmp_path, model)
    owner.limits.max_model_calls = 5
    query = AsyncMock(side_effect=AssertionError('no repair budget'))
    monkeypatch.setattr(runner, 'inspect_layout', query)
    await runner.run(ir)
    assert not query.called
    assert 'inspect_layout' not in model.plans[0]['tools']


@pytest.mark.asyncio
async def test_selected_detail_cannot_bypass_rejected_candidate_or_change_source(tmp_path):
    model = LayoutModel()
    model.reject = True
    ir, _, runner, _ = await with_cache(tmp_path, model)
    before = (tmp_path / 'outputs/documents/main.md').read_bytes()
    out = await runner.run(ir, max_new_actions=1)
    assert out is ir and runner.state['steps'][-1]['status'] == 'rejected'
    assert (tmp_path / 'outputs/documents/main.md').read_bytes() == before
    assert 'accepted' not in runner.state


@pytest.mark.asyncio
async def test_word_image_details_use_original_pixels_and_remain_separate_from_mineru(tmp_path):
    from PIL import Image

    from app.pipeline.layout_evidence import LayoutEvidence

    model = LayoutModel()
    ir, owner, runner, _ = await with_cache(tmp_path, model)
    original = runner.inventory.directory / 'original-word.png'
    Image.new('RGB', (640, 480), 'red').save(original)
    runner.inventory.word_images['photo'] = original
    runner.inventory.image_positions['photo'] = [{'page_idx': 0, 'bbox': [100, 100, 300, 300]}]
    result = await runner.layout_store.inspect(0)
    assert result['regions'][0]['provider'] == 'word_native'
    assert sum(r['provider'] == 'mineru' for r in result['regions']) == 2
    runner.state['layout_pages']['0'] = result
    decision = RecoveryDecision(action='read_regions', reason='Read original picture detail.',
                                region_ids=[result['regions'][0]['region_id']])
    await runner.read_regions(ir, {'page_idx': 0}, decision, tmp_path / 'candidate', '')
    assert model.details[0][1][-1].read_bytes() == original.read_bytes()
    original.write_bytes(original.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Unknown'):
        runner.layout_store.selected(0, decision.region_ids, result)
    # Resuming a controller reuses existing evidence, not a new MinerU parse.
    renewed = LayoutEvidence(runner.inventory, owner.config.mineru, tmp_path,
                            runner.source_sha, runner.layout_store.parse_cache)
    assert renewed.cached(0)['cache_hit']


@pytest.mark.asyncio
async def test_slow_query_is_cancelled_with_followup_time_still_reserved(tmp_path, monkeypatch):
    from app.pipeline import review_budget

    model = RecoveryModel(actions=['inspect_layout', 'defer'])
    ir, owner, runner = await setup(tmp_path, model=model)
    owner.limits.max_seconds = .5
    monkeypatch.setattr(review_budget, 'call_seconds', lambda *a: .1)
    monkeypatch.setattr(runner, 'estimate_tool', lambda *a, **kw: {'model_calls': 0, 'seconds': .2, 'followup_seconds': .2})
    cancelled = []

    async def slow(*args):
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(runner, 'inspect_layout', slow)
    assert await runner.run(ir) is ir
    assert cancelled and runner.state['steps'][0]['status'] == 'rejected'
    assert 'TimeoutError' in runner.state['steps'][0]['result']
    assert runner.state['actions'] == 0 and owner._remaining() > .15
