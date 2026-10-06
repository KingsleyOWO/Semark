"""Recovery publication is conditional on complete source-scoped final review."""
import copy
import shutil
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from test_document_recovery import RecoveryModel, cfg, setup

from app.models.document_ir import BlockType
from app.pipeline.document_recovery import RecoveryCommitError, restore_recovery


@pytest.mark.asyncio
async def test_failed_final_candidate_audit_preserves_publication_and_spent_usage(tmp_path, monkeypatch):
    ir, owner, runner = await setup(tmp_path, RecoveryModel(reject=True), cfg(max_recovery_actions=1))
    original_records = copy.deepcopy(owner.state['pages'])
    original_output = (tmp_path / 'outputs/documents/main.md').read_bytes()
    async def locally_passes(candidate, pages, folder, **kwargs):
        await runner.renderer(owner.config, candidate, folder, tmp_path, None)
        return True
    monkeypatch.setattr(runner, 'validate', locally_passes)
    output = await runner.run(ir)
    assert output is ir and 'accepted' not in runner.state
    assert (tmp_path / 'outputs/documents/main.md').read_bytes() == original_output
    assert owner.state['pages'] == original_records
    assert owner.state['usage']['model_calls'] >= 4
    assert owner.state['calls'][-1]['task'] == 'audit_page'


@pytest.mark.asyncio
async def test_audited_candidate_commits_at_budget_limit_without_postcommit_model_call(tmp_path):
    class AtLimit(RecoveryModel):
        async def query(self, task, payload, image, schema):
            answer = await super().query(task, payload, image, schema)
            if task == 'audit_page':
                owner.limits.max_model_calls = owner.state['usage']['model_calls']
            return answer
    ir, owner, runner = await setup(tmp_path, AtLimit())
    output = await runner.run(ir)
    assert output is not ir
    assert owner.state['usage']['model_calls'] == owner.limits.max_model_calls
    assert owner.model.tasks.count('audit_page') == 1
    assert owner.state['pages']['0']['status'] == 'verified'
    assert runner.state['steps'][0]['status'] == 'applied'


@pytest.mark.asyncio
async def test_partial_rebuild_failure_restores_previous_bytes_reviews_and_budget(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    original_records = copy.deepcopy(owner.state['pages'])
    original_output = (tmp_path / 'outputs/documents/main.md').read_bytes()
    (tmp_path / 'source_map.json').write_text('{"old":"source map"}')
    async def fail(candidate):
        (tmp_path / 'source_map.json').write_text('{"candidate":"source map"}')
        (tmp_path / 'manifest.json').write_text('{"candidate":"manifest"}')
        (tmp_path / 'outputs/documents/main.md').write_text('partly overwritten output')
        raise RuntimeError('disk failure after first write')
    runner.rebuild = fail
    with pytest.raises(RecoveryCommitError, match='previous publication restored'):
        await runner.run(ir)
    assert (tmp_path / 'outputs/documents/main.md').read_bytes() == original_output
    assert owner.state['pages'] == original_records
    assert (tmp_path / 'source_map.json').read_text() == '{"old":"source map"}'
    assert not (tmp_path / 'manifest.json').exists()
    assert owner.state['usage']['model_calls'] > 0
    assert 'accepted' not in runner.state and 'pending_commit' not in runner.state


@pytest.mark.asyncio
async def test_crashed_uncommitted_publication_rolls_back_before_recovery_resume(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    original_records = copy.deepcopy(owner.state['pages'])
    original_output = (tmp_path / 'outputs/documents/main.md').read_bytes()
    backup = runner.folder / 'interrupted/publication-before'
    shutil.copytree(tmp_path / 'outputs', backup / 'outputs')
    runner.state['pending_commit'] = {'backup': str(backup.relative_to(tmp_path)), 'pages': original_records}
    (tmp_path / 'outputs/documents/main.md').write_text('uncommitted candidate')
    owner.state['pages'] = {}
    owner.state['usage']['model_calls'] = 12
    restored, changed = restore_recovery(owner, ir, tmp_path)
    assert restored is ir and not changed
    assert (tmp_path / 'outputs/documents/main.md').read_bytes() == original_output
    assert owner.state['pages'] == original_records and owner.state['usage']['model_calls'] == 12
    assert 'pending_commit' not in runner.state


@pytest.mark.asyncio
async def test_rebuild_different_from_isolated_audit_is_rolled_back_and_review_goes_on(tmp_path):
    # Spot check v20.11 aborted the whole review here; only the candidate is rejected now.
    ir, owner, runner = await setup(tmp_path, config=cfg(max_recovery_actions=1))
    original_output = (tmp_path / 'outputs/documents/main.md').read_bytes()
    async def inconsistent(candidate):
        (tmp_path / 'outputs/documents/main.md').write_text('Different content never audited')
    runner.rebuild = inconsistent
    result = await runner.run(ir)
    assert result is ir
    assert (tmp_path / 'outputs/documents/main.md').read_bytes() == original_output
    assert 'accepted' not in runner.state and 'pending_commit' not in runner.state
    step = runner.state['steps'][0]
    assert step['status'] == 'rejected' and 'rebuilt_delivery_differs_from_verified_candidate' in step['result']


@pytest.mark.asyncio
async def test_late_ocr_cannot_run_without_affordable_followup_repair(tmp_path, monkeypatch):
    config = cfg()
    config.agentic.max_model_calls = 4
    config.agentic.max_seconds = 2000
    ir, owner, runner = await setup(tmp_path, config=config)
    picture = runner.inventory.directory / 'small-original.png'
    Image.new('RGB', (40, 20), 'white').save(picture)
    ir.blocks[0].type = BlockType.IMAGE
    ir.blocks[0].payload = {'word_source': {'block_index': 1}}
    runner.inventory.word_images['b0'] = picture
    owner.state['pages']['0'].update(image_only_issues=True, image_issues=[{
        'block_id': 'b0', 'failure_kind': 'reading_disagreement', 'detail': 'Tiny sign text is disputed.'}])
    monkeypatch.setattr('app.adapters.ocr.ocr_python', lambda: '/available-ocr')
    observed = AsyncMock()
    monkeypatch.setattr(runner, 'ocr_image', observed)
    await runner.run(ir)
    observed.assert_not_called()
    assert owner.state['usage']['model_calls'] == 0
    issue = runner.state['issues']['image:b0']
    assert issue['status'] == 'budget_exhausted'
    assert issue['estimates']['ocr_image']['includes_followup'].startswith('repair_image')


@pytest.mark.asyncio
async def test_unplanned_neighbor_invalidation_requires_its_own_budget(tmp_path):
    from test_document_review import fixture, write_delivery

    from app.pipeline.agentic import BudgetExhausted
    from app.pipeline.review_sources import SourceInventory

    _, owner, runner = await setup(tmp_path)
    # Two genuinely different physical pages; page 0's boundary is page 1 context.
    (tmp_path / 'two-pages').mkdir()
    ir, source = fixture(tmp_path / 'two-pages', pages=2)
    runner.inventory = SourceInventory(ir, tmp_path, source)
    await runner.inventory.prepare()
    write_delivery(tmp_path, ir)
    owner.state['pages'] = {str(p): {'page_idx': p, 'status': 'needs_review'} for p in range(2)}
    owner.limits.max_model_calls = 1
    candidate = ir.model_copy(deep=True)
    candidate.blocks[0].payload['text'] += ' A corrected source-bound detail.'
    folder = runner.folder / 'two-pages-candidate'
    await runner.renderer(owner.config, candidate, folder, tmp_path, None)
    original_records = copy.deepcopy(owner.state['pages'])
    with pytest.raises(BudgetExhausted, match='all changed delivery scopes'):
        await runner.prepare_commit(ir, candidate, [0], folder)
    assert owner.state['pages'] == original_records
    assert owner.state['usage']['model_calls'] == 0 and 'accepted' not in runner.state


@pytest.mark.asyncio
async def test_process_crash_during_candidate_review_does_not_replace_live_certificates(tmp_path):
    ir, owner, runner = await setup(tmp_path)
    original_records = copy.deepcopy(owner.state['pages'])
    runner.state['candidate_review_original_pages'] = original_records
    # A model checkpoint can persist candidate records while their isolated audit runs.
    owner.state['pages'] = {'0': {'status': 'verified', 'scoped_fingerprint': 'candidate-only'}}
    owner.state['usage']['model_calls'] = 9
    owner._save()
    restored, changed = restore_recovery(owner, ir, tmp_path)
    assert restored is ir and not changed
    assert owner.state['pages'] == original_records
    assert owner.state['usage']['model_calls'] == 9
    assert 'candidate_review_original_pages' not in runner.state


@pytest.mark.asyncio
async def test_stale_certificate_is_reaudited_even_when_fresh_before_after_scope_matches(tmp_path):
    from test_document_review import fixture, write_delivery

    from app.pipeline.agentic import BudgetExhausted
    from app.pipeline.review_sources import SourceInventory

    _, owner, runner = await setup(tmp_path)
    (tmp_path / 'three-pages').mkdir()
    ir, source = fixture(tmp_path / 'three-pages', pages=3)
    runner.inventory = SourceInventory(ir, tmp_path, source)
    await runner.inventory.prepare()
    write_delivery(tmp_path, ir)
    owner.state['pages'] = {}
    for page in range(3):
        await owner._inspect(page, ir, runner.inventory, tmp_path)
    owner.state['pages']['2']['scoped_fingerprint'] = 'certificate-from-an-older-source-or-delivery'
    original_records = copy.deepcopy(owner.state['pages'])
    candidate = ir.model_copy(deep=True)
    candidate.blocks[0].payload['text'] += ' Corrected detail only affects pages 0 and 1.'
    folder = runner.folder / 'stale-certificate-candidate'
    await runner.renderer(owner.config, candidate, folder, tmp_path, None)
    ci = await runner.candidate_inventory(candidate, folder)
    old = owner.review_scope_fingerprints(ir, runner.inventory, tmp_path)
    new = owner.review_scope_fingerprints(candidate, ci, folder)
    assert old[2] == new[2]  # The stale review is invisible to fresh before/after comparison.
    owner.limits.max_model_calls = owner.state['usage']['model_calls'] + 2
    with pytest.raises(BudgetExhausted, match='all changed delivery scopes'):
        await runner.prepare_commit(ir, candidate, [0], folder)
    assert owner.state['pages'] == original_records
    assert owner.state['pages']['2']['scoped_fingerprint'].startswith('certificate-from-an-older')
    assert 'accepted' not in runner.state
