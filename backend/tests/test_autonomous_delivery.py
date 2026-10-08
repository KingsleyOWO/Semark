"""Partial automation must never publish guessed values or incomplete RAG context."""
import json

import pytest
from test_document_review import AuditModel, config, fixture, write_delivery

from app.models.document_ir import Block, BlockType
from app.pipeline.autonomous_delivery import checked_automatic_artifact
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.review_sources import delivery_digest


def read_chunks(root):
    return [json.loads(line) for line in (root / 'outputs/rag_chunks.jsonl').read_text().splitlines()]


@pytest.mark.asyncio
async def test_budget_exhaustion_finishes_without_human_and_exports_verified_subset(tmp_path):
    ir, source = fixture(tmp_path, pages=3)
    cfg = config(max_model_calls=1)
    model = AuditModel()
    ir, _ = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    result = finalize_delivery(ir, tmp_path, cfg)
    assert result['processing_status'] == 'partial'
    assert result['requires_human_review'] is False
    assert not result['auto_rag_ready'] and result['partial_rag_ready']
    assert {bid for c in read_chunks(tmp_path) for bid in c['block_ids']} == {'b0'}
    assert 'Source page 2:' not in (tmp_path / 'outputs/delivery.md').read_text()
    assert '〔內容未確認' in (tmp_path / 'outputs/delivery.md').read_text()
    assert 'Source page 2:' in (tmp_path / 'outputs/documents/main.md').read_text()
    assert checked_automatic_artifact(tmp_path, 'rag_chunks.jsonl').is_file()
    before = (tmp_path / 'outputs/rag_chunks.jsonl').read_bytes()
    again = finalize_delivery(ir, tmp_path, cfg)
    assert again['partial_rag_ready'] and before == (tmp_path / 'outputs/rag_chunks.jsonl').read_bytes()


@pytest.mark.asyncio
async def test_independent_good_block_survives_partial_page_and_mixed_original_chunk(tmp_path):
    good = Block(block_id='good', type=BlockType.TEXT, page_idx=0, payload={'text': 'Revenue 100.'})
    bad = Block(block_id='bad', type=BlockType.TEXT, page_idx=0, payload={'text': 'Cost 999.'})
    ir, source = fixture(tmp_path, blocks=[good, bad])
    # Old chunks mingle good and bad; filtering that chunk alone would lose good content.
    (tmp_path / 'outputs/chunks.jsonl').write_text(json.dumps({
        'chunk_id': 'mixed', 'block_ids': ['good', 'bad'], 'content': 'Revenue 100.\n\nCost 999.'}))
    def review(_, v):
        v.update(source_complete=False, issues=['Wrong cost.'])
        by_id = {b['block_id']: b for b in v['blocks']}
        by_id['good']['context_complete'] = True
        by_id['bad']['status'] = 'needs_review'
        return v
    model = AuditModel(review)
    ir, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    assert report['summary']['verified_pages'] == 0
    result = finalize_delivery(ir, tmp_path, config())
    assert result['partial_rag_ready']
    chunks = read_chunks(tmp_path)
    assert [c['block_ids'] for c in chunks] == [['good']]
    assert '999' not in json.dumps(chunks)
    assert '999' not in (tmp_path / 'outputs/delivery.md').read_text()


@pytest.mark.asyncio
async def test_missing_table_header_and_transitive_dependencies_exclude_whole_table(tmp_path):
    blocks = [Block(block_id='header', type=BlockType.TEXT, page_idx=0, payload={'text': 'Currency: USD'}),
              Block(block_id='table', type=BlockType.TABLE, page_idx=0,
                    payload={'table_body': '<table><tr><td>Revenue</td><td>100</td></tr></table>'}),
              Block(block_id='conclusion', type=BlockType.TEXT, page_idx=0, payload={'text': 'Revenue increased.'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    def review(_, v):
        v.update(source_complete=False, issues=['Currency is unclear.'])
        for b in v['blocks']:
            b['context_complete'] = True
        by_id = {b['block_id']: b for b in v['blocks']}
        by_id['header']['status'] = 'needs_review'
        by_id['table']['depends_on_block_ids'] = ['header']
        by_id['conclusion']['depends_on_block_ids'] = ['table']
        return v
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    result = finalize_delivery(ir, tmp_path, config())
    assert not result['partial_rag_ready'] and read_chunks(tmp_path) == []
    report = json.loads((tmp_path / 'outputs/processing_report.json').read_text())
    assert [b['reason'] for b in report['blocks']][1:] == ['required_context_unverified'] * 2


@pytest.mark.asyncio
async def test_unverified_heading_excludes_dependent_paragraph(tmp_path):
    blocks = [Block(block_id='title', type=BlockType.TEXT, page_idx=0, payload={'text': '2027', 'text_level': 1}),
              Block(block_id='body', type=BlockType.TEXT, page_idx=0, payload={'text': 'Revenue 100.'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    def review(_, v):
        v.update(source_complete=False, issues=['Wrong year in heading.'])
        by_id = {b['block_id']: b for b in v['blocks']}
        by_id['title']['status'] = 'needs_review'
        by_id['body']['context_complete'] = True
        by_id['body']['depends_on_block_ids'] = ['title']
        return v
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    assert read_chunks(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['evidence', 'delivery', 'export', 'report'])
async def test_stale_or_tampered_evidence_cannot_be_used(tmp_path, target):
    ir, source = fixture(tmp_path)
    model = AuditModel()
    ir, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    if target == 'evidence':
        (tmp_path / report['pages'][0]['images'][0]['path']).write_bytes(b'changed')
        result = finalize_delivery(ir, tmp_path, config())
        assert not result['partial_rag_ready'] and read_chunks(tmp_path) == []
    else:
        name = {'delivery': 'documents/main.md', 'export': 'rag_chunks.jsonl',
                'report': 'processing_report.json'}[target]
        path = tmp_path / 'outputs' / name
        path.write_text('{}' if target == 'report' else 'tampered')
        with pytest.raises(ValueError):
            checked_automatic_artifact(tmp_path, 'rag_chunks.jsonl')


@pytest.mark.asyncio
async def test_omitted_page_is_retained_as_gap_even_when_other_pages_pass(tmp_path):
    ir, source = fixture(tmp_path, pages=2)
    ir.blocks.pop()
    write_delivery(tmp_path, ir)
    def review(payload, v):
        if payload['page_idx'] == 1:
            v.update(source_complete=False, issues=['Missing entire page.'])
        return v
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    report = json.loads((tmp_path / 'outputs/processing_report.json').read_text())
    assert any(g['source_pages'] == [1] and g['reason'] == 'source_page_missing' for g in report['gaps'])
    assert len(read_chunks(tmp_path)) == 1


@pytest.mark.asyncio
async def test_explicit_conversion_veto_is_not_bypassed_by_partial_export(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel()
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    (tmp_path / 'outputs/quality_gate.json').write_text(json.dumps({'status': 'needs_review', 'issues': [],
                                                                'stats': {'auto_rag_ready': False}}))
    for _ in range(2):
        result = finalize_delivery(ir, tmp_path, config())
        assert not result['partial_rag_ready'] and not read_chunks(tmp_path)


def test_legacy_export_does_not_fall_back_to_unverified_rag(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'outputs/rag.md').write_text('Legacy guessed value 999.')
    with pytest.raises(ValueError):
        checked_automatic_artifact(tmp_path, 'rag_verified.md')


@pytest.mark.asyncio
async def test_review_disabled_finishes_unverified_without_automatic_ingestion(tmp_path):
    ir, source = fixture(tmp_path)
    cfg = config(mode='off')
    model = AuditModel()
    ir, _ = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    old_digest = delivery_digest(tmp_path)
    result = finalize_delivery(ir, tmp_path, cfg)
    assert result['processing_status'] == 'unverified' and not result['requires_human_review']
    assert not model.calls and read_chunks(tmp_path) == []
    assert delivery_digest(tmp_path) == old_digest


@pytest.mark.asyncio
async def test_oversized_table_is_not_sliced_into_headerless_rag_fragments(tmp_path):
    block = Block(block_id='table', type=BlockType.TABLE, page_idx=0, payload={'table_body':
        '<table><tr><th>Year</th><th>Revenue USD</th></tr>' +
        ''.join(f'<tr><td>{2000+i}</td><td>{100+i}</td></tr>' for i in range(90)) + '</table>'})
    ir, source = fixture(tmp_path, blocks=[block])
    cfg = config()
    cfg.package.chunk_max_tokens = 64
    model = AuditModel()
    ir, _ = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, cfg)
    from app.pipeline.stages.chunk import estimate_tokens
    chunks = read_chunks(tmp_path)
    assert len(chunks) == 90
    for i, chunk in enumerate(chunks):
        assert 'Year' in chunk['content'] and 'Revenue USD' in chunk['content']
        assert str(2000 + i) in chunk['content'] and str(100 + i) in chunk['content']
        assert chunk['metadata']['table_source_rows']['table'] == [1, i + 2]
        assert estimate_tokens(chunk['content']) <= 64
        assert chunk['block_ids'] == ['table']
    report = json.loads((tmp_path / 'outputs/processing_report.json').read_text())
    assert report['blocks'][0]['rag_eligible']
    assert 'Revenue USD' in (tmp_path / 'outputs/delivery.md').read_text()


@pytest.mark.asyncio
async def test_rag_chunk_always_includes_its_verified_dependency(tmp_path):
    blocks = [Block(block_id='header', type=BlockType.TEXT, page_idx=0, reading_order=0,
                    payload={'text': 'Currency is USD. Figures refer to year 2026.'}),
              Block(block_id='table', type=BlockType.TABLE, page_idx=0, reading_order=1,
                    payload={'table_body': '<table><tr><td>Revenue</td><td>100</td></tr></table>'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    def review(_, v):
        by_id = {b['block_id']: b for b in v['blocks']}
        by_id['table']['depends_on_block_ids'] = ['header']
        return v
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    for chunk in read_chunks(tmp_path):
        if 'table' in chunk['block_ids']:
            assert 'header' in chunk['block_ids'] and '2026' in chunk['content'] and 'USD' in chunk['content']


@pytest.mark.asyncio
async def test_downloads_and_api_only_serve_verified_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from app.api.routes import runs
    from app.api.routes.download import FileType, OutputFormat, _get_file_content
    ir, source = fixture(tmp_path)
    model = AuditModel()
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    monkeypatch.setattr(runs, 'RunRepository', lambda _: SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(doc_id='doc'))))
    monkeypatch.setattr(runs, 'settings', SimpleNamespace(get_published_run_path=lambda *_: tmp_path))
    response = await runs.get_verified_chunks('run', None)
    assert response.path.name == 'rag_chunks.jsonl'
    response = await runs.get_output('run', view='rag', db=None)
    assert response.path.name == 'rag_verified.md'
    data, name = _get_file_content(tmp_path / 'outputs', FileType.RAG_CHUNKS, OutputFormat.DOCX, 'run', 'sample')
    assert name.endswith('.jsonl') and json.loads(data.splitlines()[0])['metadata']['source_verified']
    data, name = _get_file_content(tmp_path / 'outputs', FileType.REVIEW_COVERAGE, OutputFormat.MD, 'run', 'sample')
    assert name.endswith('.json') and json.loads(data)['inventory_complete']
    (tmp_path / 'outputs/rag_chunks.jsonl').write_text('unverified')
    with pytest.raises(HTTPException) as exc:
        await runs.get_verified_chunks('run', None)
    assert exc.value.status_code == 409
    assert _get_file_content(tmp_path / 'outputs', FileType.RAG_CHUNKS, OutputFormat.MD, 'run', 'sample') is None


@pytest.mark.asyncio
async def test_document_only_delivery_can_complete_without_enabling_rag(tmp_path):
    ir, source = fixture(tmp_path)
    cfg = config()
    cfg.package.generate_chunks = False
    model = AuditModel()
    ir, _ = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    result = finalize_delivery(ir, tmp_path, cfg)
    assert result['processing_status'] == 'complete'
    assert not result['auto_rag_ready'] and not result['partial_rag_ready']


@pytest.mark.asyncio
async def test_evidence_download_is_restricted_to_recorded_source_images(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from app.api.routes import runs
    ir, source = fixture(tmp_path)
    model = AuditModel(lambda _, v: {**v, 'source_complete': False, 'issues': ['Unreadable.']})
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    monkeypatch.setattr(runs, 'RunRepository', lambda _: SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(doc_id='doc'))))
    monkeypatch.setattr(runs, 'settings', SimpleNamespace(get_published_run_path=lambda *_: tmp_path))
    report = json.loads((tmp_path / 'outputs/processing_report.json').read_text())
    path = report['gaps'][0]['evidence'][0]['path']
    response = await runs.get_processing_evidence('run', path=path, db=None)
    assert response.path == tmp_path / path
    with pytest.raises(HTTPException) as exc:
        await runs.get_processing_evidence('run', path='../../etc/passwd', db=None)
    assert exc.value.status_code == 404
    (tmp_path / path).write_bytes(b'tampered')
    with pytest.raises(HTTPException) as exc:
        await runs.get_processing_evidence('run', path=path, db=None)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_program_recorded_image_dependencies_cannot_be_overridden_by_model(tmp_path):
    blocks = [Block(block_id='label', type=BlockType.TEXT, page_idx=0, payload={'text': 'Uncertain logo name'}),
              Block(block_id='logo', type=BlockType.TEXT, page_idx=0,
                    payload={'text': 'Yellow logo design', 'agentic_depends_on_block_ids': ['label']}),
              Block(block_id='photo', type=BlockType.TEXT, page_idx=0, payload={'text': 'Shelves in a shop'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    def review(_, value):
        value.update(source_complete=False, issues=['Logo name is uncertain.'])
        for block in value['blocks']:
            block['context_complete'] = True
            if block['block_id'] == 'label':
                block['status'] = 'needs_review'
        return value
    model = AuditModel(review)
    ir, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    logo = next(b for b in report['blocks'] if b['block_id'] == 'logo')
    assert logo['depends_on_block_ids'] == ['label']
    finalize_delivery(ir, tmp_path, config())
    assert {bid for chunk in read_chunks(tmp_path) for bid in chunk['block_ids']} == {'photo'}


@pytest.mark.asyncio
async def test_independent_table_survives_neighbor_with_oversized_dependencies(tmp_path):
    blocks = [Block(block_id='long', type=BlockType.TEXT, page_idx=0,
                    payload={'text': 'Verified but lengthy context. ' * 100}),
              Block(block_id='dependent', type=BlockType.TEXT, page_idx=0, reading_order=1,
                    payload={'text': 'A short conclusion.'}),
              Block(block_id='independent', type=BlockType.TABLE, page_idx=0, reading_order=2,
                    payload={'table_body': '<table><tr><th>Item</th><th>Value</th></tr>'
                                          '<tr><td>Revenue</td><td>100</td></tr></table>'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    cfg = config()
    cfg.package.chunk_max_tokens = 100
    def review(_, value):
        for block in value['blocks']:
            block['context_complete'] = True
            if block['block_id'] == 'dependent':
                block['depends_on_block_ids'] = ['long']
        return value
    model = AuditModel(review)
    ir, _ = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, cfg)
    chunks = read_chunks(tmp_path)
    assert any('independent' in c['block_ids'] and '100' in c['content'] for c in chunks)
    assert all('dependent' not in c['block_ids'] for c in chunks)


def _picture_fixture(tmp_path, picture_status):
    blocks = [Block(block_id='b0', type=BlockType.TEXT, page_idx=0, reading_order=0,
                    payload={'text': 'Source page 1: Revenue 100, cost 80.'}),
              Block(block_id='pic', type=BlockType.IMAGE, page_idx=0, reading_order=1,
                    payload={'img_path': 'pic.png', 'caption': ''})]
    ir, source = fixture(tmp_path, blocks=blocks)
    outputs = tmp_path / 'outputs'
    (outputs / 'assets_index.jsonl').write_text(json.dumps(
        {'asset_id': 'fig0000', 'block_id': 'pic', 'retrieval_text': 'Shelves of boxed puzzles in a shop'}))
    (outputs / 'documents/main.md').write_text('Source page 1: Revenue 100, cost 80.\n\nShelves of boxed puzzles in a shop')
    (outputs / 'chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in [
        {'chunk_id': 'c0', 'block_ids': ['b0'], 'page_indices': [0], 'content': 'Source page 1: Revenue 100, cost 80.'},
        {'chunk_id': 'c1', 'block_ids': ['pic'], 'page_indices': [0], 'content': 'Shelves of boxed puzzles in a shop'}]))

    def review(_, value):
        for block in value['blocks']:
            block['context_complete'] = True
            if block['block_id'] == 'pic' and picture_status == 'needs_review':
                block['status'] = 'needs_review'
                value.update(source_complete=False, issues=['Shop sign cannot be confirmed.'])
        return value
    return ir, source, AuditModel(review)


@pytest.mark.asyncio
async def test_unverified_picture_content_reaches_rag_with_its_review_as_metadata(tmp_path):
    # 2026-10-01 policy: picture content is delivered either way; review decides the label.
    # 2026-10-07: the label is metadata for RAG text and stays only in the reading copy.
    ir, source, model = _picture_fixture(tmp_path, 'needs_review')
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    result = finalize_delivery(ir, tmp_path, config())

    chunks = read_chunks(tmp_path)
    picture = next(c for c in chunks if 'pic' in c['block_ids'])
    assert 'Shelves of boxed puzzles' in picture['content'] and 'Image content' not in picture['content']
    assert 'Image content' not in (tmp_path / 'outputs/rag_verified.md').read_text()
    assert picture['metadata']['source_verified'] is False
    assert picture['metadata']['unverified_block_ids'] == ['pic']
    # The verified paragraph shares the chunk and stays unlabelled.
    assert picture['content'].startswith('Source page 1: Revenue 100, cost 80.')
    delivery = (tmp_path / 'outputs/delivery.md').read_text()
    assert '[Image content: automatic reading, not verified]\nShelves of boxed puzzles' in delivery
    report = json.loads((tmp_path / 'outputs/processing_report.json').read_text())
    entry = next(b for b in report['blocks'] if b['block_id'] == 'pic')
    assert entry['verified'] is False and entry['rag_eligible'] is True and entry['image_label'] == 'unverified'
    assert result['processing_status'] == 'partial'


@pytest.mark.asyncio
async def test_verified_picture_content_is_verified_in_metadata_and_labelled_for_readers(tmp_path):
    ir, source, model = _picture_fixture(tmp_path, 'verified')
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())

    picture = next(c for c in read_chunks(tmp_path) if 'pic' in c['block_ids'])
    assert 'Image content' not in picture['content']
    assert picture['metadata']['source_verified'] is True
    delivery = (tmp_path / 'outputs/delivery.md').read_text()
    assert '[Image content: automatic reading, verified]\nShelves of boxed puzzles' in delivery


@pytest.mark.asyncio
async def test_oversized_picture_is_split_on_line_boundaries_instead_of_dropped(tmp_path):
    rows = [f'Brand {i:02d} | yellow and black | handwritten type | local culture' for i in range(40)]
    ir, source, model = _picture_fixture(tmp_path, 'needs_review')
    text = 'Brand comparison table\n' + '\n'.join(rows)
    outputs = tmp_path / 'outputs'
    (outputs / 'assets_index.jsonl').write_text(json.dumps(
        {'asset_id': 'fig0000', 'block_id': 'pic', 'title': 'Brand comparison table', 'retrieval_text': text}))
    (outputs / 'documents/main.md').write_text('Source page 1: Revenue 100, cost 80.\n\n' + text)
    (outputs / 'chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in [
        {'chunk_id': 'c0', 'block_ids': ['b0'], 'page_indices': [0], 'content': 'Source page 1: Revenue 100, cost 80.'},
        {'chunk_id': 'c1', 'block_ids': ['pic'], 'page_indices': [0], 'content': text}]))
    cfg = config()
    cfg.package.chunk_max_tokens = 120
    ir, _ = await DocumentReview(cfg, model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, cfg)

    parts = [c for c in read_chunks(tmp_path) if 'pic' in c['block_ids']]
    assert len(parts) > 1
    assert all('Image content' not in p['content'] and p['metadata']['unverified_block_ids'] == ['pic'] for p in parts)
    assert all('Brand comparison table' in p['content'] for p in parts)
    delivered = '\n'.join(p['content'] for p in parts)
    assert all(row in delivered for row in rows)  # every row survives, none cut in half
    assert all(len(p['content']) < len(text) for p in parts)


@pytest.mark.asyncio
async def test_unverified_picture_under_an_unverified_heading_still_ships(tmp_path):
    # A 31-page PDF held back 29 of 46 pictures: each sat under a heading its page
    # review could not verify. A heading is navigation; the picture already ships
    # labelled unverified, and the unverified heading text itself still stays out.
    blocks = [Block(block_id='heading', type=BlockType.TEXT, page_idx=0, reading_order=0,
                    payload={'text': 'Source page 1', 'text_level': 1}),
              Block(block_id='pic', type=BlockType.IMAGE, page_idx=0, reading_order=1,
                    payload={'img_path': 'pic.png', 'caption': ''})]
    ir, source = fixture(tmp_path, blocks=blocks)
    outputs = tmp_path / 'outputs'
    (outputs / 'assets_index.jsonl').write_text(json.dumps(
        {'asset_id': 'fig0000', 'block_id': 'pic', 'retrieval_text': 'Shelves of boxed puzzles in a shop'}))
    (outputs / 'documents/main.md').write_text('# Source page 1\n\nShelves of boxed puzzles in a shop')
    (outputs / 'chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in [
        {'chunk_id': 'c0', 'block_ids': ['heading'], 'page_indices': [0], 'content': '# Source page 1'},
        {'chunk_id': 'c1', 'block_ids': ['pic'], 'page_indices': [0], 'content': 'Shelves of boxed puzzles in a shop'}]))

    def review(_, value):
        for block in value['blocks']:
            block['status'] = 'needs_review'
        value.update(source_complete=False, issues=['Heading and picture unconfirmed.'])
        return value
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())

    chunks = read_chunks(tmp_path)
    picture = next(c for c in chunks if 'pic' in c['block_ids'])
    assert picture['metadata']['unverified_block_ids'] == ['pic']
    assert all('heading' not in c['block_ids'] for c in chunks)


@pytest.mark.asyncio
async def test_picture_citing_its_section_title_is_not_held_back_by_the_cover_title(tmp_path):
    # Spot check 2026-10-07: five logos under a verified 「商標演進史」-style section title
    # recorded that title as context; the review left the cover title above it
    # unconfirmed, and the whole heading chain then held the logos back.
    blocks = [Block(block_id='cover', type=BlockType.TEXT, page_idx=0, reading_order=0,
                    payload={'text': 'Example Annual Report', 'text_level': 1}),
              Block(block_id='section', type=BlockType.TEXT, page_idx=0, reading_order=1,
                    payload={'text': 'Logo history', 'text_level': 2}),
              Block(block_id='pic', type=BlockType.IMAGE, page_idx=0, reading_order=2,
                    payload={'img_path': 'pic.png', 'caption': ''})]
    ir, source = fixture(tmp_path, blocks=blocks)
    outputs = tmp_path / 'outputs'
    (outputs / 'assets_index.jsonl').write_text(json.dumps(
        {'asset_id': 'fig0000', 'block_id': 'pic', 'retrieval_text': 'A green sailing ship logo, 1982'}))
    (outputs / 'documents/main.md').write_text(
        '# Example Annual Report\n\n## Logo history\n\nA green sailing ship logo, 1982')
    (outputs / 'chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in [
        {'chunk_id': 'c0', 'block_ids': ['cover'], 'page_indices': [0], 'content': '# Example Annual Report'},
        {'chunk_id': 'c1', 'block_ids': ['section'], 'page_indices': [0], 'content': '## Logo history'},
        {'chunk_id': 'c2', 'block_ids': ['pic'], 'page_indices': [0], 'content': 'A green sailing ship logo, 1982'}]))

    def review(_, value):
        by_id = {b['block_id']: b for b in value['blocks']}
        by_id['cover'].update(context_complete=False)
        by_id['section'].update(context_complete=True)
        by_id['pic'].update(context_complete=False, depends_on_block_ids=['section'])
        value.update(source_complete=False, issues=['Cover title context unconfirmed.'])
        return value
    model = AuditModel(review)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())

    chunks = read_chunks(tmp_path)
    picture = next(c for c in chunks if 'pic' in c['block_ids'])
    assert 'section' in picture['block_ids'] and 'Logo history' in picture['content']
    assert picture['metadata']['unverified_block_ids'] == ['pic']
    assert all('cover' not in c['block_ids'] for c in chunks)
    assert '[Image content: automatic reading, not verified]\nA green sailing ship logo, 1982' in (
        outputs / 'delivery.md').read_text()


@pytest.mark.asyncio
async def test_documents_index_records_which_delivered_lines_are_image_readings_and_their_review(tmp_path):
    # The documents a RAG imports keep clean text; the index tells the importer what is an
    # automatic image reading and whether review confirmed it (decided 2026-10-07).
    ir, source, model = _picture_fixture(tmp_path, 'needs_review')
    outputs = tmp_path / 'outputs'
    (outputs / 'documents/fig0000.md').write_text('# Shelves of boxed puzzles in a shop\n\nShelves of boxed puzzles in a shop')
    (outputs / 'documents_index.json').write_text(json.dumps([
        {'document_id': 'main', 'kind': 'main', 'file': 'documents/main.md'},
        {'document_id': 'fig0000', 'kind': 'figure_asset', 'file': 'documents/fig0000.md'}]))
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())

    index = {d['document_id']: d for d in json.loads((outputs / 'documents_index.json').read_text())}
    assert index['fig0000']['content_origin'] == 'image_reading'
    assert index['fig0000']['review_status'] == 'unverified'
    assert index['main']['image_reading_spans'] == [
        {'block_id': 'pic', 'review_status': 'unverified', 'start_line': 3, 'end_line': 3}]
    assert 'review_status' not in index['main']
    for name in ('main.md', 'fig0000.md'):
        assert 'Image content' not in (outputs / 'documents' / name).read_text()


def _split_picture_fixture(tmp_path, picture_status, all_text):
    """A picture delivered as its own document, as packaging writes it."""
    ir, source, model = _picture_fixture(tmp_path, picture_status)
    outputs = tmp_path / 'outputs'
    description = 'A paper sachet of herbal ointment on a white background'
    retrieval = '\n'.join([description, 'The sachet is printed in green.', *all_text, 'ointment'])
    (outputs / 'assets_index.jsonl').write_text(json.dumps({
        'type': 'figure_asset', 'asset_id': 'fig0000', 'block_id': 'pic', 'title': description,
        'retrieval_text': retrieval, 'facts': ['The sachet is printed in green.'],
        'keywords': ['ointment'], 'all_text': all_text}))
    (outputs / 'documents/main.md').write_text('Source page 1: Revenue 100, cost 80.')
    (outputs / 'documents/fig0000.md').write_text(f'# {description}\n\n{retrieval}\n')
    (outputs / 'documents_index.json').write_text(json.dumps([
        {'document_id': 'main', 'kind': 'main', 'title': 'Report', 'file': str(outputs / 'documents/main.md')},
        {'document_id': 'fig0000', 'kind': 'figure_asset', 'title': description,
         'file': str(outputs / 'documents/fig0000.md')}]))
    (outputs / 'chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in [
        {'chunk_id': 'c0', 'block_ids': ['b0'], 'page_indices': [0], 'content': 'Source page 1: Revenue 100, cost 80.'},
        {'chunk_id': 'c1', 'block_ids': ['pic'], 'page_indices': [0], 'content': retrieval}]))
    return ir, source, model


@pytest.mark.asyncio
async def test_unverified_picture_keeps_only_the_text_read_off_it(tmp_path):
    # 2026-10-08, the user's choice: of ten unverified pictures sampled, one sachet was
    # described as a sake bottle. Without review, the description, facts and keywords go.
    ir, source, model = _split_picture_fixture(tmp_path, 'needs_review', ['Herbal Balm', '60g'])
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    outputs = tmp_path / 'outputs'

    assert (outputs / 'documents/fig0000.md').read_text() == '# Picture, page 1\n\nHerbal Balm\n60g\n'
    index = json.loads((outputs / 'documents_index.json').read_text())
    figure = next(e for e in index if e['document_id'] == 'fig0000')
    assert figure['title'] == 'Picture, page 1' and figure['review_status'] == 'unverified'
    picture = next(c for c in read_chunks(tmp_path) if 'pic' in c['block_ids'])
    assert 'Herbal Balm\n60g' in picture['content'] and 'sachet' not in picture['content']
    assert 'ointment' not in picture['content']
    delivery = (outputs / 'delivery.md').read_text()
    assert 'sachet' not in delivery and 'Herbal Balm' in delivery
    report = json.loads((outputs / 'processing_report.json').read_text())
    assert next(b for b in report['blocks'] if b['block_id'] == 'pic')['delivered_content'] == 'text_read_from_image'
    assert checked_automatic_artifact(tmp_path, 'rag_chunks.jsonl')


@pytest.mark.asyncio
async def test_unverified_picture_with_nothing_read_leaves_no_document(tmp_path):
    ir, source, model = _split_picture_fixture(tmp_path, 'needs_review', [])
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    outputs = tmp_path / 'outputs'

    assert not (outputs / 'documents/fig0000.md').exists()
    assert [e['document_id'] for e in json.loads((outputs / 'documents_index.json').read_text())] == ['main']
    assert all('pic' not in c['block_ids'] for c in read_chunks(tmp_path))
    assert 'sachet' not in (outputs / 'delivery.md').read_text()


@pytest.mark.asyncio
async def test_verified_picture_keeps_its_description(tmp_path):
    ir, source, model = _split_picture_fixture(tmp_path, 'verified', ['Herbal Balm'])
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())

    assert 'A paper sachet of herbal ointment' in (tmp_path / 'outputs/documents/fig0000.md').read_text()
    picture = next(c for c in read_chunks(tmp_path) if 'pic' in c['block_ids'])
    assert 'A paper sachet of herbal ointment' in picture['content']
