"""Recovery may change one image without changing unrelated audit evidence."""
import copy
import json
import shutil

import pytest
from PIL import Image
from test_document_review import AuditModel, fixture, write_delivery

from app.config import AgenticConfig, PipelineConfig
from app.models.document_ir import Block, BlockType
from app.pipeline.document_review import DocumentReview
from app.pipeline.review_sources import DeliveryExcerpts, SourceInventory, review_scope_digest
from app.pipeline.stages.chunk import ChunkStage


def paragraphs(root, blocks, target):
    renderer = ChunkStage()
    documents, chunks = DeliveryExcerpts(root, blocks).for_blocks(target, renderer)
    return {"delivered_documents": documents, "delivered_chunks": chunks}


def block(bid, text, order=0):
    return Block(block_id=bid, type=BlockType.TEXT, page_idx=0, reading_order=order, payload={"text": text})


@pytest.mark.asyncio
async def test_front_image_expansion_does_not_reaudit_unheaded_later_pages(tmp_path):
    ir, source = fixture(tmp_path, pages=8)
    ir.blocks[0].payload["text"] = "A short shelf photograph description."
    for i in range(8):
        for n in range(3):
            ir.blocks.append(Block(block_id=f'extra-{i}-{n}', type=BlockType.TEXT, page_idx=i,
                                   reading_order=i * 10 + n + 1,
                                   payload={"text": f"Unique page {i} entry {n}: ordinary source paragraph."}))
    for i, b in enumerate(ir.blocks[:8]):
        b.reading_order = i * 10
    ir.blocks.sort(key=lambda b: b.reading_order)
    write_delivery(tmp_path, ir)
    # Real Word reports often have no Markdown headings/blank separators.
    main = tmp_path / 'outputs/documents/main.md'
    main.write_text(main.read_text().replace('\n\n', '\n'))
    model = AuditModel()
    owner = DocumentReview(PipelineConfig(), model, model)
    await owner.run(ir, tmp_path, source_path=source)
    before = len(model.calls)
    ir.blocks[0].payload['text'] = '\n'.join(f'New shelf detail {n}: visible products.' for n in range(23))
    write_delivery(tmp_path, ir)
    main.write_text(main.read_text().replace('\n\n', '\n'))
    inventory = SourceInventory(ir, tmp_path, source)
    await inventory.prepare()
    for page in range(2, 8):
        reviewed = await owner._inspect(page, ir, inventory, tmp_path)
        assert reviewed['status'] == 'verified'
        assert reviewed['reuse_reason']
    assert len(model.calls) == before
    await owner._inspect(0, ir, inventory, tmp_path)
    assert len(model.calls) == before + 1  # Actual changed delivery never inherits approval.


def test_repeated_company_line_added_to_front_picture_stays_with_its_source(tmp_path):
    blocks = [block('photo', 'Shelf overview', 0), block('early', 'Acme company report', 1),
              block('late', 'Acme company report', 2), block('body', 'Later page financial figures 100 and 80', 3)]
    ir, _ = fixture(tmp_path, blocks=blocks)
    before = paragraphs(tmp_path, ir.blocks, ir.blocks[2:])
    ir.blocks[0].payload['text'] += '\nAcme company report\nVisible shelf products.'
    write_delivery(tmp_path, ir)
    after = paragraphs(tmp_path, ir.blocks, ir.blocks[2:])
    assert review_scope_digest('v', before, []) == review_scope_digest('v', after, [])
    assert [p['block_ids'] for p in after['delivered_documents']] == [['late'], ['body']]


def test_missing_repeated_occurrence_is_ambiguous_and_invalidates_scope(tmp_path):
    blocks = [block('early', 'Repeated identical company title', 0),
              block('late', 'Repeated identical company title', 1)]
    ir, _ = fixture(tmp_path, blocks=blocks)
    before = paragraphs(tmp_path, ir.blocks, [ir.blocks[-1]])
    (tmp_path / 'outputs/documents/main.md').write_text('Repeated identical company title\n')
    after = paragraphs(tmp_path, ir.blocks, [ir.blocks[-1]])
    assert after['delivered_documents'][0]['binding'] == 'ambiguous_source_occurrence'
    assert review_scope_digest('v', before, []) != review_scope_digest('v', after, [])


def test_source_marker_takes_precedence_over_identical_source_text(tmp_path):
    blocks = [block('early', 'Repeated identical title', 0), block('late', 'Repeated identical title', 1)]
    ir, _ = fixture(tmp_path, blocks=blocks)
    (tmp_path / 'outputs/documents/main.md').write_text(
        '<!-- source-block:late -->\nRepeated identical title\n\n'
        '<!-- source-block:early -->\nRepeated identical title\n')
    value = paragraphs(tmp_path, ir.blocks, [ir.blocks[-1]])
    assert len(value['delivered_documents']) == 1
    assert value['delivered_documents'][0]['binding'] == 'source_marker'
    assert value['delivered_documents'][0]['block_ids'] == ['late']


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'text', 'adjacency', 'chunk_members', 'chunk_text', 'image', 'unknown_tail'])
async def test_model_free_preflight_detects_real_delivery_or_evidence_changes(tmp_path, mutation):
    ir, source = fixture(tmp_path, pages=2)
    ir.blocks = [block('a', 'Revenue amount is 100 dollars', 0), block('b', 'Cost amount is 80 dollars', 1)]
    write_delivery(tmp_path, ir)
    inventory = SourceInventory(ir, tmp_path, source)
    await inventory.prepare()
    model = AuditModel()
    owner = DocumentReview(PipelineConfig(), model, model)
    before = owner.review_scope_fingerprints(ir, inventory, tmp_path)
    main = tmp_path / 'outputs/documents/main.md'
    if mutation == 'missing':
        main.write_text('Revenue amount is 100 dollars')
    elif mutation == 'reordered':
        main.write_text('Cost amount is 80 dollars\nRevenue amount is 100 dollars')
    elif mutation == 'text':
        main.write_text(main.read_text().replace('100', '999'))
    elif mutation == 'adjacency':
        main.write_text(main.read_text().replace('\n\n', '\nUnrelated inserted paragraph\n'))
    elif mutation == 'unknown_tail':
        main.write_text(main.read_text() + '\nEntirely fabricated new claim.\n')
    elif mutation in {'chunk_members', 'chunk_text'}:
        chunks = [json.loads(row) for row in (tmp_path / 'outputs/chunks.jsonl').read_text().splitlines()]
        if mutation == 'chunk_members':
            chunks[0]['block_ids'].append('unrelated')
        else:
            chunks[0]['content'] = 'Revenue amount is 999 dollars'
        (tmp_path / 'outputs/chunks.jsonl').write_text('\n'.join(json.dumps(c) for c in chunks))
    else:
        Image.new('RGB', (10, 10), 'red').save(inventory.image(0))
    after = owner.review_scope_fingerprints(ir, inventory, tmp_path)
    assert before[0] != after[0]
    assert model.calls == []


@pytest.mark.asyncio
async def test_candidate_evidence_relocation_preserves_scope_but_changed_label_does_not(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    ir, source = fixture(original)
    inventory = SourceInventory(ir, original, source)
    await inventory.prepare()
    owner = DocumentReview(PipelineConfig())
    before = owner.build_page_review_input(0, ir, inventory, original)
    candidate = tmp_path / 'candidate'
    shutil.copytree(original, candidate)
    relocated = SourceInventory(ir, candidate, candidate / 'original.pdf')
    await relocated.prepare()
    # Different cache folder with identical pixels must remain the same source.
    relocated.directory = candidate / 'agentic/evidence/different-storage-folder'
    relocated.directory.mkdir()
    after = owner.build_page_review_input(0, ir, relocated, candidate)
    assert before['scoped_fingerprint'] == after['scoped_fingerprint']
    changed = copy.deepcopy(after['evidence'])
    changed[0]['label'] = 'Different source page role'
    assert review_scope_digest('v', before['payload'], before['evidence']) != review_scope_digest('v', after['payload'], changed)


@pytest.mark.asyncio
async def test_exhausted_budget_does_not_reuse_same_delivery_after_source_change(tmp_path):
    ir, source = fixture(tmp_path)
    model = AuditModel()
    owner = DocumentReview(PipelineConfig(agentic=AgenticConfig(max_model_calls=1)), model, model)
    await owner.run(ir, tmp_path, source_path=source)
    assert owner.state['pages']['0']['status'] == 'verified'
    inventory = SourceInventory(ir, tmp_path, source)
    await inventory.prepare()
    Image.new('RGB', (30, 30), 'red').save(inventory.image(0))
    reviewed = await owner._inspect(0, ir, inventory, tmp_path)
    assert reviewed['status'] == 'not_reviewed'
    assert reviewed['reason'] == 'review_budget_exhausted'


@pytest.mark.asyncio
async def test_exported_fallback_figure_title_does_not_shift_when_earlier_image_is_replaced(tmp_path):
    from app.pipeline.stages.package import PackageStage

    ir, _ = fixture(tmp_path, blocks=[
        Block(block_id='photo-a', type=BlockType.IMAGE, page_idx=0, reading_order=0,
              payload={'img_path': 'a.png'}),
        Block(block_id='photo-b', type=BlockType.IMAGE, page_idx=0, reading_order=1,
              payload={'img_path': 'b.png'}),
    ])
    cache = tmp_path / 'parse'
    cache.mkdir()
    for name in ('a.png', 'b.png'):
        Image.new('RGB', (30, 30), 'blue').save(cache / name)
    package = PackageStage(PipelineConfig())
    _, before = await package._export_assets(ir, tmp_path / 'first-assets', cache)
    ir.blocks.pop(0)
    _, after = await package._export_assets(ir, tmp_path / 'second-assets', cache)
    assert before['photo-b'].title == after['photo-b'].title == '圖表語意'
    assert before['photo-b'].retrieval_text == after['photo-b'].retrieval_text
    assert before['photo-b'].asset_id == after['photo-b'].asset_id
    assert before['photo-b'].asset_path == after['photo-b'].asset_path


def test_logical_word_page_does_not_backfill_image_from_distant_prose(tmp_path):
    from app.pipeline.stages.package import PackageStage

    ir, _ = fixture(tmp_path, blocks=[block(f'b{i}', text, i) for i, text in enumerate([
        'Annual report cover', 'Owner Acme', 'Source is not this image', 'Unrelated later body',
    ])])
    ir.page_kind = 'logical'
    picture = Block(block_id='flow', type=BlockType.IMAGE, page_idx=0, payload={})
    ir.blocks.append(picture)
    output = {'image_type': 'flowchart', 'structured_content': ['Step A > Step B'], 'all_text': ['Step A', 'Step B']}
    assert PackageStage()._augment_visual_output_from_page_text(output, ir, picture) == output
    before, _ = PackageStage()._render_rag_md(ir, {}, {'flow': {'output': output}})
    assert 'Unrelated later body' in before  # No whole-document OCR suppression either.


def test_asset_document_ownership_beats_identical_prose_on_other_pages(tmp_path):
    ir, _ = fixture(tmp_path, blocks=[block('native', 'Acme company report'), block('picture', 'Other picture text', 1)])
    output = tmp_path / 'outputs'
    (output / 'documents/figure.md').write_text('Acme company report\n')
    (output / 'assets_index.jsonl').write_text(json.dumps({'asset_id': 'figure', 'block_id': 'picture'}))
    (output / 'documents_index.json').write_text(json.dumps([
        {'document_id': 'main', 'file': 'main.md'}, {'document_id': 'figure', 'file': 'figure.md'}]))
    native = paragraphs(tmp_path, ir.blocks, [ir.blocks[0]])
    assert all(p['file'].endswith('main.md') for p in native['delivered_documents'])
    picture = paragraphs(tmp_path, ir.blocks, [ir.blocks[1]])
    assert any(p['binding'] == 'asset_document' and p['file'].endswith('figure.md')
               for p in picture['delivered_documents'])


@pytest.mark.parametrize('valid_box', [False, True])
def test_same_physical_page_prose_outside_picture_never_becomes_picture_text(tmp_path, valid_box):
    from app.pipeline.stages.package import PackageStage

    ir, _ = fixture(tmp_path, blocks=[block(f'b{i}', text, i) for i, text in enumerate([
        'Exterior title', 'Exterior context', 'Exterior paragraph', 'Exterior conclusion',
    ])])
    for source in ir.blocks:
        source.bbox_norm = [10, 10, 100, 40]
    picture = Block(block_id='picture', type=BlockType.IMAGE, page_idx=0,
                    bbox_norm=[500, 500, 900, 900] if valid_box else [0, 0, 0, 0], payload={})
    ir.blocks.append(picture)
    output = {'image_type': 'flowchart', 'structured_content': ['Step A > Step B'], 'all_text': ['Step A', 'Step B']}
    assert PackageStage()._augment_visual_output_from_page_text(output, ir, picture) == output


def test_repeated_label_uses_local_source_boundaries_when_another_image_omits_it(tmp_path):
    blocks = [block('photo', 'Acme company', 0), block('head-a', 'Section A unique heading', 1),
              block('a', 'Acme company', 2), block('body-a', 'Section A unique body', 3),
              block('head-b', 'Section B unique heading', 4), block('b', 'Acme company', 5),
              block('body-b', 'Section B unique body', 6)]
    ir, _ = fixture(tmp_path, blocks=blocks)
    main = tmp_path / 'outputs/documents/main.md'
    main.write_text('\n'.join(b.get_text() for b in blocks[1:]))
    before = paragraphs(tmp_path, ir.blocks, blocks[4:])
    ir.blocks = blocks[1:]
    after = paragraphs(tmp_path, ir.blocks, blocks[4:])
    assert review_scope_digest('v', before, []) == review_scope_digest('v', after, [])
    assert [d['block_ids'] for d in before['delivered_documents']] == [['head-b'], ['b'], ['body-b']]


@pytest.mark.asyncio
async def test_split_page_review_keeps_actual_delivery_for_every_region(tmp_path):
    blocks = [Block(block_id=f'b{i}', type=BlockType.TEXT, page_idx=0, reading_order=i,
                    bbox_norm=[0, i * 20, 900, i * 20 + 10],
                    payload={'text': f'Unique source paragraph {i} has value {i * 17 + 1}.'})
              for i in range(48)]
    ir, source = fixture(tmp_path, blocks=blocks)

    class SplitAudit(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == 'verify_layout':
                self.calls.append((task, payload, image))
                return {'reading_order_correct': True, 'observation': 'All regions remain ordered.', 'issues': []}, 10
            delivered = '\n'.join(d['text'] for d in payload['delivered_documents'])
            assert all(ir.get_block(bid).get_text() in delivered for bid in payload['required_block_ids'])
            return await super().query(task, payload, image, schema)

    model = SplitAudit()
    _, report = await DocumentReview(PipelineConfig(), model, model).run(ir, tmp_path, source_path=source)
    assert report['summary']['verified_pages'] == 1
    assert len([c for c in model.calls if c[0] == 'audit_page']) == 4
