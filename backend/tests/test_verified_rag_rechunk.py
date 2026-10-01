"""Re-grouping preserves verified source atoms and complete table row context."""
import json

import pytest
from test_autonomous_delivery import read_chunks
from test_document_review import AuditModel, config, fixture

from app.models.document_ir import Block, BlockType
from app.pipeline.autonomous_delivery import DeliveredBlockEvidence
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.stages.chunk import ChunkStage, estimate_tokens
from app.pipeline.table_structure import table_structure


def test_owned_evidence_requires_exact_text_and_recorded_complete_split_family():
    text = 'Original measurement: temperature -100.25 C; source A. ' * 10
    renderer = ChunkStage(config())
    renderer.package_config.chunk_max_tokens = 64
    chunk = renderer._create_chunk('original', 'doc', 'run', 'rag', [
        Block(block_id='a', type=BlockType.TEXT, page_idx=0, payload={'text': text})])
    pieces = [c.to_dict() for c in renderer._bound_chunks([chunk])]
    assert len(pieces) > 1
    evidence = DeliveredBlockEvidence(pieces)
    assert evidence.covers('a', text)
    assert not evidence.covers('b', text)
    assert not evidence.covers('a', text.replace('-100.25', '100.25'))
    assert not evidence.covers('a', text.replace('source A', 'source B'))
    assert not DeliveredBlockEvidence(pieces[:-1]).covers('a', text)
    pieces[-1]['block_ids'] = ['b']
    assert not DeliveredBlockEvidence(pieces).covers('a', text)


@pytest.mark.asyncio
async def test_verified_atoms_survive_new_neighbors_without_global_boundary_matching(tmp_path):
    blocks = [Block(block_id=f'b{i}', type=BlockType.TEXT, page_idx=0, reading_order=i,
                    payload={'text': f'Item {i}: revenue is {100+i} dollars.'}) for i in range(12)]
    ir, source = fixture(tmp_path, blocks=blocks)
    def verify(_, value):
        for b in value['blocks']:
            b['context_complete'] = True
        return value
    model = AuditModel(verify)
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    chunks = read_chunks(tmp_path)
    assert {bid for chunk in chunks for bid in chunk['block_ids']} == {b.block_id for b in blocks}
    assert all(any(b.payload['text'] in chunk['content'] for chunk in chunks) for b in blocks)


@pytest.mark.asyncio
async def test_owned_evidence_does_not_borrow_identical_text_from_another_block(tmp_path):
    blocks = [Block(block_id='a', type=BlockType.TEXT, page_idx=0, payload={'text': 'Revenue -100.25 USD.'}),
              Block(block_id='b', type=BlockType.TEXT, page_idx=0, reading_order=1,
                    payload={'text': 'Revenue -100.25 USD.'})]
    ir, source = fixture(tmp_path, blocks=blocks)
    # Both texts globally exist, but only b owns a delivered occurrence.
    (tmp_path/'outputs/chunks.jsonl').write_text(json.dumps({
        'chunk_id':'only-b','block_ids':['b'],'content':blocks[1].payload['text']}))
    model = AuditModel(lambda _, v: {**v, 'blocks': [{**b, 'context_complete': True} for b in v['blocks']]})
    ir, _ = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=source)
    finalize_delivery(ir, tmp_path, config())
    chunks = read_chunks(tmp_path)
    assert {bid for c in chunks for bid in c['block_ids']} == {'b'}


@pytest.mark.asyncio
async def test_multilevel_headers_merged_rows_units_and_notes_survive_table_splitting(tmp_path):
    body = ('<table><tr><td colspan="3">Period 2026; unit USD</td></tr>'
            '<tr><th rowspan="2">Item</th><th colspan="2">Amount</th></tr>'
            '<tr><th>Government</th><th>Private</th></tr>')
    for i in range(8):
        body += f'<tr><td rowspan="2">Project {i}</td><td>{100+i}</td><td>{200+i}</td></tr>'
        body += f'<tr><td>{300+i}</td><td>{400+i}</td></tr>'
    body += '<tr><td colspan="3">Note: figures exclude tax</td></tr></table>'
    block = Block(block_id='table', type=BlockType.TABLE, page_idx=0,
                  payload={'table_body':body, 'table_footnote':['Footnote: approved budget only.']})
    ir, source = fixture(tmp_path, blocks=[block])
    cfg=config()
    cfg.package.chunk_max_tokens=220
    model=AuditModel()
    ir,_=await DocumentReview(cfg,model,model).run(ir,tmp_path,source_path=source)
    finalize_delivery(ir,tmp_path,cfg)
    chunks=read_chunks(tmp_path)
    assert chunks
    grid=table_structure(block)
    covered=set()
    for chunk in chunks:
        selected=set(chunk['metadata']['table_source_rows']['table'])
        covered.update(selected)
        assert {1,2,3}.issubset(selected)
        assert 'Period 2026; unit USD' in chunk['content']
        assert 'Note: figures exclude tax' in chunk['content']
        assert 'Footnote: approved budget only.' in chunk['content']
        for cell in grid.cells:
            span=set(range(cell.row,cell.row+cell.rowspan))
            if span & selected:
                assert span.issubset(selected)
        assert estimate_tokens(chunk['content'])<=cfg.package.chunk_max_tokens
        assert 'table_html' not in chunk['metadata']
    assert covered == set(range(1,grid.row_count+1))
    assert all(any(cell.text in c['content'] for c in chunks) for cell in grid.cells if cell.text)


@pytest.mark.asyncio
async def test_oversized_row_remains_excluded_instead_of_losing_cells(tmp_path):
    block=Block(block_id='table',type=BlockType.TABLE,page_idx=0,payload={'table_body':
        '<table><tr><th>Item</th><th>Amount</th></tr><tr><td colspan="2">'+
        ('Required context and value. '*90)+'</td></tr></table>'})
    ir,source=fixture(tmp_path,blocks=[block])
    cfg=config()
    cfg.package.chunk_max_tokens=64
    model=AuditModel()
    ir,_=await DocumentReview(cfg,model,model).run(ir,tmp_path,source_path=source)
    finalize_delivery(ir,tmp_path,cfg)
    assert read_chunks(tmp_path)==[]


def test_figure_split_keeps_all_ocr_rows_together_and_rejects_unreviewed_metadata():
    from app.pipeline.autonomous_delivery import split_verified_image
    title='Comparison of two brands and their colors.'
    facts=['Brand A uses yellow.', 'Brand B uses blue.']
    transcript='Brand | Color\nBrand A | Yellow\nBrand B | Blue\nKeywords: brands, colors'
    text='\n'.join([title,*facts,transcript])
    block=Block(block_id='image',type=BlockType.IMAGE,page_idx=0,payload={})
    class IR:
        def get_block(self,bid):
            assert bid=='image'
            return block
    renderer=ChunkStage(config())
    renderer._asset_text={'image':text}
    chunk=renderer._create_chunk('c','doc','run','rag',[block])
    records={'image':{'title':title,'facts':facts}}
    parts=split_verified_image(renderer,IR(),chunk,['image'],100,records)
    assert len(parts)==2
    assert transcript in parts[1][0].content
    assert all(title in part.content for part,_ in parts)
    records['image']['facts']=['Brand A uses red.']
    assert split_verified_image(renderer,IR(),chunk,['image'],100,records)==[]
    records['image']['facts']=facts
    assert split_verified_image(renderer,IR(),chunk,['image'],10,records)==[]
