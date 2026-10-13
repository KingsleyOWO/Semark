"""Image labels cannot certify wrong geometry or missing relationship evidence."""
import json

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError
from test_document_review import AuditModel, config, fixture, write_delivery

from app.models.document_ir import BlockType
from app.pipeline.document_review import BlockReview, DocumentReview, PageReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.image_source_review import (
    ImageSourceRelation,
    WordPictureItem,
    read_word_picture_batch,
    relationship_issue,
)
from app.pipeline.review_sources import SourceInventory


def relations():
    return [
        {'relation_id': 'r0', 'kind': 'spatial_position', 'source_fact': 'ALPHA is right of the vertical axis, above the horizontal axis.',
         'bbox': [540, 200, 760, 420], 'horizontal': 'right', 'vertical': 'above', 'certain': True},
        {'relation_id': 'r1', 'kind': 'arrow', 'source_fact': 'An arrow goes from the faded ALPHA mark toward the upper-right solid ALPHA mark.',
         'bbox': [620, 80, 890, 320], 'direction': 'up_right', 'certain': True},
    ]


def reading(**changes):
    return WordPictureItem.model_validate({'source_id': 'image0', 'text': 'ALPHA\nFUNCTION\nCULTURE',
        'kind': 'transcription', 'complete': True, 'observation': 'Original image only.', 'image_type': 'chart',
        'relationships': relations(), 'relationships_complete': True, **changes})


def checks(quote='ALPHA sits upper right; arrow goes up-right.'):
    return [
        {'relation_id': 'r0', 'status': 'matched', 'delivered_quote': quote, 'horizontal': 'right', 'vertical': 'above', 'observation': 'Read actual placement.'},
        {'relation_id': 'r1', 'status': 'matched', 'delivered_quote': quote, 'direction': 'up_right', 'observation': 'Read actual arrow.'},
    ]


@pytest.mark.parametrize('mode', ['opposite_position', 'opposite_arrow', 'label_only'])
@pytest.mark.asyncio
async def test_same_labels_cannot_certify_reversed_or_omitted_visual_meaning(tmp_path, monkeypatch, mode):
    ir, source = fixture(tmp_path)
    block = ir.blocks[0]
    block.type = BlockType.IMAGE
    correct = 'ALPHA sits upper right; arrow goes up-right.'
    wrong = {'opposite_position': 'ALPHA sits lower left; arrow goes up-right.',
             'opposite_arrow': 'ALPHA sits upper right; arrow goes down-left.',
             'label_only': 'ALPHA FUNCTION CULTURE'}[mode]
    block.payload = {'caption': 'ALPHA\nFUNCTION\nCULTURE\n' + wrong}
    write_delivery(tmp_path, ir)
    original = tmp_path / 'agentic/evidence/source-chart.png'
    original.parent.mkdir(parents=True, exist_ok=True)
    picture = Image.new('RGB', (300, 220), 'white')
    draw = ImageDraw.Draw(picture)
    draw.line((150, 5, 150, 215), fill='black')
    draw.line((5, 110, 295, 110), fill='black')
    draw.text((190, 70), 'ALPHA', fill='black')
    draw.line((220, 60, 270, 20), fill='red')
    picture.save(original)
    prepare = SourceInventory.prepare

    async def with_picture(self, **kwargs):
        await prepare(self, **kwargs)
        self.word_images = {block.block_id: original}
    monkeypatch.setattr(SourceInventory, 'prepare', with_picture)

    class Model(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == 'verify_native_reading':
                self.calls.append((task, payload, image))
                assert wrong not in json.dumps(payload)
                assert not any(key in payload for key in ['blocks', 'delivered_documents', 'independent_image_readings'])
                return {'items': [reading().model_dump()]}, 10
            value, tokens = await super().query(task, payload, image, schema)
            assert payload['independent_image_readings']['b0']['relationships'][1]['direction'] == 'up_right'
            assert 'not ground truth' in payload['image_relationship_policy']
            audited = checks(wrong)
            if mode == 'opposite_position':
                audited[0].update(horizontal='left', vertical='below')
            elif mode == 'opposite_arrow':
                audited[1]['direction'] = 'down_left'
            else:
                audited = []
            value['blocks'][0].update(context_complete=True, image_relationship_checks=audited)
            return value, tokens

    model = Model()
    owner = DocumentReview(config(recovery_enabled=False), model, model)
    _, report = await owner.run(ir, tmp_path, source_path=source)
    assert len(model.calls) == 2
    assert report['summary']['needs_review_pages'] == 1
    assert report['pages'][0]['image_issues'][0]['failure_kind'] == 'image_relationship_mismatch'
    finalize_delivery(ir, tmp_path, config())
    # Since 2026-10-01 unconfirmed picture content may ship, but only labelled as
    # not verified; what must never happen is the picture passing as verified.
    chunks = [json.loads(line) for line in (tmp_path / 'outputs/rag_chunks.jsonl').read_text().splitlines()]
    assert not any('b0' in c['block_ids'] and c['metadata'].get('source_verified') for c in chunks)
    assert correct not in (tmp_path / 'outputs/delivery.md').read_text()


@pytest.mark.parametrize('mode', ['valid', 'missing', 'duplicate', 'invented', 'uncertain_source', 'fabricated_quote', 'neighbor_quote'])
def test_relationship_accounting_and_source_ownership(mode):
    text = 'ALPHA sits upper right; arrow goes up-right.'
    source = reading()
    items = checks()
    documents = [{'block_ids': ['figure'], 'text': text}]
    chunks = [{'block_ids': ['figure'], 'content': text}]
    if mode == 'missing':
        items.pop()
    elif mode == 'duplicate':
        items.append(items[0])
    elif mode == 'invented':
        items[0]['relation_id'] = 'other'
    elif mode == 'uncertain_source':
        source.relationships[0].certain = False
    elif mode == 'fabricated_quote':
        items[0]['delivered_quote'] = 'A complete invented relationship.'
    elif mode == 'neighbor_quote':
        documents[0]['block_ids'] = ['neighbor']
        chunks[0]['block_ids'] = ['neighbor']
    finding = BlockReview(block_id='figure', status='verified', context_complete=True,
                          observation='Model claims all verified.', image_relationship_checks=items)
    issue = relationship_issue(source, finding, text, documents=documents, chunks=chunks)
    assert bool(issue) == (mode != 'valid')


def test_schema_preserves_axis_crossing_and_rejects_unbounded_or_label_only_claims():
    r = relations()[0]
    r.update(horizontal='spans_axis', vertical='above')
    assert ImageSourceRelation.model_validate(r).horizontal == 'spans_axis'
    with pytest.raises(ValidationError, match='empty label-only'):
        reading(relationships=[])
    with pytest.raises(ValidationError):
        reading(relationships=[{**r, 'relation_id': f'r{i}'} for i in range(17)])
    with pytest.raises(ValidationError, match='uncertain'):
        ImageSourceRelation.model_validate({**r, 'horizontal': 'uncertain'})


@pytest.mark.asyncio
async def test_source_contract_failure_splits_and_resume_does_not_reissue_reads(tmp_path):
    from test_document_recovery import setup

    class Model(AuditModel):
        async def query(self, task, payload, image, schema):
            self.calls.append((task, payload, image))
            if len(payload['sources']) > 1:
                return {'items': []}, 10
            return {'items': [reading(source_id=payload['sources'][0]['source_id']).model_dump()]}, 10

    model = Model()
    _, owner, runner = await setup(tmp_path, model)
    picture = runner.inventory.image(0)
    payload = {'sources': [{'source_id': 'a'}, {'source_id': 'b'}], 'image_order': ['a', 'b']}
    for _ in range(2):
        result = await read_word_picture_batch(owner, payload, [picture, picture], source_ids=['a', 'b'])
        assert [item.source_id for item in result.items] == ['a', 'b']
    assert len(model.calls) == 3
    assert owner.state['usage']['model_calls'] == 3


@pytest.mark.asyncio
async def test_large_blind_evidence_cannot_bypass_post_read_context_bound(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from app.models.document_ir import Block

    ir, source = fixture(tmp_path)
    ir.blocks = [Block(block_id=f'b{i}', type=BlockType.IMAGE, page_idx=0, reading_order=i,
                       payload={'caption': f'Figure {i}'}) for i in range(7)]
    write_delivery(tmp_path, ir)
    owner = DocumentReview(config(recovery_enabled=False), AuditModel(), AuditModel())
    # Bound schema-valid source content can still exceed a full-page context when
    # many images share a page. No page audit may run after the added payload overflows.
    original = DocumentReview._word_picture_readings
    # Even one picture's part overflows (here: a lowered limit); otherwise the
    # readings are judged in parts (next test).
    from app.pipeline import document_review
    monkeypatch.setattr(document_review, 'REGION_PAYLOAD_LIMIT', 12000)
    async def large(self, blocks, inventory):
        return {b.block_id: reading(text='字' * 10000, image_type='document', relationships=[])
                for b in blocks}
    monkeypatch.setattr(DocumentReview, '_word_picture_readings', large)
    owner._audit_page_contract = AsyncMock(side_effect=AssertionError('oversize audit must not run'))
    _, report = await owner.run(ir, tmp_path, source_path=source)
    assert report['pages'][0]['reason'] == 'page_exceeds_image_relationship_context_limit'
    assert report['pages'][0]['status'] == 'not_reviewed'
    assert not owner._audit_page_contract.await_count
    monkeypatch.setattr(DocumentReview, '_word_picture_readings', original)


@pytest.mark.asyncio
@pytest.mark.parametrize('inherited', [False, True])
async def test_native_image_relationships_cannot_bypass_via_segmented_audit(tmp_path, monkeypatch, inherited):
    from unittest.mock import AsyncMock

    from app.models.document_ir import Block

    ir, source = fixture(tmp_path)
    ir.blocks = [Block(block_id=f'b{i}', type=BlockType.TEXT, page_idx=0, reading_order=i,
                       payload={'text': f'Actual source item {i}'}) for i in range(46)]
    write_delivery(tmp_path, ir)
    original = tmp_path / 'agentic/evidence/chart.png'
    original.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (200, 150), 'white').save(original)
    prepare = SourceInventory.prepare
    async def with_original(self, **kwargs):
        await prepare(self, **kwargs)
        self.word_images = {} if inherited else {'b0': original}
    if inherited:
        ir.blocks[0].payload['agentic_relationship_evidence_id'] = 'ref-requiring-unsplit-audit'
    monkeypatch.setattr(SourceInventory, 'prepare', with_original)
    owner = DocumentReview(config(), AuditModel(), AuditModel())
    owner._inspect_segments = AsyncMock(side_effect=AssertionError('no native relation bypass'))
    _, report = await owner.run(ir, tmp_path, source_path=source)
    assert report['pages'][0]['reason'] == 'page_exceeds_native_image_relationship_review_scope'
    assert not owner._inspect_segments.await_count


@pytest.mark.parametrize('with_relation_checks', [False, True])
@pytest.mark.asyncio
async def test_image_split_text_blocks_retain_relationship_gate_in_local_and_final_audit(tmp_path, with_relation_checks):
    from test_document_recovery import setup

    from app.pipeline.image_recovery import build_image_parts_candidate

    class Model(AuditModel):
        async def query(self, task, payload, image, schema):
            self.calls.append((task, payload, image))
            assert task in {'audit_image_parts', 'audit_page'}  # No repeated blind model read after ID replacement.
            assert payload['independent_image_readings']['b0']['relationships'][1]['direction'] == 'up_right'
            assert payload['image_reading_scopes']['b0'] == payload['required_block_ids']
            return {'blocks': [{'block_id': bid, 'status': 'verified', 'context_complete': True,
                                'observation': 'Source and delivery claimed equal.',
                                'image_relationship_checks': checks() if with_relation_checks else []}
                               for bid in payload['required_block_ids']],
                    'source_complete': True, 'reading_order_correct': True, 'observation': 'All labels present.',
                    'issues': []}, 10

    model = Model()
    ir, owner, runner = await setup(tmp_path, model)
    original = ir.blocks[0]
    original.type = BlockType.IMAGE
    original.payload = {'caption': 'ALPHA FUNCTION CULTURE'}
    owner.state['pages']['0']['word_image_readings'] = {'b0': reading().model_dump()}
    source = runner.inventory.image(0)
    folder = tmp_path / 'candidate'
    parts = {'visual_facts': [], 'text_regions': [{'bbox': [0, 0, 1000, 1000],
               'text': 'ALPHA sits upper right; arrow goes up-right.'}], 'gaps': [],
             'text_dependent': True, 'information_status': 'informative'}
    candidate = build_image_parts_candidate(runner, ir, original, 0, [0, 0, 1000, 1000], source,
                                           parts, [], folder)
    replacement = candidate.blocks[0]
    assert replacement.type == BlockType.TEXT and replacement.block_id != original.block_id
    assert replacement.payload['agentic_relationship_evidence_id'] in replacement.payload['agentic_evidence_ids']
    # An unaudited relationship no longer fails a repair (decided 2026-10-13); the gate still
    # reports it, so the page review below stays needs_review without the checks.
    assert await runner.validate(candidate, [0], folder, image_origin='b0', image_pruned=True)
    validation = json.loads((folder / 'validation-image.json').read_text())['review']
    assert any('relationship coverage is incomplete' in i for i in validation['issues']) is not with_relation_checks
    inventory = await runner.candidate_inventory(candidate, folder)
    assert not inventory.word_images  # The removed original ID cannot be the protection mechanism.
    record = await owner._inspect(0, candidate, inventory, folder)
    assert (record['status'] == 'verified') is with_relation_checks
    assert len(model.calls) == 2
    assert 'b0' in record['word_image_readings']


def test_inherited_relationship_evidence_rejects_swapped_source_image(tmp_path):
    from app.models.document_ir import Block
    from app.pipeline.evidence import EvidenceStore
    from app.pipeline.image_source_review import inherited_image_readings

    source = tmp_path / 'agentic/evidence/original.png'
    source.parent.mkdir(parents=True)
    Image.new('RGB', (100, 100), 'white').save(source)
    store = EvidenceStore(tmp_path, 'document')
    eid = store.add(provider='test', kind='image_relationship_reading', page=0, block_id='origin',
                    source=source, content=reading().model_dump())
    block = Block(block_id='child', type=BlockType.TEXT, page_idx=0, payload={
        'text': 'ALPHA', 'agentic_image_origin': 'origin', 'agentic_source_pages': [0],
        'agentic_source_sha256': 'document', 'agentic_source_image': str(source.relative_to(tmp_path)),
        'agentic_relationship_evidence_id': eid})
    assert inherited_image_readings([block], tmp_path)['origin'].image_type == 'chart'
    Image.new('RGB', (100, 100), 'red').save(source)
    with pytest.raises(ValueError, match='different source'):
        inherited_image_readings([block], tmp_path)


@pytest.mark.parametrize('tool', ['read_source', 'read_regions', 'reparse_pdf'])
@pytest.mark.asyncio
async def test_whole_page_tool_cannot_drop_existing_image_relationship_obligation(tmp_path, tool):
    from test_document_recovery import setup

    from app.models.document_ir import Block

    model = AuditModel()
    ir, owner, runner = await setup(tmp_path, model)
    ir.blocks[0].type = BlockType.IMAGE
    ir.blocks[0].payload = {'img_path': 'original.png'}
    owner.state['pages']['0']['word_image_readings'] = {'b0': reading().model_dump()}
    constraint = runner.tool_constraints({'page_idx': 0}, ir.blocks)[tool]
    assert not constraint['available'] and 'repair_image' in constraint['requires']
    candidate = ir.model_copy(deep=True)
    candidate.blocks = [Block(block_id='new-native-17', type=BlockType.TEXT, page_idx=0,
        payload={'text': 'ALPHA lower left; arrow down-left.', **runner.source_payload(0),
                 'agentic_recovery_tool': tool, 'agentic_source_region': [0, 0, 1000, 1000]})]
    with pytest.raises(ValueError, match='image_relationship_provenance_lost'):
        await runner.validate(candidate, [0], tmp_path / 'candidate')
    assert not model.calls and owner.state['usage']['model_calls'] == 0


@pytest.mark.asyncio
async def test_photo_without_relationships_keeps_existing_native_page_route(tmp_path):
    from test_document_recovery import setup

    ir, owner, runner = await setup(tmp_path)
    owner.state['pages']['0']['word_image_readings'] = {'b0': reading(
        image_type='photo', relationships=[], text='People sit around a table.', kind='visual_summary').model_dump()}
    for tool in ['read_source', 'read_regions', 'reparse_pdf']:
        assert runner.tool_constraints({'page_idx': 0}, ir.blocks)[tool]['available']


def test_word_page_readings_overflowing_one_request_are_judged_in_parts():
    """2026-10-08: a Word page of 8 blocks went unreviewed once its pictures' independent
    readings pushed the request past the limit."""
    from app.pipeline.document_review import reading_parts

    readings = {f"pic{i}": {"kind": "transcription", "text": f"Printed line {i} " * 300, "complete": True}
                for i in range(3)}
    payload = {
        "required_block_ids": ["intro", "pic0", "pic0-r1", "pic1", "pic2"],
        "blocks": [{"block_id": bid, "untrusted_extracted_text": bid} for bid in
                   ["intro", "pic0", "pic0-r1", "pic1", "pic2"]],
        "context_blocks": [{"block_id": "prev", "untrusted_extracted_text": "Previous page."}],
        "delivered_documents": [], "delivered_chunks": [],
        "independent_image_readings": readings,
        "image_reading_scopes": {"pic0": ["pic0", "pic0-r1"], "pic1": ["pic1"], "pic2": ["pic2"]},
        "independent_ocr_votes": {"pic1": {"verdict": "delivery"}},
    }
    limit = len(str(payload)) * 2 // 3

    parts = reading_parts(payload, limit)

    assert len(parts) > 1 and all(len(str(part)) <= limit for part in parts)
    required = [bid for part in parts for bid in part["required_block_ids"]]
    assert sorted(required) == sorted(payload["required_block_ids"])
    assert "intro" in parts[0]["required_block_ids"]
    for part in parts:
        owned = {bid for ids in part["image_reading_scopes"].values() for bid in ids}
        assert owned <= set(part["required_block_ids"])
        assert set(part["independent_image_readings"]) == set(part["image_reading_scopes"])
        assert set(part.get("independent_ocr_votes", {})) <= set(part["required_block_ids"])
        # Everything else on the page stays in view as context.
        context = {b["block_id"] for b in part["context_blocks"]}
        assert context == {"prev"} | (set(payload["required_block_ids"]) - set(part["required_block_ids"]))
    assert reading_parts(payload, len(str(readings["pic0"]))) is None


@pytest.mark.asyncio
async def test_many_picture_readings_on_one_word_page_are_audited_in_parts(tmp_path, monkeypatch):
    from app.models.document_ir import Block
    from app.pipeline.document_review import REGION_PAYLOAD_LIMIT

    ir, source = fixture(tmp_path)
    ir.blocks = [Block(block_id=f'b{i}', type=BlockType.IMAGE, page_idx=0, reading_order=i,
                       payload={'caption': f'Figure {i}'}) for i in range(7)]
    write_delivery(tmp_path, ir)
    owner = DocumentReview(config(recovery_enabled=False), AuditModel(), AuditModel())
    async def large(self, blocks, inventory):
        return {b.block_id: reading(text='字' * 10000, image_type='document', relationships=[])
                for b in blocks}
    monkeypatch.setattr(DocumentReview, '_word_picture_readings', large)
    sizes, scopes = [], []
    async def audit(payload, images, depth=0):
        sizes.append(len(str(payload)))
        scopes.append(payload['required_block_ids'])
        return PageReview(blocks=[{'block_id': bid, 'status': 'verified', 'observation': 'Matches.'}
                                  for bid in payload['required_block_ids']],
                          source_complete=True, reading_order_correct=True, observation='Page checked.', issues=[])
    owner._audit_page_contract = audit
    _, report = await owner.run(ir, tmp_path, source_path=source)
    assert report['pages'][0]['reason'] != 'page_exceeds_image_relationship_context_limit'
    assert len(sizes) > 1 and max(sizes) <= REGION_PAYLOAD_LIMIT
    assert sorted(bid for scope in scopes for bid in scope) == [f'b{i}' for i in range(7)]
