"""Review, storage and publication must bind the same immutable content snapshot."""
import json

import pytest
from test_document_review import AuditModel, fixture, write_delivery

from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.review_sources import digest, input_identity, validate_review_coverage
from app.pipeline.stages.normalize import load_document_ir, save_document_ir


def table():
    return Block(block_id='recovered', type=BlockType.TABLE, page_idx=0,
                 payload={'table_body': '<table><tr><th>Revenue</th><th>Cost</th></tr>'
                          '<tr><td>100</td><td>80</td></tr></table>'})


def test_save_is_serialization_without_materializing_or_mutating_tables(tmp_path):
    ir, _ = fixture(tmp_path, blocks=[table()])
    before = ir.to_dict()
    save_document_ir(ir, tmp_path)
    assert ir.to_dict() == before
    assert load_document_ir(tmp_path).to_dict() == before
    first = (tmp_path / 'document_ir.json').read_bytes()
    save_document_ir(load_document_ir(tmp_path), tmp_path)
    assert (tmp_path / 'document_ir.json').read_bytes() == first


def test_relationship_evidence_reference_is_part_of_review_identity(tmp_path):
    ir, _ = fixture(tmp_path, blocks=[table()])
    original = digest(input_identity(ir))
    ir.blocks[0].payload['agentic_relationship_evidence_id'] = 'ev_' + 'a' * 64
    first = digest(input_identity(ir))
    assert first != original
    ir.blocks[0].payload['agentic_relationship_evidence_id'] = 'ev_' + 'b' * 64
    assert digest(input_identity(ir)) != first


@pytest.mark.asyncio
async def test_recovery_table_remains_valid_after_save_reload_and_publish_checks(tmp_path, monkeypatch):
    from app.pipeline.document_recovery import RecoveryController
    ir, source = fixture(tmp_path)
    cfg = PipelineConfig()

    async def recover(self, candidate):
        candidate.blocks = [table()]
        self.inventory.ir = candidate
        self.inventory.mapping = {'recovered': [0]}
        write_delivery(tmp_path, candidate)
        return candidate

    monkeypatch.setattr(RecoveryController, 'run', recover)
    model = AuditModel()
    result, report = await DocumentReview(cfg, model, model).run(
        ir, tmp_path, source_path=source, rebuild=lambda _: None)
    assert result.blocks[0].payload.get('table_structure')
    reviewed = digest(input_identity(result))
    assert reviewed == report['ir_sha256']
    save_document_ir(result, tmp_path)
    restored = load_document_ir(tmp_path)
    assert reviewed == digest(input_identity(restored))
    assert validate_review_coverage(restored, tmp_path, cfg)['evidence_valid']
    status = finalize_delivery(restored, tmp_path, cfg)
    assert status['partial_rag_ready']
    assert '100' in (tmp_path / 'outputs/rag_verified.md').read_text()
    # Normalization stability must never excuse a real source/content change.
    restored.blocks[0].payload['table_body'] = restored.blocks[0].payload['table_body'].replace('100', '999')
    assert not validate_review_coverage(restored, tmp_path, cfg)['evidence_valid']
    coverage = json.loads((tmp_path / 'outputs/review_coverage.json').read_text())
    image = tmp_path / coverage['pages'][0]['images'][0]['path']
    image.write_bytes(b'changed source pixels')
    assert not validate_review_coverage(result, tmp_path, cfg)['evidence_valid']
