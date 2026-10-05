"""Evidence a page record cites must still verify in the live run.

Validation v20.1 spot check: a repair candidate's audit certified page 2's text
from the text layer and wrote the proof inside the candidate folder. The
committed record cited it, the live run never had the file, and delivery
treated the whole review as tampered: nothing reached RAG.
"""
import hashlib
from types import SimpleNamespace

import fitz
import pytest

from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType
from app.pipeline.document_recovery import adopt_candidate_evidence
from app.pipeline.document_review import DocumentReview


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_recorded_text_layer_proof_survives_a_later_proof_of_the_same_page(tmp_path):
    pdf = tmp_path / "rendered.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(pdf)
    inventory = SimpleNamespace(directory=tmp_path / "agentic/evidence/review", pdf=pdf)
    inventory.directory.mkdir(parents=True)
    evidence = SimpleNamespace(lines={0: ["一、計畫摘要"]})
    review = DocumentReview(PipelineConfig())
    blocks = [Block(block_id=f"b{i}", type=BlockType.TEXT, page_idx=0, payload={"text": "一、計畫摘要"}) for i in range(2)]

    first = review._native_text_proof(0, blocks[:1], inventory, tmp_path, evidence)
    second = review._native_text_proof(0, blocks, inventory, tmp_path, evidence)

    for entry in (first, second):
        assert _sha(tmp_path / entry["path"]) == entry["sha256"]


def _record(path, sha):
    return {"1": {"images": [{"path": path, "sha256": sha, "label": "proof"}]}}


def test_committing_a_candidate_brings_the_evidence_its_records_cite(tmp_path):
    folder, root = tmp_path / "candidate", tmp_path / "live"
    staged = folder / "agentic/evidence/review/native-text-p0001.json"
    staged.parent.mkdir(parents=True)
    staged.write_text('{"page_idx": 1}')
    root.mkdir()

    adopt_candidate_evidence(_record("agentic/evidence/review/native-text-p0001.json", _sha(staged)), [1], folder, root)

    assert _sha(root / "agentic/evidence/review/native-text-p0001.json") == _sha(staged)


def test_a_candidate_cannot_replace_different_live_evidence(tmp_path):
    folder, root = tmp_path / "candidate", tmp_path / "live"
    for base, text in ((folder, "candidate"), (root, "live")):
        path = base / "agentic/evidence/review/context-0001.png"
        path.parent.mkdir(parents=True)
        path.write_text(text)

    with pytest.raises(ValueError, match="candidate_evidence"):
        adopt_candidate_evidence(_record("agentic/evidence/review/context-0001.png",
                                         _sha(folder / "agentic/evidence/review/context-0001.png")), [1], folder, root)
    assert (root / "agentic/evidence/review/context-0001.png").read_text() == "live"


@pytest.mark.asyncio
async def test_a_committed_repair_keeps_the_evidence_its_candidate_audit_wrote(tmp_path, monkeypatch):
    import json

    from test_document_recovery import setup

    ir, owner, runner = await setup(tmp_path)
    inspect = type(owner)._inspect

    async def with_local_proof(self, page, working, inventory, run_path):
        # Like the text-layer proof: written under whichever run the audit inspects.
        record = await inspect(self, page, working, inventory, run_path)
        proof = run_path / "agentic/evidence/local-proof.json"
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text(json.dumps({"page": page, "blocks": [b.block_id for b in working.blocks]}))
        record["images"] = record.get("images", []) + [
            {"path": str(proof.relative_to(run_path)), "sha256": _sha(proof), "label": "local proof"}]
        return record
    monkeypatch.setattr(type(owner), "_inspect", with_local_proof)

    await runner.run(ir)

    assert runner.state["steps"][0]["status"] == "applied"
    cited = [i for r in owner.state["pages"].values() for i in r.get("images", [])]
    assert any(i["label"] == "local proof" for i in cited)
    for image in cited:
        assert _sha(tmp_path / image["path"]) == image["sha256"]
