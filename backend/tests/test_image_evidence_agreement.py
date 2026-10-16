"""Readings that agree are not a conflict, and one real conflict costs one region (2026-10-16).

Rerun 2026-10-15: 11 full re-reads (49 min) failed whole on "conflicting readings". Of their
31 pairs, 24 were one reading holding the other (OCR's 「No.」 against 「No.1」, one region over
two OCR lines), 4 differed only in script (simplified OCR against traditional readings) and
3 really differed. A chart read by OCR at scale 1 and again at 2 counted every line twice and
was refused as over capacity.
"""
import pytest
from test_document_recovery import RecoveryModel, setup
from test_image_recovery import parts

from app.pipeline.image_evidence import (
    one_ocr_observation,
    resolve_image_evidence,
    select_agreeing_readings,
)
from app.pipeline.image_reading import reconcile_image_parts


def _reading(text, bbox, provider="paddleocr"):
    return {"text": text, "bbox": bbox, "provider": provider}


def test_containment_and_script_agree_and_the_longer_reading_is_kept():
    kept, disputed = select_agreeing_readings([
        _reading("No.", [100, 100, 300, 200]), _reading("No.1", [100, 100, 320, 200], "vlm_primary"),
        _reading("晴諾米凸轮", [100, 400, 600, 500]), _reading("晴諾米凸輪", [100, 400, 600, 500], "vlm_primary")])
    assert [k["text"] for k in kept] == ["No.1", "晴諾米凸轮"] and not disputed


def test_a_real_disagreement_holds_back_its_region_only():
    kept, disputed = select_agreeing_readings([
        _reading("著作權歸屬發報", [0, 900, 1000, 990], "vlm_primary"),
        _reading("著作權歸屬發聲", [0, 905, 1000, 995], "vlm_independent"),
        _reading("晴諾米", [0, 0, 400, 100]),
        _reading("著作權歸屬", [0, 910, 1000, 990])])
    assert [k["text"] for k in kept] == ["晴諾米"]
    assert disputed == [[0, 900, 1000, 995]]


def test_one_ocr_observation_per_picture_revision():
    def observation(sha, scale, region, lines):
        return {"source": {"sha256": sha}, "bbox": region,
                "content": {"region": region, "scale": scale, "lines": [{"text": "x"}] * lines}}
    whole1, whole2 = observation("a", 1, [0, 0, 1000, 1000], 59), observation("a", 2, [0, 0, 1000, 1000], 64)
    crop = observation("a", 4, [0, 0, 500, 500], 20)
    stale = observation("b", 2, [0, 0, 1000, 1000], 70)
    assert one_ocr_observation([whole1, crop, whole2, stale], "a") == [whole2]
    assert one_ocr_observation([stale], "a") == []


@pytest.mark.asyncio
async def test_a_conflict_becomes_a_gap_instead_of_failing_the_picture(tmp_path):
    class Select(RecoveryModel):
        async def query(self, task, payload, image, schema):
            rows = [{"source_ref": c["source_ref"], "decision": "supported", "reason": "Pixels"}
                    for c in payload["candidates"]]
            return {"selections": rows, "accounted_for": True, "observation": "Needs source audit"}, 10

    ir, _, runner = await setup(tmp_path, Select())
    block = ir.blocks[0]
    source = runner.inventory.image(0)
    for scale in (1, 2):
        runner.evidence.add(provider="paddleocr", kind="transcription", page=0, block_id=block.block_id,
                            source=source, content={"region": [0, 0, 1000, 1000], "scale": scale, "lines": [
                                {"line_id": "line0", "text": "Shop B", "score": .7, "bbox": [0, 0, 1000, 200]}]})
    primary, independent = parts("Shop A", kind="photo"), parts("Shop A", kind="photo")
    candidate, _ = await resolve_image_evidence(runner, block, 0, source, primary, independent, {},
                                                reconcile_image_parts(primary, independent), tmp_path)
    assert not candidate["text_regions"]
    assert [g["reason"] for g in candidate["gaps"]] == ["image_text_readings_conflict"]


@pytest.mark.asyncio
async def test_a_table_keeps_its_supported_cells_beside_one_disputed_cell(tmp_path):
    """A text-dependent picture lost every reading over one disputed cell (2026-10-16)."""
    class Select(RecoveryModel):
        async def query(self, task, payload, image, schema):
            rows = [{"source_ref": c["source_ref"], "decision": "supported", "reason": "Pixels"}
                    for c in payload["candidates"]]
            return {"selections": rows, "accounted_for": True, "observation": "Needs source audit"}, 10

    ir, _, runner = await setup(tmp_path, Select())
    block = ir.blocks[0]
    source = runner.inventory.image(0)
    runner.evidence.add(provider="paddleocr", kind="transcription", page=0, block_id=block.block_id,
                        source=source, content={"region": [0, 0, 1000, 1000], "scale": 2, "lines": [
                            {"line_id": "line0", "text": "Shop B", "score": .7, "bbox": [0, 0, 1000, 200]},
                            {"line_id": "line1", "text": "晴諾米", "score": .9, "bbox": [0, 300, 1000, 500]}]})
    regions = [{"region_id": "name", "bbox": [0, 300, 1000, 500], "readable": True, "text": "晴諾米"},
               {"region_id": "sign", "bbox": [0, 0, 1000, 200], "readable": True, "text": "Shop A"}]
    primary = parts(kind="table", text_regions=regions)
    independent = parts(kind="table", text_regions=regions)
    candidate, _ = await resolve_image_evidence(runner, block, 0, source, primary, independent, {},
                                                reconcile_image_parts(primary, independent), tmp_path)
    assert [r["text"] for r in candidate["text_regions"]] == ["晴諾米"]
    assert [g["reason"] for g in candidate["gaps"]] == ["image_text_readings_conflict"]
    assert candidate["text_dependent"] and not candidate["visual_facts"]
