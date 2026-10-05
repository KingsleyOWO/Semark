"""When the review's own reading disagrees with the delivery, a different reader settles the lines.

Lumio v20.1: the second model reading "corrected" the slide's 「Alphabet Samles」
(the author's typo, faithfully drafted) to 「Samples」 and read 「把散落的日常」 as
「日常生活」, failing correct drafts. PaddleOCR, which shares none of the vision
model's habits, read both exactly as drafted. It is a third vote, not truth: a
line it cannot settle stays unverified.
"""
import json

import pytest
from PIL import Image
from test_document_review import AuditModel, config, fixture, write_delivery

from app.models.document_ir import BlockType
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.line_adjudication import adjudicate
from app.pipeline.review_sources import SourceInventory

DELIVERED = ("右列為晴諾米拼圖視覺識別系統之主要英文字體。\nAlphabet Samles\nABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
             "把尋常的每一天，拼成更自在、更明亮的自己；把散落的日常，拼成你想要的生活。")


def test_ocr_agreeing_with_the_delivery_overrules_the_second_reading():
    verdicts = adjudicate(["Alphabet Samples", "把尋常的每一天，拼成更自在、更明亮的自己；把散落的日常生活，拼成你想要的生活。"],
                          DELIVERED, ["Alphabet Samles", "把尋常的每一天，拼成更自在、更明亮的自己；把散落的日", "常，拼成你想要的生活。"])

    assert [v["verdict"] for v in verdicts] == ["delivery", "delivery"]


def test_ocr_agreeing_with_the_second_reading_keeps_the_disagreement():
    assert adjudicate(["Alphabet Samples"], DELIVERED, ["Alphabet Samples"])[0]["verdict"] == "reading"


def test_a_line_missing_from_the_delivery_that_ocr_sees_is_not_overruled():
    assert adjudicate(["值得信任的"], DELIVERED, ["值得信任的", "Alphabet Samles"])[0]["verdict"] == "reading"


@pytest.mark.parametrize("ocr", [["Alphabet Sanmlcs"], ["LUMIO Brand Guidelines", "87"], []])
def test_garbled_or_absent_ocr_settles_nothing(ocr):
    delivered = DELIVERED + "\nLUMIO Brand Guidelines | 07"
    verdicts = adjudicate(["Alphabet Samples", "LUMIO Brand Guidelines | 01"], delivered, ocr)
    assert {v["verdict"] for v in verdicts} == {"unresolved"}


def _picture_page(tmp_path, monkeypatch, ocr_lines):
    ir, source = fixture(tmp_path)
    block = ir.blocks[0]
    block.type = BlockType.IMAGE
    block.payload = {"caption": DELIVERED}
    write_delivery(tmp_path, ir)
    original = tmp_path / "agentic/evidence/slide.png"
    original.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (700, 360), "white").save(original)
    prepare = SourceInventory.prepare

    async def with_picture(self, **kwargs):
        await prepare(self, **kwargs)
        self.word_images = {block.block_id: original}
    monkeypatch.setattr(SourceInventory, "prepare", with_picture)
    from app.adapters import ocr

    async def read_ocr(source, region, scale, folder, *, timeout=90):
        return {"status": "ok", "lines": [{"text": t, "score": 0.98} for t in ocr_lines], "version": "test",
                "detector": "test", "recognizer": "test", "device": "cpu", "source_sha256": "x",
                "region": region, "scale": scale}
    monkeypatch.setattr(ocr, "read_ocr", read_ocr)
    monkeypatch.setattr(ocr, "ocr_python", lambda: "/usr/bin/python3")

    class Model(AuditModel):
        async def query(self, task, payload, image, schema):
            if task == "verify_native_reading":
                self.calls.append((task, payload, image))
                return {"items": [{"source_id": "image0", "kind": "transcription", "complete": True,
                                   "text": DELIVERED.replace("Samles", "Samples"), "observation": "Slide.",
                                   "image_type": "document", "relationships": [], "relationships_complete": True}]}, 10
            value, tokens = await super().query(task, payload, image, schema)
            value["blocks"][0].update(context_complete=True)
            return value, tokens
    return ir, source, Model()


@pytest.mark.asyncio
async def test_a_picture_the_ocr_vote_settles_is_verified_with_hashed_evidence(tmp_path, monkeypatch):
    ir, source, model = _picture_page(tmp_path, monkeypatch, DELIVERED.splitlines())

    output, report = await DocumentReview(config(recovery_enabled=False), model, model).run(ir, tmp_path, source_path=source)

    assert report["pages"][0]["status"] == "verified"
    proof = [i for i in report["pages"][0]["images"] if i["label"].startswith("PaddleOCR line adjudication")]
    assert len(proof) == 1
    assert json.loads((tmp_path / proof[0]["path"]).read_text())["adjudications"][0]["verdict"] == "delivery"
    assert finalize_delivery(output, tmp_path, config())["review"]["evidence_valid"]


@pytest.mark.asyncio
async def test_a_picture_the_ocr_vote_cannot_settle_stays_unverified(tmp_path, monkeypatch):
    ir, source, model = _picture_page(tmp_path, monkeypatch, ["Alphabet Samples"])

    _, report = await DocumentReview(config(recovery_enabled=False), model, model).run(ir, tmp_path, source_path=source)

    assert report["pages"][0]["status"] == "needs_review"
    assert report["pages"][0]["image_issues"][0]["failure_kind"] == "reading_disagreement"


@pytest.mark.asyncio
async def test_a_picture_with_more_lines_than_one_ocr_pass_is_read_in_overlapping_bands(tmp_path, monkeypatch):
    # A sticky-note wall of 148 lines exceeds one PaddleOCR response; bands overlap so no line is cut.
    from types import SimpleNamespace

    from app.adapters import ocr

    regions = []

    async def read_ocr(source, region, scale, folder, *, timeout=90):
        regions.append(region)
        if region == [0, 0, 1000, 1000]:
            raise ValueError("ocr_output_capacity_exceeded: select a smaller region")
        return {"lines": [{"text": f"band {region[1]}"}]}
    monkeypatch.setattr(ocr, "read_ocr", read_ocr)
    monkeypatch.setattr(ocr, "ocr_python", lambda: "/usr/bin/python3")
    picture = tmp_path / "wall.png"
    Image.new("RGB", (1190, 1476), "white").save(picture)
    review = DocumentReview(config())
    review.state = {"usage": {"model_calls": 0, "seconds": 0.0}}
    review.state_path = tmp_path / "state.json"
    review.limits = config().agentic

    lines = await review._ocr_lines(picture, SimpleNamespace(directory=tmp_path))

    assert regions[0] == [0, 0, 1000, 1000] and len(regions) == 5
    assert all(b[1] < a[3] for a, b in zip(regions[1:], regions[2:], strict=False))  # consecutive bands overlap
    assert lines == [f"band {r[1]}" for r in regions[1:]]
