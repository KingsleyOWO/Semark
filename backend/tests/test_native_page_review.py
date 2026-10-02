"""Source review certifies plain native-Word pages from the text layer, without a model call.

The model audit still decides every page the text layer cannot prove.
"""

import json
from unittest.mock import AsyncMock

import fitz
import pytest

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.document_review import DocumentReview
from app.pipeline.final_delivery import finalize_delivery
from app.pipeline.stages.parse import ParseStage
from tests.test_document_review import AuditModel, config, write_delivery


def _word_fixture(tmp_path, monkeypatch, printed_pages, texts):
    pdf_path = tmp_path / "rendered.pdf"
    with fitz.open() as pdf:
        for lines in printed_pages:
            page = pdf.new_page()
            for row, line in enumerate(lines):
                page.insert_text((60, 60 + 24 * row), line, fontname="china-t", fontsize=11)
        pdf.save(pdf_path)
    word = tmp_path / "report.docx"
    word.write_bytes(b"word conversion fixture")
    monkeypatch.setattr(ParseStage, "_convert_office_to_pdf", AsyncMock(return_value=(pdf_path, None)))
    ir = DocumentIR(
        doc_id="doc", run_id="run", page_kind="logical",
        source=SourceInfo(path=str(word), ext="docx", sha256="word", size_bytes=1),
        engine=EngineInfo(name="native_docx", backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[Block(block_id=f"b{i}", type=BlockType.TEXT, page_idx=0, reading_order=i,
                      payload={"text": text, "word_source": {"source": "native_docx", "block_index": i}})
                for i, text in enumerate(texts)],
    )
    write_delivery(tmp_path, ir)
    return ir, word


@pytest.mark.asyncio
async def test_native_word_pages_are_verified_without_a_model_call(tmp_path, monkeypatch):
    ir, word = _word_fixture(tmp_path, monkeypatch, [["一、計畫摘要", "本計畫協助企業建立品牌識別。"],
                                                     ["二、執行成果", "完成品牌手冊與官網改版。"]],
                             ["一、計畫摘要", "本計畫協助企業建立品牌識別。", "二、執行成果", "完成品牌手冊與官網改版。"])
    model = AuditModel()

    output, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=word)

    assert model.calls == []
    assert report["summary"]["verified_pages"] == 2
    assert report["summary"]["verified_blocks"] == 4
    assert {p["reason"] for p in report["pages"]} == {"native_word_text_layer_match"}
    evidence = [i for p in report["pages"] for i in p["images"] if i["label"].startswith("Word-rendered PDF text")]
    assert len(evidence) == 2
    assert json.loads((tmp_path / evidence[0]["path"]).read_text())["method"] == "native_word_text_layer"
    assert finalize_delivery(output, tmp_path, config())["auto_rag_ready"]


@pytest.mark.asyncio
async def test_page_the_text_layer_cannot_prove_still_gets_the_model_audit(tmp_path, monkeypatch):
    ir, word = _word_fixture(tmp_path, monkeypatch, [["結案報告書", "合約編號：示範001"], ["二、執行成果"]],
                             ["結案報告書", "二、執行成果"])
    model = AuditModel()

    _, report = await DocumentReview(config(), model, model).run(ir, tmp_path, source_path=word)

    assert [call[1]["page_idx"] for call in model.calls] == [0]
    pages = {p["page_idx"]: p for p in report["pages"]}
    assert pages[1]["reason"] == "native_word_text_layer_match"
    assert pages[0]["reason"] != "native_word_text_layer_match"


@pytest.mark.asyncio
async def test_review_switched_off_certifies_nothing(tmp_path, monkeypatch):
    ir, word = _word_fixture(tmp_path, monkeypatch, [["一、計畫摘要"]], ["一、計畫摘要"])
    model = AuditModel()

    _, report = await DocumentReview(config(mode="off"), model, model).run(ir, tmp_path, source_path=word)

    assert report["summary"]["verified_pages"] == 0
