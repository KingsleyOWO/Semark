"""Regression contracts for mixed documents and actual delivered content."""
import json

import pytest

from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.content_coverage import TextCoverage
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.structured_rag import (
    _collect_supporting_page_sections,
    build_form_documents_rag,
    is_form_like_document,
)


def document(blocks, ext="pdf", page_kind="unknown"):
    return DocumentIR(
        doc_id="doc", run_id="run",
        source=SourceInfo(path=f"report.{ext}", ext=ext, sha256="source", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        page_kind=page_kind,
        pages=[PageInfo(page_idx=p) for p in sorted({b.page_idx for b in blocks})],
        blocks=blocks,
    )


def block(identifier, text, page=0, kind=BlockType.TEXT):
    key = "caption" if kind == BlockType.IMAGE else "table_body" if kind == BlockType.TABLE else "text"
    return Block(block_id=identifier, type=kind, page_idx=page, payload={key: text})


@pytest.mark.parametrize("ext", ["docx", "pdf", "doc", "xlsx"])
def test_repeated_budget_terms_do_not_make_a_report_a_form(ext):
    ir = document([block("body", "計畫金額與合計由各組核定。年度報告說明執行成果。" * 50)], ext)
    assert not is_form_like_document(ir)
    assert build_form_documents_rag(ir, {}).plan.document_type == "generic_document"


def test_supporting_sections_keep_short_budget_image_and_long_body():
    ir = document([
        block("budget", "數位行銷預算：1,100,000 元。", 0),
        block("chart", "工作 A 在一月至三月執行，四月驗收。", 1, BlockType.IMAGE),
        block("long", "執行計畫及年度成果。" * 700 + "最末重要結論：通過驗收。", 2),
    ])
    sections = _collect_supporting_page_sections(ir, [], "zh-TW")
    text = "\n".join(s["content"] for s in sections)
    assert "1,100,000" in text
    assert "一月至三月" in text
    assert "最末重要結論" in text
    assert {bid for section in sections for bid in section["block_ids"]} == {"budget", "chart", "long"}


def test_form_on_page_does_not_cover_other_content_on_same_page():
    ir = document([block("budget", "本計畫另編列數位行銷預算 1,100,000 元。", 0)])
    output = build_form_documents_rag(ir, {
        "form_page_0000": {
            "kind": "form_asset", "input": {"page_idx": 0},
            "output": {
                "title": "人員申請表", "document_type": "form",
                "field_schema": [{"name": "姓名", "type": "text"}, {"name": "簽章", "type": "signature"}],
                "retrieval_text": "填寫姓名與簽章後提出申請。",
            },
        },
    })
    assert "1,100,000" in output.rag_markdown
    assert any("1,100,000" in c["content"] for c in output.chunks)


@pytest.mark.asyncio
async def test_partial_structured_chunks_cannot_replace_whole_document(tmp_path):
    ir = document([
        block("form", "姓名：王小明", 0),
        block("budget", "數位行銷預算 1,100,000 元", 73),
        block("figure", "工作 A 一月至三月，工作 B 四月至六月。", 74, BlockType.IMAGE),
    ])
    out = tmp_path / "outputs"
    out.mkdir()
    structured = {
        "chunk_id": "form_chunk", "doc_id": "doc", "run_id": "run", "content": "姓名：王小明",
        # Even lying references must not authorize dropping missing content.
        "block_ids": ["form", "budget", "figure"], "page_indices": [0, 73, 74],
    }
    (out / "structured_chunks.jsonl").write_text(json.dumps(structured, ensure_ascii=False))
    result = await ChunkStage(PipelineConfig()).run("doc", "run", ir, tmp_path)
    assert result.success
    delivered = "\n".join(chunk.content for chunk in result.chunks)
    assert "1,100,000" in delivered
    assert "一月至三月" in delivered
    assert delivered.count("姓名：王小明") == 1
    assert len({c.chunk_id for c in result.chunks}) == len(result.chunks)


def test_missing_number_is_not_hidden_by_large_text_overlap():
    source = "年度計畫執行說明與成果。" * 100 + "預算 1,100,000 元。"
    assert not TextCoverage([source.replace("1,100,000", "100,000")]).covers(source)


@pytest.mark.parametrize("original,changed", [("100", "1100"), ("-100", "100"), ("1,100,000", "100,000")])
def test_numeric_substring_is_not_the_same_amount(original, changed):
    source = "執行成果。" * 200 + f"預算 {original} 元。"
    assert not TextCoverage([source.replace(original, changed)]).covers(source)


def test_native_word_logical_page_is_not_physical_page():
    ir = document([block("b", "申請人：姓名：")], "docx", "logical")
    assert not ir.has_physical_pages()
    assert DocumentIR.from_dict(ir.to_dict()).page_kind == "logical"



def test_numeric_inequality_is_not_an_html_tag():
    source = "自由餘氯 <0.01 mg/L，總三氯甲烷 0.00117，硬度 121。"
    delivered = source + "\n<table><tr><td>補充資料</td></tr></table>"
    assert TextCoverage([delivered]).covers(source)
    assert not TextCoverage(["自由餘氯 0.1 mg/L"]).covers("自由餘氯 <0.01 mg/L")


@pytest.mark.asyncio
async def test_large_structured_chunk_is_bounded_without_dropping_content(tmp_path):
    from app.pipeline.stages.chunk import estimate_tokens
    text = "".join(f"第{i}項執行計畫與年度成果，預算為{i + 100}元。\n" for i in range(300))
    ir = document([block("long", text)])
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "structured_chunks.jsonl").write_text(json.dumps({
        "chunk_id": "long", "doc_id": "doc", "run_id": "run",
        "content": text, "block_ids": ["long"], "page_indices": [0],
    }, ensure_ascii=False))
    config = PipelineConfig()
    config.package.chunk_max_tokens = 256
    result = await ChunkStage(config).run("doc", "run", ir, tmp_path)
    assert result.success, result.error
    assert len(result.chunks) > 1
    assert all(estimate_tokens(c.content) <= 256 for c in result.chunks)
    assert len({c.chunk_id for c in result.chunks}) == len(result.chunks)
    assert all(c.block_ids == ["long"] for c in result.chunks)
    coverage = TextCoverage(c.content for c in result.chunks)
    assert all(coverage.covers(line) for line in text.splitlines())
