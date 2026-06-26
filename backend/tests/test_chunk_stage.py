import asyncio
import json

from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.pipeline.stages.chunk import ChunkStage


def test_structured_chunks_replace_raw_block_chunks(tmp_path):
    run_path = tmp_path / "run"
    outputs = run_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "structured_chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "sr_repair_form_0001_0000",
                "doc_id": "doc:form:0001",
                "run_id": "run",
                "view": "semantic_repair",
                "content": "LLM/VLM final semantic chunk with NT$100,000.",
                "block_ids": ["semantic_repair:form_0001"],
                "page_indices": [1],
                "attachments": [],
                "metadata": {"auto_rag_ready": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (outputs / "document_plan.json").write_text(
        json.dumps({"document_type": "form_collection"}),
        encoding="utf-8",
    )
    document_ir = DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="raw0",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Raw MinerU fallback that must not enter final chunks."},
            )
        ],
    )

    result = asyncio.run(ChunkStage().run("doc", "run", document_ir, run_path))

    assert result.success
    chunks_text = (outputs / "chunks.jsonl").read_text(encoding="utf-8")
    assert "LLM/VLM final semantic chunk" in chunks_text
    assert "Raw MinerU fallback" not in chunks_text
    assert len(chunks_text.splitlines()) == 1


def test_empty_structured_chunks_fall_back_to_raw_block_chunks(tmp_path):
    run_path = tmp_path / "run"
    outputs = run_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "document_plan.json").write_text(
        json.dumps({"document_type": "form_document"}),
        encoding="utf-8",
    )
    (outputs / "structured_chunks.jsonl").write_text("", encoding="utf-8")
    (outputs / "semantic_repair.json").write_text(
        json.dumps({"fallback_count": 1, "items": [{"reason": "repaired_markdown_not_usable"}]}),
        encoding="utf-8",
    )
    document_ir = DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="fallback-form.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="raw0",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Raw MinerU fallback is the last-resort non-empty output."},
            )
        ],
    )

    result = asyncio.run(ChunkStage().run("doc", "run", document_ir, run_path))

    assert result.success
    chunks_text = (outputs / "chunks.jsonl").read_text(encoding="utf-8")
    assert "Raw MinerU fallback is the last-resort non-empty output" in chunks_text
    assert len(chunks_text.splitlines()) == 1
