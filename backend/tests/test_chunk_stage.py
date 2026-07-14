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


def test_image_block_chunk_carries_figure_retrieval_text_not_placeholder(tmp_path):
    """Default block chunks weave the VLM figure retrieval text (searchable)
    instead of the bare ``[Image: caption]`` placeholder, reading the
    ``assets_index.jsonl`` the package stage writes before chunking runs."""
    run_path = tmp_path / "run"
    outputs = run_path / "outputs"
    outputs.mkdir(parents=True)
    # Package stage writes assets_index.jsonl with per-figure retrieval_text keyed by block_id.
    (outputs / "assets_index.jsonl").write_text(
        json.dumps(
            {
                "type": "figure_asset",
                "asset_id": "fig0000",
                "doc_id": "doc",
                "run_id": "run",
                "title": "會議室環控面板",
                "page_idx": 0,
                "asset_path": "assets/figures/fig0000.jpg",
                "block_id": "img0",
                "retrieval_text": "會議室環控面板\n投影機 開關\nHDMI 訊號切換\n黑幕 升降控制",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    document_ir = DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="208.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="h0",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "208 會議室操作", "text_level": 1},
            ),
            Block(
                block_id="img0",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "figures/fig0000.jpg", "caption": "控制面板"},
            ),
        ],
    )

    result = asyncio.run(ChunkStage().run("doc", "run", document_ir, run_path))

    assert result.success
    chunks = [
        json.loads(line)
        for line in (outputs / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_chunks = [c for c in chunks if "img0" in c["block_ids"]]
    assert image_chunks, "expected a chunk covering the image block"
    content = "\n".join(c["content"] for c in image_chunks)
    # Figure information is now retrievable as body text.
    assert "投影機" in content
    assert "HDMI" in content
    assert "黑幕" in content
    # The impoverished placeholder no longer stands in for the figure.
    assert "[Image:" not in content
    # No broken internal image link leaks into the searchable body.
    assert "asset://" not in content


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
