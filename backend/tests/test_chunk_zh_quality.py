"""Level 1 zh-TW chunk quality: CJK-aware token estimation, heading paths,
small-section merge, and HTML-table-to-markdown conversion."""

import asyncio

from app.config import PackageConfig, PipelineConfig
from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.pipeline.stages import chunk as chunk_stage_module
from app.pipeline.stages.chunk import ChunkStage


def _text_block(block_id: str, text: str, level: int = 0) -> Block:
    payload: dict = {"text": text}
    if level:
        payload["text_level"] = level
    return Block(block_id=block_id, type=BlockType.TEXT, page_idx=0, payload=payload)


def _table_block(block_id: str, table_body: str) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TABLE,
        page_idx=0,
        payload={"table_body": table_body},
    )


def _make_ir(blocks: list[Block]) -> DocumentIR:
    return DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=blocks,
    )


def _run_chunks(tmp_path, blocks: list[Block], config: PipelineConfig | None = None):
    stage = ChunkStage(config=config)
    result = asyncio.run(stage.run("doc", "run", _make_ir(blocks), tmp_path / "run"))
    assert result.success, result.error
    return result.chunks


# ---------------------------------------------------------------------------
# 1. CJK-aware token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_cjk_counts_two_thirds_of_chars():
    assert chunk_stage_module.estimate_tokens("春" * 300) == 200


def test_estimate_tokens_ascii_counts_quarter_of_chars():
    assert chunk_stage_module.estimate_tokens("a" * 400) == 100


def test_estimate_tokens_mixed_text_sums_both_rates():
    assert chunk_stage_module.estimate_tokens("春" * 150 + "a" * 200) == 150


def test_estimate_tokens_counts_cjk_punctuation_and_fullwidth_as_cjk():
    # U+3002 ideographic full stop, U+FF0C fullwidth comma, U+FF21 fullwidth A
    assert chunk_stage_module.estimate_tokens("。，Ａ") == 2
    # U+3400 CJK extension A
    assert chunk_stage_module.estimate_tokens("㐀" * 3) == 2


def test_estimate_tokens_empty_string_is_zero():
    assert chunk_stage_module.estimate_tokens("") == 0


def test_cjk_heavy_section_splits_below_embedder_budget(tmp_path):
    # Under the old len(text) // 3 estimate this whole section looked like
    # ~400 tokens and stayed one chunk of ~1200 CJK chars, which a 512-token
    # embedder silently truncates.
    blocks = [
        _text_block("h1", "第三章 請假", level=1),
        _text_block("h2", "第五條", level=2),
    ]
    for i in range(5):
        blocks.append(_text_block(f"p{i}", "假" * 240))

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk_stage_module.estimate_tokens(chunk.content) <= 512 + 64


# ---------------------------------------------------------------------------
# 2. Heading path metadata on every chunk
# ---------------------------------------------------------------------------


def test_every_chunk_gets_heading_path_metadata(tmp_path):
    blocks = [
        _text_block("h1", "第三章 請假", level=1),
        _text_block("b1", "假" * 200),
        _text_block("h2", "第五條", level=2),
        _text_block("b2", "條" * 200),
    ]

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) == 2
    assert chunks[0].metadata.get("heading_path") == ["第三章 請假"]
    assert chunks[1].metadata.get("heading_path") == ["第三章 請假", "第五條"]


def test_preamble_chunk_gets_empty_heading_path(tmp_path):
    blocks = [
        _text_block("p0", "緒" * 150),
        _text_block("h1", "總則", level=1),
        _text_block("b1", "則" * 150),
    ]

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) == 2
    assert chunks[0].metadata.get("heading_path") == []
    assert chunks[1].metadata.get("heading_path") == ["總則"]


def test_sibling_heading_replaces_stack_entry(tmp_path):
    blocks = [
        _text_block("h1", "第三章 請假", level=1),
        _text_block("b1", "假" * 150),
        _text_block("h2", "第五條", level=2),
        _text_block("b2", "五" * 150),
        _text_block("h3", "第六條", level=2),
        _text_block("b3", "六" * 150),
    ]

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) == 3
    assert chunks[2].metadata.get("heading_path") == ["第三章 請假", "第六條"]


def test_continuation_chunks_carry_heading_path_prefix_line(tmp_path):
    blocks = [
        _text_block("h1", "第三章 請假", level=1),
        _text_block("h2", "第五條", level=2),
    ]
    for i in range(5):
        blocks.append(_text_block(f"p{i}", "假" * 240))

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) >= 2
    assert chunks[0].content.startswith("# 第三章 請假")
    for chunk in chunks[1:]:
        first_line = chunk.content.splitlines()[0]
        assert first_line == "第三章 請假 > 第五條（續）"
    for chunk in chunks:
        assert chunk.metadata.get("heading_path") == ["第三章 請假", "第五條"]


# ---------------------------------------------------------------------------
# 3. Small-section merge
# ---------------------------------------------------------------------------


def test_consecutive_small_sections_merge_into_one_chunk(tmp_path):
    blocks = [
        _text_block("h1", "第一條", level=2),
        _text_block("b1", "一" * 30),
        _text_block("h2", "第二條", level=2),
        _text_block("b2", "二" * 30),
        _text_block("h3", "第三條", level=2),
        _text_block("b3", "三" * 30),
    ]

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) == 1
    for expected in ("第一條", "第二條", "第三條", "一" * 30, "二" * 30, "三" * 30):
        assert expected in chunks[0].content


def test_small_section_merge_respects_chunk_max_tokens(tmp_path):
    config = PipelineConfig(package=PackageConfig(chunk_max_tokens=60))
    blocks = [
        _text_block("h1", "第一條", level=2),
        _text_block("b1", "一" * 30),
        _text_block("h2", "第二條", level=2),
        _text_block("b2", "二" * 30),
        _text_block("h3", "第三條", level=2),
        _text_block("b3", "三" * 30),
    ]

    chunks = _run_chunks(tmp_path, blocks, config=config)

    assert len(chunks) == 2
    assert "第一條" in chunks[0].content
    assert "第二條" in chunks[0].content
    assert "第三條" in chunks[1].content


def test_heading_only_section_merges_into_following_section(tmp_path):
    blocks = [
        _text_block("h1", "第三章 請假", level=1),
        _text_block("h2", "第五條", level=2),
        _text_block("b1", "假" * 200),
    ]

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) == 1
    assert chunks[0].content.startswith("# 第三章 請假\n\n## 第五條")
    assert chunks[0].metadata.get("heading_path") == ["第三章 請假", "第五條"]


def test_trailing_heading_only_section_merges_backward(tmp_path):
    blocks = [
        _text_block("h1", "第一章 總則", level=1),
        _text_block("b1", "總" * 200),
        _text_block("h2", "附則", level=2),
    ]

    chunks = _run_chunks(tmp_path, blocks)

    assert len(chunks) == 1
    assert chunks[0].content.endswith("## 附則")


# ---------------------------------------------------------------------------
# 4. Table HTML -> markdown pipe tables
# ---------------------------------------------------------------------------

_TABLE_HTML = (
    "<table><tr><th>假別</th><th>日數</th></tr>"
    "<tr><td>事假</td><td>14日</td></tr></table>"
)


def test_table_html_becomes_markdown_pipe_table(tmp_path):
    chunks = _run_chunks(tmp_path, [_table_block("t0", _TABLE_HTML)])

    assert len(chunks) == 1
    content = chunks[0].content
    assert "| 假別 | 日數 |" in content
    assert "| --- | --- |" in content
    assert "| 事假 | 14日 |" in content
    assert "<table" not in content
    assert chunks[0].metadata.get("table_html") == [_TABLE_HTML]


def test_table_rowspan_colspan_repeat_values(tmp_path):
    html = (
        '<table><tr><th>假別</th><th colspan="2">說明</th></tr>'
        '<tr><td rowspan="2">事假</td><td>甲</td><td>乙</td></tr>'
        "<tr><td>丙</td><td>丁</td></tr></table>"
    )

    chunks = _run_chunks(tmp_path, [_table_block("t0", html)])

    content = chunks[0].content
    assert "| 假別 | 說明 | 說明 |" in content
    assert "| 事假 | 甲 | 乙 |" in content
    assert "| 事假 | 丙 | 丁 |" in content


def test_non_table_markup_tags_stripped(tmp_path):
    chunks = _run_chunks(tmp_path, [_table_block("t0", "<div>不是表格</div>")])

    assert chunks[0].content == "不是表格"
    assert "<div>" not in chunks[0].content
