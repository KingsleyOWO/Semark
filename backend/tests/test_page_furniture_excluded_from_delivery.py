"""Page furniture must not reach the delivered surfaces.

Live evidence (2026-08-10, 100-document store): every rag.md carried the
journal's running heads, volume lines and bare page numbers as body
paragraphs — 1,826 lines, median 20 per document — and 851 of 1,843 chunks
contained at least one of them. ``is_page_furniture`` already existed and was
unit-tested, but no production code ever called it, and MinerU only types a
fraction of the furniture (the rest arrives as plain ``text``), so both halves
are covered here: detection at normalize time and exclusion at delivery.
"""

import asyncio
from types import SimpleNamespace

from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.pipeline.quality_gate import _check_authored_text_survival
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.normalize import NormalizeStage, is_page_furniture
from app.pipeline.stages.package import PackageStage

BODY = (
    "國內生產方面，受惠於人工智慧與雲端服務需求續強，資訊電子產業成為支撐國內生產的"
    "核心動能，工業生產指數較上年同期成長一成六，其中製造業年增率達一成七。"
)


def _ir(blocks: list[Block], pages: int = 1) -> DocumentIR:
    return DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=idx) for idx in range(pages)],
        blocks=blocks,
    )


def _furniture(block_id: str, text: str, page_idx: int, order: int) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=page_idx,
        bbox_norm=[80, 960, 200, 985],
        reading_order=order,
        payload={"text": text, "text_level": 0, "origin": "page_furniture"},
    )


def _body(block_id: str, text: str, page_idx: int, order: int) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=page_idx,
        bbox_norm=[87, 400, 480, 600],
        reading_order=order,
        payload={"text": text, "text_level": 0},
    )


# ---------------------------------------------------------------------------
# Delivery surfaces honour the tag
# ---------------------------------------------------------------------------


def test_tagged_furniture_is_not_rendered_into_rag_md():
    document_ir = _ir(
        [
            _furniture("f0", "示範景氣報導", 0, 0),
            _body("b0", BODY, 0, 1),
            _furniture("f1", "第49卷第1期 115年1月", 0, 2),
            _furniture("f2", "21", 0, 3),
        ]
    )

    source_md, _ = PackageStage()._render_rag_md(
        document_ir=document_ir, asset_map={}, enrichments={}
    )

    assert "核心動能" in source_md
    assert "示範景氣報導" not in source_md
    assert "第49卷第1期" not in source_md
    assert "\n21\n" not in source_md


def test_tagged_furniture_is_not_chunked(tmp_path):
    document_ir = _ir(
        [
            _furniture("f0", "示範經濟研究月刊", 0, 0),
            _body("b0", BODY, 0, 1),
            _furniture("f1", "22", 0, 2),
        ]
    )

    result = asyncio.run(ChunkStage().run("doc", "run", document_ir, tmp_path / "run"))

    assert result.success
    body = (tmp_path / "run" / "outputs" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "核心動能" in body
    assert "示範經濟研究月刊" not in body


def test_document_with_only_furniture_still_produces_no_crash(tmp_path):
    """Guard the degenerate case: excluding every block must not explode."""
    document_ir = _ir([_furniture("f0", "示範經濟研究月刊", 0, 0)])

    source_md, _ = PackageStage()._render_rag_md(
        document_ir=document_ir, asset_map={}, enrichments={}
    )
    result = asyncio.run(ChunkStage().run("doc", "run", document_ir, tmp_path / "run"))

    assert "示範經濟研究月刊" not in source_md
    assert result.success


# ---------------------------------------------------------------------------
# Detection: MinerU types only some furniture, the rest arrives as plain text
# ---------------------------------------------------------------------------


def test_running_head_repeated_in_the_margin_is_detected():
    """The journal title runs across every page top; MinerU types it 'text'."""
    title = "2025~2026年示範經濟景氣回顧與展望"
    blocks = [
        # The real title, in the body of page 0 — must survive.
        Block(
            block_id="b000000",
            type=BlockType.TEXT,
            page_idx=0,
            bbox_norm=[127, 189, 600, 230],
            reading_order=0,
            payload={"text": title, "text_level": 1},
        )
    ]
    for page in range(1, 4):
        blocks.append(
            Block(
                block_id=f"b{page:06d}",
                type=BlockType.TEXT,
                page_idx=page,
                bbox_norm=[593, 110, 900, 132],
                reading_order=page,
                payload={"text": title, "text_level": 0},
            )
        )

    tagged = NormalizeStage()._tag_layout_furniture(blocks, page_count=4)
    by_id = {block.block_id: block for block in tagged}

    assert is_page_furniture(by_id["b000000"]) is False, "the body title must survive"
    assert all(is_page_furniture(by_id[f"b{page:06d}"]) for page in range(1, 4))


def test_bare_page_number_in_the_margin_is_detected():
    blocks = [
        Block(
            block_id=f"b{page:06d}",
            type=BlockType.TEXT,
            page_idx=page,
            bbox_norm=[885, 960, 905, 980],
            reading_order=page,
            payload={"text": str(20 + page), "text_level": 0},
        )
        for page in range(3)
    ]

    tagged = NormalizeStage()._tag_layout_furniture(blocks, page_count=3)

    assert all(is_page_furniture(block) for block in tagged)


def test_a_year_inside_body_prose_is_not_mistaken_for_a_page_number():
    """Anti-over-correction: only bare numbers sitting in the margin qualify."""
    blocks = [
        Block(
            block_id="b000000",
            type=BlockType.TEXT,
            page_idx=0,
            bbox_norm=[87, 400, 480, 600],
            reading_order=0,
            payload={"text": BODY, "text_level": 0},
        ),
        # A short numeric label inside the body column, e.g. a figure index.
        Block(
            block_id="b000001",
            type=BlockType.TEXT,
            page_idx=0,
            bbox_norm=[300, 500, 330, 520],
            reading_order=1,
            payload={"text": "2025", "text_level": 0},
        ),
    ]

    tagged = NormalizeStage()._tag_layout_furniture(blocks, page_count=1)

    assert not any(is_page_furniture(block) for block in tagged)


def test_prose_repeated_in_the_margin_band_is_kept():
    """A long paragraph that happens to sit low on the page is not furniture."""
    blocks = [
        Block(
            block_id=f"b{page:06d}",
            type=BlockType.TEXT,
            page_idx=page,
            bbox_norm=[87, 905, 480, 985],
            reading_order=page,
            payload={"text": BODY, "text_level": 0},
        )
        for page in range(3)
    ]

    tagged = NormalizeStage()._tag_layout_furniture(blocks, page_count=3)

    assert not any(is_page_furniture(block) for block in tagged)


def test_text_appearing_on_a_single_page_only_is_kept():
    """A one-off footer line (DOI, source note) carries information; keep it."""
    blocks = [
        Block(
            block_id="b000000",
            type=BlockType.TEXT,
            page_idx=0,
            bbox_norm=[670, 960, 950, 980],
            reading_order=0,
            payload={"text": "DOI:10.12345/DEMO.202601_49(1).0004", "text_level": 0},
        ),
        _body("b000001", BODY, 0, 1),
    ]

    tagged = NormalizeStage()._tag_layout_furniture(blocks, page_count=3)

    assert not any(is_page_furniture(block) for block in tagged)


# ---------------------------------------------------------------------------
# The completeness gate must agree that furniture was dropped on purpose
# ---------------------------------------------------------------------------


def test_authored_text_survival_ignores_page_furniture():
    """Excluding furniture must not read as silent block-dropping."""
    prose = [
        "第一段內容說明本文的研究背景與問題意識，並交代資料來源與涵蓋期間。",
        "第二段說明分析方法，包含指標選取的理由以及與既有文獻的差異之處。",
        "第三段整理主要發現，並就政策意涵提出具體的建議方向供後續參考。",
    ]
    blocks = [_body(f"b{idx}", text, 0, idx) for idx, text in enumerate(prose)]
    blocks.append(_furniture("f0", "第49卷第1期 115年1月，示範經濟研究月刊發行", 0, 9))
    structured_output = SimpleNamespace(plan=SimpleNamespace(document_type="generic_document"))

    issues = _check_authored_text_survival(_ir(blocks), structured_output, "\n".join(prose))

    assert issues == []
