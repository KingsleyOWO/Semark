"""A back-page book advert must not be delivered as article content.

These journals bind a publisher's advert into the right-hand column of the
article's **last** page: book title, price, publication month, a blurb, an
ordering hotline, cover shots, a QR code and partner logos. It has nothing to
do with the article, yet the whole column reached rag.md and the chunks.

Live evidence (2026-08-11, 167-document store): 32 documents carry one, 479
blocks in total, of which 105 are images that were sent to the VLM for
figure enrichment — paid captions for book covers and QR codes.

The fix has to be **column**-level, never page-level: on 141 of the 167 last
pages the right column holds something, and the *left* column of every
advert page still holds the article's ``■參考文獻`` / ``■注釋`` and the tail
of the prose (121-1,469 characters live). Dropping the page would delete the
references, so the tests below pin the left column down as hard as they pin
the advert.
"""

import asyncio
import json
from types import SimpleNamespace

from app.config import EnrichConfig, PipelineConfig
from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.pipeline.corpus_rules import RULES_PATH_ENV_VAR, reset_rules_cache
from app.pipeline.quality_gate import _check_authored_text_survival
from app.pipeline.stages.chunk import ChunkStage
from app.pipeline.stages.enrich import EnrichStage
from app.pipeline.stages.normalize import (
    NormalizeStage,
    is_non_content,
    is_page_furniture,
    is_promotional_insert,
)
from app.pipeline.stages.package import PackageStage

BODY = (
    "國內生產方面，受惠於人工智慧與雲端服務需求續強，資訊電子產業成為支撐國內生產的"
    "核心動能，工業生產指數較上年同期成長一成六，其中製造業年增率達一成七。"
)
REFERENCE_TAIL = "上述趨勢是否延續，仍有待後續觀察，本文將持續關注相關發展。"
REFERENCES = [
    "1.示範研究院(2026)，示範島能源轉型年度回顧，示範研究院出版。",
    "2.示範統計處(2026)，示範島工業生產統計月報，2026/03。",
    "3.示範能源署(2026)，示範島再生能源發展白皮書，2026/01。",
]

# Two independent commerce signals plus a blurb — what a bound-in advert says.
ADVERT_LINES = [
    "示範能源轉型全解析",
    "【示範島二版】",
    "售價：NT$500",
    "115年3月出版",
    "本書分析示範島能源轉型的政策脈絡，並借鏡各國經驗提出布局觀點。",
    "洽詢專線：(00)1234-5678",
]

# Column geometry of the live journals, in the 0-1000 ``bbox_norm`` space:
# the text column runs 90-480, the second column 550-900, so the gutter the
# detector has to *derive* sits near x=515.
LEFT_X = (90, 480)
RIGHT_X = (550, 900)


def _ir(blocks: list[Block], pages: int = 2) -> DocumentIR:
    return DocumentIR(
        doc_id="doc",
        run_id="run",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=idx) for idx in range(pages)],
        blocks=blocks,
    )


def _text(
    block_id: str,
    text: str,
    page_idx: int,
    order: int,
    x_range: tuple[int, int],
    y: int,
    origin: str | None = None,
) -> Block:
    payload: dict = {"text": text, "text_level": 0}
    if origin:
        payload["origin"] = origin
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=page_idx,
        bbox_norm=[x_range[0], y, x_range[1], y + 30],
        reading_order=order,
        payload=payload,
    )


def _image(block_id: str, page_idx: int, order: int, x_range: tuple[int, int], y: int) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.IMAGE,
        page_idx=page_idx,
        bbox_norm=[x_range[0], y, x_range[1], y + 120],
        reading_order=order,
        payload={"image_path": f"images/{block_id}.jpg", "caption": ""},
    )


def _document_with_back_page_advert(
    advert_lines: list[str] = ADVERT_LINES,
    left_x: tuple[int, int] = LEFT_X,
    right_x: tuple[int, int] = RIGHT_X,
    advert_page: int = 1,
    pages: int = 2,
    advert_images: int = 2,
) -> list[Block]:
    """Two-column journal: body on page 0, references + advert on the last page."""
    blocks: list[Block] = [
        _text("b000000", BODY, 0, 0, left_x, 200),
        _text("b000001", BODY, 0, 1, right_x, 200),
    ]
    order = 2
    # Left column of the advert page: the article's own tail and references.
    for idx, text in enumerate([REFERENCE_TAIL, "■參考文獻", *REFERENCES]):
        blocks.append(_text(f"l{idx:06d}", text, advert_page, order, left_x, 200 + idx * 60))
        order += 1
    # Right column: the bound-in advert, interleaved with its cover images.
    for idx, text in enumerate(advert_lines):
        blocks.append(_text(f"a{idx:06d}", text, advert_page, order, right_x, 220 + idx * 60))
        order += 1
    for idx in range(advert_images):
        blocks.append(_image(f"g{idx:06d}", advert_page, order, right_x, 620 + idx * 130))
        order += 1
    if pages > advert_page + 1:  # pad so ``advert_page`` is not the last page
        for extra in range(advert_page + 1, pages):
            blocks.append(_text(f"x{extra:06d}", BODY, extra, order, left_x, 200))
            order += 1
            blocks.append(_text(f"y{extra:06d}", BODY, extra, order, right_x, 200))
            order += 1
    return blocks


def _tag(blocks: list[Block]) -> dict[str, Block]:
    tagged = NormalizeStage()._tag_promotional_inserts(blocks)
    return {block.block_id: block for block in tagged}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_back_page_advert_column_is_tagged():
    by_id = _tag(_document_with_back_page_advert())

    assert all(is_promotional_insert(by_id[f"a{idx:06d}"]) for idx in range(len(ADVERT_LINES)))
    assert all(is_promotional_insert(by_id[f"g{idx:06d}"]) for idx in range(2))


def test_left_column_references_are_never_tagged():
    """The regression that rules page-level dropping out: references live here."""
    by_id = _tag(_document_with_back_page_advert())

    for block_id, block in by_id.items():
        if block_id.startswith("l"):
            assert not is_promotional_insert(block), block.payload["text"]
    assert not is_promotional_insert(by_id["l000001"])  # ■參考文獻
    assert not is_promotional_insert(by_id["l000000"])  # the prose the article ends on


def test_body_column_on_an_earlier_page_is_never_tagged():
    """Only the last page can carry the bound-in advert (0 exceptions live)."""
    blocks = _document_with_back_page_advert(advert_page=1, pages=3)

    by_id = _tag(blocks)

    assert not any(is_promotional_insert(block) for block in by_id.values())


def test_one_signal_is_not_enough():
    """A single commerce word can occur in prose; two independent ones cannot."""
    lines = [
        "示範能源轉型全解析",
        "【示範島二版】",
        "示範島的能源政策在近年出現明顯轉折，值得後續持續追蹤觀察其成效。",
        "洽詢專線：(00)1234-5678",
    ]

    by_id = _tag(_document_with_back_page_advert(advert_lines=lines))

    assert not any(is_promotional_insert(block) for block in by_id.values())


def test_a_short_column_is_not_enough():
    """Three blocks is a figure with a caption, not a bound-in advert."""
    lines = ["售價：NT$500", "115年3月出版", "洽詢專線：(00)1234-5678"]

    by_id = _tag(_document_with_back_page_advert(advert_lines=lines, advert_images=0))

    assert not any(is_promotional_insert(block) for block in by_id.values())


def test_the_journals_own_article_series_number_is_not_a_signal():
    """Anti-over-correction: the journal numbers its own articles 「系列3-6」.

    A bare ``系列\\s*\\d`` rule matched those and would have deleted real
    article columns, so it is deliberately absent from the ruleset.
    """
    lines = [
        "系列3-6",
        "系列1-4",
        "示範島的能源政策在近年出現明顯轉折，值得後續持續追蹤觀察其成效。",
        "本節整理主要發現，並就政策意涵提出具體的建議方向供後續參考。",
        "資料來源：示範研究院整理，2026年3月。",
        "註：本文觀點僅代表作者個人立場。",
    ]

    by_id = _tag(_document_with_back_page_advert(advert_lines=lines))

    assert not any(is_promotional_insert(block) for block in by_id.values())


def test_column_boundary_is_derived_from_the_page_not_hardcoded():
    """x≈520 is this corpus's gutter, not a setting: a narrower one must work."""
    blocks = _document_with_back_page_advert(left_x=(60, 300), right_x=(380, 960))

    by_id = _tag(blocks)

    assert all(is_promotional_insert(by_id[f"a{idx:06d}"]) for idx in range(len(ADVERT_LINES)))
    assert not is_promotional_insert(by_id["l000001"])


def test_single_column_document_is_left_alone():
    """No gutter to find: never guess a column split on a one-column page."""
    lines = [
        "示範能源轉型全解析",
        "售價：NT$500",
        "115年3月出版",
        "洽詢專線：(00)1234-5678",
    ]
    blocks = [
        _text(f"s{idx:06d}", text, 0, idx, (90, 900), 200 + idx * 60)
        for idx, text in enumerate([BODY, *lines])
    ]

    by_id = _tag(blocks)

    assert not any(is_promotional_insert(block) for block in by_id.values())


def test_running_head_inside_the_advert_column_keeps_its_own_tag():
    """The advert pass must not overwrite what the furniture pass decided."""
    blocks = _document_with_back_page_advert()
    blocks.append(
        _text("f000000", "示範經濟月刊 第49卷第3期", 1, 99, (700, 900), 60, origin="page_furniture")
    )

    by_id = _tag(blocks)

    assert is_page_furniture(by_id["f000000"])
    assert not is_promotional_insert(by_id["f000000"])
    assert is_non_content(by_id["f000000"])


def _with_ruleset(tmp_path, monkeypatch, ruleset: dict):
    path = tmp_path / "ruleset.json"
    path.write_text(json.dumps(ruleset, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv(RULES_PATH_ENV_VAR, str(path))
    reset_rules_cache()
    return path


def test_a_ruleset_without_the_key_falls_back_to_the_bundled_default(tmp_path, monkeypatch):
    """A corpus ruleset replaces the whole file, it does not layer onto it.

    Every deployed ruleset predates this key, so keying off "absent" is what
    stops the whole pass from silently becoming a no-op in production.
    """
    _with_ruleset(tmp_path, monkeypatch, {"document_markers": {"watermark_terms": []}})
    try:
        by_id = _tag(_document_with_back_page_advert())
    finally:
        reset_rules_cache()

    assert is_promotional_insert(by_id["a000002"])  # 售價：NT$500


def test_a_ruleset_with_an_empty_list_turns_the_pass_off(tmp_path, monkeypatch):
    """"Configured empty" is a real intent and must differ from "not configured"."""
    _with_ruleset(
        tmp_path, monkeypatch, {"document_markers": {"promotional_insert_patterns": []}}
    )
    try:
        by_id = _tag(_document_with_back_page_advert())
    finally:
        reset_rules_cache()

    assert not any(is_promotional_insert(block) for block in by_id.values())


def test_is_non_content_covers_both_origins():
    body = _text("b", BODY, 0, 0, LEFT_X, 200)
    furniture = _text("f", "27", 0, 1, LEFT_X, 960, origin="page_furniture")
    advert = _text("a", "售價：NT$500", 0, 2, RIGHT_X, 400, origin="promotional_insert")

    assert is_non_content(body) is False
    assert is_non_content(furniture) is True
    assert is_non_content(advert) is True


# ---------------------------------------------------------------------------
# Delivery surfaces honour the tag
# ---------------------------------------------------------------------------


def _tagged_ir() -> DocumentIR:
    blocks = NormalizeStage()._tag_promotional_inserts(_document_with_back_page_advert())
    return _ir(blocks)


def test_advert_is_not_rendered_into_rag_md():
    source_md, _ = PackageStage()._render_rag_md(
        document_ir=_tagged_ir(), asset_map={}, enrichments={}
    )

    assert "■參考文獻" in source_md
    assert "示範能源署" in source_md
    assert "售價" not in source_md
    assert "洽詢專線" not in source_md
    assert "本書分析" not in source_md


def test_advert_is_not_chunked(tmp_path):
    result = asyncio.run(ChunkStage().run("doc", "run", _tagged_ir(), tmp_path / "run"))

    assert result.success
    body = (tmp_path / "run" / "outputs" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "示範能源署" in body
    assert "售價" not in body
    assert "洽詢專線" not in body


def test_authored_text_survival_ignores_the_advert():
    """The hidden regression: advert copy reads as authored prose to the gate.

    Live, the 32 advert documents land at survival 0.87-0.98 once the advert
    is dropped — under 0.90 for four of them — so leaving the new origin out
    of the gate would put the whole corpus one or two blocks from a bogus
    ``authored_text_dropped`` failure.
    """
    document_ir = _tagged_ir()
    delivered = "\n".join(
        str(block.payload.get("text", ""))
        for block in document_ir.blocks
        if not is_non_content(block)
    )
    structured_output = SimpleNamespace(plan=SimpleNamespace(document_type="generic_document"))

    issues = _check_authored_text_survival(document_ir, structured_output, delivered)

    assert issues == []


def test_authored_text_survival_still_reports_a_real_drop():
    """Anti-over-correction: the gate must keep catching genuine losses."""
    document_ir = _tagged_ir()
    structured_output = SimpleNamespace(plan=SimpleNamespace(document_type="generic_document"))

    issues = _check_authored_text_survival(document_ir, structured_output, "")

    assert [issue.code for issue in issues] == ["authored_text_dropped"]


# ---------------------------------------------------------------------------
# Enrichment: the only part of this with a money cost
# ---------------------------------------------------------------------------


def test_advert_images_are_not_sent_to_the_vlm():
    """105 book covers / QR codes / logos went to the VLM live. None should."""
    document_ir = _tagged_ir()
    config = PipelineConfig(enrich=EnrichConfig(enable_vlm=True, vlm_enrich_figures=True))
    stage = EnrichStage(db=None, config=config)

    blocks_to_enrich, gating_stats = stage._apply_gating(document_ir)

    assert [block.block_id for block, _, _ in blocks_to_enrich] == []
    assert gating_stats["skip_reasons"].get("promotional_insert") == 2


def test_ordinary_figures_are_still_enriched():
    """Anti-over-correction: the skip must be scoped to the advert column."""
    blocks = _document_with_back_page_advert()
    blocks.append(_image("fig0001", 0, 500, LEFT_X, 500))
    document_ir = _ir(NormalizeStage()._tag_promotional_inserts(blocks))
    config = PipelineConfig(enrich=EnrichConfig(enable_vlm=True, vlm_enrich_figures=True))
    stage = EnrichStage(db=None, config=config)

    blocks_to_enrich, _ = stage._apply_gating(document_ir)

    assert [block.block_id for block, _, _ in blocks_to_enrich] == ["fig0001"]
