"""MinerU's own text layer must reach zh-TW, not just the VLM's output.

Live evidence (2026-08-10, 100-document store): 849 unambiguously simplified
characters survived into rag.md across 88 of the 100 documents — 税賦優惠,
脱碳, 生质甲烷, 潔淨氢, 随燃料一併移轉. The converter existed and worked, but
its single call site was ``render_vlm_text``, so every glyph MinerU misread
went out untouched.

The anti-over-correction half matters just as much: the 2026-08-07 regression
came from converting text that was already correct (台灣→臺灣, 零組件→零元件),
so the Taiwanese spellings are pinned here too.
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import NormalizeStage
from app.pipeline.zh_text import to_taiwan_traditional


def _text_block(text: str, block_id: str = "b0") -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=0,
        bbox_norm=[87, 400, 480, 600],
        reading_order=0,
        payload={"text": text, "text_level": 0},
    )


def _converted(text: str) -> str:
    blocks = NormalizeStage()._normalize_zh_text([_text_block(text)])
    return str(blocks[0].payload["text"])


# ---------------------------------------------------------------------------
# The glyphs that actually leaked
# ---------------------------------------------------------------------------


def test_ocr_simplified_glyphs_from_the_text_layer_are_converted():
    out = _converted("並提供節能設備投資的税賦優惠，以鼓勵工業部門脱碳。")

    assert "稅賦優惠" in out
    assert "脫碳" in out
    assert "税" not in out and "脱" not in out


def test_every_leaked_glyph_class_is_covered():
    samples = {
        "潔淨氢生產": "潔淨氫生產",
        "生质甲烷": "生質甲烷",
        "随燃料一併移轉": "隨燃料一併移轉",
        "資金的用途内容": "資金的用途內容",
        "虚實整合": "虛實整合",
        "混掺義務": "混摻義務",
        "碳边境調整機制": "碳邊境調整機制",
        "兑現國家承諾": "兌現國家承諾",
        "女性經濟参與": "女性經濟參與",
    }
    for source, expected in samples.items():
        assert _converted(source) == expected, source


# ---------------------------------------------------------------------------
# Anti-over-correction: the 2026-08-07 regression must not come back
# ---------------------------------------------------------------------------


def test_taiwanese_spellings_survive_conversion():
    source = (
        "台灣的零組件廠商公布新台幣計價的占比資料，說明干擾因素與布局方向，"
        "並提供節能設備投資的税賦優惠。"
    )

    out = _converted(source)

    assert "稅賦優惠" in out, "the one genuinely simplified glyph must still be fixed"
    for term in ("台灣", "零組件", "公布", "新台幣", "占比", "干擾", "布局"):
        assert term in out, term
    assert "臺灣" not in out and "零元件" not in out


def test_text_without_simplified_content_passes_through_unchanged():
    source = "資訊電子產業成為支撐國內生產的核心動能，工業生產指數較上年同期成長16.3%。"

    assert _converted(source) == source


def test_japanese_citations_are_left_alone():
    """Kana marks the text as Japanese; 会社 is correct there, not simplified."""
    source = "日本自然エネルギー株式会社(2017)，「再生可能エネルギー」報告。"

    assert _converted(source) == source
    assert to_taiwan_traditional(source) == source


def test_a_bibliography_converts_its_chinese_lines_and_spares_the_japanese_ones():
    """A single Japanese entry must not shield the Chinese prose around it."""
    source = (
        "在兼顧能源安全、經濟成長與脱碳目標之間取得平衡。\n"
        "13.日本自然エネルギー株式会社(2017)，「非化石価値」報告。\n"
        "14.經濟部能源署(2025)，節能設備税賦優惠說明。"
    )

    out = _converted(source)
    lines = out.split("\n")

    assert lines[0] == "在兼顧能源安全、經濟成長與脫碳目標之間取得平衡。"
    assert lines[1] == "13.日本自然エネルギー株式会社(2017)，「非化石価値」報告。"
    assert lines[2] == "14.經濟部能源署(2025)，節能設備稅賦優惠說明。"


def test_english_only_text_is_untouched():
    source = "Integrated Sensing and Communication (ISAC), Release 18."

    assert _converted(source) == source


# ---------------------------------------------------------------------------
# Every text-bearing payload field, not only ``text``
# ---------------------------------------------------------------------------


def test_table_bodies_are_converted_without_disturbing_the_markup():
    block = Block(
        block_id="t0",
        type=BlockType.TABLE,
        page_idx=0,
        reading_order=0,
        payload={
            "table_body": "<table><tr><td>税率</td><td>脱碳投資</td></tr></table>",
            "table_caption": ["表1 碳边境調整機制"],
            "table_footnote": ["資料來源：本研究整理，含生质甲烷。"],
        },
    )

    converted = NormalizeStage()._normalize_zh_text([block])[0].payload

    assert "<td>稅率</td>" in converted["table_body"]
    assert "<td>脫碳投資</td>" in converted["table_body"]
    assert converted["table_caption"] == ["表1 碳邊境調整機制"]
    assert converted["table_footnote"] == ["資料來源：本研究整理，含生質甲烷。"]


def test_image_captions_are_converted():
    block = Block(
        block_id="i0",
        type=BlockType.IMAGE,
        page_idx=0,
        reading_order=0,
        payload={"img_path": "images/a.jpg", "caption": ["圖1 潔淨氢供應鏈"], "footnote": None},
    )

    converted = NormalizeStage()._normalize_zh_text([block])[0].payload

    assert converted["caption"] == ["圖1 潔淨氫供應鏈"]
    assert converted["footnote"] is None


def test_supplemented_pymupdf_blocks_are_converted_too():
    """The post-pass runs over every block, whatever produced it."""
    supplement = Block(
        block_id="s000075",
        type=BlockType.TEXT,
        page_idx=1,
        bbox_norm=[95, 192, 400, 218],
        reading_order=5,
        payload={"text": "随燃料一併移轉的憑證。", "text_level": 0, "origin": "pymupdf_supplement"},
    )

    converted = NormalizeStage()._normalize_zh_text([supplement])[0].payload

    assert converted["text"] == "隨燃料一併移轉的憑證。"
    assert converted["origin"] == "pymupdf_supplement"
