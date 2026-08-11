"""An OCR-damaged heading must be rebuilt from the running head that repeats it.

Live evidence (2026-08-10, 167-document store): these journals set the cover
title *vertically*. MinerU's OCR of a vertical column garbles characters —
「AI」 came back as 「A」, 「6G」 as 「G」, 「全球」 as 「至球」, 「戰略」 as
「戦略」, enumeration commas vanished — and the block carries ``text_level=1``,
so the damaged string became the document's H1. The very same title is printed
horizontally as a running head on pages 2/4/6, is read from the PDF's text
layer, and is character-perfect — but ``_tag_layout_furniture`` marks it
``origin="page_furniture"`` and all three delivery surfaces then drop it. The
broken spelling was kept and the clean one thrown away in 49 of 167 documents.

The running head is therefore treated as the character-level authority, with
four anti-over-correction guards pinned below:

* a heading a running head repeats *verbatim* is never rewritten (71 documents
  in the store; they must survive byte-identical, whitespace included);
* material only the heading has, at its front, is kept — that is the 眉題, and
  the running head routinely omits it;
* material only the *running head* has is never adopted — the head carries
  series labels 「【…篇】系列3-9」 and subtitles the cover does not print;
* where the two disagree character for character, the document's own body
  prose breaks the tie, so a running head that is itself misread
  (「淨零」→「浮零」, live) cannot overwrite a heading that is right.

Every string below is fictional; the structure of each defect is taken from
the store.
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import NormalizeStage

# Authored prose used as the corroboration pool. Long enough to look like a
# real page, and it spells 「全球」 and 「淨零」 the way the articles do.
BODY = (
    "本文觀察示範島產業在全球供應鏈重組下的處境，並整理主要經濟體的因應措施。"
    "受惠於人工智慧與雲端服務需求續強，資訊電子產業成為支撐生產的核心動能。"
    "淨零轉型的時程壓力使中小企業的資本支出配置出現明顯調整。"
)


def _heading(text: str, block_id: str = "b000000", page_idx: int = 0) -> Block:
    """The cover title: one tall, narrow block of vertically set type."""
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=page_idx,
        bbox_norm=[124, 147, 277, 744],
        reading_order=0,
        payload={"text": text, "text_level": 1},
    )


def _running_head(text: str, page_idx: int, block_id: str | None = None) -> Block:
    """The horizontal running head, already tagged by _tag_layout_furniture."""
    return Block(
        block_id=block_id or f"b{page_idx:06d}",
        type=BlockType.TEXT,
        page_idx=page_idx,
        bbox_norm=[521, 95, 907, 112],
        reading_order=10 + page_idx,
        payload={"text": text, "text_level": 0, "origin": "page_furniture"},
    )


def _body(text: str = BODY, page_idx: int = 1, block_id: str = "b000900") -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=page_idx,
        bbox_norm=[87, 400, 480, 600],
        reading_order=500,
        payload={"text": text, "text_level": 0},
    )


def _repair(blocks: list[Block]) -> dict[str, str]:
    repaired = NormalizeStage()._repair_headings_from_running_heads(blocks)
    return {block.block_id: str(block.payload.get("text") or "") for block in repaired}


# ---------------------------------------------------------------------------
# The defect: characters the vertical OCR lost are restored from the head
# ---------------------------------------------------------------------------


def test_dropped_digit_and_line_break_space_are_restored():
    """「邁向6G世代」 came back as 「邁向G世代」 with a stray column break."""
    blocks = [
        _heading("邁向G世代：示範島頻譜 使用現況與未來釋出規劃"),
        _body(),
        *[
            _running_head("邁向6G世代：示範島頻譜使用現況與未來釋出規劃", page)
            for page in (2, 4, 6)
        ],
    ]

    assert _repair(blocks)["b000000"] == "邁向6G世代：示範島頻譜使用現況與未來釋出規劃"


def test_dropped_letters_and_misread_glyph_are_restored():
    """「AI」→「A」 twice and 「全」→「至」 in one heading, all from one head."""
    blocks = [
        _heading("示範國《A基本法》上路：讓A合規 成為示範企業的至球市場通行證"),
        _body(),
        *[
            _running_head("示範國《AI基本法》上路：讓AI合規成為示範企業的全球市場通行證", page)
            for page in (2, 4, 6)
        ],
    ]

    assert _repair(blocks)["b000000"] == (
        "示範國《AI基本法》上路：讓AI合規成為示範企業的全球市場通行證"
    )


def test_punctuation_the_vertical_column_lost_is_restored():
    """Enumeration commas vanish in the vertical column; the head still has them."""
    blocks = [
        _heading("政策座標力，創新深布局 示範城示範港新創政策與表現"),
        _body(),
        *[
            _running_head("政策座標力，創新深布局—示範城、示範港新創政策與表現", page)
            for page in (2, 4, 6)
        ],
    ]

    assert _repair(blocks)["b000000"] == "政策座標力，創新深布局—示範城、示範港新創政策與表現"


def test_a_misread_first_character_is_corrected():
    """Live miss: 「至球品牌…」 kept its 「至」 because it opened the string."""
    blocks = [
        _heading("至球品牌格局轉變， 示範島中小企業的破局之鑰"),
        _body(),
        *[
            _running_head("全球品牌格局轉變，示範島中小企業的破局之鑰", page)
            for page in (2, 4, 6, 8)
        ],
    ]

    assert _repair(blocks)["b000000"] == "全球品牌格局轉變，示範島中小企業的破局之鑰"


def test_a_misread_last_character_is_corrected():
    """Live miss: 「…實施現況研標」 kept its 「標」 because it closed the string."""
    blocks = [
        _heading("全球線上安全治理趨勢與 示範國《線上安全法》實施現況研標"),
        _body(),
        *[
            _running_head("全球線上安全治理趨勢與示範國《線上安全法》實施現況研析", page)
            for page in (2, 4, 6, 8)
        ],
    ]

    assert _repair(blocks)["b000000"] == (
        "全球線上安全治理趨勢與示範國《線上安全法》實施現況研析"
    )


def test_a_whole_phrase_at_the_boundary_is_not_treated_as_a_misread():
    """The cover opens on a 眉題 where the head opens on a section label.

    Both strings have material there, but a misread is a glyph or two — a
    phrase is authored text, and the heading's own must survive.
    """
    blocks = [
        _heading("與時俱進，科技賦能 示範城2026年中小微企業政策轉向的至球啟示"),
        _body(),
        *[
            _running_head("【專題】示範城2026年中小微企業政策轉向的全球啟示", page)
            for page in (1, 3, 5)
        ],
    ]

    assert _repair(blocks)["b000000"] == (
        "與時俱進，科技賦能 示範城2026年中小微企業政策轉向的全球啟示"
    )


def test_repair_runs_on_heads_that_furniture_tagging_has_just_labelled():
    """The repair depends on _tag_layout_furniture having run first."""
    stage = NormalizeStage()
    head = "示範島生成式AI之產業應用發展趨勢"
    blocks = [
        _heading("示範島生成式A之產業應用發展趨勢"),
        _body(),
        # Untagged: MinerU typed these as ordinary text, as it does in the store.
        *[
            Block(
                block_id=f"b{page:06d}",
                type=BlockType.TEXT,
                page_idx=page,
                bbox_norm=[521, 95, 907, 112],
                reading_order=10 + page,
                payload={"text": head, "text_level": 0},
            )
            for page in (2, 4, 6)
        ],
    ]

    tagged = stage._tag_layout_furniture(blocks, page_count=7)
    repaired = stage._repair_headings_from_running_heads(tagged)

    assert repaired[0].payload["text"] == head


# ---------------------------------------------------------------------------
# Anti-over-correction 1: a heading the head confirms is never touched
# ---------------------------------------------------------------------------


def test_heading_a_running_head_repeats_verbatim_is_left_byte_identical():
    """71 of 167 documents are already correct; the column break must stay too."""
    original = "示範經濟景氣 回顧與展望"
    blocks = [
        _heading(original),
        _body(),
        *[_running_head("示範經濟景氣回顧與展望", page) for page in (2, 4)],
    ]

    assert _repair(blocks)["b000000"] == original


def test_one_page_disagreeing_cannot_override_a_head_that_matches_verbatim():
    """A single page read 「戰略-2026」; three others agree with the heading."""
    original = "全球變局下的示範戰略2026年景氣展望研討會側記"
    blocks = [
        _heading(original),
        _body(),
        _running_head("全球變局下的示範戰略2026年景氣展望研討會側記", 4, "b000400"),
        _running_head("全球變局下的示範戰略-2026年景氣展望研討會側記", 6, "b000600"),
    ]

    assert _repair(blocks)["b000000"] == original


def test_a_running_head_that_is_a_different_string_never_overwrites_the_heading():
    """The journal name and the folio share a few glyphs; that is not a match."""
    original = "示範島新創環境發展可以更好"
    blocks = [
        _heading(original),
        _body(),
        *[_running_head("示範經濟研究月刊", page) for page in (1, 2, 3)],
        *[_running_head("第40卷第1期 115年1月", page) for page in (1, 2, 3)],
    ]

    assert _repair(blocks)["b000000"] == original


def test_a_bare_folio_cannot_be_chosen_as_the_running_head():
    """Live bug: 「20」 scored a perfect containment against a heading holding a 2050.

    Two characters agreeing is coincidence, so short furniture is not a
    candidate at all — and the genuine head must still win.
    """
    blocks = [
        _heading("低碳之橋：示範島2050淨零路徑下的銜接挑戰"),
        _body(),
        *[_running_head("20", page, f"n{page:06d}") for page in (1, 2, 3)],
        *[
            _running_head("低碳之橋：示範島2050淨零路徑下的銜接挑戰與展望", page)
            for page in (1, 3, 5)
        ],
    ]

    assert _repair(blocks)["b000000"] == "低碳之橋：示範島2050淨零路徑下的銜接挑戰"


# ---------------------------------------------------------------------------
# Anti-over-correction 2: the 眉題 the running head omits must survive
# ---------------------------------------------------------------------------


def test_the_kicker_only_the_cover_prints_is_kept_while_the_core_is_repaired():
    """The head drops the 眉題; adopting it wholesale would delete authored text."""
    blocks = [
        _heading("與時俱進，科技賦能 示範城2026年中小微企業政策轉向的至球啟示"),
        _body(),
        *[
            _running_head("示範城2026年中小微企業政策轉向的全球啟示", page)
            for page in (1, 3, 5, 7)
        ],
    ]

    assert _repair(blocks)["b000000"] == (
        "與時俱進，科技賦能 示範城2026年中小微企業政策轉向的全球啟示"
    )


def test_a_latin_kicker_is_kept_even_though_the_running_head_abbreviates_it():
    """The head names the article without its technology label; the label stands."""
    blocks = [
        _heading("B5G 示範島高空平台的崛起： 頻譜規劃與DEMS應用"),
        _body(),
        *[
            _running_head("示範島高空平台的崛起：頻譜規劃與DEMOS應用", page)
            for page in (1, 3)
        ],
    ]

    assert _repair(blocks)["b000000"] == "B5G 示範島高空平台的崛起：頻譜規劃與DEMOS應用"


def test_a_heading_differing_from_the_head_only_in_column_breaks_is_untouched():
    """Nothing is wrong with the characters, so the block is not rewritten."""
    original = "B5G 示範島高空平台的崛起： 頻譜規劃與DEMOS應用"
    blocks = [
        _heading(original),
        _body(),
        *[
            _running_head("示範島高空平台的崛起：頻譜規劃與DEMOS應用", page)
            for page in (1, 3)
        ],
    ]

    assert _repair(blocks)["b000000"] == original


# ---------------------------------------------------------------------------
# Anti-over-correction 3: material only the running head carries is not adopted
# ---------------------------------------------------------------------------


def test_the_series_label_only_the_running_head_prints_is_not_adopted():
    """The head prefixes a series label; the cover title must keep its own scope."""
    blocks = [
        _heading("示範農業科技服務崛起之路： 政策商模及挑戰"),
        _body(),
        *[
            _running_head("【示範創新篇】 系列3-9示範農業科技服務崛起之路：政策、商模及挑戰", page)
            for page in (1, 3, 5)
        ],
    ]

    assert _repair(blocks)["b000000"] == "示範農業科技服務崛起之路：政策、商模及挑戰"


def test_a_subtitle_only_the_running_head_prints_is_not_appended():
    original = "示範地方創生進化論"
    blocks = [
        _heading(original),
        _body(),
        *[_running_head("示範地方創生進化論——從1.0到3.0的轉變", page) for page in (1, 3, 5)],
    ]

    assert _repair(blocks)["b000000"] == original


# ---------------------------------------------------------------------------
# Anti-over-correction 4: the body prose settles character-level disagreements
# ---------------------------------------------------------------------------


def test_each_side_right_about_a_different_glyph_is_resolved_by_the_body():
    """Live shape: the heading has 「淨」 right and 「戦」 wrong, the head vice versa.

    Resolved here without any help from the zh-TW converter that runs later.
    """
    blocks = [
        _heading("示範顧問：淨零國際戦略視角， 零碳競合下的定位"),
        _body(),
        *[
            _running_head("示範顧問：凈零國際戰略視角，零碳競合下的定位", page)
            for page in (1, 3, 5, 7)
        ],
    ]

    assert _repair(blocks)["b000000"] == "示範顧問：淨零國際戰略視角，零碳競合下的定位"


def test_a_running_head_the_ocr_misread_cannot_overwrite_a_correct_heading():
    """Live case: every head read 「淨零」 as 「浮零」 while the cover was right."""
    original = "低碳之橋：示範島2050淨零路徑下的銜接挑戰"
    blocks = [
        _heading(original),
        _body(),
        *[
            _running_head("低碳之橋：示範島2050浮零路徑下的銜接挑戰", page)
            for page in (1, 3, 5, 7)
        ],
    ]

    assert _repair(blocks)["b000000"] == original


# ---------------------------------------------------------------------------
# Logo and folio glyphs swept into the title box
# ---------------------------------------------------------------------------


def test_a_latin_logo_trailing_the_title_column_is_dropped():
    blocks = [
        _heading("示範加速器如何助攻新創國際化DEMOHUB"),
        _body(),
        *[_running_head("示範加速器如何助攻新創國際化", page) for page in (1, 3, 5, 7)],
    ]

    assert _repair(blocks)["b000000"] == "示範加速器如何助攻新創國際化"


def test_glyphs_bled_into_the_middle_of_the_title_are_dropped():
    """A folio and a watermark fragment landed inside the vertical column."""
    blocks = [
        _heading("示範島淨零轉型政策與配套措施， 20 協助降低中小企業轉型壓力"),
        _body(),
        *[
            _running_head("示範島淨零轉型政策與配套措施，協助降低中小企業轉型壓力", page)
            for page in (1, 3, 5)
        ],
    ]

    assert _repair(blocks)["b000000"] == "示範島淨零轉型政策與配套措施，協助降低中小企業轉型壓力"


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_a_document_without_running_heads_is_untouched():
    original = "示範島產業趨勢研析"
    blocks = [_heading(original), _body()]

    assert _repair(blocks)["b000000"] == original


def test_a_document_without_a_heading_leaves_every_block_alone():
    blocks = [_body(), *[_running_head("示範經濟研究月刊", page) for page in (1, 2)]]
    before = [str(block.payload.get("text")) for block in blocks]

    NormalizeStage()._repair_headings_from_running_heads(blocks)

    assert [str(block.payload.get("text")) for block in blocks] == before


def test_the_running_heads_themselves_are_never_rewritten():
    """Only the heading is repaired; the furniture blocks stay as read."""
    head = "示範島生成式AI之產業應用發展趨勢"
    blocks = [
        _heading("示範島生成式A之產業應用發展趨勢"),
        _body(),
        *[_running_head(head, page) for page in (2, 4, 6)],
    ]

    texts = _repair(blocks)

    assert texts["b000000"] == head
    assert all(texts[f"b{page:06d}"] == head for page in (2, 4, 6))
