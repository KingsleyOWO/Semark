"""Non-Taiwanese variant glyphs that the simplified-Chinese detector cannot see.

``to_taiwan_traditional`` only converts when opencc's ``s2t`` pass reports a
change, which is the right gate for genuinely simplified glyphs (税/脱/质/氢).
It is blind to a second class of OCR damage: variant forms that are encodable,
that ``s2t`` and ``s2tw`` both leave untouched, and that are still not the
Taiwan standard form. Those pass the gate and ship as-is.

Live evidence (2026-08-10, 167-document run):

    凈  should be 淨   51 occurrences / 30 documents  — 51 Chinese, 0 Japanese
    説  should be 說   25 occurrences / 21 documents  — 24 Chinese, 1 Japanese

and, deliberately NOT covered by the fix under test:

    経 31 / 産 19 / 関 14 / 戦 11 / 対 7 — overwhelmingly Japanese citations

The corpus proves these are OCR noise rather than authorial spelling: one line
carried 說 and 説 side by side within a single clause.

The anti-over-correction half of this file is the load-bearing half. A
bibliography mixes zh-TW prose with Japanese titles, and rewriting 経営 to
經營 or 関係 to 關係 corrupts a citation. The fictional institution names here
are fictional on purpose — real corpus identifiers stay in the gitignored
ruleset, never in a tracked test.
"""

from app.models.document_ir import Block, BlockType
from app.pipeline.stages.normalize import NormalizeStage
from app.pipeline.stages.package import render_vlm_text
from app.pipeline.zh_text import to_taiwan_traditional


def _text_block(text: str) -> Block:
    return Block(
        block_id="b0",
        type=BlockType.TEXT,
        page_idx=0,
        bbox_norm=[87, 400, 480, 600],
        reading_order=0,
        payload={"text": text, "text_level": 0},
    )


def _via_parser(text: str) -> str:
    """The MinerU text-layer path."""
    return str(NormalizeStage()._normalize_zh_text([_text_block(text)])[0].payload["text"])


# ---------------------------------------------------------------------------
# The glyphs that leak past the detection gate
# ---------------------------------------------------------------------------


def test_jing_variant_reaches_the_taiwan_standard_form():
    source = "本文分析國際永續凈零推動趨勢，並探討潔凈能源的投資結構。"

    assert to_taiwan_traditional(source) == "本文分析國際永續淨零推動趨勢，並探討潔淨能源的投資結構。"


def test_shuo_variant_reaches_the_taiwan_standard_form():
    source = "所以本文擬再補充説明國民所得恆等式的更多道理。"

    assert to_taiwan_traditional(source) == "所以本文擬再補充說明國民所得恆等式的更多道理。"


def test_the_gate_is_the_bug_no_simplified_glyph_is_present_at_all():
    """Both sentences above are otherwise perfect zh-TW, so ``s2t`` reports no
    change and the old code returned them byte-identical. Pin that the fix does
    not depend on some other glyph tripping the detector."""
    from opencc import OpenCC

    for source in ("國際永續凈零推動趨勢", "補充説明國民所得恆等式"):
        assert OpenCC("s2t").convert(source) == source, "detector must still see nothing"
        assert to_taiwan_traditional(source) != source, "fix must run anyway"


def test_inconsistent_ocr_spellings_in_one_clause_converge():
    """Live: a single line carried both spellings, which is what makes this OCR
    noise rather than the author's choice."""
    source = "誇大來吸引支持者的方法，說説聽聽就好。"

    assert to_taiwan_traditional(source) == "誇大來吸引支持者的方法，說說聽聽就好。"


def test_variant_fix_composes_with_a_genuine_simplified_conversion():
    """A document that *does* trip the gate must get both repairs."""
    source = "節能設備的税賦優惠有助於凈零轉型，詳如説明。"

    assert to_taiwan_traditional(source) == "節能設備的稅賦優惠有助於淨零轉型，詳如說明。"


def test_headings_and_short_fragments_are_covered():
    for source, expected in {
        "## 國際凈零局勢": "## 國際淨零局勢",
        "### 説明": "### 說明",
        "在歐盟凈營收1億歐元以上": "在歐盟淨營收1億歐元以上",
        "數位賦能金融業協助中小企業凈零轉型": "數位賦能金融業協助中小企業淨零轉型",
        "1.第八屆遴選作業說明會簡介及申請説明": "1.第八屆遴選作業說明會簡介及申請說明",
    }.items():
        assert to_taiwan_traditional(source) == expected, source


# ---------------------------------------------------------------------------
# Anti-over-correction: Japanese must not be rewritten
# ---------------------------------------------------------------------------


def test_a_kana_bearing_japanese_line_keeps_its_variant_glyphs():
    """The single Japanese-context 説 in the corpus sits on a line carrying
    kana, so the existing per-line kana guard already covers it. Pin that the
    new rule runs *inside* that guard and does not weaken it."""
    source = '6.示範団体(まちあげパン太),"移住・定住施策の現状と課題について解説！".'

    assert to_taiwan_traditional(source) == source
    assert "解説" in to_taiwan_traditional(source)


def test_katakana_energy_citation_is_untouched():
    source = "再生可能エネルギー・水素等関係閣僚会議，脱炭素社会の実現に関する報告。"

    assert to_taiwan_traditional(source) == source
    assert "関係" in to_taiwan_traditional(source) and "會議" not in to_taiwan_traditional(source)


def test_japanese_only_variants_are_deliberately_left_alone():
    """The trade-off this fix deliberately accepts.

    経/産/関/戦/対 are overwhelmingly Japanese in the corpus, and — unlike 凈
    and 説 — their Chinese-context occurrences are almost all kana-free
    Japanese bibliography entries that the line guard cannot catch. Converting
    them would corrupt 経営 → 經營 and 経済産業省 → 經濟產業省 in a citation.
    Residual mis-spellings in Chinese prose are the accepted cost.
    """
    kana_free_japanese = "13.示範省，“就農準備資金‧経営開始資金”，脱炭素成長型経済構造推進法案。"

    out = to_taiwan_traditional(kana_free_japanese)

    for keep in ("経営", "経済"):
        assert keep in out, keep
    for never in ("經營", "經済", "經濟"):
        assert never not in out, never


def test_the_other_japanese_variant_glyphs_pass_through_untouched():
    """Kana-free Japanese, so the line guard cannot help — only the exclusion
    from the rule table keeps these intact.

    Scope note: neighbouring glyphs that opencc genuinely classes as simplified
    (会→會, 来→來, 脱→脫) are still converted by the pre-existing gate. That is
    long-standing behaviour, untouched here, so the samples below avoid them.
    """
    for source in ("経営開始資金", "水素等関係省庁", "示範産業省報告書", "戦略本部", "費用対効果"):
        assert to_taiwan_traditional(source) == source, source


def test_a_bibliography_fixes_its_chinese_lines_and_spares_the_japanese_ones():
    source = (
        "本書分析市場新局及國際永續凈零推動趨勢。\n"
        "13.示範団体(2017)，「再生可能エネルギー」報告，費用対効果について解説。\n"
        "14.示範研究院(2025)，節能設備税賦優惠説明。"
    )

    lines = to_taiwan_traditional(source).split("\n")

    assert lines[0] == "本書分析市場新局及國際永續淨零推動趨勢。"
    assert lines[1] == "13.示範団体(2017)，「再生可能エネルギー」報告，費用対効果について解説。"
    assert lines[2] == "14.示範研究院(2025)，節能設備稅賦優惠說明。"


# ---------------------------------------------------------------------------
# 合 vs 閤 — opencc rewrites 合中 → 閤中
# ---------------------------------------------------------------------------


def test_he_zhong_does_not_trip_the_detector_on_clean_taiwan_prose():
    """``s2t`` maps 合中 → 閤中, so ordinary zh-TW prose carrying 結合中央 holds
    no simplified glyph yet still reads as simplified to the detection gate —
    the same false positive 台→臺 produced. It shipped: 6 documents of the
    167-document run went out with 結閤中央 / 投資組閤中 / 能源組閤中 /
    淨零競閤中 / 聯閤中小型製作公司.
    """
    for source in (
        "政府結合中央與地方資源推動產業升級。",
        "揭露其投資組合中永續投資的比重。",
        "在再生能源組合中仍具基礎地位。",
        "在全球淨零競合中累積防禦能量。",
        "聯合中小型製作公司共同投入資金。",
        "本案符合中華民國法令規定。",
    ):
        assert to_taiwan_traditional(source) == source, source


def test_he_zhong_survives_when_a_conversion_really_runs():
    """The gate-level fix above is not enough on its own: once any genuinely
    simplified glyph hands the text to s2tw, s2tw applies the same 合中 → 閤中
    rule to the whole document. Pin the restore step, not just the detector.
    """
    source = (
        "政府結合中央資源，揭露其投資組合中永續投資的比重，"
        "在全球淨零競合中聯合中小型製作公司，並提供節能設備的税賦優惠。"
    )

    out = to_taiwan_traditional(source)

    assert "稅賦優惠" in out, "a conversion must actually have run"
    assert "税" not in out
    for keep in ("結合中央", "投資組合中", "淨零競合中", "聯合中小型製作公司"):
        assert keep in out, keep
    assert "閤" not in out


def test_legitimate_he_spelling_is_not_flattened():
    """閤家/閤第 spell 閤 deliberately. Both s2t and s2tw leave 閤 alone, so the
    restore step's ``conv`` branch hands the original glyph straight back —
    folding 閤 into 合 for *detection* must not leak into the output.
    """
    assert to_taiwan_traditional("恭賀新禧，閤家平安。") == "恭賀新禧，閤家平安。"

    out = to_taiwan_traditional("恭賀新禧，閤家平安，並享節能設備的税賦優惠。")

    assert "閤家平安" in out and "合家平安" not in out
    assert "稅賦優惠" in out, "a conversion must actually have run"


def test_the_five_pre_existing_variant_pairs_do_not_regress():
    source = "台灣公布的占比資料顯示干擾下降，展示了成果，並提供節能設備的税賦優惠。"

    out = to_taiwan_traditional(source)

    assert "稅賦優惠" in out, "a conversion must actually have run"
    for keep in ("台灣", "公布", "占比", "干擾", "展示了"):
        assert keep in out, keep
    for never in ("臺灣", "公佈", "佔比", "幹擾", "展示瞭"):
        assert never not in out, never


# ---------------------------------------------------------------------------
# Existing behaviour must not regress
# ---------------------------------------------------------------------------


def test_taiwanese_spellings_still_survive_conversion():
    source = (
        "示範研究院公布新台幣計價的占比資料，說明干擾因素與布局方向，"
        "台灣關鍵零組件的税賦優惠有助於凈零轉型。"
    )

    out = to_taiwan_traditional(source)

    assert "稅賦優惠" in out and "淨零轉型" in out
    for term in ("示範研究院", "台灣", "零組件", "公布", "新台幣", "占比", "干擾", "布局"):
        assert term in out, term
    assert "臺灣" not in out and "零元件" not in out and "公佈" not in out


def test_particle_le_still_survives_the_roundtrip():
    text = "這張截圖展示了如何將檔案拖放到資料夾中。"

    assert to_taiwan_traditional(text) == text


def test_genuine_liao_words_still_kept_when_a_conversion_runs():
    text = "该功能设定一目瞭然，站在瞭望台上對系統瞭如指掌，也展示了完整流程。"

    out = to_taiwan_traditional(text)

    assert "该" not in out and "设定" not in out
    for keep in ("一目瞭然", "瞭望台", "瞭如指掌", "展示了"):
        assert keep in out, keep


def test_clean_taiwan_prose_still_passes_through_byte_identical():
    source = "資訊電子產業成為支撐國內生產的核心動能，工業生產指數較上年同期成長16.3%。"

    assert to_taiwan_traditional(source) == source


def test_english_and_empty_input_untouched():
    assert to_taiwan_traditional("Integrated Sensing and Communication (ISAC).") == (
        "Integrated Sensing and Communication (ISAC)."
    )
    assert to_taiwan_traditional("") == ""


def test_conversion_is_idempotent():
    source = "國際永續凈零推動趨勢，補充説明如後。"

    once = to_taiwan_traditional(source)

    assert to_taiwan_traditional(once) == once


# ---------------------------------------------------------------------------
# Both producers, not just one — the 2026-08-10 bug was a missing call site
# ---------------------------------------------------------------------------


def test_the_mineru_text_layer_path_applies_the_variant_fix():
    assert _via_parser("推動2050凈零轉型的補充説明。") == "推動2050淨零轉型的補充說明。"


def test_the_vlm_output_path_applies_the_variant_fix():
    assert render_vlm_text("圖中標示凈零路徑的説明文字。") == "圖中標示淨零路徑的說明文字。"


def test_the_vlm_path_still_spares_japanese_and_taiwan_spellings():
    source = "示範団体「再生可能エネルギー・水素等関係閣僚会議」的資料，台灣公布的占比。"

    assert render_vlm_text(source) == source


def test_table_bodies_and_captions_get_the_variant_fix():
    block = Block(
        block_id="t0",
        type=BlockType.TABLE,
        page_idx=0,
        reading_order=0,
        payload={
            "table_body": "<table><tr><td>凈零</td><td>説明</td></tr></table>",
            "table_caption": ["表1 凈零轉型路徑"],
            "table_footnote": ["資料來源：本研究整理，詳如説明。"],
        },
    )

    converted = NormalizeStage()._normalize_zh_text([block])[0].payload

    assert converted["table_body"] == "<table><tr><td>淨零</td><td>說明</td></tr></table>"
    assert converted["table_caption"] == ["表1 淨零轉型路徑"]
    assert converted["table_footnote"] == ["資料來源：本研究整理，詳如說明。"]
