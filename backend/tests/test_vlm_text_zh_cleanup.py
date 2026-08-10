"""Simplified-Chinese cleanup for VLM output.

The local VLM occasionally emits simplified characters and mainland terms in
its figure descriptions (live: 「界面标题…默认…」 blocks inside zh-TW rag.md).
Conversion runs only when simplified content is detected, so genuine zh-TW
text passes through byte-identical.
"""

from app.pipeline.stages.package import render_vlm_text


def test_simplified_description_is_converted_to_taiwan_traditional():
    text = "界面标题为路径显示，默认下载图标。"

    out = render_vlm_text(text)

    assert out == "介面標題為路徑顯示，預設下載圖示。"


def test_mainland_terms_are_localized_when_simplified_detected():
    text = "视频分辨率设置后，用户可继续操作。"

    out = render_vlm_text(text)

    assert "影片" in out
    assert "解析度" in out
    assert "簡" not in out  # sanity: no stray characters invented
    assert "视频" not in out and "分辨率" not in out


def test_pure_taiwan_traditional_text_is_untouched():
    text = "請點選『工具』選單，選擇「清理舊項目」後按確定，設定即完成。"

    out = render_vlm_text(text)

    assert out == text


def test_taiwan_prose_using_tai_and_bu_variants_is_untouched():
    """台/臺 and 布/佈 are both standard zh-TW spellings, but opencc's s2t maps
    台→臺 and 布→佈, so the detector read ordinary Taiwanese prose as
    simplified and handed the whole document to s2twp.

    Live damage (2026-08-07, run 01KZDNKM51YQXGJDXGQAA2RZXC): a reviewer-repaired
    economics report shipped with 台灣→臺灣, 公布→公佈 and 關鍵零組件→關鍵零元件
    (the institution name in the byline was mangled the same way), and the
    mangled text then tripped the gate's authored_text_dropped check
    (survival 0.11) into a false high issue.
    """
    text = (
        "2025年台灣經濟展現韌性，主計總處公布第三季GDP概估值，"
        "關鍵零組件需求大增，確立了科技產業的核心地位。"
    )

    out = render_vlm_text(text)

    assert out == text


def test_org_name_and_component_term_survive_a_long_report_paragraph():
    """Regression pin for the delivered surface: the reviewer hands whole
    documents to render_vlm_text, so a false detection rewrites the entire
    body. Institution names and domain vocabulary must come out byte-identical.

    The institution here is fictional on purpose — the real corpus identifiers
    stay in the gitignored ruleset, never in a tracked test.
    """
    text = (
        "作者／示範研究院景氣預測中心暨企業發展研究中心\n"
        "回顧2025年，台灣經濟展現強勁韌性。根據主計總處於2025年10月底公布"
        "第三季國內生產毛額(GDP)概估值，關鍵零組件之需求大幅推升，"
        "全年出口值以新台幣計價創下新高。"
    )

    out = render_vlm_text(text)

    assert out == text
    assert "示範研究院" in out
    assert "台灣" in out and "新台幣" in out  # 台 must not become 臺
    assert "零組件" in out and "零元件" not in out


def test_mixed_text_converts_only_because_simplified_present():
    text = "步驟一：開啟設定頁面。\n该按钮用于确认操作。"

    out = render_vlm_text(text)

    assert "步驟一：開啟設定頁面。" in out
    assert "該按鈕用於確認操作。" in out
    assert "该" not in out


def test_particle_le_survives_conversion_roundtrip():
    # opencc's s2t pass reads verb+了 as liǎo (展示了→展示瞭), so the detection
    # gate itself trips on pure-traditional prose and hands it to s2twp, which
    # ships the mangled particle (live: chunks carried 「展示瞭如何」 ×8).
    text = "這張截圖展示了如何將檔案拖放到資料夾中。"

    assert render_vlm_text(text) == text


def test_genuine_liao_words_kept_when_a_conversion_really_runs():
    """The particle is restored from the source, so genuine 瞭 words — which the
    source spells with 瞭 — are left alone even mid-conversion."""
    text = "该功能设定一目瞭然，站在瞭望台上對系統瞭如指掌，說明相當明瞭，也展示了完整流程。"

    out = render_vlm_text(text)

    assert "该" not in out and "设定" not in out  # conversion really ran
    for keep in ("一目瞭然", "瞭望台", "瞭如指掌", "明瞭", "展示了"):
        assert keep in out


def test_mainland_vocabulary_fixed_in_traditional_text():
    # VLM prose written in traditional characters still carries mainland
    # vocabulary (圖標/界面/分辨率) — no simplified chars, so the opencc
    # detection gate never sees it. A small unambiguous word map runs always.
    text = "界面左側有一個下載圖標，點選後可調整分辨率。"

    out = render_vlm_text(text)

    assert "介面" in out and "圖示" in out and "解析度" in out
    assert "界面" not in out and "圖標" not in out and "分辨率" not in out


def test_vocab_fix_spares_surfactant_and_taiwan_prose():
    # 界面活性劑 is legitimate chemistry vocabulary — not a UI word.
    text = "本產品含界面活性劑，操作介面顯示注意圖示。"

    out = render_vlm_text(text)

    assert "界面活性劑" in out
    assert "操作介面" in out
