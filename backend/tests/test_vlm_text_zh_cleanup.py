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


def test_genuine_liao_words_kept_after_conversion():
    from app.pipeline.stages.package import _fix_opencc_liao

    assert _fix_opencc_liao("展示瞭如何將檔案拖放到資料夾。") == "展示了如何將檔案拖放到資料夾。"
    assert _fix_opencc_liao("設定一目瞭然，便於瞭解流程。") == "設定一目瞭然，便於瞭解流程。"
    assert _fix_opencc_liao("站在瞭望台上對系統瞭如指掌。") == "站在瞭望台上對系統瞭如指掌。"
    assert _fix_opencc_liao("說明相當明瞭。") == "說明相當明瞭。"


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
