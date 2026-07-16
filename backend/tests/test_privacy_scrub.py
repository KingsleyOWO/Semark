"""Deterministic privacy scrub for VLM screenshot transcriptions.

The enrichment prompt asks the model not to transcribe unrelated personal
content, but compliance is probabilistic — a live batch shipped an inbox
screenshot's private mail subjects/senders and a NAS dialog's real employee
account list into rag.md. The scrub is the deterministic backstop at the
single point where VLM text enters the delivered surface (render_vlm_text).
"""

from app.pipeline.stages.package import render_vlm_text


def test_truncated_mail_list_lines_are_dropped():
    text = (
        "圖片顯示 Outlook 收件匣畫面。\n"
        "王.. 報價\n"
        "示範資訊股份有限...\n"
        "林.. RE: 關於隨身碟\n"
        "感謝通知 我來買文具...\n"
        "陳.. 【示範瑜伽課程】\n"
        "點選「清理舊項目」開始封存。"
    )

    out = render_vlm_text(text)

    assert "報價" not in out
    assert "隨身碟" not in out
    assert "瑜伽" not in out
    assert "示範資訊股份有限" not in out
    assert "我來買文具" not in out
    assert "Outlook 收件匣畫面" in out
    assert "清理舊項目" in out


def test_reply_and_forward_subject_lines_are_dropped():
    text = "RE: 會議時間確認\nFW: 內部公告轉知\n步驟一：開啟設定。"

    out = render_vlm_text(text)

    assert "會議時間確認" not in out
    assert "內部公告轉知" not in out
    assert "步驟一：開啟設定。" in out


def test_domain_account_digits_are_masked():
    text = "選擇網域使用者 DEMO\\d32213 與 DEMO\\d32755 加入分享。"

    out = render_vlm_text(text)

    assert "d32213" not in out
    assert "d32755" not in out
    assert "DEMO\\d*****" in out
    assert "加入分享" in out


def test_account_format_instruction_is_preserved():
    # The login teaching itself (no digits) must survive untouched.
    text = "帳號請輸入 demo\\員工編號，密碼為開機密碼。"

    out = render_vlm_text(text)

    assert out == text


def test_ui_ellipsis_line_without_mail_context_is_preserved():
    # 「清理信箱(M)...」 is a faithful UI transcription; a trailing ellipsis
    # alone (not adjacent to a dropped sender line) is not a privacy signal.
    text = "點選工具選單。\n清理信箱(M)...\n完成設定。"

    out = render_vlm_text(text)

    assert "清理信箱(M)..." in out


def test_subject_line_in_the_middle_of_text_is_dropped():
    text = "步驟一：開啟郵件。\nRE: 私人往來主旨\n步驟二：繼續設定。"

    out = render_vlm_text(text)

    assert "私人往來主旨" not in out
    assert "步驟一" in out and "步驟二" in out


def test_consecutive_truncated_inbox_lines_are_dropped():
    # This transcription style has no 1-3 char sender column — just a run of
    # cropped subject lines (live: 「感謝通知 我來買金士...」「瑜珈課新的一期
    # 即將...」 survived the sender-anchored rule).
    text = (
        "圖片顯示收件匣列表。\n"
        "日期: 今天\n"
        "示範資訊股份有...\n"
        "感謝通知 我來買文具...\n"
        "示範課程新的一期即將...\n"
        "點選「封存」完成操作。"
    )

    out = render_vlm_text(text)

    assert "示範資訊股份有" not in out
    assert "我來買文具" not in out
    assert "新的一期即將" not in out
    assert "圖片顯示收件匣列表。" in out
    assert "點選「封存」完成操作。" in out


def test_consecutive_ui_menu_items_with_accelerators_are_preserved():
    # A transcribed File menu is consecutive trailing-ellipsis lines too, but
    # the accelerator marker (「另存新檔(A)...」) identifies complete UI labels.
    text = "另存新檔(A)...\n列印(P)...\n傳送(D)...\n關閉(C)"

    out = render_vlm_text(text)

    assert "另存新檔(A)..." in out
    assert "列印(P)..." in out


def test_domain_account_adjacent_to_cjk_is_masked():
    # Python's \b treats CJK as word chars, so 「為demo\d29652。」 and
    # 「DEMO\d32755分享給您」 slipped the first mask (live: 2 of 13 survived).
    text = "使用者名稱為demo\\d29652。描述：DEMO\\d32755分享給您。"

    out = render_vlm_text(text)

    assert "d29652" not in out
    assert "d32755" not in out
    assert out.count("d*****") == 2


def test_inbox_list_region_is_dropped_regardless_of_line_shape():
    # Fresh enrichments re-roll the transcription shape (with/without trailing
    # ellipsis), so shape-based rules whack-a-mole. The inbox list pane has
    # stable UI markers (全部/未讀取/日期: 今天/收件匣 N) — everything short
    # and non-sentence after them is subjects/senders.
    text = (
        "圖片顯示 Outlook 主畫面，左側為資料夾窗格。\n"
        "收件匣 21\n"
        "全部\n"
        "未讀取\n"
        "日期: 今天\n"
        "示範資訊股份有限\n"
        "感謝通知 我來買文具\n"
        "示範課程新的一期即將\n"
        "日期: 昨天\n"
        "待領出設備\n"
        "點選「封存」資料夾即可查看封存的郵件。"
    )

    out = render_vlm_text(text)

    assert "示範資訊股份有限" not in out
    assert "我來買文具" not in out
    assert "新的一期即將" not in out
    assert "待領出設備" not in out
    assert "圖片顯示 Outlook 主畫面" in out
    assert "點選「封存」資料夾即可查看封存的郵件。" in out


def test_privacy_scrub_can_be_disabled():
    # 設定頁的「遮蔽個人資訊」開關:關閉時輸出保持原樣(僅去除頭尾空白)。
    from app.pipeline.stages.package import set_privacy_scrub_enabled

    text = "王.. 報價\n選擇網域使用者 DEMO\\d32213 加入分享。"
    try:
        set_privacy_scrub_enabled(False)
        out = render_vlm_text(text)
        assert "報價" in out
        assert "d32213" in out
    finally:
        set_privacy_scrub_enabled(True)

    out_enabled = render_vlm_text(text)
    assert "報價" not in out_enabled
    assert "d32213" not in out_enabled


def test_inbox_region_is_dropped_when_output_is_a_list():
    # all_text arrives as a list; per-item scrubbing loses the region context
    # (「感謝通知 我來買金士」 alone matches nothing). The scrub must see the
    # joined text.
    items = [
        "圖片顯示收件匣。",
        "收件匣 21",
        "全部",
        "未讀取",
        "日期: 今天",
        "示範資訊股份有限",
        "感謝通知 我來買文具",
        "點選「封存」完成。",
    ]

    out = render_vlm_text(items)

    assert "我來買文具" not in out
    assert "示範資訊股份有限" not in out
    assert "點選「封存」完成。" in out


def test_finalize_delivered_markdown_masks_parser_content():
    # MinerU OCR of a screenshot region lands in TEXT blocks and never passes
    # render_vlm_text (live: 「描述：CORP\\d54321分享給您」 reached rag.md as a
    # parser text block). The delivered files get one final pass.
    from app.pipeline.stages.package import PackageStage

    md = "# 標題\n\n[[asset:tbl0000]]\n\n描述：DEMO\\d32755分享給您。狀態：仍在等待中"

    out = PackageStage._finalize_delivered_markdown(md)

    assert "[[asset:" not in out
    assert "d32755" not in out
    assert "DEMO\\d*****" in out


def test_email_account_digits_masked():
    # Live (web-drive guide): 「收件者為 d23456@demo.example.tw」 sailed through —
    # the domain-account rule only knew the DOMAIN\x12345 shape, not emails.
    text = "分享時收件者信箱填寫為 d23456@demo.example.tw，主旨會自動帶入。"

    from app.pipeline.privacy import scrub_transcribed_privacy

    out = scrub_transcribed_privacy(text)

    assert "d23456" not in out
    assert "d*****@demo.example.tw" in out
    assert "主旨會自動帶入" in out


def test_email_without_personal_numeric_id_untouched():
    text = "問題請寄到 support@demo.example.tw 信箱，或洽分機一二三。"

    from app.pipeline.privacy import scrub_transcribed_privacy

    assert scrub_transcribed_privacy(text) == text
