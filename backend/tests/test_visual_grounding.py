"""Brand/term grounding for VLM visual output on retrieval surfaces.

Live regression (doc 9bc9e370421766b5, web-drive how-to guide): the VLM
captioned three QNAP File Station screenshots as "Synology" — a brand that
appears nowhere in the document text or in any screenshot's OCR. rag.md was
clean (it weaves facts/all_text), but the assets_index retrieval_text carried
the caption + keywords into chunks.jsonl, so the hallucinated brand reached
the retrieval surface three times. Two mechanisms drove it:

1. ``_augment_visual_output_from_page_text`` reclassified the screenshot as a
   flowchart (its structured_content held a UI navigation path with " > "),
   which switched the render to the flow template built around the caption.
2. Caption/keywords are free-form VLM prose — nothing checked their Latin
   terms against text actually visible in the document.
"""

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, SourceInfo
from app.pipeline.stages.package import AssetEntry, PackageStage


def _doc_with_page_text(lines: list[str]) -> DocumentIR:
    blocks = [
        Block(
            block_id=f"t{idx:03d}",
            type=BlockType.TEXT,
            page_idx=0,
            payload={"text": text},
        )
        for idx, text in enumerate(lines)
    ]
    return DocumentIR(
        doc_id="doc-g",
        run_id="run-g",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=blocks,
    )


def test_augment_does_not_reclassify_screenshot_as_flowchart():
    # A screenshot whose structured_content holds a UI navigation path must stay
    # a screenshot: " > " lines there are folder trees / menus, not a workflow.
    document_ir = _doc_with_page_text(
        [
            "二、檔案上傳方式說明如下所示",
            "(1)進入雲端硬碟後，點選File Station檔案總管，選擇左側「Qsync」節點",
            "(2)可點選上方箭頭將檔案或資料夾上傳至雲端硬碟空間",
            "(3)或將檔案以滑鼠拖拉方式拉進雲端硬碟視窗中完成上傳",
        ]
    )
    image_block = Block(block_id="img1", type=BlockType.IMAGE, page_idx=0, payload={})

    out = PackageStage()._augment_visual_output_from_page_text(
        {
            "image_type": "screenshot",
            "semantic_caption": "檔案總管介面截圖。",
            "structured_content": "File Station > 左側導航欄 > home > Qsync",
            "all_text": ["File Station 檔案總管"],
        },
        document_ir,
        image_block,
    )

    assert str(out.get("image_type", "")).lower() == "screenshot"


def test_ungrounded_latin_brand_stripped_from_caption_and_keywords():
    output = {
        "image_type": "screenshot",
        "semantic_caption": "這是一張 Synology NAS File Station 5 的螢幕截圖，展示了如何將檔案拖放到 Qsync 資料夾中。",
        "structured_content": "File Station > 左側導航欄 > home > Qsync",
        "all_text": ["File Station 檔案總管", "請將檔案拖拉到此處"],
        "facts": ["操作指示是將檔案拖放到 Qsync 資料夾中。"],
        "keywords": ["File Station", "Qsync", "Synology", "NAS", "拖放操作"],
    }
    grounding = PackageStage._visual_grounding_tokens_from_texts(
        [
            "二、檔案上傳方式",
            "(1)進入雲端硬碟後，點選File Station檔案總管→左側「Qsync」",
            "File Station 檔案總管",
            "請將檔案拖拉到此處",
        ]
    )

    grounded = PackageStage._ground_visual_output_terms(
        output, grounding, semantic_output_language="zh-TW"
    )

    caption = grounded["semantic_caption"]
    assert "Synology" not in caption, caption
    assert "NAS" not in caption, caption
    assert "File Station" in caption, caption
    # Chinese prose must pass through untouched.
    assert "展示了如何將檔案拖放到 Qsync 資料夾中" in caption, caption
    assert "Synology" not in grounded["keywords"]
    assert "NAS" not in grounded["keywords"]
    assert "Qsync" in grounded["keywords"]
    assert "拖放操作" in grounded["keywords"]


def test_grounding_leaves_english_corpus_untouched():
    # Grounding is a zh-corpus guard: an English corpus caption is legitimate
    # English prose, and token-dropping would mangle it.
    output = {
        "semantic_caption": "A Synology NAS dashboard screenshot.",
        "keywords": ["Synology", "dashboard"],
    }
    grounded = PackageStage._ground_visual_output_terms(
        output, set(), semantic_output_language="en"
    )
    assert grounded["semantic_caption"] == "A Synology NAS dashboard screenshot."
    assert grounded["keywords"] == ["Synology", "dashboard"]


def test_caption_title_cut_at_sentence_boundary():
    # Live: title = caption[:100] produced 「…畫面左側顯示了檔案系統的樹狀結構，中」 —
    # a mid-sentence, mid-clause cut embedded into every retrieval surface.
    caption = (
        "這是一張檔案總管的螢幕截圖，展示了如何將檔案拖放到同步資料夾中。"
        "畫面左側顯示了檔案系統的樹狀結構，中間區域為檔案列表，右側則是預覽窗格與詳細資訊面板。"
    )
    title = PackageStage()._infer_visual_asset_title("", caption, "Figure 3", all_text=[])
    assert title == "這是一張檔案總管的螢幕截圖，展示了如何將檔案拖放到同步資料夾中。", title


def test_split_asset_document_respects_stored_screenshot_type():
    # The per-asset document export used to re-derive image_type from
    # " > " in structured_content, turning UI screenshots into fake flowcharts
    # (「…的流程圖語意化內容。流程從…開始」 built around the caption).
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0001",
        doc_id="doc-g",
        run_id="run-g",
        title="File Station 檔案總管",
        page_idx=3,
        asset_path="assets/figures/fig0001.jpg",
        block_id="img1",
        retrieval_text="",
        semantic_caption="這是一張檔案總管的螢幕截圖，展示了拖放檔案的操作。",
        structured_content="File Station > 左側導航欄 > home > Qsync",
        facts=["操作指示是將檔案拖放到 Qsync 資料夾中。"],
        keywords=["Qsync"],
        image_type="screenshot",
    )

    rendered = PackageStage()._render_split_asset_document(
        asset, "使用說明", "guide.pdf", semantic_output_language="zh-TW"
    )

    assert "流程圖語意化內容" not in rendered, rendered
    assert "拖放到 Qsync 資料夾" in rendered, rendered


def test_asset_entry_serializes_image_type():
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0001",
        doc_id="doc-g",
        run_id="run-g",
        title="t",
        page_idx=0,
        asset_path="p",
        block_id="b",
        retrieval_text="r",
        image_type="screenshot",
    )
    assert asset.to_dict()["image_type"] == "screenshot"


def test_low_value_icon_output_detection():
    # p11-13 live: ~15 tiny UI icons (叉號/盾牌/下載箭頭/模糊月牙) each got a
    # caption and wove into rag.md; 5 of 22 chunks were icon noise.
    icon = {
        "image_type": "other",
        "semantic_caption": "這是一個黑色的叉號符號，通常用於關閉或取消。",
        "all_text": [],
    }
    assert PackageStage._is_low_value_icon_output(icon) is True

    blurry = {
        "image_type": "screenshot",
        "semantic_caption": "這是一張極度模糊的圖片，顯示難以辨認的深色圖案。",
        "all_text": [],
    }
    assert PackageStage._is_low_value_icon_output(blurry) is True

    with_ocr_text = {
        "image_type": "other",
        "semantic_caption": "QNAP Authenticator 應用程式的圖示。",
        "all_text": ["QNAP Authenticator"],
    }
    assert PackageStage._is_low_value_icon_output(with_ocr_text) is False

    informative_screenshot = {
        "image_type": "screenshot",
        "semantic_caption": "這是一張顯示兩步驟驗證設定的截圖，包含備援信箱欄位與開始使用按鈕。",
        "all_text": [],
    }
    assert PackageStage._is_low_value_icon_output(informative_screenshot) is False


def test_decorative_icon_not_woven_into_rag():
    document_ir = _doc_with_page_text(
        ["(7)掃描完畢後，手機出現兩階段驗證號碼，請輸入至網頁完成驗證。"]
    )
    document_ir.blocks.append(
        Block(block_id="icon1", type=BlockType.IMAGE, page_idx=0, payload={})
    )
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0001",
        doc_id="doc-g",
        run_id="run-g",
        title="叉號圖示",
        page_idx=0,
        asset_path="assets/figures/fig0001.jpg",
        block_id="icon1",
        retrieval_text="這是一個黑色的叉號符號。",
        semantic_caption="這是一個黑色的叉號符號。",
        image_type="other",
        decorative=True,
    )

    markdown, _ = PackageStage()._render_rag_md(
        document_ir=document_ir,
        asset_map={"icon1": asset},
        enrichments={
            "icon1": {
                "kind": "figure_description",
                "output": {
                    "image_type": "other",
                    "semantic_caption": "這是一個黑色的叉號符號。",
                    "facts": ["圖片中心有一個黑色的叉號符號。"],
                },
            }
        },
    )

    assert "叉號" not in markdown
    assert "兩階段驗證號碼" in markdown


def test_single_char_ocr_does_not_rescue_icon():
    # Fresh enrichments re-roll: the close-button icon came back with
    # all_text=["X"] — one junk OCR char must not count as retrievable text.
    icon = {
        "image_type": "other",
        "semantic_caption": "圖像中心有一個黑色的叉號，圖像非常模糊。",
        "all_text": ["X"],
    }
    assert PackageStage._is_low_value_icon_output(icon) is True
