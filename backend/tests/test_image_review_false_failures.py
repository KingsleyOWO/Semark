"""Image reviews must not fail correct deliveries on bookkeeping, nor RAG drop delivered meaning.

Validation v20.1 spot check, 20 unverified pictures: 15 failed on relationship
accounting rather than misreading. Workshop photos whose every relation the
audit matched verbatim stayed unverified because sideless "positions" had been
turned into uncertainties; slides with no relationships failed on an ID the
audit invented; a quote that skipped one delivered line or crossed a chunk
split was "ungrounded". Separately, a picture's RAG text kept its description
only as an 80-character title and dropped the facts pairing each colour with
its codes.
"""
from types import SimpleNamespace

from app.pipeline.image_source_review import WordPictureItem, relationship_issue
from app.pipeline.stages.package import PackageStage


def _reading(relationships, image_type="photo"):
    return WordPictureItem.model_validate({
        "source_id": "image0", "text": "晴諾米", "complete": True, "observation": "Picture.",
        "image_type": image_type, "relationships": relationships, "relationships_complete": True})


def _relation(relation_id, kind="spatial_position", fact="與會者圍坐長桌進行工作坊", **sides):
    return {"relation_id": relation_id, "kind": kind, "source_fact": fact, "bbox": [0, 0, 1000, 1000],
            "certain": True, **sides}


def _finding(*checks):
    return SimpleNamespace(block_id="b0", image_relationship_checks=[SimpleNamespace(
        relation_id=relation_id, status="matched", delivered_quote=quote, direction="none",
        horizontal="unspecified", vertical="unspecified") for relation_id, quote in checks])


def test_a_sideless_position_with_no_axes_is_a_description_that_can_certify():
    reading = _reading([_relation("r0")])

    assert (reading.relationships[0].kind, reading.relationships[0].certain) == ("other", True)
    assert relationship_issue(reading, _finding(("r0", "與會者圍坐長桌進行工作坊")), "照片中與會者圍坐長桌進行工作坊。") == ""


def test_a_sideless_position_on_a_picture_with_axes_stays_uncertain():
    reading = _reading([_relation("r0", "axis_mapping", "水平軸左端為極簡保守"),
                        _relation("r1", fact="Clementoni 靠近水平軸")], image_type="diagram")

    assert (reading.relationships[1].horizontal, reading.relationships[1].certain) == ("uncertain", False)
    delivered = "水平軸左端為極簡保守。Clementoni 靠近水平軸。"
    assert "uncertain" in relationship_issue(
        reading, _finding(("r0", "水平軸左端為極簡保守"), ("r1", "Clementoni 靠近水平軸")), delivered)


def test_relation_ids_the_audit_invents_certify_nothing_and_fail_nothing():
    delivered = "品牌標語：一步一步，拼出想要的生活"
    assert relationship_issue(_reading([]), _finding(("b0_text", "一步一步，拼出想要的生活")), delivered) == ""
    reading = _reading([_relation("r0", "other", "品牌標語為一步一步")])
    assert relationship_issue(reading, _finding(("r0", "一步一步"), ("r9", "遇見")), delivered) == ""
    assert "omitted" in relationship_issue(reading, _finding(("r9", "遇見")), delivered)


def test_a_quote_may_skip_a_delivered_line_between_the_lines_it_cites():
    reading = _reading([_relation("r0", "other", "Chuck Pinson 拼圖售價 US$11.97")])
    delivered = "Chuck Pinson - 海邊小屋\n4.200\n過去一個月有1000個以上顧客購買\nUS$11.97 定價: US$15.00"
    assert relationship_issue(reading, _finding(("r0", "Chuck Pinson - 海邊小屋\n4.200\nUS$11.97")), delivered) == ""


def test_lines_far_apart_are_not_a_grounded_relationship_quote():
    reading = _reading([_relation("r0", "value_pair", "Engagement Rate 0.26% 與 Average Likes 8")])
    delivered = "Engagement Rate\n0.26%\n" + "\n".join(["其他社群指標的說明文字"] * 30) + "\n8"
    assert "grounded" in relationship_issue(reading, _finding(("r0", "Engagement Rate\n0.26%\n8")), delivered)


def test_a_quote_crossing_its_own_chunk_split_is_found():
    reading = _reading([_relation("r0", "other", "PIECE TALES 為美式卡通風格")])
    delivered = "PIECE TALES\n美式卡通風格"
    chunks = [{"block_ids": ["b0"], "content": "〔圖片內容〕\nPIECE TALES"},
              {"block_ids": ["b0"], "content": "〔圖片內容〕\n美式卡通風格"}]
    assert relationship_issue(reading, _finding(("r0", "PIECE TALES\n美式卡通風格")), delivered, chunks=chunks) == ""


def test_picture_rag_text_keeps_the_full_description_and_chinese_facts_carrying_codes():
    caption = ("此圖為品牌識別系統的色彩規範說明，以左右兩欄並列呈現兩種標準色：左側為「Harbor Yellow」黃色背景，"
               "右側為「Slate Gray」灰色背景。每欄皆展示品牌方形圖示。")
    fact = "左欄標準色為「Harbor Yellow」，色碼為 PANTONE 1235 C、RGB 255,200,70、CMYK 0,25,80,0、HEX #f9c74f。"
    stage = PackageStage()
    semantic = stage._render_visual_semantic_content(caption[:80], {"semantic_caption": caption, "facts": [fact],
                                                                     "all_text": ["Harbor Yellow"]})

    assert fact in semantic
    assert caption in stage._figure_retrieval_text(caption[:80], caption, semantic, [])


def test_a_quote_whose_first_line_also_appears_earlier_is_still_found():
    reading = _reading([_relation("r0", "table_relation", "晴諾米列的四欄")])
    delivered = ("此圖比較晴諾米拼圖工作室與七個品牌。\n" + "\n".join(["其他品牌的設計語言說明"] * 20)
                 + "\n晴諾米拼圖工作室\nLUMIO\n拼圖就是生活®\n黃＋黑色調")
    assert relationship_issue(reading, _finding(("r0", "晴諾米拼圖工作室\n拼圖就是生活®\n黃＋黑色調")), delivered) == ""


def test_repeated_transcription_lines_survive_into_picture_rag_text():
    # Two colour columns each label their codes: dropping the second "CODE" leaves #777777 unlabelled.
    output = {"facts": ["此頁並列兩種品牌標準色與各自的色碼。"],
              "all_text": ["Harbor Yellow", "CODE", "#f9c74f", "Slate Gray", "CODE", "#777777", "100%", "100%"]}
    lines = PackageStage()._render_visual_semantic_content("品牌標準色", output).splitlines()

    assert lines.count("CODE") == 2 and lines.count("100%") == 2
    assert lines[lines.index("#777777") - 1] == "CODE"
