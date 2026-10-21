"""A full re-read of a picture never drops lines read off it, and a picture has one gap marker (2026-10-16).

Rerun 2026-10-15: a re-read that came back as one whole-picture gap replaced 48 logo
names an audit had called omitted, not wrong; others lost a brand name and a seal their
issues asked for. One meeting photo shipped eleven gap markers naming internal IDs.
"""
from types import SimpleNamespace

from app.models.document_ir import Block, BlockType
from app.pipeline.image_recovery import (
    distinct_text_regions,
    gap_marker,
    keep_drafted_read_lines,
    merge_picture_gaps,
    picture_gap,
)


def _parts(texts=(), facts=(), gaps=()):
    return {"text_regions": [{"text": t, "bbox": [0, 0, 500, 500]} for t in texts], "visual_facts": list(facts),
            "gaps": list(gaps), "text_dependent": False, "information_status": "informative"}


def test_a_gap_only_reread_carries_every_drafted_name_an_audit_called_omitted():
    drafted = ["晴諾米", "晴川拼圖", "Ollie Bank"]
    issues = ["The partner logo grid omits about 35 legible logos (晴諾米, 晴川拼圖, …)."]
    parts, carried, named, contested = keep_drafted_read_lines(_parts(gaps=[{"bbox": [0, 0, 1000, 1000]}]),
                                                               drafted, issues)
    assert carried == drafted and named == [] and contested == []
    assert [r["text"] for r in parts["text_regions"] if r.get("carried")] == drafted


def test_quoting_a_correct_heading_is_not_naming_it_wrong():
    drafted = ["品牌定位建構", "品牌溝通體驗", "透過消費者調查，"]
    issues = ["交付文本將「品牌定位建構」欄的「精準定位品牌的目標消費族群」誤植為「透過消費者調查」的後續結果。"]
    _, carried, named, _ = keep_drafted_read_lines(_parts(texts=["內部行動"]), drafted, issues)
    assert carried == drafted and named == []


def test_a_correction_the_reread_confirms_drops_the_wrong_line():
    issues = ["b000016 交付文字將玻璃牆上的「TAIWAN」誤寫為「TAIWES」。"]
    _, carried, named, contested = keep_drafted_read_lines(_parts(texts=["No.1", "TAIWAN"]), ["No.1", "TAIWES"], issues)
    assert named == ["TAIWES"] and carried == [] and contested == []


def test_an_unsettled_correction_keeps_the_line_and_says_it_is_disputed():
    issues = ["b000016 交付文字將玻璃牆上的「TAIWAN」誤寫為「TAIWES」。"]
    _, carried, named, contested = keep_drafted_read_lines(_parts(texts=["No.1"]), ["No.1", "TAIWES"], issues)
    assert carried == ["TAIWES"] and named == [] and contested == ["TAIWES"]
    # OCR can settle it as well as the re-read.
    _, carried, named, _ = keep_drafted_read_lines(_parts(texts=["No.1"]), ["No.1", "TAIWES"], issues,
                                                   ocr_lines=["No.1", "TAIWAN"])
    assert named == ["TAIWES"] and carried == []


def test_wording_an_audit_says_is_not_in_the_picture_goes_unless_something_reads_it():
    issues = ["交付文字中的「橡皮擦」並未出現在圖中。"]
    _, carried, named, _ = keep_drafted_read_lines(_parts(texts=["鉛筆"]), ["鉛筆", "橡皮擦"], issues)
    assert named == ["橡皮擦"] and carried == []
    _, carried, named, _ = keep_drafted_read_lines(_parts(texts=["鉛筆"]), ["鉛筆", "橡皮擦"], issues,
                                                   ocr_lines=["橡皮擦"])
    assert carried == ["橡皮擦"] and named == []


def _gap_block(bid, reason, bbox, depends=()):
    return Block(block_id=bid, type=BlockType.TEXT, page_idx=0, bbox_norm=[0, 0, 1000, 1000], payload={
        "agentic_image_origin": "b000412", "text": "〔old marker〕", "agentic_image_region": bbox,
        "agentic_gap": {"reason": reason, "source_id": "b000412", "image_bbox": bbox},
        "agentic_depends_on_block_ids": list(depends)})


def test_a_pictures_gap_markers_fold_into_one_line_keeping_every_region():
    text = Block(block_id="t0", type=BlockType.TEXT, page_idx=0, bbox_norm=[0, 0, 1000, 1000], payload={
        "agentic_image_origin": "b000412", "text": "圖片文字：晴諾米", "agentic_depends_on_block_ids": ["g1"]})
    blocks = [text, _gap_block("g0", "image_text_unreadable", [0, 0, 100, 100]),
              _gap_block("g1", "region_not_interpreted", [500, 500, 600, 700], depends=["t0"]),
              _gap_block("g2", "image_text_unreadable", [200, 0, 300, 50])]
    merged = merge_picture_gaps(blocks, "b000412", 4)
    assert [b.block_id for b in merged] == ["t0", "g0"]
    gap = merged[1].payload["agentic_gap"]
    assert [r["image_bbox"] for r in gap["regions"]] == [[0, 0, 100, 100], [500, 500, 600, 700], [200, 0, 300, 50]]
    assert gap["image_bbox"] == [0, 0, 600, 700] and gap["reason"] == "image_text_unreadable"
    assert merged[1].payload["text"] == ("〔原文第 5 頁圖片：無法可靠辨識的文字或必要內容 2 處、"
                                         "未納入文字解讀的區域（保留來源圖） 1 處〕")
    # A dependency on a folded marker now points at the one that stays.
    assert text.payload["agentic_depends_on_block_ids"] == ["g0"]
    assert merged[1].payload["agentic_depends_on_block_ids"] == ["t0"]


def test_a_marker_names_no_internal_block_and_a_single_region_keeps_its_reasons():
    regions = [{"reason": "image_essential_content_unverified", "image_bbox": [0, 0, 1000, 1000],
                "source_reasons": ["image_text_unreadable_or_disputed"]}]
    assert gap_marker(0, regions) == "〔原文第 1 頁圖片：無法可靠辨識的文字或必要內容 1 處〕"
    gap = picture_gap("b000412", regions)
    assert gap["image_bbox"] == [0, 0, 1000, 1000] and gap["source_reasons"] == ["image_text_unreadable_or_disputed"]
    assert merge_picture_gaps([SimpleNamespace(payload={})], "b000412", 0)  # nothing to fold


def _reading(relations):
    from app.pipeline.image_source_review import WordPictureItem

    return WordPictureItem.model_validate({
        "source_id": "image0", "text": "定位\n策略", "complete": True, "observation": "Diagram.",
        "image_type": "diagram", "relationships_complete": True,
        "relationships": [{"relation_id": rid, "kind": "value_pair", "source_fact": fact, "certain": True}
                          for rid, fact in relations]})


def _finding(bid, checks):
    from app.pipeline.document_review import BlockReview
    from app.pipeline.image_source_review import ImageRelationshipCheck

    return BlockReview(block_id=bid, status="verified", image_relationship_checks=[
        ImageRelationshipCheck(relation_id=rid, status=status, delivered_quote=quote) for rid, status, quote in checks])


def test_a_split_pictures_relations_are_judged_across_its_pieces():
    from app.pipeline.image_source_review import relationship_group_issues

    reading = _reading([("r0", "定位 對應 晴諾米"), ("r1", "策略 對應 晴川")])
    delivered = {"k0": "定位：晴諾米", "r0": "圖片關係：策略 對應 晴川"}
    # r1 checked on the piece without its quote and again on the statement block that has it.
    findings = [_finding("k0", [("r0", "matched", "定位：晴諾米"), ("r1", "matched", "策略 對應 晴川")]),
                _finding("r0", [("r1", "matched", "策略 對應 晴川")])]
    assert relationship_group_issues(reading, findings, delivered, documents=None, chunks=None) == {}
    # A relation no piece checks is reported once, softly; a contradicted one is not excused.
    findings = [_finding("k0", [("r0", "contradicted", "定位：晴諾米")]), _finding("r0", [])]
    issues = relationship_group_issues(reading, findings, delivered, documents=None, chunks=None)
    assert list(issues) == ["k0"]
    assert issues["k0"].startswith("Image relationship r0 remains contradicted") and "r1 not checked" in issues["k0"]
    from app.pipeline.review_policy import soft_review_remark
    assert not soft_review_remark(issues["k0"])
    findings = [_finding("k0", [("r0", "matched", "定位：晴諾米")]), _finding("r0", [])]
    assert relationship_group_issues(reading, findings, delivered, documents=None, chunks=None) == {
        "k0": "Split image relationship coverage is incomplete: r1 not checked."}


def test_a_pruned_read_line_stays_delivered_beside_its_marker():
    """2026-10-16: a verified subset turned two legible slide bullets into a marker over one disputed word."""
    from app.pipeline.image_refinement import mark_region_omitted

    blocks = [Block(block_id=f"t{i}", type=BlockType.TEXT, page_idx=0, bbox_norm=[0, 0, 1000, 1000], payload={
        "agentic_image_origin": "b000412", "text": "圖片文字：" + line, "agentic_image_region": [0, i * 100, 1000, i * 100 + 90]})
        for i, line in enumerate(["晴諾米的橘色取自晴川", "與晴川拼圖統一的品牌體驗"])]
    for block in blocks:
        mark_region_omitted(block, "b000412", 48)
    merged = merge_picture_gaps(blocks, "b000412", 48)
    assert merged[0].payload["text"].split("\n") == [
        "〔原文第 49 頁圖片：未納入文字解讀的區域（保留來源圖） 2 處〕",
        "圖片文字（有疑義，未核對）：晴諾米的橘色取自晴川",
        "圖片文字（有疑義，未核對）：與晴川拼圖統一的品牌體驗"]


def test_a_drafted_line_the_reread_reads_better_with_ocr_is_not_carried_beside_it():
    """2026-10-16: 「由晴偃恩發佈」 was carried beside the re-read's 「由晴偃思發佈」 with no audit naming it."""
    parts = _parts(texts=["晴諾米\n由晴偃思發佈 · 6月6日"])
    _, carried, named, _ = keep_drafted_read_lines(parts, ["由晴偃恩發佈 · 6月6日"], [], ocr_lines=["由晴偃思發佈 · 6月6日"])
    assert named == ["由晴偃恩發佈 · 6月6日"] and carried == []
    # Without OCR for the re-read's version, the drafted line stays: two model readings, neither settled.
    _, carried, named, _ = keep_drafted_read_lines(parts, ["由晴偃恩發佈 · 6月6日"], [], ocr_lines=["晴諾米"])
    assert carried == ["由晴偃恩發佈 · 6月6日"] and named == []


def test_a_piece_of_a_drafted_line_or_another_drafted_line_is_no_other_version_of_it():
    """Test run 2026-10-19: an award wall lost 「THE COMMUNICATOR AWARDS」 to OCR's 「COMMUNICATOR」 and
    「GEWINNER」 to the 「BERLINER」 of another logo, both read as better versions of the drafted lines."""
    drafted = ["THE COMMUNICATOR AWARDS", "GEWINNER", "BERLINER TYPE AWARD", "由晴偃恩發佈"]
    parts = _parts(texts=["COMMUNICATOR", "BERLINER", "BERLINER TYPE AWARD", "由晴偃思發佈"])
    _, carried, named, _ = keep_drafted_read_lines(parts, drafted, [],
                                                   ocr_lines=["COMMUNICATOR", "BERLINER", "由晴偃思發佈"])
    assert carried == ["THE COMMUNICATOR AWARDS", "GEWINNER"] and named == ["由晴偃恩發佈"]


def test_a_drafted_word_or_count_inside_a_longer_one_is_not_read():
    """Test run 2026-10-21: 「Active」 counted as re-read inside 「Attractive」 and was dropped; so would a
    count 「×5」 inside 「15」. A line the re-read wraps differently is still read."""
    parts = _parts(texts=["Accurate", "Advanced", "Attractive", "15", "如果沒有經歷失敗", "表示你還不夠創新"])
    drafted = ["Active", "Attractive", "×5", "如果沒有經歷失敗，表示你還不夠創新"]
    _, carried, named, _ = keep_drafted_read_lines(parts, drafted, [])
    assert carried == ["Active", "×5"] and named == []
    _, carried, _, _ = keep_drafted_read_lines(_parts(texts=["① Active", "x5", "Location Traffic Share"]),
                                               ["Active", "×5", "Traffic"], [])
    assert carried == []


def test_a_word_inside_a_longer_one_is_its_own_region():
    """Test run 2026-10-21: a whiteboard's 「Active」 was merged away into 「Attractive」 in three runs."""
    regions = [{"text": t} for t in ["Active", "Attractive", "承", "傳承", "9.6", "9.61", "Active"]]
    assert [r["text"] for r in distinct_text_regions(regions)] == ["Active", "Attractive", "傳承", "9.6", "9.61"]
