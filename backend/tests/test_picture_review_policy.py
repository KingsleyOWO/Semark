"""A picture is judged on its own review, and a remark about layout alone finds nothing wrong (2026-10-13)."""
import pytest

from app.pipeline.review_policy import picture_content_confirmed, soft_review_remark


@pytest.mark.parametrize("remark", [
    # Written by relationship_issue, word_picture_issue and the blind reading.
    "Image relationship r0 remains missing: 標題「晨光茶坊」位於畫面左上角。",
    "Image relationship r2 has no grounded quote in its own delivered content.",
    "Image relationship r3 has no owned quote in actual chunks.",
    "Independent image relationships are incomplete; labels alone cannot verify visual meaning.",
    "Independent Word image reading is incomplete: 會議室照片，螢幕文字過小無法辨識。",
    # Written by the review model.
    "b000412 交付未保留三個圓形左中右空間關係。",
    "b000412: Left/right spatial relationship between Bold and Calm is not preserved in the delivered text.",
    "b000412 delivery does not explicitly state the left-to-right order of the four icons.",
    "b000412 delivered text is a flat list of labels rather than a structured description of the page layout.",
    "b000412 白板便利貼上的手寫文字過小且模糊，無法可靠辨識。",
    "b000412: lower descriptive text block is illegible in source; delivery marks it as unreadable",
])
def test_remarks_that_find_nothing_wrong_in_the_delivered_wording(remark):
    assert soft_review_remark(remark)


@pytest.mark.parametrize("remark", [
    "Image relationship r1 remains contradicted: 箭頭由左指向右。",
    "Image relationship r1 direction differs between source and delivery.",
    "Independent original image wording is absent or different in delivery: 晨光茶坊",
    "b000412 標籤文字誤植為「側窗圓弧」，來源為「側前圓弧」。",
    "b000412 交付將原始背板「4.28」讀作「4228」。",
    "b000412 右側海報底部店名：交付為 NORTH BOOKS，獨立讀取為 SOUTH BOOKS。",
    "b000412 copyright line delivered as '晨光[?]' but source image reads '晨光茶坊有限公司'",
    "Source caption reads 青草膏-5片裝, but the delivered text changes the unit to 5片入.",
    "b000412 交付所述「兩名白衣人物圍坐桌旁」無法在影像中確認。",
    "b000412 delivered text descriptions are plausible but not fully verifiable from source pixels",
    "b000412 包裝正面小字標籤（10倍、4重、30包/盒）未完整交付，且上下位置關係未說明。",
    "b000412 交付描述將可辨識桌牌姓名降級為「文字過小無法辨識」，未保留來源可讀資訊。",
    "b000412 delivery claims text on the glass partition, but the independent image reading found none.",
    "b000412 表格中「廣告少」在來源對應「行銷支援」，交付逐行卻配對「運送服務」。",
    # A needs_review finding whose words find no gap is not taken as soft.
    "Source values 100 and 80 match the delivered row.",
])
def test_remarks_about_wrong_missing_or_unsupported_wording(remark):
    assert not soft_review_remark(remark)


def _page(status, observation, issues=(), image_issues=(), reading=None, adjudications=None):
    page = {"images": [{"path": "p.png"}], "image_issues": list(image_issues),
            "review": {"blocks": [{"block_id": "b000412", "status": status, "observation": observation},
                                  {"block_id": "b000413", "status": "needs_review", "observation": "Wrong total."}],
                       "issues": list(issues), "source_complete": False, "reading_order_correct": True}}
    if reading:
        page["word_image_readings"] = {"b000412": reading}
    if adjudications:
        page["line_adjudications"] = {"b000412": adjudications}
    return page


def confirmed(page, delivered="晨光茶坊\n本店招牌"):
    return picture_content_confirmed(page, "b000412", {"b000412"}, delivered)


def test_a_verified_picture_is_not_dragged_down_by_its_page():
    assert confirmed(_page("verified", "Logo and wording match.", ["b000413: total differs from source."]))


def test_a_picture_whose_only_remark_is_layout_is_confirmed():
    remark = "Image relationship r0 remains missing: 招牌位於門口上方。"
    page = _page("needs_review", remark, ["b000412: " + remark, "b000413: total differs."],
                 [{"block_id": "b000412", "failure_kind": "image_relationship_mismatch", "detail": remark}])
    assert confirmed(page)


def test_a_wording_remark_anywhere_on_the_picture_keeps_it_unverified():
    page = _page("needs_review", "Image relationship r0 remains missing: 招牌位於門口上方。",
                 ["b000412 標籤文字誤植為「晨光茶房」，來源為「晨光茶坊」。"])
    assert not confirmed(page)
    disagreement = _page("needs_review", "b000412 的左右位置關係未說明。", image_issues=[
        {"block_id": "b000412", "failure_kind": "reading_disagreement", "detail": "b000412 的左右位置關係未說明。"}])
    assert not confirmed(disagreement)


def test_a_wording_remark_naming_no_block_keeps_the_picture_unverified():
    page = _page("needs_review", "b000412 的左右位置關係未說明。", ["交付文字「晨光茶房」與原圖「晨光茶坊」不符"])
    assert not confirmed(page)


def test_a_layout_remark_hiding_a_reading_the_delivery_lacks_keeps_it_unverified():
    # relationship_issue is reported instead of the lettering check; the lettering still has to hold.
    reading = {"text": "晨光茶坊\n每日現烤", "complete": True, "kind": "transcription"}
    page = _page("needs_review", "Image relationship r0 remains missing: 招牌位於門口上方。", reading=reading)
    assert not confirmed(page)
    settled = _page("needs_review", "Image relationship r0 remains missing: 招牌位於門口上方。", reading=reading,
                    adjudications=[{"reading_line": "每日現烤", "verdict": "delivery"}])
    assert confirmed(settled)
    assert confirmed(page, delivered="晨光茶坊\n每日現烤")


def test_an_unreviewed_or_unmatched_picture_is_not_confirmed():
    assert not picture_content_confirmed({"images": []}, "b000412", {"b000412"}, "")
    page = _page("needs_review", "Picture content is not covered by the text layer and was not confirmed.")
    assert not confirmed(page)
