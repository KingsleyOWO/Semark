"""Shared content acceptance rules; source binding and delivery checks stay with callers."""
import re


SOURCE_COMPLETENESS_POLICY = (
    "Judge completeness relative to information actually present in the source, not an ideal expanded dataset. "
    "A source-unlabelled bar has no exact value to transcribe; do not demand invented values or declare "
    "an extraction gap solely for labels, years or fields the source itself does not supply. "
    "A faithful statement that a source label is absent is not an unverified extraction. "
    "Still require every visible substantive label, number, unit and their correct relationships; "
    "illegible visible lettering and missing essential context remain genuine uncertainty. "
    "Distinguish an incidental decorative watermark from meaningful branding, labels or diagram symbols. "
    "A decorative watermark with no readable wording or role in the source meaning may be omitted; "
    "never dismiss a meaningful logo, legend or label merely because it is small or hard to read."
)


def reviewed_block_usable(page, finding):
    """A partial page needs an explicit context-complete finding for each usable block."""
    return bool(finding and finding.get("status") == "verified"
                and page.get("review", {}).get("reading_order_correct")
                and (page.get("status") == "verified" or finding.get("context_complete")))


def partial_review_accepted(review, gap_ids, required_ids=None):
    """Only declared gaps may remain; all retained content needs complete context."""
    gaps = set(gap_ids)
    findings = review.get("blocks", [])
    ids = [item.get("block_id") for item in findings]
    if (not gaps or not review.get("declared_gaps_only") or review.get("source_complete")
            or not review.get("reading_order_correct") or len(ids) != len(set(ids))
            or None in ids or not gaps <= set(ids)
            or required_ids is not None and set(ids) != set(required_ids)):
        return False
    return all(item.get("status") == "needs_review" if item["block_id"] in gaps else
               item.get("status") == "verified" and item.get("context_complete") for item in findings)


def dependency_closed_subset(candidates, dependencies):
    """Remove anything whose required context is absent, including transitive dependants."""
    retained = set(candidates)
    while True:
        rejected = {bid for bid in retained if not set(dependencies.get(bid, ())) <= retained}
        if not rejected:
            return retained
        retained -= rejected


# A picture's own content counts as confirmed when its review found nothing wrong or missing in the
# delivered wording (decided 2026-10-13). Of 1899 pictures a 44-report batch labelled unverified, 436
# drew only remarks that a relationship (a side, an arrow, a grid position) was not spelled out or that
# small print is illegible, and 178 were verified themselves on a page that failed elsewhere; only 25
# had a wrong description. Remarks were sorted by hand on samples of the batch's review records.
_SOFT_CODE_REMARK = re.compile(
    r"Independent image relationships are incomplete|Image relationship audit omitted or duplicated|"
    r"Image relationship \S+ remains (?:missing|uncertain)|has no (?:grounded|owned) quote|"
    r"Split image relationship coverage is incomplete|Independent Word image reading is incomplete|"
    r"Independent Word image reading supplied no content")
_HARD_CODE_REMARK = re.compile(r"remains contradicted|direction differs|position differs|"
                               r"Independent original image wording is absent or different")
_RELATION = re.compile(
    r"\b(?:relationships?|relations?|spatial|layout|arrangement|positions?|placement|arrows?|axis|axes|"
    r"directions?|grid|connectors?|flow|left|right|above|below|order|sequence|structure|columns?|rows?)\b|"
    r"位置|關係|空間|排列|排版|版面|佈局|布局|箭頭|方向|左右|上下|左側|右側|上方|下方|對應|連線|流程|順序|座標|軸", re.I)
# A relationship remark is soft only when it says the relationship is not spelled out.
_NOT_SPELLED_OUT = re.compile(
    r"not (?:explicitly |clearly |fully |completely )?(?:stated|preserved|expressed|described|mentioned|conveyed|"
    r"captured|shown|represented|retained|maintained|indicated|specified|reflected|spelled out|clear)|"
    r"(?:does|do|did)(?: not|n't) (?:explicitly |clearly |fully )?(?:state|preserve|express|describe|mention|convey|"
    r"capture|show|represent|indicate|specify|reflect|spell out|retain|keep|include)|"
    r"no (?:explicit|spatial|layout|positional)|without|flat list|only (?:lists?|as|by)|rather than|incomplete|"
    r"未(?:明確|完整|清楚)?(?:說明|保留|呈現|表達|描述|交付|標示|提及|標明|體現|反映|指出|連結|對應|區分)|"
    r"未在.{0,12}(?:說明|保留|呈現|表達|描述|交付|標示)|僅以|僅列|僅呈現|沒有說明|無法(?:從|由)?.{0,8}(?:得知|判斷)", re.I)
_ILLEGIBLE = re.compile(
    r"illegible|legib|too small|small (?:text|print|lettering)|low resolution|resolution|blurr|unclear|"
    r"cannot be (?:fully )?(?:read|verified|confirmed|resolved)|could not be (?:read|verified|confirmed)|"
    r"unable to (?:read|verify|confirm)|not (?:fully )?(?:legible|readable)|uncertain|unresolved|"
    r"無法(?:清楚|完整|完全)?(?:辨識|判讀|辨認|確認|閱讀|核對|驗證)|看不清|不清楚|模糊|字跡過小|過小|太小|解析度|小字|"
    r"難以(?:辨識|判讀)", re.I)
# Anything saying delivered wording is wrong, missing, added or unsupported, in the reviewers' own phrasing.
_WORDING_PROBLEM = re.compile(
    r"誤植|誤讀|誤寫|誤認|錯|寫成|讀成|打成|應為|應是|應作|不一致|不符|不同|矛盾|相反|顛倒|多寫|多出|少寫|"
    r"缺少|缺漏|遺漏|漏掉|漏列|漏植|未見|未列|未轉錄|未包含|未收錄|未納入|未呈現.{0,6}(?:文字|字樣|標籤|數值|名稱)|"
    r"沒有.{0,8}(?:文字|字樣|標籤|數值|名稱)|不存在|虛構|捏造|推測|臆測|無中生有|並未出現|未出現|"
    r"(?:未保留|未交付|未列出|未提及|未記載|未寫出|沒有|無)\S{0,12}(?:文字|字樣|標籤|數值|數字|名稱|標示|標題|字)|"
    r"交付.{0,40}無法.{0,12}(?:確認|核對|證實|驗證)|未驗證|交付(?:描述|文字)?(?:為|是|寫|稱|所述)|"
    r"獨立讀(?:取|圖)?為|讀取為|未逐字|卻|配對錯|錯配|原圖(?:為|是|寫|作)|來源(?:為|是|寫|作)|原文(?:為|是|寫|作)|"
    r"未完整(?:交付|轉錄|呈現|列出|保留)|細節.{0,10}無法(?:確認|核對|證實)|降級|讀作|讀為|記為|轉為|變成|改為|"
    r"可讀(?:資訊|文字|內容)|未(?:明確)?保留(?!.{0,20}(?:關係|位置|方向|箭頭|排列|順序|結構|對應|佈局|布局|版面|連線|網格))|"
    r"absent|missing|omit|lacks?\b|differ|mismatch|misread|incorrect|wrong|typo|instead|should be|"
    r"hallucin|fabricat|invent|not (?:present|visible|shown|found) in|asserts|unsupported|not supported|"
    r"extra\b|added|discrepan|vs\.?\s|truncat|截斷|duplicat|重複|delivered as|delivery (?:says|reads|has|gives|shows)|"
    r"source (?:image )?(?:reads|says|shows|has)|reads? ['\"「]|but (?:the )?(?:source|original)|but the independent|"
    r"\bclaim(?:s|ed)?\b|mis-?pair|\breads\b|\badds?\b|\bchang(?:e|es|ed|ing)\b|\bunverified\b|plausible|"
    r"not (?:fully )?verifiable|unverifiable|not (?:fully |completely )?(?:transcribed|delivered)|"
    r"\b(?:text|qualifiers?|labels?|wording|words?|numbers?|values?|names?)\b.{0,30}"
    r"not (?:fully |completely |explicitly )?(?:captured|included|listed|present)|"
    r"(?:delivered|delivery)\b.{0,60}(?:cannot|could not|can't|not) be (?:verified|confirmed)", re.I)
_BLOCK_ID = re.compile(r"(?<![\w-])(?:b\d{6}|recovery_[\w-]+)(?![\w-])")


def soft_review_remark(text):
    """True for a remark that only misses a spelled-out relationship or meets illegible print.

    Such a remark finds nothing wrong in what is delivered; a remark that wording is wrong,
    missing, added or unsupported, or that a relationship is contradicted, is not soft.
    """
    if _SOFT_CODE_REMARK.search(text) and not _HARD_CODE_REMARK.search(text):
        return True
    if _HARD_CODE_REMARK.search(text) or _WORDING_PROBLEM.search(text):
        return False
    return bool(_RELATION.search(text) and _NOT_SPELLED_OUT.search(text) or _ILLEGIBLE.search(text))


def _names(remark, ids):
    return any(found in ids for found in _BLOCK_ID.findall(remark))


def picture_content_confirmed(page, block_id, names, delivered):
    """Whether the review of one source page confirms a picture block's own content.

    ``names`` are the IDs a remark may call the picture by (its own and its original
    picture's); ``delivered`` is the block's delivered text. The block's own verified
    finding confirms it whatever failed elsewhere on the page. A finding left at
    needs_review confirms it only when every remark naming the picture is soft, no
    remark on the page naming no block at all finds a wording problem, and the blind
    reading's lettering is delivered or settled for the delivery by OCR.
    """
    from app.pipeline.content_coverage import normalized_text
    from app.pipeline.image_source_review import appears_in

    review = page.get("review") or {}
    findings = [f for f in review.get("blocks", []) if f.get("block_id") == block_id]
    if not page.get("images") or len(findings) != 1:
        return False
    if findings[0].get("status") == "verified":
        return True
    image_issues = [i for i in page.get("image_issues") or [] if i.get("block_id") == block_id]
    if any(i.get("failure_kind") == "reading_disagreement" for i in image_issues):
        return False
    issues = review.get("issues") or []
    remarks = [findings[0].get("observation", ""), *(i.get("detail", "") for i in image_issues),
               *(i for i in issues if _names(i, names))]
    if not all(soft_review_remark(r) for r in remarks if r.strip()):
        return False
    if any(not _BLOCK_ID.search(i) and not soft_review_remark(i) for i in issues):
        return False
    # A relationship failure is reported in place of the lettering check; run that check here.
    reading = (page.get("word_image_readings") or {}).get(block_id) or {}
    if reading.get("complete") and reading.get("kind", "transcription") == "transcription":
        settled = {v.get("reading_line") for v in (page.get("line_adjudications") or {}).get(block_id, [])
                   if v.get("verdict") == "delivery"}
        if any(normalized_text(line) and not appears_in(line, delivered) and line not in settled
               for line in str(reading.get("text") or "").splitlines()):
            return False
    return True
