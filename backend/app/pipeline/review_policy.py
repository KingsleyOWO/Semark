"""Shared content acceptance rules; source binding and delivery checks stay with callers."""


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
