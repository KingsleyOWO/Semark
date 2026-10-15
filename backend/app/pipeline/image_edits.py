"""Fix a picture's delivery where its audits pointed, with wording an independent reader saw.

Validation v20.6: a delivery patch added a missing footer relation correctly, then the
re-audit failed it on older faults in the kept text the first audit never named (an
invented 「橡皮」, page 66 drafted as 06, a 「最佳賣家」 gloss). A full re-read of a
text-heavy picture failed while merging its evidence. Here the model only names
exact quotes to change; the system applies an edit when it deletes description
wording, or brings in wording PaddleOCR or the blind second reading contain. A
transcription line is never deleted and changes only into a line OCR reads, and no
edit turns description wording OCR reads into wording it does not (spot check
v20.10 B: 光 was deleted from the card name 「晴空光年」 on a misreading). The
same per-image and page audits then decide.
"""
import json
import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import Field, model_validator

from app.pipeline.content_coverage import normalized_text
from app.pipeline.image_source_review import appears_in
from app.pipeline.recovery_models import Contract
from app.pipeline.zh_text import to_taiwan_traditional

EDIT_POLICY = """Fix delivered_text only where the original image shows it is wrong or incomplete.
known_issues come from earlier audits of this same delivery: check each against the pixels and fix every one
that is real, together with any other statement the image does not show. Each edit quotes delivered_text exactly.
A quoted span that occurs several times is edited everywhere it occurs, which the system allows only when the
image shows that span nowhere; otherwise quote a longer span that occurs once. op=delete removes description wording the image does not support (invented
objects, added translations or glosses, duplicated sentences). op=replace changes the quoted span; any new wording
must be copied verbatim from ocr_lines or independent_reading, or the edit is discarded. A line listed in
transcription_lines may only be replaced by that line exactly as ocr_lines reads it, and is never deleted.
op=add_relation adds the statement of one independent_reading relation_id the delivery omits. Never paraphrase
correct text and never add facts of your own. Return an empty edits list when nothing is wrong."""


class ImageEdit(Contract):
    op: Literal["replace", "delete", "add_relation"]
    quote: str = Field(default="", max_length=600)
    replacement: str = Field(default="", max_length=600)
    relation_id: str = Field(default="", max_length=24)
    reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def complete(self):
        if self.op == "add_relation" and not self.relation_id:
            raise ValueError("add_relation requires relation_id")
        if self.op != "add_relation" and not self.quote:
            raise ValueError(f"{self.op} requires an exact delivered quote")
        if self.op == "replace" and not self.replacement:
            raise ValueError("replace requires a replacement; use delete to remove wording")
        return self


class ImageEditProposal(Contract):
    edits: list[ImageEdit] = Field(max_length=16)
    observation: str = Field(min_length=1, max_length=600)


def _folded(text):
    return normalized_text(to_taiwan_traditional(text))


def _lines(text, start, end):
    """The lines of text that [start, end) touches; an empty span touches the line it sits in."""
    first = text[:start].count("\n")
    last = text[:max(start, end - 1)].count("\n")
    return text.split("\n")[first:last + 1]


def _new_wording(old, new):
    """The replacement's changed span, widened to whole Latin words and numbers (6 in 「66」 is 66)."""
    ops = [op for op in SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
           if op[0] != "equal" and op[4] > op[3]]
    if not ops:
        return ""
    start, end = ops[0][3], ops[-1][4]
    while start and new[start - 1].isascii() and new[start - 1].isalnum():
        start -= 1
    while end < len(new) and new[end].isascii() and new[end].isalnum():
        end += 1
    return new[start:end]


OCR_CONTEXT = 3  # characters each side of an edit: enough to hold a short CJK name such as 「晴空光年」


def _breaks_ocr_wording(text, changed, start, quote, replacement, ocr):
    """The edit turns wording PaddleOCR reads into wording it does not (光 deleted from 「晴空光年」)."""
    ops = [op for op in SequenceMatcher(None, quote, replacement, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return False
    before = text[max(0, start + ops[0][1] - OCR_CONTEXT):start + ops[-1][2] + OCR_CONTEXT]
    after = changed[max(0, start + ops[0][3] - OCR_CONTEXT):start + ops[-1][4] + OCR_CONTEXT]
    return bool(normalized_text(before)) and appears_in(before, ocr) and not appears_in(after, ocr)


def apply_edits(delivered, edits, *, transcription, ocr_lines, reading):
    """Apply the edits the rules allow, in order; return (text, statements, applied, rejected)."""
    ocr = "\n".join(ocr_lines or [])
    evidence = "\n".join([ocr, *([reading.text, *(r.source_fact for r in reading.relationships)] if reading else [])])
    relations = {r.relation_id: r for r in reading.relationships if r.certain} if reading else {}
    protected = {_folded(line) for line in transcription if _folded(line)}
    text, statements, applied, rejected = delivered, [], [], []

    def refuse(edit, why):
        rejected.append({**edit.model_dump(), "refused": why})

    def edited(text, start, edit):
        """The text with the occurrence at start edited, or unchanged with why the rules refuse it."""
        replacement = edit.replacement if edit.op == "replace" else ""
        changed = text[:start] + replacement + text[start + len(edit.quote):]
        if any(_folded(line) in protected for line in _lines(text, start, start + len(edit.quote))):
            if edit.op == "delete":
                return text, "a transcription line is never deleted"
            if not ocr or not all(appears_in(line, ocr) for line in _lines(changed, start, start + len(replacement))
                                  if normalized_text(line)):
                return text, "a transcription line changes only into the line PaddleOCR reads"
        elif normalized_text(wording := _new_wording(edit.quote, replacement)) and not appears_in(wording, evidence):
            return text, "new wording is in neither PaddleOCR nor the independent reading"
        elif ocr and _breaks_ocr_wording(text, changed, start, edit.quote, replacement, ocr):
            # The second reading can be the one that is wrong; OCR's wording is never edited away.
            return text, "would change description wording PaddleOCR reads"
        return changed, None

    for edit in edits:
        if edit.op == "add_relation":
            relation = relations.get(edit.relation_id)
            if relation is None:
                refuse(edit, "not a certain relationship of the independent reading")
                continue
            statement = "圖片關係：" + to_taiwan_traditional(relation.source_fact)
            if statement in statements or appears_in(relation.source_fact, text):
                refuse(edit, "already delivered")
                continue
            statements.append(statement)
            applied.append(edit.model_dump())
            continue
        starts = [match.start() for match in re.finditer(re.escape(edit.quote), text)]
        if not starts:
            refuse(edit, "quote is not in the delivered text")
            continue
        # A misreading the draft repeats (its line, then its description) is fixed everywhere, but
        # only when the picture shows the quoted wording nowhere: then no occurrence can be right.
        # 晴嵐 on v20.23: 12 of 13 refused corrections quoted a span written 2-7 times.
        if len(starts) > 1 and appears_in(edit.quote, evidence):
            refuse(edit, "quote occurs more than once and the picture shows it, so not every occurrence is wrong")
            continue
        # Changing every occurrence needs PaddleOCR to read the new wording: replayed on 814 stored
        # proposals, the blind reading alone renamed a company everywhere (qingya 晴雅 → qingva 晴微).
        if len(starts) > 1 and edit.op == "replace" and not (ocr and appears_in(edit.replacement, ocr)):
            refuse(edit, "quote occurs more than once and PaddleOCR does not read the replacement")
            continue
        result, why = text, None
        for start in reversed(starts):
            result, why = edited(result, start, edit)
            if why:
                break
        if why:
            refuse(edit, why)
            continue
        text = result
        applied.append(edit.model_dump())
    text = "\n".join(line for line in text.split("\n") if line.strip())
    return text, statements, applied, rejected


def transcription_lines(outputs, block_id):
    """The picture's drafted literal lines, which an edit may correct but never drop."""
    from app.pipeline.stages.package import split_vlm_lines

    path = outputs / "enrichments.jsonl"
    for line in path.read_text().splitlines() if path.is_file() else []:
        record = json.loads(line)
        if record.get("block_id") == block_id:
            return split_vlm_lines((record.get("output") or {}).get("all_text", ""), unique=False)
    return []


def known_issues(owner_state, recovery_state, page, block_id):
    """Every fault any audit named for this picture so far, page review and rejected candidates alike."""
    record = owner_state.get("pages", {}).get(str(page), {})
    issues = [i["detail"] for i in record.get("image_issues", []) if i["block_id"] == block_id]
    finding = next((b for b in record.get("review", {}).get("blocks", []) if b["block_id"] == block_id), None)
    if finding and finding.get("status") != "verified":
        issues.append(finding.get("observation", ""))
        issues += [f"Relationship {c['relation_id']} {c['status']}: {c.get('observation', '')}"
                   for c in finding.get("image_relationship_checks", []) if c.get("status") != "matched"]
    for step in recovery_state.get("steps", []):
        if step.get("block_id") == block_id:
            for feedback in step.get("validation_feedback") or []:
                issues += feedback.get("issues", []) + [b.get("observation", "") for b in feedback.get("blocks", [])]
    return list(dict.fromkeys(str(i)[:300] for i in issues if str(i).strip()))[:20]


# A remark that spells out a correction: 「Talvin」 should be 「Tolvin」, 誤植為「側窗圓弧」，來源為「側前圓弧」.
_CORRECTION = re.compile(r"should (?:be|read)|instead of|rather than|delivered as|\bnot\b.{0,40}\bas delivered|"
                         r"incorrect|misread|mis-?spel|typo|應為|應是|應作|誤植|誤寫|誤讀|讀作|讀為|寫成|讀成|錯寫|而非|有誤",
                         re.I)
_QUOTED = re.compile(r"「([^「」\n]{1,80})」|『([^『』\n]{1,80})』|“([^“”\n]{1,80})”|‘([^‘’\n]{1,80})’|"
                     r"\"([^\"\n]{1,80})\"|'([^'\n]{1,80})'")
# A wrong span shorter than this is replaced only where it occurs once (「tt」 in 「其中「tt」為藍色」).
REPEATED_SPAN_CHARS = 4


def spelled_out_corrections(remarks):
    """The (a, b) quote pairs of hard remarks that mark a correction; which one is right is for evidence to say."""
    from app.pipeline.review_policy import soft_review_remark

    pairs = []
    for remark in remarks:
        if soft_review_remark(remark) or not _CORRECTION.search(remark):
            continue
        quoted = list(dict.fromkeys(next(g for g in m.groups() if g is not None).strip()
                                    for m in _QUOTED.finditer(remark)))
        if len(quoted) == 2 and all(normalized_text(q) for q in quoted):
            pairs.append(tuple(quoted))
    return pairs


def review_corrections(remarks, delivered, ocr_lines):
    """Replacements a review spelled out and PaddleOCR confirms, as {wrong: right}.

    The user's rule (2026-10-13): when the review names the correct wording and text
    recognition agrees, change the wording directly instead of re-reading the picture.
    Only a hard remark marking a correction and quoting exactly two strings counts, and
    which of the two is wrong is left to evidence rather than the remark's grammar: the
    wrong one is delivered and OCR does not read it, the right one OCR reads. Spot check
    of the 2026-10-08 batch: the blind reading and OCR alone agreed on a wrong character
    in a company's name, so a remark is required.
    """
    ocr = "\n".join(ocr_lines or [])
    found = {}
    for pair in spelled_out_corrections(remarks) if ocr else []:
        for wrong, right in (pair, pair[::-1]):
            occurrences = delivered.count(wrong)
            if (occurrences and (occurrences == 1 or len(normalized_text(wrong)) >= REPEATED_SPAN_CHARS)
                    and not appears_in(wrong, ocr) and appears_in(right, ocr)
                    and SequenceMatcher(None, _folded(wrong), _folded(right)).ratio() >= 0.5):
                found.setdefault(wrong, set()).add(right)
    return {wrong: rights.pop() for wrong, rights in found.items() if len(rights) == 1}


def apply_corrections(text, corrections):
    """Every occurrence of each wrong span replaced, longest span first so none is cut inside another."""
    for wrong in sorted(corrections, key=len, reverse=True):
        text = text.replace(wrong, corrections[wrong])
    return text
