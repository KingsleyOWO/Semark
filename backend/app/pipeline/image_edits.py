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
import unicodedata
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
    """The replacement's changed span, widened to whole Latin words and numbers (6 in 「66」 is 66,
    5 in 1.5 is 1.5)."""
    ops = [op for op in SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
           if op[0] != "equal" and op[4] > op[3]]
    if not ops:
        return ""
    start, end = ops[0][3], ops[-1][4]
    while start and _word_class(new, start - 1) == "latin":
        start -= 1
    while end < len(new) and _word_class(new, end) == "latin":
        end += 1
    return new[start:end]


def _misreading(quote, replacement):
    """At most half the quote's characters change and the rest keep their order (腸 → 暢 in 好腸安)."""
    old, new = _folded(quote), _folded(replacement)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2
                  in SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if tag != "equal")
    return bool(old) and changed * 2 <= len(old)


def _word_class(text, index):
    """A Latin word or number (a '.', ',', '/' or ':' between two digits joins one, as in 3.33), other
    letters (a CJK run), or nothing (spaces and punctuation)."""
    char = text[index]
    if char.isascii():
        if char.isalnum():
            return "latin"
        joins = char in ".,/:" and 0 < index < len(text) - 1
        return "latin" if joins and text[index - 1].isdigit() and text[index + 1].isdigit() else None
    return "letters" if char.isalnum() else None


def _changed_words(new, start, quote, replacement):
    """The words of new (a text whose quote at start was replaced) that the edit changes.

    A Latin word or number is taken whole from the edited text, so a quote cut inside one still
    names it (6 in 「66」 is 66, 2 in 3.23 is 3.33). Other letters have no word boundary to find:
    their run is taken within the replacement, as the edit's own wording.
    """
    words = []
    end = start + len(replacement)
    for tag, _, _, j1, j2 in SequenceMatcher(None, quote, replacement, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        positions = range(start + j1, start + j2) if j2 > j1 else [start + j1 - 1, start + j1]
        for position in positions:
            if not 0 <= position < len(new) or not (kind := _word_class(new, position)):
                continue
            low, high = (0, len(new)) if kind == "latin" else (start, end)
            if not low <= position < high:
                continue
            left = right = position
            while left > low and _word_class(new, left - 1) == kind:
                left -= 1
            while right + 1 < high and _word_class(new, right + 1) == kind:
                right += 1
            if (word := new[left:right + 1]) not in words:
                words.append(word)
    return words


_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:[.,/:][0-9]+)*")
_NUMBER = re.compile(r"[0-9]+(?:[.,/:][0-9]+)*")
_LABEL = re.compile(r"[^\W\d_]")  # a letter or CJK character: wording beside a number
# Bopomofo letters printed like a Latin letter or a CJK character.
_LOOKALIKES = str.maketrans({"ㄚ": "Y", "ㄒ": "T", "ㄨ": "X", "ㄧ": "一", "ㄇ": "冂", "ㄈ": "匚", "ㄏ": "厂", "ㄩ": "凵"})


def _lookalike(text):
    return unicodedata.normalize("NFKC", text).translate(_LOOKALIKES)


def _reads(word, text):
    """The text has the word: one Latin word or number whole (Leu is not read inside Fleur, nor 1.5
    as 15 or inside 1.57), anything else as appears_in compares it."""
    if _LATIN_WORD.fullmatch(word):
        pattern = r"(?<![a-z0-9])" + re.escape(word.casefold()) + r"(?![a-z0-9])"
        return any(re.search(pattern, line.casefold()) for line in text.split("\n"))
    return appears_in(word, text)


def _closeness(phrase, lines):
    """The share of the phrase one OCR line holds in order, at best; only ever compared, never a bar."""
    folded = _folded(phrase)
    if not folded:
        return 0.0
    return max((sum(block.size for block in SequenceMatcher(None, folded, _folded(line), autojunk=False)
                    .get_matching_blocks()) / len(folded) for line in lines if _folded(line)), default=0.0)


# Wording a delivery uses for lettering nobody could read: a placeholder ([?], [手寫簽名]) or a remark.
_UNREAD = re.compile(r"\[\?\]|\[[^\[\]\n]{0,24}(?:手寫|簽名|簽章|模糊|無法|不清|illegible|unreadable|handwrit|signature)"
                     r"[^\[\]\n]{0,24}\]|〔無法可靠辨識|模糊|難以?辨|無法(?:可靠|清楚|完整)?(?:辨識|辨認|判讀)|看不清|"
                     r"illegible|unreadable", re.I)
# Wording that says a reading is uncertain; removing it states the reading as fact.
_HEDGE = re.compile(r"或(?:相近|類似|近似)|相近字樣|疑似|似乎|看似|推測|可能(?:是|為)|不確定|無法確認|字樣不清|\[\?\]|"
                    r"\b(?:possibly|probably|likely|uncertain|unclear)\b", re.I)
# Quoted wording in a relation's statement: 「…」, 『…』, “…” or "…".
_RELATION_QUOTE = re.compile(r"「([^「」\n]+)」|『([^『』\n]+)』|“([^“”\n]+)”|\"([^\"\n]+)\"")


OCR_CONTEXT = 3  # characters each side of an edit: enough to hold a short CJK name such as 「晴空光年」


def _breaks_ocr_wording(text, changed, start, quote, replacement, ocr):
    """The edit turns wording PaddleOCR reads into wording it does not (光 deleted from 「晴空光年」)."""
    ops = [op for op in SequenceMatcher(None, quote, replacement, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return False
    before = text[max(0, start + ops[0][1] - OCR_CONTEXT):start + ops[-1][2] + OCR_CONTEXT]
    after = changed[max(0, start + ops[0][3] - OCR_CONTEXT):start + ops[-1][4] + OCR_CONTEXT]
    return bool(normalized_text(before)) and appears_in(before, ocr) and not appears_in(after, ocr)


def apply_edits(delivered, edits, *, transcription, ocr_lines, reading, document_ocr=()):
    """Apply the edits the rules allow, in order; return (text, statements, applied, rejected).

    ``document_ocr`` lists the PaddleOCR lines of the document's other pictures (one list each).
    """
    ocr = "\n".join(ocr_lines or [])
    evidence = "\n".join([ocr, *([reading.text, *(r.source_fact for r in reading.relationships)] if reading else [])])
    relations = {r.relation_id: r for r in reading.relationships if r.certain} if reading else {}
    protected = {_folded(line) for line in transcription if _folded(line)}
    blind = "\n".join([reading.text, *(r.source_fact for r in reading.relationships)]) if reading else ""
    others = ["\n".join(lines) for lines in document_ocr if lines and list(lines) != list(ocr_lines or [])]
    text, statements, applied, rejected = delivered, [], [], []

    def refuse(edit, why):
        rejected.append({**edit.model_dump(), "refused": why})

    def backing(text, start, edit):
        """Which OCR reads the words this occurrence's replacement changes: "here" when this picture's
        OCR reads them; "elsewhere" when another picture's OCR reads them and this picture's own,
        garbled, comes closer to the replacement than to the quote; else None.

        Rerun 2026-10-15: on page-sized screenshots whose print is about 5 px tall, OCR read
        「品牌统力」 for a radar chart's 「品牌統御力」, while a sharper form in the same report
        printed all eight indicator names; the chart's draft had invented four of them.
        """
        if edit.op != "replace" or not ocr:
            return None
        changed = text[:start] + edit.replacement + text[start + len(edit.quote):]
        words = _changed_words(changed, start, edit.quote, edit.replacement)
        if not words:
            return None
        if all(_reads(word, ocr) for word in words):
            return "here"
        # Only for Chinese wording, read whole in another picture, and only when this picture's own
        # garbled OCR holds a character the replacement brings in and the quote lacks: replayed on the
        # stored proposals, word-by-word support from other pictures turned a date into 「LINE 2020」.
        new_chars = set(_folded(edit.replacement)) - set(_folded(edit.quote))
        if (others and new_chars and not any(_LATIN_WORD.fullmatch(word) for word in words)
                and any(appears_in(edit.replacement, other) for other in others)
                and any(new_chars & set(_folded(line)) for line in ocr_lines or [])
                and _closeness(edit.replacement, ocr_lines) > _closeness(edit.quote, ocr_lines)):
            return "elsewhere"
        return None

    def seconded(text, start, edit):
        """The blind reading reads the words this occurrence brings in, and none of those it replaces."""
        changed = text[:start] + edit.replacement + text[start + len(edit.quote):]
        brought = _changed_words(changed, start, edit.quote, edit.replacement)
        replaced = _changed_words(text, start, edit.replacement, edit.quote)
        return (bool(blind) and all(_reads(word, blind) for word in brought)
                and not any(_reads(word, blind) for word in replaced))

    def edited(text, start, edit):
        """The text with the occurrence at start edited, or unchanged with why the rules refuse it."""
        replacement = edit.replacement if edit.op == "replace" else ""
        changed = text[:start] + replacement + text[start + len(edit.quote):]
        if (edit.op == "replace" and _UNREAD.search(edit.quote) and normalized_text(_new_wording(edit.quote, replacement))
                and backing(text, start, edit) != "here"):
            # The blind reading guessed the same signature as three different names on three forms.
            return text, "an unreadable spot is filled only with wording this picture's PaddleOCR reads"
        if edit.op == "replace" and (numbers := [word for word in _changed_words(changed, start, edit.quote, replacement)
                                                 if _NUMBER.fullmatch(word)]) and not (ocr and all(
                                                     _reads(number, ocr) for number in numbers)):
            # The blind reading names a chart's values axis by axis; one of them is no evidence for another
            # axis (a radar chart's 好感度 3.5 was edited from 1.67 to 1.5, another axis's value).
            return text, "a number changes only into one this picture's PaddleOCR reads"
        if (edit.op == "replace" and ocr and any(_NUMBER.fullmatch(word) and _reads(word, ocr)
                                                 for word in _changed_words(text, start, replacement, edit.quote))
                and not (_LABEL.search(replacement) and appears_in(replacement, ocr))
                and not all(appears_in(line, ocr) for line in _lines(changed, start, start + len(replacement))
                            if normalized_text(line))):
            # A picture printing both numbers backs neither: a slide-thumbnail sheet labelled 54 to 71 had a
            # description's 「第66頁」 edited to 「第71頁」 on where the model thought the slide sat. OCR reading
            # the number with its label (「Top 3 0」) or the whole new line does back it.
            return text, "a number this picture's PaddleOCR also reads changes only where OCR reads it with its label"
        if _HEDGE.search(edit.quote) and not _HEDGE.search(replacement):
            # 「NORIA」（或相近字樣） became 「NORIA」: the screen shows "?ORIA", the same logo another
            # photo reads SORIA. What the hedge held back must be what this picture's PaddleOCR reads.
            line = "\n".join(_lines(changed, start, start + len(replacement)))
            words = [next(g for g in m.groups() if g) for m in _RELATION_QUOTE.finditer(line)] or [_new_wording(edit.quote, replacement) or line]
            if not (ocr and all(_reads(word, ocr) for word in words if normalized_text(word))):
                return text, "a hedge is removed only when this picture's PaddleOCR reads what it held back"
        touched = _lines(text, start, start + len(edit.quote))
        if any(_folded(line) in protected for line in touched):
            # Nor dropped by a replacement spanning it: 「晴諾米\nBright Forever\n晴川銀行」 became
            # 「晴諾米\n晴川銀行」 on a remark the badge was no brand; the badge reads "Bright Together".
            if edit.op == "delete" or (sum(bool(normalized_text(line)) for line in _lines(changed, start, start + len(replacement)))
                                       < sum(bool(normalized_text(line)) for line in touched)):
                return text, "a transcription line is never deleted"
            whole = ocr and all(appears_in(line, ocr) for line in _lines(changed, start, start + len(replacement))
                                if normalized_text(line))
            # A line also changes when the blind reading reads what the edit brings in and none of what it
            # replaces, never by losing lettering, and either the change is a number this picture's OCR
            # reads (3.23 → 3.33 in a line whose garbled label OCR cannot read) or a Chinese name read
            # whole in another picture. Letters OCR and the blind reading read alike are not enough: both
            # took a crowdfunding site's 「嘖嘖」 for 「噴噴」 and a logo's K for V; OCR alone
            # read 200g where the blind reading kept 290g.
            support = backing(text, start, edit) if len(_folded(replacement)) >= len(_folded(edit.quote)) else None
            numbers = all(_NUMBER.fullmatch(word) for word in _changed_words(changed, start, edit.quote, replacement))
            if not whole and not (seconded(text, start, edit)
                                  and (support == "here" and numbers or support == "elsewhere")):
                return text, "a transcription line changes only into the line PaddleOCR reads"
        elif normalized_text(wording := _new_wording(edit.quote, replacement)) and not _reads(wording, evidence):
            return text, "new wording is in neither PaddleOCR nor the independent reading"
        elif edit.op == "replace" and (latin := [word for word in _changed_words(changed, start, edit.quote, replacement)
                                                 if re.search(r"[A-Za-z]", word)]) and not (ocr and all(
                                                     _reads(word, ocr) for word in latin)):
            # As in a transcription line, a respelt Latin word needs this picture's OCR: the blind
            # reading alone turned a label's 「Lumora」, which OCR read 「Lumara」, into 「Lumosa」.
            return text, "a Latin word changes only into one this picture's PaddleOCR reads"
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
            # A relation quoting lettering nobody else saw states it as fact: an audit of a workshop
            # poster matched 「我們為誰解決什麼問題」, which the poster does not print, to itself.
            # A form's blank (「__ 數位行銷 __」) is no lettering.
            quoted = [next(g for g in m.groups() if g).replace("_", "")
                      for m in _RELATION_QUOTE.finditer(relation.source_fact)]
            if any(not appears_in(q, text) and not (ocr and appears_in(q, ocr)) for q in quoted):
                refuse(edit, "the relation quotes wording neither delivered nor read by PaddleOCR")
                continue
            statements.append(statement)
            applied.append(edit.model_dump())
            continue
        if edit.op == "replace" and to_taiwan_traditional(edit.quote) == edit.quote:
            # PaddleOCR answers in simplified glyphs (健锋, 片装) for traditional print; the delivery
            # keeps its zh-TW script. The rules below compare both scripts alike.
            edit = edit.model_copy(update={"replacement": to_taiwan_traditional(edit.replacement)})
        if edit.op == "replace" and edit.replacement == edit.quote:
            # Replayed, 75 proposed edits repeated their quote and 20 only turned zh-TW glyphs simplified.
            refuse(edit, "the edit changes nothing but the script")
            continue
        if edit.op == "replace" and _lookalike(edit.quote) == _lookalike(edit.replacement):
            # A chat partner's 「晴諾茶葉ㄚ倫」 (ㄚ倫, a nickname) became 「…Y倫」 on OCR's reading: the two
            # glyphs print alike, so neither the readers nor OCR can tell them apart (test run 2026-10-19).
            refuse(edit, "a lookalike letter of another script is the same print")
            continue
        starts = [match.start() for match in re.finditer(re.escape(edit.quote), text)]
        # An edit names what was delivered, never what another edit of the same proposal wrote.
        if not starts or edit.quote not in delivered:
            refuse(edit, "quote is not in the delivered text")
            continue
        backed = [backing(text, start, edit) for start in starts] if len(starts) > 1 else []
        # Only a misreading is repeated: most of the quote stays in place. Replayed, the rewrites that
        # passed the checks below were wrong: a title OCR read as "T", a right-to-left couplet put in
        # OCR's left-to-right order, a generic noun swapped for a product name. Wording read in
        # another picture of the document is not a rewrite: the draft invented the quoted name.
        if (len(starts) > 1 and edit.op == "replace" and not _misreading(edit.quote, edit.replacement)
                and set(backed) != {"elsewhere"}):
            refuse(edit, "quote occurs more than once and the edit rewrites it rather than fixing a misreading")
            continue
        # A misreading the draft repeats (its line, then its description) is fixed everywhere, but
        # only when the picture shows the quoted wording nowhere: then no occurrence can be right.
        # Rerun on v20.23: 12 of 13 refused corrections quoted a span written 2-7 times.
        if len(starts) > 1 and appears_in(edit.quote, evidence):
            refuse(edit, "quote occurs more than once and the picture shows it, so not every occurrence is wrong")
            continue
        # Changing every occurrence needs PaddleOCR to read the words it changes: replayed on 814
        # stored proposals, the blind reading alone renamed a company everywhere.
        if len(starts) > 1 and edit.op == "replace" and None in backed:
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


def known_issues(owner_state, recovery_state, page, block_id, part_ids=()):
    """Every fault any audit named for this picture so far, page review and rejected candidates alike.

    A repaired picture's are also on the blocks it is delivered as (``part_ids``), and in the
    page remarks naming them.
    """
    from app.pipeline.review_policy import _BLOCK_ID

    record = owner_state.get("pages", {}).get(str(page), {})
    names = {block_id, *part_ids}
    issues = [i["detail"] for i in record.get("image_issues", []) if i["block_id"] in names]
    for finding in record.get("review", {}).get("blocks", []):
        if finding["block_id"] in names and finding.get("status") != "verified":
            issues.append(finding.get("observation", ""))
            issues += [f"Relationship {c['relation_id']} {c['status']}: {c.get('observation', '')}"
                       for c in finding.get("image_relationship_checks", []) if c.get("status") != "matched"]
    if part_ids:
        issues += [i for i in record.get("review", {}).get("issues") or [] if set(_BLOCK_ID.findall(i)) & names]
    for step in recovery_state.get("steps", []):
        if step.get("block_id") == block_id:
            for feedback in step.get("validation_feedback") or []:
                issues += feedback.get("issues", []) + [b.get("observation", "") for b in feedback.get("blocks", [])]
    return list(dict.fromkeys(str(i)[:300] for i in issues if str(i).strip()))[:20]


# A remark that spells out a correction: 「Talvin」 should be 「Tolvin」, 誤植為「側窗圓弧」，來源為「側前圓弧」.
# 誤記 and 誤標 too: a radar chart's legend 「貴公司診斷分數」 誤記為 another went unrepaired (rerun 2026-10-16).
_CORRECTION = re.compile(r"should (?:be|read)|instead of|rather than|delivered as|\bnot\b.{0,40}\bas delivered|"
                         r"incorrect|misread|mis-?spel|typo|應為|應是|應作|誤植|誤寫|誤讀|誤記|誤標|讀作|讀為|寫成|讀成|錯寫|而非|有誤",
                         re.I)
_QUOTED = re.compile(r"「([^「」\n]{1,80})」|『([^『』\n]{1,80})』|“([^“”\n]{1,80})”|‘([^‘’\n]{1,80})’|"
                     r"\"([^\"\n]{1,80})\"|'([^'\n]{1,80})'")


def _whole_latin_words(text, span):
    """No occurrence of the span sits inside a longer Latin word or number (「tt」 inside 「Matt」)."""
    edges = (span[:1].isascii() and span[:1].isalnum(), span[-1:].isascii() and span[-1:].isalnum())
    for match in re.finditer(re.escape(span), text):
        before = text[match.start() - 1:match.start()]
        after = text[match.end():match.end() + 1]
        if (edges[0] and before.isascii() and before.isalnum()) or (edges[1] and after.isascii() and after.isalnum()):
            return False
    return True


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
        # One quote inside the other is wording added or dropped, not a misreading: OCR always reads the
        # shorter, so 「社群策略 建議」, which the picture shows, became 「建議」 (rerun 2026-10-16).
        if _folded(pair[0]) in _folded(pair[1]) or _folded(pair[1]) in _folded(pair[0]):
            continue
        for wrong, right in (pair, pair[::-1]):
            occurrences = delivered.count(wrong)
            # A span written several times changes everywhere under the targeted edits' rule: a
            # misreading, never a piece of a longer Latin word (「tt」 in 「其中「tt」為藍色」 and in Matt).
            if (occurrences and (occurrences == 1 or _misreading(wrong, right) and _whole_latin_words(delivered, wrong))
                    and not appears_in(wrong, ocr) and appears_in(right, ocr)
                    and SequenceMatcher(None, _folded(wrong), _folded(right)).ratio() >= 0.5):
                found.setdefault(wrong, set()).add(right)
    return {wrong: rights.pop() for wrong, rights in found.items() if len(rights) == 1}


def apply_corrections(text, corrections):
    """Every occurrence of each wrong span replaced, longest span first so none is cut inside another."""
    for wrong in sorted(corrections, key=len, reverse=True):
        text = text.replace(wrong, corrections[wrong])
    return text
