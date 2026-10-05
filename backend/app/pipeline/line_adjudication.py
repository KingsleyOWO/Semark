"""A third, independent reader settles lines the review's reading disagrees with the delivery on.

The second model reading is not ground truth: it shares the drafting model's
weights and sometimes "corrects" what the source actually prints. PaddleOCR
reads the same original with unrelated machinery. Only the differing spans are
compared, so OCR noise elsewhere in a long line does not matter; a span OCR
cannot settle leaves the line unverified.
"""
import difflib

from app.pipeline.content_coverage import normalized_text
from app.pipeline.zh_text import to_taiwan_traditional

CONTEXT = 3        # characters kept on each side of a differing span
MIN_EVIDENCE = 4   # a snippet shorter than this is too common to be a vote
MIN_SIMILARITY = 0.6


def _folded(text):
    return normalized_text(to_taiwan_traditional(text))


def _counterpart(line, delivered):
    target = _folded(line)
    best, score = None, 0.0
    for candidate in delivered.splitlines():
        folded = _folded(candidate)
        if folded:
            ratio = difflib.SequenceMatcher(None, target, folded, autojunk=False).ratio()
            if ratio > score:
                best, score = folded, ratio
    return best if score >= MIN_SIMILARITY else None


def _snippets(reading, delivered):
    """(delivered_snippet, reading_snippet) around each span where the two differ."""
    matcher = difflib.SequenceMatcher(None, delivered, reading, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            yield (delivered[max(0, i1 - CONTEXT):i2 + CONTEXT], reading[max(0, j1 - CONTEXT):j2 + CONTEXT])


def _ocr_window(text, ocr_lines):
    """The one to three consecutive OCR lines that read this line (OCR wraps long lines).

    Comparing against the whole picture would let 「Numeral Samples」 elsewhere
    vote on 「Alphabet Samles」.
    """
    folded = [_folded(line) for line in ocr_lines]
    best, score = "", 0.0
    for start in range(len(folded)):
        for end in range(start + 1, min(start + 3, len(folded)) + 1):
            window = "".join(folded[start:end])
            ratio = difflib.SequenceMatcher(None, text, window, autojunk=False).ratio()
            if ratio > score:
                best, score = window, ratio
    return best if score >= MIN_SIMILARITY else ""


def adjudicate(disputed, delivered, ocr_lines):
    """Verdict per disputed reading line: delivery (OCR backs the delivery), reading, or unresolved."""
    verdicts = []
    for line in disputed:
        reading = _folded(line)
        counterpart = _counterpart(line, delivered)
        verdict = "unresolved"
        if counterpart is None:
            if len(reading) >= MIN_EVIDENCE and reading in _ocr_window(reading, ocr_lines):
                verdict = "reading"
        else:
            ocr = _ocr_window(counterpart, ocr_lines)
            votes = set()
            for ours, theirs in _snippets(reading, counterpart):
                backs_delivery = len(ours) >= MIN_EVIDENCE and ours in ocr
                backs_reading = len(theirs) >= MIN_EVIDENCE and theirs in ocr
                votes.add("delivery" if backs_delivery and not backs_reading else
                          "reading" if backs_reading and not backs_delivery else "unresolved")
            verdict = ("reading" if "reading" in votes else
                       "delivery" if votes == {"delivery"} else "unresolved")
        verdicts.append({"reading_line": line, "delivered_line": counterpart, "verdict": verdict})
    return verdicts
