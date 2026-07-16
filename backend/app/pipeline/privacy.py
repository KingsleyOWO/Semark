"""Deterministic privacy scrub for delivered pipeline text.

Source documents are built from real screenshots, and both extraction layers
leak personal content that is unrelated to the teaching material:

- the VLM transcribes inbox panes (private mail subjects/senders) despite the
  prompt asking it not to — compliance is probabilistic;
- MinerU OCRs screenshot regions into plain TEXT/TABLE blocks (live: a real
  domain account 「CORP\\x12345」 reached rag.md as a parser text block).

The scrub therefore runs on every delivered surface (rag.md, main_text.md,
chunks.jsonl, document exports) rather than only on VLM output. Teaching
content survives: account-format instructions carry no digits, and UI menu
labels (「另存新檔(A)...」) are explicitly preserved.

The module-level toggle mirrors PackageConfig.scrub_private_info (settings
page switch); stages set it at run start. Concurrent runs share the same
stored setting, so the flag cannot diverge between them.
"""

from __future__ import annotations

import re

__all__ = [
    "scrub_transcribed_privacy",
    "set_privacy_scrub_enabled",
]

_PRIVACY_SCRUB_ENABLED = True

# A mail-list row transcribed from an inbox screenshot: a 1-3 char truncated
# sender name followed by an ellipsis (「王.. 報價」「d… 2016/10/21 待領」).
_MAIL_SENDER_LINE_RE = re.compile(r"^\s*[^\W\d_]{1,3}(?:\.{2,}|…)", re.MULTILINE)
# A bare reply/forward subject line.
_MAIL_SUBJECT_LINE_RE = re.compile(r"^\s*(?:RE|FW|回覆|轉寄)[:：]", re.IGNORECASE | re.MULTILINE)
# Windows-domain accounts with a personal numeric id (DOMAIN\x12345). The
# format teaching (「網域\員工編號」, no digits) never matches. Explicit
# lookarounds instead of \b: Python counts CJK as word chars, so 「為demo\…」
# and a trailing 「…分享」 would defeat word boundaries.
_DOMAIN_ACCOUNT_RE = re.compile(
    r"(?<![A-Za-z0-9\\])([A-Za-z][A-Za-z0-9]{1,15}\\[A-Za-z])\d{4,6}(?![0-9])"
)
# The same personal id in email form (d23456@example.tw). Service aliases
# without a numeric id (support@...) pass through.
_EMAIL_ACCOUNT_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z][A-Za-z._%+-]*)\d{4,6}(?=@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
# Any line cropped by a screenshot edge (trailing ellipsis).
_TRUNCATED_LINE_RE = re.compile(r"(?:\.{2,}|…)\s*$", re.MULTILINE)
# Stable UI markers of a mail-client message-list pane. Everything short and
# non-sentence after one of these is subject/sender content — the VLM re-rolls
# the exact line shape on every fresh enrichment, so region detection is the
# only rule that survives resampling.
_INBOX_REGION_MARKER_RE = re.compile(
    r"^(?:全部|未讀取|已讀取|收件匣\s*\d*|日期[:：]\s*(?:今天|昨天|本週|上週|上個月|較舊|更舊))$"
)


def set_privacy_scrub_enabled(enabled: bool) -> None:
    global _PRIVACY_SCRUB_ENABLED
    _PRIVACY_SCRUB_ENABLED = bool(enabled)


def scrub_transcribed_privacy(text: str) -> str:
    """Drop mail-list content and mask domain-account ids.

    Everything else passes through unchanged; returns the input untouched
    when the settings toggle is off.
    """
    if not text or not _PRIVACY_SCRUB_ENABLED:
        return text
    if not (
        _MAIL_SENDER_LINE_RE.search(text)
        or _MAIL_SUBJECT_LINE_RE.search(text)
        or _DOMAIN_ACCOUNT_RE.search(text)
        or _EMAIL_ACCOUNT_RE.search(text)
        or _TRUNCATED_LINE_RE.search(text)
        or any(_INBOX_REGION_MARKER_RE.match(line.strip()) for line in text.splitlines())
    ):
        return text
    lines = text.splitlines()

    # A run of >= 2 consecutive cropped lines is an inbox-list transcription
    # (subjects cut off by the screenshot edge). Complete UI labels carry an
    # accelerator before the ellipsis (「另存新檔(A)...」) and are kept.
    def _is_cropped_list_line(stripped: str) -> bool:
        if not re.search(r"(?:\.{2,}|…)\s*$", stripped):
            return False
        if re.search(r"\([A-Za-z0-9]\)(?:\.{2,}|…)\s*$", stripped):
            return False
        return len(re.sub(r"\s+", "", stripped)) <= 30

    cropped = [_is_cropped_list_line(line.strip()) for line in lines]
    in_cropped_run = [
        flag and ((idx > 0 and cropped[idx - 1]) or (idx + 1 < len(cropped) and cropped[idx + 1]))
        for idx, flag in enumerate(cropped)
    ]

    kept: list[str] = []
    previous_dropped = False
    in_inbox_region = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if _INBOX_REGION_MARKER_RE.match(stripped):
            in_inbox_region = True
            previous_dropped = True
            continue
        if in_inbox_region:
            is_sentence_like = bool(re.search(r"[。！？；：]", stripped)) or stripped.startswith("#")
            if stripped and not is_sentence_like and len(re.sub(r"\s+", "", stripped)) <= 30:
                previous_dropped = True
                continue
            in_inbox_region = False
        if _MAIL_SENDER_LINE_RE.match(stripped) or _MAIL_SUBJECT_LINE_RE.match(stripped):
            previous_dropped = True
            continue
        if in_cropped_run[idx]:
            previous_dropped = True
            continue
        # A truncated fragment (trailing ellipsis) right after a dropped
        # mail-list line is the subject/company continuation of that entry.
        if previous_dropped and re.search(r"(?:\.{2,}|…)\s*$", stripped):
            continue
        previous_dropped = False
        kept.append(line)
    result = _DOMAIN_ACCOUNT_RE.sub(r"\1*****", "\n".join(kept))
    return _EMAIL_ACCOUNT_RE.sub(r"\1*****", result)
