"""Fact-preservation guard for reviewer-driven semantic repairs.

The semantic-repair reviewer rewrites whole documents from parser/VLM
evidence. A rewrite that reads well but silently drops numbers, dates,
amounts, field labels, or legal references is worse than the original,
because downstream RAG answers become confidently wrong.

`repair_preserves_facts` extracts fact tokens from the pre-repair markdown
(numbers including fullwidth digits, 民國-style dotted dates, percentages
and money amounts; CJK field labels ending with a colon; and 第N條/項/款
legal references) and requires at least ``MIN_SURVIVAL_RATIO`` of them to
survive in the repaired markdown. Tokens that only appear in the source
evidence, but not in the original markdown, are never required.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "MIN_SURVIVAL_RATIO",
    "extract_fact_tokens",
    "extract_value_tokens",
    "repair_preserves_facts",
]

MIN_SURVIVAL_RATIO = 0.9

_FULLWIDTH_TRANSLATION = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "％": "%",
        "．": ".",
        "，": ",",
    }
)

# Numbers (after fullwidth normalization): plain integers, dotted 民國 dates
# such as 114.12.11, decimals, thousand-separated amounts, and percentages.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")

# A contiguous label run that ends with a halfwidth/fullwidth colon.
# Punctuation and whitespace break the run so sentence text is not captured.
_CJK_LABEL_RE = re.compile(
    r"([^\s:：;；,，.。、!！?？…‧·•|()（）【】\[\]{}<>\"'「」『』]{1,24})[:：]"
)
_CJK_CHAR_RE = re.compile(r"[㐀-䶿一-鿿]")
_LABEL_STRIP_CHARS = "#*_`>~—–-‧·•　 \t"

# Legal references: 第N條 / 第N項 / 第N款 with arabic or CJK numerals.
_LEGAL_REF_RE = re.compile(r"第\s*([0-9〇零一二三四五六七八九十百千兩]{1,8})\s*([條条項项款])")

_COMPACT_RE = re.compile(r"[\s,，]+")


def _normalize(text: Any) -> str:
    return str(text or "").translate(_FULLWIDTH_TRANSLATION)


def _compact(text: str) -> str:
    return _COMPACT_RE.sub("", text)


def extract_fact_tokens(text: str) -> set[str]:
    """Extract normalized fact tokens (numbers, CJK labels, legal refs)."""

    normalized = _normalize(text)
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(normalized):
        tokens.add(match.group(0))
    for match in _CJK_LABEL_RE.finditer(normalized):
        label = match.group(1).strip(_LABEL_STRIP_CHARS)
        if label and _CJK_CHAR_RE.search(label):
            tokens.add(label)
    for match in _LEGAL_REF_RE.finditer(normalized):
        tokens.add(f"第{match.group(1)}{match.group(2)}")
    return tokens


def extract_value_tokens(text: str) -> set[str]:
    """Extract only *value* facts — numbers/dates/amounts/percentages and legal
    references — excluding CJK field labels. A blank form's labels (申請日期：)
    are structure already captured as fields, not values worth preserving, so
    the source-text-dump coverage decision must not treat them as facts.
    """
    normalized = _normalize(text)
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(normalized):
        tokens.add(match.group(0))
    for match in _LEGAL_REF_RE.finditer(normalized):
        tokens.add(f"第{match.group(1)}{match.group(2)}")
    return tokens


def _is_noise_fact_token(token: str) -> bool:
    """Tokens that read as facts to the extractor but are layout noise.

    Bare one/two-digit integers are step markers, page numbers and list
    indices from screenshot transcriptions — a legitimate rewrite reorders
    or drops them. Real values keep protection: three+ digits, decimals,
    percentages and dotted dates all fail these patterns. 欄位N/Column N are
    the renderer's fallback names for headerless table columns.
    """
    if re.fullmatch(r"\d{1,2}", token):
        return True
    return bool(re.fullmatch(r"(?:欄位|Column\s*)\d+", token))


def _token_survives(token: str, repaired_normalized: str, repaired_compact: str) -> bool:
    if token in repaired_normalized:
        return True
    compact_token = _compact(token)
    return bool(compact_token) and compact_token in repaired_compact


def repair_preserves_facts(
    original_md: str,
    evidence_text: str,
    repaired_md: str,
    *,
    min_survival_ratio: float = MIN_SURVIVAL_RATIO,
) -> tuple[bool, dict[str, Any]]:
    """Check that a repaired markdown keeps the original markdown's fact tokens.

    Returns ``(ok, details)`` where ``details`` records the checked/survived
    counts, the missing tokens, and the survival ratio. Only tokens present in
    ``original_md`` are required; evidence-only tokens never count against the
    repair. Documents without fact tokens always pass.
    """

    from app.pipeline.corpus_rules import get_rules

    # Template scaffolding labels (主要欄位分組:, 常見查詢關鍵字:, …) are
    # renderer output, not document facts; the reviewer is explicitly asked
    # to remove template filler, so their removal must not count as loss.
    template_labels = set(get_rules().marker_list("template_section_labels"))
    # Internal asset anchors ([[asset:tbl0000]]) would otherwise contribute
    # fake numeric facts like 「0000」.
    original_md = re.sub(r"\[\[asset:[^\]]+\]\]", " ", str(original_md or ""))
    original_tokens = extract_fact_tokens(original_md) - template_labels
    original_tokens = {
        token for token in original_tokens if not _is_noise_fact_token(token)
    }
    evidence_tokens = extract_fact_tokens(evidence_text)
    repaired_normalized = _normalize(repaired_md)
    repaired_compact = _compact(repaired_normalized)

    missing = [
        token
        for token in sorted(original_tokens)
        if not _token_survives(token, repaired_normalized, repaired_compact)
    ]
    checked = len(original_tokens)
    survived = checked - len(missing)
    ratio = 1.0 if checked == 0 else survived / checked
    ok = ratio >= min_survival_ratio
    details: dict[str, Any] = {
        "checked_token_count": checked,
        "survived_token_count": survived,
        "missing_tokens": missing,
        "survival_ratio": round(ratio, 4),
        "min_survival_ratio": min_survival_ratio,
        "evidence_only_token_count": len(evidence_tokens - original_tokens),
    }
    return ok, details
