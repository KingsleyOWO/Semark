"""zh-TW normalization shared by the parser text layer and the VLM output.

This lived inside the package stage and was reachable only through
``render_vlm_text``, so it never saw MinerU's own text. Live evidence
(2026-08-10, 100-document store): 849 unambiguously simplified characters
reached rag.md across 88 of the 100 documents — 税賦優惠, 脱碳, 生质甲烷,
潔淨氢 — every one of them an OCR glyph out of the parser, not the model.
"""

from __future__ import annotations

import re
from typing import Any

_OPENCC_CONVERTERS: dict[str, Any] = {}


def _get_opencc(profile: str) -> Any:
    if profile not in _OPENCC_CONVERTERS:
        try:
            from opencc import OpenCC

            _OPENCC_CONVERTERS[profile] = OpenCC(profile)
        except Exception:
            _OPENCC_CONVERTERS[profile] = None
    return _OPENCC_CONVERTERS[profile]


# VLM prose written in traditional characters still carries mainland
# vocabulary (圖標/界面/分辨率) — no simplified chars, so the opencc detection
# gate never sees it. Small, unambiguous pairs only; 界面活性劑 (chemistry)
# is spared.
_MAINLAND_VOCAB_FIXES = (
    (re.compile(r"界面(?!活性)"), "介面"),
    (re.compile(r"默認"), "預設"),
    (re.compile(r"圖標"), "圖示"),
    (re.compile(r"分辨率"), "解析度"),
    (re.compile(r"視頻"), "影片"),
    (re.compile(r"軟件"), "軟體"),
    (re.compile(r"硬盤"), "硬碟"),
    (re.compile(r"鼠標"), "滑鼠"),
)

_HAN_RE = re.compile(r"[一-鿿]")

# Hiragana and katakana. 会社/学会/実 are correct Japanese, not simplified
# Chinese; converting a cited Japanese company name to 會社 corrupts it.
_KANA_RE = re.compile(r"[぀-ヿ]")


def fix_mainland_vocab(text: str) -> str:
    if not text or not _HAN_RE.search(text):
        return text
    for pattern, replacement in _MAINLAND_VOCAB_FIXES:
        text = pattern.sub(replacement, text)
    return text


# opencc rewrites 台→臺, 布→佈, 占→佔, 干→幹, 了→瞭 — but every left-hand form
# is standard zh-TW (台灣, 新台幣, 公布, 布局, 占比, 干擾, 展示了). Two
# consequences, both seen live on 2026-08-07: the detector below read ordinary
# Taiwanese prose as simplified, and once a document did carry a genuinely
# simplified glyph the converter rewrote these characters throughout.
# Trade-off: a genuinely simplified 干净/一只 keeps the simplified glyph. The
# converter only ever runs on mostly-zh-TW text here, where preserving the
# author's spelling is the safer error.
_ZH_TW_VARIANT_PAIRS = (("臺", "台"), ("佈", "布"), ("佔", "占"), ("幹", "干"), ("瞭", "了"))
_ZH_TW_VARIANT_CHARS = frozenset(source for _, source in _ZH_TW_VARIANT_PAIRS)


def _normalize_variants_for_detection(text: str) -> str:
    """Fold the ambiguous pairs together so only genuinely simplified glyphs
    count as evidence that a conversion is needed."""
    for converted, source in _ZH_TW_VARIANT_PAIRS:
        text = text.replace(converted, source)
    return text


def _restore_zh_tw_variants(source: str, converted: str) -> str:
    """Put the author's own variant choice back after conversion, so a document
    written with 台灣 keeps 台灣 and one written with 臺灣 keeps 臺灣."""
    if len(source) != len(converted):
        return converted
    return "".join(
        src if src in _ZH_TW_VARIANT_CHARS else conv
        for src, conv in zip(source, converted, strict=True)
    )


def to_taiwan_traditional(text: str) -> str:
    """Convert text that carries simplified Chinese to zh-TW.

    Two producers feed this: MinerU's text layer, whose OCR misreads single
    glyphs (稅→税, 脫→脱), and the local VLM, which occasionally ruminates in
    simplified Chinese and mainland terms. Detection first: text that the plain
    s2t pass leaves unchanged has no simplified content and passes through
    byte-identical, so genuine zh-TW prose (and English) is never rewritten.
    The 了/瞭 particle ambiguity is repaired on both sides of the gate — the
    detector itself trips on it.
    """
    if not text or not _HAN_RE.search(text):
        return text
    if _KANA_RE.search(text):
        # Guarding the whole string is too coarse for a bibliography, which
        # mixes zh-TW prose with Japanese titles line by line — one Japanese
        # entry let 脱碳 survive in the Chinese sentences around it. Recursion
        # terminates: each line carries no newline of its own.
        if "\n" in text:
            return "\n".join(to_taiwan_traditional(line) for line in text.split("\n"))
        return text
    detector = _get_opencc("s2t")
    if detector is None:
        return text
    if _normalize_variants_for_detection(detector.convert(text)) == _normalize_variants_for_detection(text):
        return text
    # s2tw, not s2twp. One stray simplified glyph (税/脱 out of MinerU's text
    # layer) is enough to trip the gate, and s2twp then applies its phrase
    # table to the WHOLE document — rewriting correct zh-TW as it goes
    # (數據→資料, 設備→裝置, 零組件→零元件, 台→臺). s2tw fixes the glyphs and
    # leaves vocabulary alone; deliberate term localisation stays in the small,
    # reviewed _MAINLAND_VOCAB_FIXES table above.
    converter = _get_opencc("s2tw")
    if converter is None:
        return text
    return _restore_zh_tw_variants(text, converter.convert(text))
