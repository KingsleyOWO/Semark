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

# Variant glyphs MinerU's OCR emits from the same fonts that produce 税/脱, but
# which opencc's s2t leaves untouched — so the detection gate in
# ``to_taiwan_traditional`` reads a document full of them as pure zh-TW and
# returns it byte-identical. They are encodable and traditional, just not the
# Taiwan standard form, so nothing downstream flags them either.
#
# Live evidence (2026-08-10, 167-document run), counted per line and split by
# whether the line carries kana:
#
#     凈 → 淨   51 occurrences / 30 documents — 51 Chinese,  0 Japanese
#     説 → 說   25 occurrences / 21 documents — 24 Chinese,  1 Japanese
#
# OCR noise rather than the author's spelling: one line carried both forms
# inside a single clause (「說説聽聽就好」), another spelled the same word both
# ways one sentence apart (潔淨能源的投資 … 潔凈能源投資結構), and a third put
# 小説 next to 非小說. The single Japanese-context 説 sits on a line with kana
# (…について解説！), which the guard above already covers.
#
# Deliberately NOT in this table: 経(31) 産(19) 関(14) 戦(11) 対(7). Those run
# the other way — mostly Japanese citations — and, unlike the two above, their
# kana-free occurrences are Japanese too, where the line guard cannot help:
# all three kana-free 経 lines are Japanese bibliography (就農準備資金‧経営開始
# 資金 / 経済産業省商務情報政策局), 関 and 対 have no Chinese-context
# occurrence at all, and two of the four kana-free 戦 lines quote the Japanese
# body name 「區域未來戦略本部」 inside Chinese prose. Adding them would corrupt
# citations to repair prose; the residual mis-spellings are the cheaper error.
# Re-run that Chinese/Japanese split before extending this table.
_NON_TAIWAN_VARIANT_GLYPHS = str.maketrans({"凈": "淨", "説": "說"})


def fix_mainland_vocab(text: str) -> str:
    if not text or not _HAN_RE.search(text):
        return text
    for pattern, replacement in _MAINLAND_VOCAB_FIXES:
        text = pattern.sub(replacement, text)
    return text


# opencc rewrites 台→臺, 布→佈, 占→佔, 干→幹, 了→瞭, 合→閤 — but every
# left-hand form is standard zh-TW (台灣, 新台幣, 公布, 布局, 占比, 干擾,
# 展示了, 結合中央). Two consequences, both seen live on 2026-08-07: the
# detector below read ordinary Taiwanese prose as simplified, and once a
# document did carry a genuinely simplified glyph the converter rewrote these
# characters throughout.
#
# 合 joined the table on 2026-08-11 from the same 167-document run. opencc
# applies 合→閤 only before 中 (結合中央, 投資組合中, 能源組合中, 淨零競合中,
# 聯合中小型製作公司 — while 符合中華民國 is untouched), which is why it went
# unnoticed: 6 documents shipped with 閤中, and because s2t makes the rewrite
# too, prose with no simplified glyph at all tripped the gate and was handed to
# s2tw wholesale. Folding here fixes both halves — the false detection and the
# rewrite. Deliberate 閤 (閤家平安, 閤第光臨) is safe: s2t and s2tw both leave
# 閤 alone, so _restore_zh_tw_variants' ``conv`` branch returns it unchanged.
#
# Trade-off: a genuinely simplified 干净/一只 keeps the simplified glyph. The
# converter only ever runs on mostly-zh-TW text here, where preserving the
# author's spelling is the safer error.
_ZH_TW_VARIANT_PAIRS = (
    ("臺", "台"),
    ("佈", "布"),
    ("佔", "占"),
    ("幹", "干"),
    ("瞭", "了"),
    ("閤", "合"),
)
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

    One class of damage cannot use that gate at all: variant glyphs the
    detector does not recognise as simplified (凈, 説). They are fixed
    unconditionally by ``_NON_TAIWAN_VARIANT_GLYPHS`` below, placed after the
    kana guard so Japanese keeps 解説.
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
    # Past the kana guard this text is not Japanese, so the variant fix is
    # safe. It has to run *before* the gate below and independently of it:
    # s2t maps these glyphs to themselves, so a document whose only defect is
    # 凈零/説明 returns unchanged from the detector and never reaches s2tw.
    # 1:1 substitution keeps the length, which _restore_zh_tw_variants needs.
    text = text.translate(_NON_TAIWAN_VARIANT_GLYPHS)
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
