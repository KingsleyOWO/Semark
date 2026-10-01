"""Conservative transcription accounting for restructuring and final delivery."""
import html
import re
from collections.abc import Iterable


def plain_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<!--[\s\S]*?-->|</?[A-Za-z][\w:-]*(?:\s[^<>]*|/?)>", " ", text or ""))).strip()


def normalized_text(text: str) -> str:
    return re.sub(r"[^\w]", "", plain_text(text)).casefold()


def _numbers(text: str) -> list[str]:
    return re.findall(r"[+-]?\d+(?:\.\d+)?", plain_text(text).replace(",", "").replace("，", ""))


class TextCoverage:
    """References alone cannot prove coverage; check delivered content too.

    This measures transcription survival, not semantic truth.
    """

    def __init__(self, texts: Iterable[str]):
        texts = list(texts)
        self.numbers = {number for text in texts for number in _numbers(text)}
        self.parts = [normalized_text(text) for text in texts]
        self.text = "\n".join(self.parts)
        self.grams = {part[i:i + 4] for part in self.parts for i in range(len(part) - 3)}

    def covers(self, source: str) -> bool:
        target = normalized_text(source)
        if not target:
            return True
        if not set(_numbers(source)).issubset(self.numbers):
            return False
        if target in self.text:
            return True
        if len(target) < 12:
            return False
        grams = [target[i:i + 4] for i in range(len(target) - 3)]
        return sum(gram in self.grams for gram in grams) / len(grams) >= 0.98
