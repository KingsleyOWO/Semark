"""MinerU emits captions as lists; every package-stage caption read must
coerce them to text or figure export crashes (list[:100] → .startswith)."""

import unittest

from app.pipeline.package_utils import caption_text


class CaptionTextTest(unittest.TestCase):
    def test_list_caption_joins_non_empty_parts(self):
        self.assertEqual(caption_text(["圖一", "", "組織架構"]), "圖一 組織架構")

    def test_string_caption_passes_through(self):
        self.assertEqual(caption_text("表三 出差單"), "表三 出差單")

    def test_empty_values_become_empty_string(self):
        self.assertEqual(caption_text(None), "")
        self.assertEqual(caption_text([]), "")
        self.assertEqual(caption_text(""), "")

    def test_non_string_items_are_stringified(self):
        self.assertEqual(caption_text([1, "圖"]), "1 圖")
