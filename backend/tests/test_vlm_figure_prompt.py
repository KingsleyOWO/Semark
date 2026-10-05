"""The figure prompt keeps neighbouring text out of the picture and lets unread text stay unread.

v6 asked for "ALL text visible" with no way to say "unreadable", and appended
the neighbouring paragraphs as "Context from document:". On the Validation
figures the model then wrote the cover's company and meeting names into a
storefront photo's facts and filled unreadable slide text with invented
recommendations. Replay of the same 26 figures (enlarged): context leaks in
3 figures -> 1, anchors 150 -> 149 of 150, and every unreadable mark it emitted
sat on text that is genuinely too small to read.

The mark is [?], not □: □ is a checkbox to the form detection and field
splitter (structured_rag / semantic.normalizer), so it would turn a figure's
unread characters into form signals.
"""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if importlib.util.find_spec("openai") is None:
    raise unittest.SkipTest("openai package is required to import VLMAdapter")

from PIL import Image

from app.adapters.vlm import VLMAdapter


class _StopBeforeNetwork(Exception):
    pass


def _sent_prompt(context: str) -> str:
    sink: dict = {}
    adapter = VLMAdapter()

    async def create(**kwargs):
        sink.update(kwargs)
        raise _StopBeforeNetwork

    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    import asyncio

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "figure.png"
        Image.new("RGB", (400, 300)).save(path)
        asyncio.run(adapter.enrich_figure(path, context))
    user = sink["messages"][-1]["content"]
    return "\n".join(part["text"] for part in user if part.get("type") == "text")


class FigurePromptTest(unittest.TestCase):
    def test_neighbouring_text_is_labelled_as_not_part_of_the_image(self):
        prompt = _sent_prompt("企業名稱： 示範有限公司")

        label, context = prompt.rsplit("\n", 1)
        self.assertEqual(context, "企業名稱： 示範有限公司")
        self.assertIn("NOT part of the image", label)
        self.assertNotIn("Context from document:", prompt)

    def test_unreadable_text_has_a_marker_instead_of_a_guess(self):
        prompt = _sent_prompt("")

        self.assertIn("write [?] for each character you cannot read", prompt)
        self.assertIn("Never invent", prompt)

    def test_unreadable_marker_is_not_the_checkbox_glyph(self):
        self.assertNotIn("□", _sent_prompt(""))

    def test_chart_values_stay_with_their_labels(self):
        self.assertIn("keep each value together with its label", _sent_prompt(""))

    def test_prompt_change_moves_the_enrich_cache_version(self):
        adapter = VLMAdapter()
        # The enrich cache keys on figure_description's own version, not the
        # aliased figure_caption's, so both have to move together.
        self.assertEqual(adapter.get_prompt_version("figure_caption"), "v7")
        self.assertEqual(adapter.get_prompt_version("figure_description"), "v6")
