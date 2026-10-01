"""Small figures are enlarged before the model reads them.

qwen3-vl turns every 32x32 px into one visual token and never enlarges an
image itself: a 190x142 storefront photo reaches the model as ~24 tokens, so
the sign 「晴諾米拼圖工作坊」 is a fraction of a token tall and comes back
guessed as 「晴陸米拼圖工作店」. The same photo enlarged 6x reads correctly
(Lumio replay, 2026-10-01).
"""

import base64
import importlib.util
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if importlib.util.find_spec("openai") is None:
    raise unittest.SkipTest("openai package is required to import VLMAdapter")

from PIL import Image

from app.adapters.vlm import VLMAdapter
from app.config import VLMConfig


class _StopBeforeNetwork(Exception):
    pass


def _adapter(sink: dict, **config) -> VLMAdapter:
    adapter = VLMAdapter(VLMConfig(**config))

    async def create(**kwargs):
        sink.clear()
        sink.update(kwargs)
        raise _StopBeforeNetwork

    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return adapter


def _sent_images(sink: dict) -> list[Image.Image]:
    images = []
    for part in sink["messages"][-1]["content"]:
        if part.get("type") == "image_url":
            data = part["image_url"]["url"].split(",", 1)[1]
            images.append(Image.open(io.BytesIO(base64.b64decode(data))))
    return images


def _run(coro):
    import asyncio

    return asyncio.run(coro)


class FigureUpscaleTest(unittest.TestCase):
    def _figure(self, tmp: str, size: tuple[int, int]) -> Path:
        path = Path(tmp) / "figure.png"
        Image.new("RGB", size, "white").save(path)
        return path

    def test_small_figure_is_enlarged_up_to_the_factor_limit(self):
        sink: dict = {}
        with TemporaryDirectory() as tmp:
            _run(_adapter(sink).enrich_figure(self._figure(tmp, (190, 142)), "context"))

        self.assertEqual(_sent_images(sink)[0].size, (1140, 852))

    def test_mid_size_figure_is_enlarged_to_the_target_area(self):
        sink: dict = {}
        with TemporaryDirectory() as tmp:
            _run(_adapter(sink).enrich_figure(self._figure(tmp, (792, 443)), "context"))

        width, height = _sent_images(sink)[0].size
        self.assertAlmostEqual(width * height, 3_000_000, delta=10_000)
        self.assertAlmostEqual(width / height, 792 / 443, places=2)

    def test_figure_already_at_the_target_is_sent_as_is(self):
        sink: dict = {}
        with TemporaryDirectory() as tmp:
            _run(_adapter(sink).enrich_figure(self._figure(tmp, (2400, 1600)), "context"))

        self.assertEqual(_sent_images(sink)[0].size, (2400, 1600))

    def test_enlarging_can_be_turned_off(self):
        sink: dict = {}
        with TemporaryDirectory() as tmp:
            _run(_adapter(sink, figure_target_pixels=0).enrich_figure(self._figure(tmp, (190, 142)), "context"))

        self.assertEqual(_sent_images(sink)[0].size, (190, 142))

    def test_forms_keep_their_original_pixels(self):
        sink: dict = {}
        with TemporaryDirectory() as tmp:
            _run(_adapter(sink).enrich_form(self._figure(tmp, (190, 142)), "context"))

        self.assertEqual(_sent_images(sink)[0].size, (190, 142))

    def test_enlarging_setting_is_part_of_the_enrich_cache_identity(self):
        self.assertNotEqual(
            VLMAdapter.compute_config_hash(VLMConfig()),
            VLMAdapter.compute_config_hash(VLMConfig(figure_target_pixels=0)),
        )
