"""Undersized crops must never reach the VLM.

qwen3-vl's image processor requires both sides to exceed its patch factor of
32 px. A smaller image makes the ollama model runner panic and drop the
connection mid-request:

    http: panic serving 127.0.0.1:51220:
        height:31 or width:34 must be larger than factor:32
    qwen3vl.(*ImageProcessor).SmartResize ... imageprocessor.go:54
    level=ERROR source=server.go:1610 msg="post predict" error="...: EOF"

Live (2026-08-07): MinerU extracted four figure crops of 32x31, 34x31 and
37x31 px from doc ba4242775667a51c. _get_block_image handed them over as-is —
it returns MinerU's own image files and never checks their size — and each was
sent twice, producing exactly 8 runner panics and 8 lost enrichments.

The guard lives in _enrich because that is the single funnel every image
passes through; the three producers (YOLO crops, MinerU image files, page
renders) cannot each be trusted to check.
"""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if importlib.util.find_spec("openai") is None:
    raise unittest.SkipTest("openai package is required to import VLMAdapter")

from PIL import Image

from app.adapters.vlm import VLM_MIN_IMAGE_SIDE, VLMAdapter


def _write_image(directory: Path, name: str, size: tuple[int, int]) -> Path:
    path = directory / name
    Image.new("RGB", size).save(path)
    return path


class _ExplodingClient:
    """Any attribute access means the adapter tried to talk to the model."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"VLM client must not be reached (accessed .{name})")


class _StopBeforeNetwork(Exception):
    """Raised by the stub client once the request payload is captured."""


def _capturing_client(sink: dict) -> SimpleNamespace:
    async def create(**kwargs):
        sink.clear()
        sink.update(kwargs)
        raise _StopBeforeNetwork

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _image_parts(sink: dict) -> list[dict]:
    content = sink["messages"][-1]["content"]
    return [part for part in content if part.get("type") == "image_url"]


class ImageSideThresholdTest(unittest.TestCase):
    def test_threshold_matches_the_qwen3_vl_patch_factor(self):
        self.assertEqual(VLM_MIN_IMAGE_SIDE, 32)

    def test_crop_shorter_than_the_patch_factor_is_rejected(self):
        with TemporaryDirectory() as tmp:
            # The exact geometry that panicked the runner in production.
            path = _write_image(Path(tmp), "fig0004.jpg", (32, 31))

            self.assertTrue(VLMAdapter()._image_is_too_small(path))

    def test_crop_exactly_at_the_patch_factor_is_allowed(self):
        with TemporaryDirectory() as tmp:
            path = _write_image(Path(tmp), "ok.png", (32, 32))

            self.assertFalse(VLMAdapter()._image_is_too_small(path))

    def test_unreadable_image_is_not_blocked_by_the_guard(self):
        """We only block what we can prove is too small; a corrupt file keeps
        its existing failure path instead of being silently reclassified."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.png"
            path.write_bytes(b"not an image")

            self.assertFalse(VLMAdapter()._image_is_too_small(path))


class EnrichSkipsUndersizedImagesTest(unittest.IsolatedAsyncioTestCase):
    async def test_undersized_main_crop_short_circuits_without_calling_the_model(self):
        with TemporaryDirectory() as tmp:
            path = _write_image(Path(tmp), "fig0003.jpg", (34, 31))
            adapter = VLMAdapter()
            adapter.client = _ExplodingClient()

            result = await adapter._enrich(image_path=path, kind="figure_caption")

            self.assertFalse(result.success)
            self.assertIn("34x31", result.error or "")

    async def test_undersized_page_thumbnail_is_dropped_but_the_crop_still_goes(self):
        """The thumbnail is optional context — losing it must not cost us the
        enrichment, only the extra image."""
        sink: dict = {}
        with TemporaryDirectory() as tmp:
            crop = _write_image(Path(tmp), "crop.png", (512, 512))
            thumbnail = _write_image(Path(tmp), "thumb.png", (20, 20))
            adapter = VLMAdapter()
            adapter.client = _capturing_client(sink)

            await adapter._enrich(
                image_path=crop,
                kind="figure_caption",
                page_thumbnail_path=thumbnail,
            )

            self.assertEqual(len(_image_parts(sink)), 1)

    async def test_normal_crop_and_thumbnail_both_reach_the_model(self):
        sink: dict = {}
        with TemporaryDirectory() as tmp:
            crop = _write_image(Path(tmp), "crop.png", (512, 512))
            thumbnail = _write_image(Path(tmp), "thumb.png", (256, 256))
            adapter = VLMAdapter()
            adapter.client = _capturing_client(sink)

            await adapter._enrich(
                image_path=crop,
                kind="figure_caption",
                page_thumbnail_path=thumbnail,
            )

            self.assertEqual(len(_image_parts(sink)), 2)
