"""A transparent picture reaches the models and OCR on white, as Word prints it (2026-10-16).

PNGs exported from slides keep black under their transparent pixels. convert("RGB"),
and the model's own processor, used that black: dark lettering vanished from a 4P/4C
table (its header and two of four rows), a logo wall and two panel headings, the
reader wrote 「黑色背景」, and the per-image audit, shown the same black picture,
agreed, while the page review saw the page on white and kept flagging the text missing.
"""
import asyncio
import base64
import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.adapters import ocr
from app.adapters.metafile import has_transparency, on_white


def _black_under_transparency(path: Path, size=(400, 120)) -> Path:
    """Black lettering on a transparent background stored as black: the slide export case."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((40, 40, 120, 80), fill=(0, 0, 0, 255))
    image.save(path)
    return path


def test_transparent_pixels_become_white_and_opaque_ones_keep_their_colour(tmp_path):
    with Image.open(_black_under_transparency(tmp_path / "p.png")) as image:
        assert has_transparency(image)
        flat = on_white(image)
    assert flat.mode == "RGB"
    assert flat.getpixel((5, 5)) == (255, 255, 255)
    assert flat.getpixel((60, 60)) == (0, 0, 0)

    palette = Image.new("P", (10, 10), 0)
    palette.putpalette([0, 0, 0, 255, 0, 0])
    palette.info["transparency"] = 0
    assert on_white(palette).getpixel((0, 0)) == (255, 255, 255)

    opaque = Image.new("RGB", (10, 10), (0, 0, 0))
    assert not has_transparency(opaque)
    assert on_white(opaque).getpixel((0, 0)) == (0, 0, 0)


def _sent_pixels(sink):
    part = next(p for p in sink["messages"][-1]["content"] if p.get("type") == "image_url")
    return Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))).convert("RGB")


@pytest.mark.parametrize("target", [3_000_000, 0])
def test_the_model_sees_a_transparent_picture_on_white_enlarged_or_not(tmp_path, target):
    pytest.importorskip("openai")
    from types import SimpleNamespace

    from app.adapters.vlm import VLMAdapter
    from app.config import VLMConfig

    sink = {}
    adapter = VLMAdapter(VLMConfig(figure_target_pixels=target))

    class Stop(Exception):
        pass

    async def create(**kwargs):
        sink.update(kwargs)
        raise Stop

    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    asyncio.run(adapter.enrich_figure(_black_under_transparency(tmp_path / "p.png"), "context"))
    sent = _sent_pixels(sink)
    assert sent.getpixel((2, 2)) == (255, 255, 255)
    # The lettering itself is still there, not washed out.
    assert sent.getpixel((sent.width * 80 // 400, sent.height * 60 // 120)) == (0, 0, 0)


def test_ocr_reads_a_transparent_picture_on_white(tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    async def no_worker(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(ocr, "ocr_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_worker)
    with pytest.raises(Stop):
        asyncio.run(ocr.read_ocr(_black_under_transparency(tmp_path / "p.png"), [0, 0, 1000, 1000], 1,
                                 tmp_path / "ocr"))
    with Image.open(tmp_path / "ocr" / "ocr-input.png") as sent:
        assert sent.mode == "RGB"
        assert sent.getpixel((5, 5)) == (255, 255, 255)
        assert sent.getpixel((60, 60)) == (0, 0, 0)


def test_a_reading_cached_from_the_black_picture_is_not_reused(monkeypatch):
    pytest.importorskip("openai")
    from app.adapters import vlm
    from app.config import VLMConfig

    current = vlm.VLMAdapter.compute_config_hash(VLMConfig())
    monkeypatch.setattr(vlm, "IMAGE_PREPARATION", "as-stored")
    assert vlm.VLMAdapter.compute_config_hash(VLMConfig()) != current
