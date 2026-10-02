"""The independent review reading sees the picture as large as the draft did, in either script.

Lumio v20: the draft read the brand-guideline slides enlarged (~2,900 visual
tokens) while the review read them as stored (~250), so the review misread
「晴諾米拼圖工作坊」 as 「集諾米所屬工作」 and rejected the correct draft. Two
review readings also came back in simplified script and could never match
the zh-TW delivery.
"""
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.adapters.vlm import VLMAdapter
from app.config import VLMApiMode, VLMConfig
from app.pipeline.agentic import RegionModel
from app.pipeline.document_review import word_picture_issue
from app.pipeline.image_source_review import WordPictureItem, relationship_issue
from app.pipeline.recovery_models import NativeReading


class _Sent(Exception):
    pass


async def _sent_size(tmp_path, task):
    picture = tmp_path / "logo.png"
    Image.new("RGB", (190, 142), "yellow").save(picture)
    adapter = VLMAdapter(VLMConfig(api_mode=VLMApiMode.OLLAMA))
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        raise _Sent
    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(_Sent):
        await RegionModel(adapter, 4096).query(task, {}, picture, NativeReading)
    url = next(p["image_url"]["url"] for p in sent["messages"][-1]["content"] if p["type"] == "image_url")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["verify_native_reading", "read_image_parts", "verify_image_parts"])
async def test_picture_readings_in_review_are_enlarged_like_the_draft(tmp_path, task):
    assert await _sent_size(tmp_path, task) == (190 * 6, 142 * 6)


@pytest.mark.asyncio
async def test_a_page_audit_keeps_its_many_images_as_stored(tmp_path):
    # A page audit already carries up to seven images; enlarging them all would overflow the context.
    assert await _sent_size(tmp_path, "audit_page") == (190, 142)


def _reading(text):
    return WordPictureItem.model_validate({"source_id": "image0", "text": text, "complete": True,
                                           "observation": "Slide text.", "image_type": "document",
                                           "relationships": [], "relationships_complete": True})


def test_a_simplified_review_reading_matches_the_zh_tw_delivery():
    delivered = "右列為晴諾米拼圖視覺識別系統之主要英文字體。"
    assert word_picture_issue(_reading("右列为雷诺瓦拼图视觉识别系统之主要英文字体。"), delivered) == ""
    assert word_picture_issue(_reading("右列為集諾米拼圖視覺識別系統之主要英文字體。"), delivered)


def test_a_simplified_relationship_quote_is_found_in_the_zh_tw_delivery():
    reading = WordPictureItem.model_validate({
        "source_id": "image0", "text": "晴諾米", "complete": True, "observation": "Logo.", "image_type": "logo",
        "relationships": [{"relation_id": "r0", "kind": "other", "source_fact": "標誌位於品牌名上方",
                           "bbox": [0, 0, 1000, 500], "certain": True}], "relationships_complete": True})
    finding = SimpleNamespace(block_id="b0", image_relationship_checks=[SimpleNamespace(
        relation_id="r0", status="matched", delivered_quote="标志位于品牌名上方", direction="none",
        horizontal="unspecified", vertical="unspecified")])
    assert relationship_issue(reading, finding, "晴諾米的標誌位於品牌名上方。") == ""
