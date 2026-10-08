"""A figure's own document carries everything its retrieval text does.

2026-10-08, eight delivered documents: the figure document left out the text read off
the picture, the keywords, and any description written mostly in Latin script, and put
the description after the details. The delivery check therefore found nearly every
figure missing (264 of 282 in one report) and appended each one whole to main.md under
「補充原文內容」, headed by the English placeholder "Figure semantics" when the picture
had no caption. Every figure was delivered twice and main.md doubled in size.
"""
import json

import pytest
from PIL import Image

from app.config import PipelineConfig
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.content_coverage import TextCoverage
from app.pipeline.stages.package import PackageStage

MEETING = {
    "image_type": "photo",
    "semantic_caption": "一張會議室座談會的照片，約七位與會者圍坐在長桌旁。背景可見白板與投影幕。",
    "facts": ["場景為一間會議室，牆上掛有白板與投影幕。", "桌上擺放多張姓名名牌與數瓶礦泉水。"],
    "all_text": ["王小明", "陳美華"],
    "keywords": ["座談會", "會議", "名牌"],
}
TEST_REPORT = {
    "image_type": "document",
    "semantic_caption": "此圖為 Example Testing Laboratory Inc. 出具的 IEC 60950-1 Safety Test Report 封面頁，報告編號 T-1234。",
    "facts": ["報告機構為 Example Testing Laboratory Inc.，報告編號 T-1234。"],
    "all_text": ["Example Testing Laboratory Inc.", "Safety Test Report", "IEC 60950-1", "Report No.: T-1234"],
    "keywords": ["測試報告", "IEC 60950-1"],
}


def _ir(*image_ids):
    return DocumentIR(
        doc_id="doc", run_id="run", page_kind="physical",
        source=SourceInfo(path="示範報告.pdf", ext="pdf", sha256="source", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[Block(block_id="intro", type=BlockType.TEXT, page_idx=0, reading_order=0,
                      payload={"text": "示範公司年度活動紀錄。"}),
                *[Block(block_id=block_id, type=BlockType.IMAGE, page_idx=0, reading_order=i + 1,
                        bbox_norm=[100, 200 + 200 * i, 900, 380 + 200 * i], payload={"img_path": f"{block_id}.png"})
                  for i, block_id in enumerate(image_ids)]],
    )


def _cache(tmp_path, *image_ids):
    cache = tmp_path / "parse"
    cache.mkdir()
    for block_id in image_ids:
        Image.new("RGB", (40, 30), "blue").save(cache / f"{block_id}.png")
    return cache


async def _assets(tmp_path, outputs):
    ir = _ir(*outputs)
    enrichments = {block_id: {"block_id": block_id, "kind": "figure", "output": output, "quality": {}}
                   for block_id, output in outputs.items()}
    _, by_block = await PackageStage(PipelineConfig())._export_assets(
        ir, tmp_path / "assets", _cache(tmp_path, *outputs), enrichments)
    return by_block


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [MEETING, TEST_REPORT], ids=["photo", "latin-description"])
async def test_figure_document_covers_its_retrieval_text(tmp_path, output):
    asset = (await _assets(tmp_path, {"pic": output}))["pic"]
    document = PackageStage()._render_split_asset_document(asset, "示範報告", "示範報告.pdf")

    assert TextCoverage([document]).covers(asset.retrieval_text)
    for line in output["all_text"]:
        assert line in document
    assert "、".join(output["keywords"]) in document
    assert output["semantic_caption"] in document
    # The description leads, as in the retrieval text.
    assert document.index(output["semantic_caption"]) < document.index(output["facts"][0])


@pytest.mark.asyncio
async def test_a_figure_without_caption_is_named_by_its_description(tmp_path):
    assets = await _assets(tmp_path, {"meeting": MEETING, "report": TEST_REPORT})

    assert assets["meeting"].title == "一張會議室座談會的照片，約七位與會者圍坐在長桌旁。"
    assert assets["report"].title == "此圖為 Example Testing"
    for asset in assets.values():
        assert "Figure semantics" not in asset.retrieval_text


@pytest.mark.asyncio
async def test_a_figure_with_nothing_read_has_no_placeholder_in_its_text(tmp_path):
    asset = (await _assets(tmp_path, {"blank": {}}))["blank"]

    assert asset.title == "圖表語意"
    assert "圖表語意" not in asset.retrieval_text
    assert "Figure semantics" not in asset.retrieval_text


@pytest.mark.asyncio
async def test_delivered_figures_are_not_appended_to_main_again(tmp_path):
    outputs = {"meeting": MEETING, "report": TEST_REPORT}
    ir = _ir(*outputs)
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    (run / "outputs/enrichments.jsonl").write_text("\n".join(
        json.dumps({"block_id": block_id, "kind": "figure", "output": output, "quality": {}}, ensure_ascii=False)
        for block_id, output in outputs.items()))
    config = PipelineConfig()
    config.enrich.enable_vlm = False
    config.package.enable_semantic_repair = False

    result = await PackageStage(config).run("doc", "run", ir, run, parse_cache_path=_cache(tmp_path, *outputs))

    assert result.success, result.error
    main = (run / "outputs/documents/main.md").read_text()
    assert "補充原文內容" not in main
    assert "Figure semantics" not in main
    figures = [json.loads(line) for line in (run / "outputs/assets_index.jsonl").read_text().splitlines()]
    for figure in figures:
        document = (run / "outputs/documents" / f"{figure['asset_id']}.md").read_text()
        assert TextCoverage([document]).covers(figure["retrieval_text"])
