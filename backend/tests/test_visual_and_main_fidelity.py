"""Source occurrences and visual relationships must survive publication unchanged."""
import pytest

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, SourceInfo
from app.pipeline.stages.package import AssetEntry, PackageStage


@pytest.mark.parametrize("image_type", ["chart", "table", "org_chart", "diagram", "other", "figure", "screenshot", "photo", ""])
@pytest.mark.parametrize("language", ["en", "zh-TW"])
def test_visual_path_notation_does_not_invent_a_process(image_type, language):
    path = "理性功能 > 風格強烈 > 品牌甲"
    output = {"image_type": image_type, "structured_content": [path],
              "all_text": ["理性功能", "風格強烈", "品牌甲"]}
    rendered = PackageStage()._render_visual_semantic_content(
        "定位矩陣", output, semantic_output_language=language)
    assert path in rendered
    assert "flow" not in rendered.lower()
    assert "流程" not in rendered
    assert "開始" not in rendered


def test_explicit_flowchart_retains_its_process_edges():
    rendered = PackageStage()._render_visual_semantic_content(
        "申請流程", {"image_type": "flowchart", "structured_content": ["收到申請 > 審核 > 完成"]})
    assert "詳細流程路徑" in rendered
    assert "收到申請 > 審核 > 完成" in rendered


def test_untyped_legacy_asset_preserves_relations_without_guessing_a_workflow():
    asset = AssetEntry(type="figure_asset", asset_id="matrix", doc_id="doc", run_id="run",
                       title="定位矩陣", page_idx=0, asset_path="matrix.png", block_id="image",
                       structured_content="功能 > 樣式 > 品牌", retrieval_text="功能 > 樣式 > 品牌")
    text = PackageStage()._render_split_asset_document(asset, "報告", "report.docx")
    assert "功能 > 樣式 > 品牌" in text
    assert "流程" not in text


@pytest.mark.parametrize("image_type", ["chart", "org_chart", "table", "other"])
def test_page_ocr_does_not_promote_path_notation_to_flow(image_type):
    image = Block(block_id="image", type=BlockType.IMAGE, page_idx=0,
                  bbox_norm=[0, 0, 1000, 1000], payload={})
    ir = DocumentIR(doc_id="doc", run_id="run",
                    source=SourceInfo(path="matrix.pdf", ext="pdf", sha256="abc", size_bytes=1),
                    engine=EngineInfo(backend="pipeline", method="ocr"),
                    blocks=[image, *[Block(block_id=f"text{i}", type=BlockType.TEXT, page_idx=0,
                                          bbox_norm=[20, 20 + 50*i, 500, 60 + 50*i],
                                          reading_order=i+1, payload={"text": value})
                                     for i, value in enumerate(["功能", "樣式", "品牌甲", "品牌乙"])]] )
    output = {"image_type": image_type, "structured_content": ["功能 > 樣式 > 品牌甲"]}
    assert PackageStage()._augment_visual_output_from_page_text(output, ir, image) == output
    assert not PackageStage._visual_reference_labels_for_page({
        "image": {"input": {"page_idx": 0}, "output": output}}, 0)


def test_main_keeps_equal_steps_and_urls_owned_by_different_cases():
    shared = "3.建立產品分析，完成品牌與產品策略方向。"
    url = "https://example.test/awards/winners"
    first = f"## 案例甲\n1.訪談客戶。\n{shared}\n獎項甲\n{url}"
    second = f"## 案例乙\n1.市場研究。\n{shared}\n獎項乙\n{url}\n獎項丙\n{url}"
    rendered = PackageStage()._render_split_main_document(
        source_md=first + "\n" + second, source_title="案例報告", source_filename="report.docx",
        form_entries=[], semantic_output_language="zh-TW")
    assert rendered.count(shared) == 2
    assert rendered.count(url) == 3
    case_b = rendered.split("## 案例乙", 1)[1]
    assert shared in case_b
    assert case_b.count(url) == 2


def test_equal_native_paragraphs_in_same_section_are_not_ocr_duplicates():
    paragraph = "本項經費由政府補助與企業自籌各分擔一半。"
    text = "## 第一部分\n" + paragraph + "\n小計：一百萬元。\n" + paragraph
    rendered = PackageStage()._clean_split_main_body(text, "報告", [])
    assert rendered.count(paragraph) == 2
