"""A figure's numbers are content, not list markers.

split_vlm_lines used to lstrip every leading digit, dot and dash, so a fact
「2023 年的市場規模標註為 1.89 十億美元」 shipped as 「年的市場規模…」 and an OCR
label 「1.89」 vanished. Across the 44-document store that dropped or truncated
8,214 of 81,400 figure lines (10%) in 1,276 of 3,049 figures.
"""

from app.pipeline.stages.package import PackageStage, split_vlm_lines


def test_split_vlm_lines_keeps_a_leading_year():
    assert split_vlm_lines(["2023 年的市場規模標註為 1.89 十億美元。"]) == [
        "2023 年的市場規模標註為 1.89 十億美元。"
    ]


def test_split_vlm_lines_keeps_numeric_ocr_labels():
    labels = ["1.89", "2018", "100%", "0.26%", "-0.1", "30 cm / 167 pieces", ".NET"]
    assert split_vlm_lines(labels) == labels


def test_split_vlm_lines_still_drops_bullet_markers_and_duplicates():
    assert split_vlm_lines(["- 品牌承諾", "• 品牌個性", "* 品牌價值", "- 品牌承諾"]) == [
        "品牌承諾",
        "品牌個性",
        "品牌價值",
    ]


def test_chart_facts_keep_their_years_in_the_rendered_figure_text():
    output = {
        "semantic_caption": "此圖為拼圖市場長條圖。",
        "image_type": "chart",
        "structured_content": "",
        "all_text": ["Jigsaw Puzzle Market", "1.89", "2023"],
        "facts": [
            "2023 年的市場規模標註為 1.89 十億美元。",
            "2032 年的市場規模標註為 2.4 十億美元，為圖中最高值。",
        ],
        "keywords": [],
    }

    rendered = PackageStage()._render_visual_semantic_content("拼圖市場", output)

    assert "2023 年的市場規模標註為 1.89 十億美元。" in rendered
    assert "2032 年的市場規模標註為 2.4 十億美元" in rendered
    assert "1.89" in rendered.splitlines()
