"""Table names from adjacent headings, and attributable VLM figure sections.

Task A — ``infer_table_asset_title`` used to fall back to the *document* title
whenever MinerU produced no ``table_caption`` (52 of 128 tables in the 2026-08
corpus). A table's name is a **local** thing: in a magazine layout it sits on
the line directly above the grid. ``_nearest_table_caption`` recovers it.

Task B — ``_render_visual_semantic_content`` renders the flowchart template as
fixed ``##`` sections. That renderer is shared by the split-asset document
(where ``##`` sits under the document's own ``#``) and by the rag.md body weave
(where ``##`` collides with the article's own chapter headings, and where the
five sections never say *which figure* they describe).
"""

import importlib.util
import json
import re
import unittest
from pathlib import Path

if importlib.util.find_spec("pydantic_settings") is None:
    raise unittest.SkipTest("pydantic_settings is required to import package stage")

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, SourceInfo
from app.pipeline.package_utils import infer_table_asset_title
from app.pipeline.stages.package import AssetEntry, PackageStage

RULESET_DIR = Path(__file__).resolve().parents[1] / "app" / "pipeline" / "rulesets"


def _ir(blocks: list[Block], path: str = "示範經濟月刊.pdf") -> DocumentIR:
    return DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path=path, ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=blocks,
    )


def _text(block_id, page, text, *, level=0, bbox=None, order=0) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.TEXT,
        page_idx=page,
        bbox_norm=bbox or [110, 200, 880, 220],
        reading_order=order,
        payload={"text": text, "text_level": level},
    )


def _table(block_id, page, *, caption=None, bbox=None, order=0) -> Block:
    payload = {"table_body": "<table><tr><td>項目</td><td>金額</td></tr><tr><td>甲</td><td>10</td></tr></table>"}
    if caption is not None:
        payload["table_caption"] = caption
    return Block(
        block_id=block_id,
        type=BlockType.TABLE,
        page_idx=page,
        bbox_norm=bbox or [110, 260, 880, 520],
        reading_order=order,
        payload=payload,
    )


def _image(block_id, page, *, caption="", bbox=None, order=0) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.IMAGE,
        page_idx=page,
        bbox_norm=bbox or [110, 260, 880, 520],
        reading_order=order,
        payload={"img_path": f"{block_id}.png", "caption": caption},
    )


# ---------------------------------------------------------------------------
# Task A — table names recovered from the adjacent heading
# ---------------------------------------------------------------------------


class NearestTableCaptionTest(unittest.TestCase):
    def test_recovers_heading_above_table_across_a_unit_annotation(self):
        """The real layout: 「<heading>」 then 「單位：…」 then the grid."""
        document_ir = _ir(
            [
                _text("b1", 2, "示範島輸美前30大產品之出口額與市占率", level=2, bbox=[176, 198, 509, 218], order=0),
                _text("b2", 2, "單位：百萬美元；%", bbox=[747, 242, 887, 257], order=1),
                _table("b3", 2, bbox=[110, 261, 887, 822], order=2),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "示範島輸美前30大產品之出口額與市占率")

    def test_skips_bare_table_number_label_and_keeps_scanning(self):
        """「附表」 is a marker printed to the left of the name, and reading order
        can emit it after the name. It is not a table name on its own."""
        document_ir = _ir(
            [
                _text("b1", 3, "再生能源設置容量", level=2, bbox=[176, 198, 352, 216], order=0),
                _text("b2", 3, "附表", bbox=[107, 199, 151, 216], order=1),
                _text("b3", 3, "單位：MW", bbox=[810, 242, 889, 257], order=2),
                _table("b4", 3, bbox=[110, 261, 887, 479], order=3),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "再生能源設置容量")

    def test_stops_at_adjacent_body_prose(self):
        """Adjacency is the whole point: a paragraph above the table means the
        table's name was not printed, not that the paragraph is the name."""
        prose = (
            "隨著關稅政策持續升溫，示範島對外出口結構遂成為評估其衝擊影響的核心，"
            "本文透過量化方法檢視替代彈性。"
        )
        document_ir = _ir(
            [
                _text("b1", 2, "第二章 出口結構分析", level=2, bbox=[176, 100, 509, 120], order=0),
                _text("b2", 2, prose, bbox=[110, 180, 880, 250], order=1),
                _table("b3", 2, bbox=[110, 261, 887, 822], order=2),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "")

    def test_ignores_headings_from_a_previous_page(self):
        document_ir = _ir(
            [
                _text("b1", 1, "示範島產業競爭力評估", level=2, bbox=[176, 800, 509, 820], order=0),
                _table("b2", 2, bbox=[110, 261, 887, 822], order=1),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "")

    def test_ignores_running_header_far_above_the_table(self):
        """A same-page running header (page furniture at the very top) is not
        adjacent to a table that starts a third of the way down the page."""
        document_ir = _ir(
            [
                _text("b1", 2, "專題探索", level=2, bbox=[151, 80, 243, 101], order=0),
                _table("b2", 2, bbox=[110, 261, 887, 822], order=1),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "")

    def test_ignores_blocks_below_the_table(self):
        """A footnote emitted before the table in reading order must not become
        the table's name — a caption is printed above the grid."""
        document_ir = _ir(
            [
                _text("b1", 2, "資料來源整理", level=2, bbox=[110, 850, 500, 870], order=0),
                _table("b2", 2, bbox=[110, 261, 887, 822], order=1),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "")

    def test_stops_at_an_intervening_figure(self):
        document_ir = _ir(
            [
                _text("b1", 1, "示範島研發支出排名", level=2, bbox=[176, 198, 509, 218], order=0),
                _image("b2", 1, bbox=[126, 244, 872, 323], order=1),
                _table("b3", 1, bbox=[115, 342, 366, 468], order=2),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "")

    def test_accepts_an_explicit_table_number_caption_in_body_text(self):
        document_ir = _ir(
            [
                _text("b1", 4, "表3 示範研究院年度預算配置", bbox=[176, 220, 509, 240], order=0),
                _table("b2", 4, bbox=[110, 261, 887, 520], order=1),
            ]
        )

        caption = PackageStage()._nearest_table_caption(document_ir, document_ir.blocks[-1])

        self.assertEqual(caption, "表3 示範研究院年度預算配置")


class InferTableAssetTitleNearbyCaptionTest(unittest.TestCase):
    def test_nearby_caption_beats_the_document_title(self):
        title = infer_table_asset_title(
            caption=[],
            nearby_caption="示範島輸美前30大產品之出口額與市占率",
            source_title="示範島輸美產品進口替代彈性分析與政策意涵",
            page_idx=2,
            table_idx=0,
        )

        self.assertEqual(title, "示範島輸美前30大產品之出口額與市占率")

    def test_mineru_caption_still_wins_over_the_nearby_heading(self):
        """Over-correction guard: the parser's own caption is stronger evidence."""
        title = infer_table_asset_title(
            caption="示範島低軌衛星業者商業模式類型",
            nearby_caption="附表 2",
            source_title="示範島衛星產業觀察",
            page_idx=1,
            table_idx=0,
        )

        self.assertEqual(title, "示範島低軌衛星業者商業模式類型")

    def test_generic_nearby_caption_does_not_displace_the_document_title(self):
        title = infer_table_asset_title(
            caption=[],
            nearby_caption="表格",
            source_title="示範島衛星產業觀察",
            page_idx=0,
            table_idx=0,
        )

        self.assertEqual(title, "示範島衛星產業觀察 第 1 頁 表格 1")

    def test_document_title_fallback_survives_without_a_nearby_caption(self):
        title = infer_table_asset_title(
            caption=[],
            source_title="表一示範研究院檔案保存年限區分表",
            page_idx=0,
            table_idx=0,
        )

        self.assertEqual(title, "表一示範研究院檔案保存年限區分表 第 1 頁 表格 1")


class RagTableNameUsesRecoveredCaptionTest(unittest.TestCase):
    def test_rag_body_names_the_table_after_the_adjacent_heading(self):
        document_ir = _ir(
            [
                _text("b1", 2, "示範島輸美前30大產品之出口額與市占率", level=2, bbox=[176, 198, 509, 218], order=0),
                _text("b2", 2, "單位：百萬美元；%", bbox=[747, 242, 887, 257], order=1),
                _table("b3", 2, bbox=[110, 261, 887, 822], order=2),
            ]
        )

        markdown, _ = PackageStage()._render_rag_md(
            document_ir=document_ir,
            asset_map={},
            enrichments={},
        )

        self.assertIn("表格名稱：示範島輸美前30大產品之出口額與市占率", markdown)
        self.assertNotIn("表格名稱：示範經濟月刊", markdown)

    def test_the_recovered_heading_is_not_repeated_in_the_outline(self):
        """The name now comes *from* a heading the document already prints, so
        the table renderer's own copy of it would double the outline entry."""
        document_ir = _ir(
            [
                _text("b1", 2, "示範島輸美前30大產品之出口額與市占率", level=2, bbox=[176, 198, 509, 218], order=0),
                _text("b2", 2, "單位：百萬美元；%", bbox=[747, 242, 887, 257], order=1),
                _table("b3", 2, bbox=[110, 261, 887, 822], order=2),
            ]
        )

        markdown, _ = PackageStage()._render_rag_md(
            document_ir=document_ir,
            asset_map={},
            enrichments={},
        )

        headings = re.findall(r"(?m)^#{1,6} 示範島輸美前30大產品之出口額與市占率$", markdown)
        self.assertEqual(len(headings), 1, markdown)
        self.assertIn("表格名稱：示範島輸美前30大產品之出口額與市占率", markdown)

    def test_a_mineru_captioned_table_keeps_its_own_heading(self):
        """Guard: only the *recovered* heading is deduplicated. A caption that
        came from the parser has no heading block above it to collide with."""
        document_ir = _ir(
            [
                _text("b1", 2, "示範島衛星產業觀察", level=2, bbox=[176, 198, 509, 218], order=0),
                _table("b2", 2, caption="全球主要低軌衛星系統", bbox=[110, 261, 887, 822], order=1),
            ]
        )

        markdown, _ = PackageStage()._render_rag_md(
            document_ir=document_ir,
            asset_map={},
            enrichments={},
        )

        self.assertIn("## 全球主要低軌衛星系統", markdown)
        self.assertIn("表格名稱：全球主要低軌衛星系統", markdown)


# ---------------------------------------------------------------------------
# Task B — heading level and attribution of VLM figure sections
# ---------------------------------------------------------------------------

FLOW_OUTPUT = {
    "image_type": "flowchart",
    "semantic_caption": "這張圖片展示了示範島氫能政策的組織架構圖。圖中列出主管機關與所屬單位的層級關係。",
    "structured_content": [
        "示範島氫能推動小組 > 政策組 > 法規調適",
        "示範島氫能推動小組 > 技術組 > 示範場域",
    ],
    "all_text": [
        "示範島氫能推動小組",
        "政策組",
        "技術組",
        "法規調適",
        "示範場域",
        # Not reachable from any path line, so it lands in 「圖中文字」.
        "民國一百一十五年一月版",
        "示範島能源署製表",
    ],
    "facts": ["示範島氫能推動小組下設政策組與技術組，分別負責法規調適與示範場域推動。"],
    "keywords": ["氫能", "政策組", "技術組"],
}


class VisualSemanticHeadingLevelTest(unittest.TestCase):
    def test_default_heading_level_is_unchanged(self):
        text = PackageStage()._render_visual_semantic_content("示範島氫能政策架構", FLOW_OUTPUT)

        self.assertIn("## 語意摘要", text)
        self.assertIn("## 詳細流程路徑", text)
        self.assertNotIn("### 語意摘要", text)

    def test_heading_level_three_demotes_every_section(self):
        text = PackageStage()._render_visual_semantic_content(
            "示範島氫能政策架構", FLOW_OUTPUT, heading_level=3
        )

        self.assertIn("### 語意摘要", text)
        self.assertIn("### 詳細流程路徑", text)
        self.assertNotRegex(text, r"(?m)^## ")

    def test_heading_level_zero_renders_bold_section_labels(self):
        text = PackageStage()._render_visual_semantic_content(
            "示範島氫能政策架構", FLOW_OUTPUT, heading_level=0
        )

        self.assertIn("**語意摘要**", text)
        self.assertIn("**詳細流程路徑**", text)
        self.assertNotRegex(text, r"(?m)^#{1,6} ")

    def test_container_title_is_emitted_once_above_the_bold_sections(self):
        text = PackageStage()._render_visual_semantic_content(
            "示範島氫能政策架構",
            FLOW_OUTPUT,
            heading_level=0,
            container_title="圖3 示範島氫能推動組織",
        )

        self.assertEqual(re.findall(r"(?m)^#{1,6} .*$", text), ["### 圖3 示範島氫能推動組織"])
        self.assertTrue(text.startswith("### 圖3 示範島氫能推動組織"), text[:80])

    def test_english_sections_follow_the_same_level(self):
        text = PackageStage()._render_visual_semantic_content(
            "Hydrogen policy chart",
            {
                "image_type": "flowchart",
                "structured_content": ["Task force > Policy team > Regulation"],
                "all_text": ["Task force", "Policy team"],
                "facts": ["The task force has a policy team."],
                "keywords": ["hydrogen"],
            },
            semantic_output_language="en",
            heading_level=0,
        )

        self.assertIn("**Semantic Summary**", text)
        self.assertNotRegex(text, r"(?m)^#{1,6} ")


class VisualContainerTitleTest(unittest.TestCase):
    def setUp(self):
        self.stage = PackageStage()

    def test_prefers_the_parser_caption(self):
        block = _image("fig-a", 1, caption="圖2 示範島氫能推動組織")
        document_ir = _ir([block])

        title = self.stage._visual_container_title(document_ir, block, None, FLOW_OUTPUT)

        self.assertEqual(title, "圖2 示範島氫能推動組織")

    def test_picks_the_geometrically_nearest_figure_label_not_reading_order(self):
        """Three figures on one page: reading order would hand 「圖1」 to all of
        them. Each figure must take the label physically closest to it."""
        blocks = [
            _text("lab1", 4, "圖1 示範島能源結構", bbox=[110, 240, 400, 258], order=0),
            _text("lab2", 4, "圖2 示範島電網布局", bbox=[110, 520, 400, 538], order=1),
            _text("lab3", 4, "圖3 示範島儲能配置", bbox=[110, 800, 400, 818], order=2),
            _image("fig-a", 4, bbox=[110, 264, 880, 500], order=3),
            _image("fig-b", 4, bbox=[110, 544, 880, 780], order=4),
            _image("fig-c", 4, bbox=[110, 824, 880, 950], order=5),
        ]
        document_ir = _ir(blocks)
        by_id = {block.block_id: block for block in blocks}

        self.assertEqual(
            self.stage._visual_container_title(document_ir, by_id["fig-a"], None, FLOW_OUTPUT),
            "圖1 示範島能源結構",
        )
        self.assertEqual(
            self.stage._visual_container_title(document_ir, by_id["fig-b"], None, FLOW_OUTPUT),
            "圖2 示範島電網布局",
        )
        self.assertEqual(
            self.stage._visual_container_title(document_ir, by_id["fig-c"], None, FLOW_OUTPUT),
            "圖3 示範島儲能配置",
        )

    def test_joins_a_bare_label_with_the_name_typeset_beside_it(self):
        """「圖1」 and the figure's name are two blocks on one printed line."""
        blocks = [
            _text("lab", 3, "圖1", bbox=[110, 200, 144, 216], order=0),
            _text("name", 3, "示範島能源來源證明轉換核發", bbox=[176, 198, 410, 218], order=1),
            _image("fig-a", 3, bbox=[203, 242, 818, 484], order=2),
        ]
        document_ir = _ir(blocks)
        output = {key: value for key, value in FLOW_OUTPUT.items() if key != "semantic_caption"}

        title = self.stage._visual_container_title(document_ir, blocks[2], None, output)

        self.assertEqual(title, "圖1 示範島能源來源證明轉換核發")

    def test_a_label_far_from_the_figure_is_not_claimed(self):
        """One label, three panels: only the panel it touches may claim it. A
        source-note logo further down the page must not become 「圖2」."""
        blocks = [
            _text("lab", 4, "圖1 示範島設施影響區域示意圖", bbox=[110, 199, 426, 216], order=0),
            _image("fig-a", 4, bbox=[282, 246, 766, 396], order=1),
            _image("fig-b", 4, bbox=[194, 402, 786, 631], order=2),
            _image("fig-c", 4, bbox=[193, 641, 810, 872], order=3),
        ]
        document_ir = _ir(blocks)
        output = {key: value for key, value in FLOW_OUTPUT.items() if key != "semantic_caption"}
        by_id = {block.block_id: block for block in blocks}

        self.assertEqual(
            self.stage._visual_container_title(document_ir, by_id["fig-a"], None, output),
            "圖1 示範島設施影響區域示意圖",
        )
        for block_id in ("fig-b", "fig-c"):
            self.assertEqual(
                self.stage._visual_container_title(document_ir, by_id[block_id], None, output),
                "圖表語意",
            )

    def test_truncates_the_vlm_caption_at_the_first_sentence(self):
        block = _image("fig-a", 1)
        document_ir = _ir([block])

        title = self.stage._visual_container_title(document_ir, block, None, FLOW_OUTPUT)

        self.assertEqual(title, "這張圖片展示了示範島氫能政策的組織架構圖")
        self.assertNotIn("。", title)

    def test_cuts_a_long_caption_at_a_clause_boundary_not_mid_phrase(self):
        block = _image("fig-a", 1)
        document_ir = _ir([block])
        output = dict(FLOW_OUTPUT)
        output["semantic_caption"] = (
            "這張圖表展示了示範島產業創新政策的三個階段演進：從2017年的第一期計畫過渡到第二期"
        )

        title = self.stage._visual_container_title(document_ir, block, None, output)

        self.assertEqual(title, "這張圖表展示了示範島產業創新政策的三個階段演進")

    def test_truncates_a_run_on_vlm_caption_by_length(self):
        block = _image("fig-a", 1)
        document_ir = _ir([block])
        output = dict(FLOW_OUTPUT)
        output["semantic_caption"] = "這張圖片以非常詳細的方式展示了示範島氫能政策推動組織的完整架構關係以及各單位分工"

        title = self.stage._visual_container_title(document_ir, block, None, output)

        self.assertLessEqual(len(title), 24)
        self.assertTrue(title)

    def test_falls_back_to_an_unnumbered_generic_label(self):
        """81% of figures have neither a caption nor a page label. Inventing 「圖7」
        would contradict the article's own numbering — say nothing instead."""
        block = _image("fig-a", 1)
        document_ir = _ir([block])
        output = {key: value for key, value in FLOW_OUTPUT.items() if key != "semantic_caption"}

        title = self.stage._visual_container_title(document_ir, block, None, output)

        self.assertEqual(title, "圖表語意")
        self.assertNotRegex(title, r"[0-9０-９]")

    def test_does_not_borrow_a_label_from_another_page(self):
        blocks = [
            _text("lab1", 3, "圖1 示範島能源結構", bbox=[110, 240, 400, 258], order=0),
            _image("fig-a", 4, bbox=[110, 264, 880, 500], order=1),
        ]
        document_ir = _ir(blocks)
        output = {key: value for key, value in FLOW_OUTPUT.items() if key != "semantic_caption"}

        title = self.stage._visual_container_title(document_ir, blocks[1], None, output)

        self.assertEqual(title, "圖表語意")


class ListCaptionImageBlockTest(unittest.TestCase):
    """Incidental find while replaying the 167 cached IRs: MinerU emits
    ``img_caption`` as a list, and an assetless figure block appended it raw,
    so ``"\\n".join(block_lines)`` raised TypeError and killed the stage."""

    def test_list_caption_on_an_assetless_figure_renders_as_text(self):
        document_ir = _ir(
            [_image("fig-a", 1, bbox=[110, 264, 880, 500])],
        )
        document_ir.blocks[0].payload["caption"] = ["示範島新創企業統計", "資料來源：示範研究院。"]

        markdown, _ = PackageStage()._render_rag_md(
            document_ir=document_ir,
            asset_map={},
            enrichments={},
        )

        self.assertIn("示範島新創企業統計", markdown)


class RagBodyVisualAttributionTest(unittest.TestCase):
    def _render(self, *, caption="") -> str:
        block = _image("fig-a", 1, caption=caption, bbox=[110, 264, 880, 500])
        document_ir = _ir(
            [
                _text("h1", 1, "示範島氫能政策的推動架構", level=2, bbox=[110, 120, 880, 140], order=0),
                block,
            ]
        )
        markdown, _ = PackageStage()._render_rag_md(
            document_ir=document_ir,
            asset_map={},
            enrichments={"fig-a": {"kind": "figure_description", "output": dict(FLOW_OUTPUT)}},
        )
        return markdown

    def test_body_weave_does_not_collide_with_article_chapter_headings(self):
        markdown = self._render()

        self.assertIn("## 示範島氫能政策的推動架構", markdown)
        self.assertNotIn("## 語意摘要", markdown)
        self.assertNotIn("## 詳細流程路徑", markdown)
        self.assertIn("**語意摘要**", markdown)
        self.assertIn("**詳細流程路徑**", markdown)

    def test_body_weave_names_the_figure_once(self):
        markdown = self._render(caption="圖4 示範島氫能推動組織")

        self.assertIn("### 圖4 示範島氫能推動組織", markdown)
        self.assertEqual(markdown.count("### 圖4 示範島氫能推動組織"), 1)

    def test_body_weave_keeps_the_flow_content_searchable(self):
        markdown = self._render()

        self.assertIn("法規調適", markdown)
        self.assertIn("示範場域", markdown)


class SplitAssetDocumentUnchangedTest(unittest.TestCase):
    def test_split_asset_document_keeps_h2_sections(self):
        """The split document has its own ``#`` title, so ``##`` is correct there."""
        asset = AssetEntry(
            type="figure_asset",
            asset_id="fig0000",
            doc_id="doc-a",
            run_id="run-a",
            title="示範島氫能推動組織",
            page_idx=0,
            asset_path="assets/figures/fig0000.jpg",
            block_id="fig-a",
            retrieval_text=(
                "示範島氫能推動組織\n"
                "## 詳細流程路徑\n"
                "- 示範島氫能推動小組 > 政策組\n\n"
                "## 圖中文字\n"
                "- 法規調適\n"
                "- 示範場域"
            ),
            structured_content="示範島氫能推動小組 > 政策組",
            semantic_caption="這張圖片展示了示範島氫能政策的組織架構圖。",
            image_type="flowchart",
        )

        text = PackageStage()._render_split_asset_document(
            asset=asset,
            source_title="示範島氫能政策觀察",
            source_filename="hydrogen.pdf",
        )

        self.assertIn("# 示範島氫能推動組織", text)
        self.assertIn("## 詳細流程路徑", text)
        self.assertIn("法規調適", text)
        self.assertIn("示範場域", text)


class VisualTextRoundTripTest(unittest.TestCase):
    """``_extract_visual_text_lines_from_retrieval_text`` parses back what the
    renderer wrote. Demoting or bolding the section labels must not silently
    break 「圖中文字」 extraction — the failure mode has no symptom in the output.
    """

    EXPECTED = ["口頭及電話提交", "回覆申訴人", "結案或製成處理"]

    def _body(self, label: str) -> str:
        return (
            "示範島申訴處理作業流程圖\n"
            f"{label('詳細流程路徑')}\n"
            "- 收到民眾申訴 > 送承辦單位辦理\n\n"
            f"{label('圖中文字')}\n"
            "- 口頭及電話提交\n"
            "- 回覆申訴人\n"
            "- 結案或製成處理"
        )

    def test_h2_h3_and_bold_forms_yield_the_same_lines(self):
        stage = PackageStage()
        forms = {
            "h2": lambda name: f"## {name}",
            "h3": lambda name: f"### {name}",
            "bold": lambda name: f"**{name}**",
        }

        for name, label in forms.items():
            with self.subTest(form=name):
                lines = stage._extract_visual_text_lines_from_retrieval_text(self._body(label), "zh-TW")
                self.assertEqual(lines, self.EXPECTED)

    def test_a_following_section_still_terminates_collection(self):
        stage = PackageStage()
        for label, following in (
            (lambda name: f"## {name}", "## 重要事實"),
            (lambda name: f"### {name}", "### 重要事實"),
            (lambda name: f"**{name}**", "**重要事實**"),
        ):
            with self.subTest(following=following):
                body = self._body(label) + f"\n\n{following}\n- 不應被收進圖中文字"
                lines = stage._extract_visual_text_lines_from_retrieval_text(body, "zh-TW")
                self.assertEqual(lines, self.EXPECTED)

    def test_rendered_output_round_trips_at_every_heading_level(self):
        stage = PackageStage()
        baseline = None
        for kwargs in ({}, {"heading_level": 3}, {"heading_level": 0}):
            with self.subTest(**kwargs):
                rendered = stage._render_visual_semantic_content(
                    "示範島氫能政策架構", FLOW_OUTPUT, **kwargs
                )
                lines = stage._extract_visual_text_lines_from_retrieval_text(rendered, "zh-TW")
                if baseline is None:
                    baseline = lines
                    self.assertTrue(baseline, rendered)
                self.assertEqual(lines, baseline)


class TemplateSectionLabelRulesetTest(unittest.TestCase):
    """The VLM section labels are renderer scaffolding, not document facts; the
    repair fact guard must not count them toward the survival ratio."""

    def test_default_ruleset_lists_the_visual_section_labels(self):
        rules = json.loads((RULESET_DIR / "default.json").read_text(encoding="utf-8"))
        labels = set(rules["document_markers"]["template_section_labels"])

        for label in ("語意摘要", "重要事實", "詳細流程路徑", "圖中文字", "常見查詢主題"):
            self.assertIn(label, labels)

    def test_default_ruleset_labels_stay_corpus_neutral(self):
        """Real organization/journal names belong in the untracked local ruleset;
        the bundled default must stay generic renderer scaffolding."""
        labels = json.loads(
            (RULESET_DIR / "default.json").read_text(encoding="utf-8")
        )["document_markers"]["template_section_labels"]

        for label in labels:
            self.assertNotRegex(label, r"研究院|月刊|大學|股份有限公司|\d{2}\.\d{5}")


if __name__ == "__main__":
    unittest.main()
