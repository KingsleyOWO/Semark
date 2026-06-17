import importlib.util
import unittest

if importlib.util.find_spec("pydantic_settings") is None:
    raise unittest.SkipTest("pydantic_settings is required to import package stage")

from app.config import PROFILES, ProfileName
from app.models.document_ir import Block, BlockType
from app.pipeline.stages.package import (
    PackageStage,
    html_table_to_text,
    infer_table_asset_title,
    semantic_table_to_text,
)


class PackageSerializationTest(unittest.TestCase):
    def test_html_table_to_text_preserves_columns_and_rows(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Amount</th></tr>"
            "<tr><td>Alpha</td><td>10</td></tr>"
            "<tr><td>Beta</td><td>20</td></tr>"
            "</table>"
        )

        text = html_table_to_text(html, caption="Sample")

        self.assertIn("TABLE: Sample", text)
        self.assertIn("COLUMNS: Name | Amount", text)
        self.assertIn("ROW: Alpha | 10", text)
        self.assertIn("ROW: Beta | 20", text)

    def test_infer_table_asset_title_uses_source_title_for_empty_caption(self):
        title = infer_table_asset_title(
            caption=[],
            source_title="表一示範研究院檔案保存年限區分表",
            page_idx=0,
            table_idx=0,
        )

        self.assertEqual(title, "表一示範研究院檔案保存年限區分表 第 1 頁 表格 1")
        self.assertNotIn("Table", title)


    def test_semantic_table_to_text_renders_headerless_fragments_as_notes(self):
        html = (
            "<table>"
            "<tr><td>三、研究成果轉讓或授權使用文件</td><td>永久</td><td>1.委託/補助計畫成果：各單位</td></tr>"
            "<tr><td></td><td></td><td>2.本院自主性研究成果：資服中心</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, caption="保存年限區分表 第 6 頁 表格 2")

        self.assertIn("表格名稱：保存年限區分表 第 6 頁 表格 2", text)
        self.assertIn("內容類型：表格片段或續接資料", text)
        self.assertIn("三、研究成果轉讓或授權使用文件；永久；1.委託/補助計畫成果：各單位", text)
        self.assertIn("2.本院自主性研究成果：資服中心", text)
        self.assertNotIn("TABLE:", text)
        self.assertNotIn("COLUMNS:", text)


    def test_rag_table_renderer_uses_semantic_text_not_raw_html(self):
        block = Block(
            block_id="tbl",
            type=BlockType.TABLE,
            page_idx=0,
            payload={
                "table_caption": "保存年限片段",
                "table_body": (
                    "<table>"
                    "<tr><td>三、研究成果轉讓或授權使用文件</td><td>永久</td><td>各單位</td></tr>"
                    "</table>"
                ),
            },
        )

        text = "\n".join(PackageStage()._render_block_rag(block, {}))

        self.assertIn("內容類型：表格片段或續接資料", text)
        self.assertNotIn("<table", text)
        self.assertNotIn("TABLE:", text)
        self.assertNotIn("COLUMNS:", text)


    def test_accurate_profile_keeps_table_vlm_summary_disabled(self):
        self.assertFalse(PROFILES[ProfileName.ACCURATE].enrich.vlm_enrich_tables)


    def test_semantic_table_to_text_serializes_reference_rows(self):
        html = (
            "<table>"
            "<tr><td colspan=\"6\">01院務發展管理類</td></tr>"
            "<tr><th>分類號</th><th>項目</th><th>內容描述</th><th>保存年限</th><th>文件保管單位</th><th>備註</th></tr>"
            "<tr><td>01001</td><td>組織與規章</td><td>本院設立文件</td><td>永久</td><td>院長室</td><td></td></tr>"
            "<tr><td></td><td></td><td>法人印鑑及印信之樣式</td><td>永久</td><td>主秘辦公室</td><td></td></tr>"
            "<tr><td>03002</td><td>執行規劃</td><td>會議活動籌備文件</td><td>5年</td><td>資服中心</td><td></td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, caption="保存年限區分表")

        self.assertIn("表格名稱：保存年限區分表", text)
        self.assertIn("分類或範圍：01院務發展管理類", text)
        self.assertIn("### 01001 組織與規章", text)
        self.assertIn("- 內容描述：法人印鑑及印信之樣式", text)
        self.assertIn("- 文件保管單位：主秘辦公室", text)
        self.assertIn("### 03002 執行規劃", text)
        self.assertIn("- 保存年限：5年", text)
        self.assertIn("- 文件保管單位：資服中心", text)


if __name__ == "__main__":
    unittest.main()
