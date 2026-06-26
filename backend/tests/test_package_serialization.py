import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

if importlib.util.find_spec("pydantic_settings") is None:
    raise unittest.SkipTest("pydantic_settings is required to import package stage")

from app.config import PROFILES, ProfileName
from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.stages.package import (
    AssetEntry,
    PackageStage,
    html_table_to_text,
    infer_table_asset_title,
    semantic_table_to_text,
)
from app.pipeline.structured_rag import _infer_field_section


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


    def test_infer_table_asset_title_localizes_english_fallback(self):
        title = infer_table_asset_title(
            caption=[],
            source_title="Step 8: Payments and Refundable Credit",
            page_idx=0,
            table_idx=0,
            semantic_output_language="en",
        )

        self.assertEqual(title, "Step 8: Payments and Refundable Credit Page 1 Table 1")
        self.assertNotIn("第", title)
        self.assertNotIn("表格", title)


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




    def test_semantic_repair_rejects_json_shaped_markdown_after_title_prefix(self):
        markdown = '# Form\n\n{"status":"repaired","repaired_markdown":"# Actual"}'

        assert not PackageStage._semantic_repair_markdown_is_usable(markdown, "old semantic output" * 20, "en")


    def test_repair_evidence_includes_parse_failed_vlm_caption(self):
        stage = PackageStage()

        evidence = stage._repair_enrichment_text_for_page(
            {
                "fig": {
                    "kind": "figure_description",
                    "input": {"page_idx": 0},
                    "output": {
                        "_error": "JSON_PARSE_FAILED: truncated",
                        "semantic_caption": "{\"structured_content\":[\"Victim files complaint > Applicable law\"]}",
                    },
                }
            },
            0,
        )

        self.assertIn("model_error", evidence)
        self.assertIn("model_caption_or_raw_output", evidence)
        self.assertIn("Victim files complaint", evidence)


    def test_semantic_table_to_text_localizes_english_fragments(self):
        html = (
            "<table>"
            "<tr><td>Name</td><td>Amount</td></tr>"
            "<tr><td>Alpha</td><td>10</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, caption="Sample Table", semantic_output_language="en")

        self.assertIn("## Sample Table", text)
        self.assertIn("Content type: form/table fragment", text)
        self.assertIn("### Table Content", text)
        self.assertIn("- Name | Amount", text)
        self.assertNotIn("Row 1", text)
        self.assertNotIn("表格名稱", text)
        self.assertNotIn("第 1 列", text)


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


    def test_infer_source_title_skips_english_table_section_headings(self):
        source_md = (
            "## *60012251W* Page 1 Table 1\n"
            "Table name: *60012251W* Page 1 Table 1\n"
            "\n"
            "## Content\n"
            "- Row 1: placeholder\n"
            "\n"
            "## Step 8: Payments and Refundable Credit\n"
            "25 Illinois Income Tax withheld.\n"
        )

        title = PackageStage()._infer_source_title(source_md, "il-1040.pdf")

        self.assertEqual(title, "Step 8: Payments and Refundable Credit")


    def test_infer_source_title_prefers_english_document_title_from_body(self):
        source_md = (
            "## Table Content\n"
            "Content type: form/table fragment\n"
            "\n"
            "## Self Spouse Dependent(s)\n"
            "Filing status checkboxes.\n"
            "\n"
            "## Step 1: Personal Information\n"
            "Enter personal information and Social Security numbers.\n"
            "\n"
            "Illinois Department of Revenue 2025 Form IL-1040 Individual Income Tax Return\n"
            "\n"
            "## Step 2: Income\n"
            "Enter federal adjusted gross income.\n"
        )

        title = PackageStage()._infer_source_title(source_md, "il-1040.pdf")

        self.assertEqual(title, "Illinois Department of Revenue 2025 Form IL-1040 Individual Income Tax Return")


    def test_promotes_form_title_over_file_stub_main_title(self):
        title = PackageStage()._promoted_source_title_from_forms(
            source_title="internationaltravelform",
            source_filename="internationaltravelform.pdf",
            form_entries=[{"title": "INTERNATIONAL TRAVEL REIMBURSEMENT CLAIM FORM"}],
        )

        assert title == "INTERNATIONAL TRAVEL REIMBURSEMENT CLAIM FORM"


    def test_promotes_form_title_over_instruction_page_title(self):
        title = PackageStage()._promoted_source_title_from_forms(
            source_title="Chart for individual transcripts (Form 1040 series and Form W-2 and Form 1099)",
            source_filename="f4506t.pdf",
            form_entries=[{"title": "Form 4506-T Request for Transcript of Tax Return"}],
        )

        assert title == "Form 4506-T Request for Transcript of Tax Return"


    def test_generic_form_asset_title_does_not_override_structured_form_title(self):
        stage = PackageStage()
        asset = AssetEntry(
            type="form_asset",
            asset_id="form0000",
            doc_id="doc",
            run_id="run",
            title="Form 1",
            page_idx=0,
            asset_path="",
            block_id="form_page_0000",
            retrieval_text="Form 1",
        )

        assert stage._form_asset_titles_by_page([asset], "尋職簽證申請檢核表-final") == {}
        title = stage._best_form_export_title(
            {"title": "尋職簽證申請檢核表-final", "page_indices": [0]},
            "20180122_尋職簽證申請檢核表-final",
            {0: "Form 1"},
        )

        assert title == "尋職簽證申請檢核表-final"


    def test_clean_export_title_removes_english_form_instructions(self):
        assert (
            PackageStage._clean_export_title(
                "INTERNATIONAL TRAVEL REIMBURSEMENT CLAIM FORM Form and receipts must be submitted within 45 days"
            )
            == "INTERNATIONAL TRAVEL REIMBURSEMENT CLAIM FORM"
        )
        assert (
            PackageStage._clean_export_title(
                "Form 4506-T Request for Transcript of Tax Return Form Do not sign this form unless all applicable lines have been completed"
            )
            == "Form 4506-T Request for Transcript of Tax Return"
        )


    def test_infer_source_title_prefers_form_4506t_over_transcript_chart(self):
        source_md = (
            "Chart for individual transcripts (Form 1040 series and Form W-2 and Form 1099)\n"
            "4506-T Request for Transcript of Tax Return Form Do not sign this form unless all applicable lines have been completed.\n"
            "Taxpayer name and address fields.\n"
        )

        title = PackageStage()._infer_source_title(source_md, "f4506t.pdf")

        assert title == "Form 4506-T Request for Transcript of Tax Return"


    def test_infer_source_title_prefers_authorization_title_over_return_instructions(self):
        source_md = (
            "Washington State Health Care Authority\n"
            "Authorization for Release of Information\n"
            "Health Care Authority is authorized to release information or records about\n"
            "\n"
            "## Person or organization authorized to receive information or records\n"
            "Name and address fields.\n"
            "\n"
            "## Please return completed form to:\n"
            "Health Care Authority P.O. Box 42722 Olympia, WA\n"
        )

        title = PackageStage()._infer_source_title(source_md, "80-020-release-information-authorization.pdf")

        self.assertEqual(title, "Authorization for Release of Information")


    def test_infer_source_title_skips_toc_and_keeps_versioned_english_title(self):
        source_md = (
            "# Table of Contents\n"
            "Overview 1\n"
            "Sections on the Form 3\n"
            "\n"
            "# 50.34 version 2.0 | Human Specimen Submission Form Training Guide\n"
            "The purpose of this training is to assist providers in completing the specimen submission form.\n"
        )

        title = PackageStage()._infer_source_title(
            source_md,
            "human-50-34-specimen-submission-form-training-guide.pdf",
        )

        self.assertEqual(title, "50.34 version 2.0 | Human Specimen Submission Form Training Guide")


    def test_split_main_document_keeps_metadata_out_of_body(self):
        text = PackageStage()._render_split_main_document(
            source_md="""## Body
Text
[[asset:tbl0000]]""",
            source_title="English Document",
            source_filename="sample.pdf",
            form_entries=[],
            semantic_output_language="en",
        )

        self.assertTrue(text.startswith("# English Document"))
        self.assertIn("## Body", text)
        self.assertNotIn("Source file: sample.pdf", text)
        self.assertNotIn("Document type: main/regulation", text)
        self.assertNotIn("[[asset:", text)
        self.assertNotIn("來源檔案", text)


    def test_split_main_document_removes_display_noise(self):
        title = "Illinois Department of Revenue 2025 Form IL-1040 Individual Income Tax Return"
        text = PackageStage()._render_split_main_document(
            source_md=(
                "*60012251W*\n\n"
                f"{title}\n\n"
                "## Step 1: Personal Information\n\n"
                "Content type: form/table fragment\n\n"
                "### Step 1: Personal Information Enter personal information and Social Security numbers (SSN)\n"
                "- You must provide the entire SSN.\n"
            ),
            source_title=title,
            source_filename="il-1040.pdf",
            form_entries=[],
            semantic_output_language="en",
        )

        self.assertTrue(text.startswith(f"# {title}"))
        self.assertNotIn("*60012251W*", text)
        self.assertNotIn("Content type: form/table fragment", text)
        self.assertEqual(text.count("Step 1: Personal Information"), 1)
        self.assertNotIn("Enter personal information and Social Security numbers", text)
        self.assertEqual(text.count(title), 1)


    def test_split_asset_document_keeps_metadata_out_of_body(self):
        asset = AssetEntry(
            type="table_asset",
            asset_id="tbl0000",
            doc_id="doc",
            run_id="run",
            title="Sample Table",
            page_idx=0,
            asset_path="",
            block_id="block",
            retrieval_text="Table name: Sample Table",
            structured_content="Table name: Sample Table",
        )

        text = PackageStage()._render_split_asset_document(
            asset=asset,
            source_title="English Document",
            source_filename="sample.pdf",
            semantic_output_language="en",
        )

        self.assertTrue(text.startswith("# Sample Table"))
        self.assertIn("Table name: Sample Table", text)
        self.assertNotIn("Source document: English Document", text)
        self.assertNotIn("Source page: Page 1", text)
        self.assertNotIn("Document type: table", text)
        self.assertNotIn("來源文件", text)
        self.assertNotIn("文件類型", text)


    def test_split_form_document_uses_asset_title_and_removes_rag_prefixes(self):
        raw_markdown = """# Source Plan

## 政府機關著作權約定文件範本
頁碼：第 14 頁

表單：政府機關著作權約定文件範本。來源：sample.pdf，第 14 頁。

### 表單用途
表單：政府機關著作權約定文件範本。區塊：表單用途。本文件為講座授權書範本。

### RAG 查詢摘要
表單：政府機關著作權約定文件範本。區塊：RAG 查詢摘要。本文件可回答查詢。

### 表單欄位與欄位說明
- 表單欄位: 講座姓名(明確必填, name)。
"""

        text = PackageStage()._render_split_form_document(
            raw_markdown=raw_markdown,
            item={"form_id": "form_0000", "title": "範本 1-2：講座授權書", "page_label": "第 14 頁"},
            source_title="政府機關著作權約定文件範本",
            source_filename="sample.pdf",
        )

        self.assertTrue(text.startswith("# 範本 1-2：講座授權書"))
        self.assertIn("本文件為講座授權書範本。", text)
        self.assertIn("### 表單欄位與欄位說明", text)
        self.assertNotIn("表單：政府機關著作權約定文件範本。區塊", text)
        self.assertNotIn("RAG 查詢摘要", text)
        self.assertNotIn("來源檔案", text)


    def test_single_page_flowchart_source_does_not_export_figure_subdocument(self):
        stage = PackageStage()
        asset = AssetEntry(
            type="figure_asset",
            asset_id="fig0000",
            doc_id="doc-flow",
            run_id="run-flow",
            title="不同性騷擾申訴對象標準作業流程圖",
            page_idx=0,
            asset_path="assets/figures/fig0000.jpg",
            block_id="fig",
            retrieval_text="流程圖：被害人提出申訴，判斷適用法律並啟動調查程序。",
            semantic_caption="不同性騷擾申訴對象標準作業流程圖",
            structured_content="被害人提出申訴 > 判斷適用法律 > 啟動調查程序",
        )
        document_ir = DocumentIR(
            doc_id="doc-flow",
            run_id="run-flow",
            source=SourceInfo(path="202408221407355384561.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=0)],
            blocks=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            paths = stage._write_document_exports(
                outputs_dir=outputs,
                source_md="# 不同性騷擾申訴對象標準作業流程圖\n\n[[asset:fig0000]]",
                assets=[asset],
                structured_paths={},
                document_ir=document_ir,
                semantic_output_language="zh-TW",
            )

            index = json.loads(Path(paths["documents_index"]).read_text(encoding="utf-8"))
            self.assertEqual([item["document_id"] for item in index], ["main"])
            self.assertFalse((Path(paths["documents_dir"]) / "fig0000.md").exists())


    def test_normalize_repaired_markdown_replaces_weak_pdf_heading(self):
        stage = PackageStage()

        markdown = stage._normalize_repaired_markdown(
            "# 202408221407355384561.pdf\n\n## 不同性騷擾申訴對象標準作業流程圖\n流程內容。",
            "不同性騷擾申訴對象標準作業流程圖",
        )

        self.assertTrue(markdown.startswith("# 不同性騷擾申訴對象標準作業流程圖"))
        self.assertNotIn("# 202408221407355384561.pdf", markdown)


    def test_semantic_repair_rejects_ellipsis_in_final_markdown(self):
        repaired = (
            "# Visa Checklist\n\n"
            "## Conditions\n"
            "The applicant must satisfy the listed condition and not yet...\n"
            "Additional text makes this candidate long enough for the repair validator. " * 5
        )

        self.assertFalse(
            PackageStage._semantic_repair_markdown_is_usable(
                repaired,
                "# Visa Checklist\n\n## Conditions\nCurrent parser output. " * 8,
                "zh-TW",
            )
        )


    def test_decorative_figure_assets_are_not_split_documents(self):
        asset = AssetEntry(
            type="figure_asset",
            asset_id="fig0000",
            doc_id="doc",
            run_id="run",
            title="Figure 1",
            page_idx=0,
            asset_path="assets/figures/fig0000.jpg",
            block_id="block",
            retrieval_text="The image contains a downward-pointing arrow. There is no text visible within the image itself.",
            semantic_caption="The image contains a single downward-pointing arrow with no text visible.",
            facts=["The image contains a downward-pointing arrow with a rectangular shaft and triangular arrowhead."],
        )

        self.assertFalse(PackageStage()._should_export_asset_document(asset, "body", [asset]))


    def test_simple_arrow_figure_without_no_text_signal_is_decorative(self):
        asset = AssetEntry(
            type="figure_asset",
            asset_id="fig0000",
            doc_id="doc",
            run_id="run",
            title="Figure 1",
            page_idx=0,
            asset_path="assets/figures/fig0000.jpg",
            block_id="block",
            retrieval_text=(
                "Figure 1\n"
                "The image contains a single graphical element: a downward-pointing arrow.\n"
                "The arrow is black and has a rectangular shaft with a triangular head."
            ),
            semantic_caption="The image displays a simple, solid black arrow pointing downwards against a white background.",
            facts=[
                "The image contains a single graphical element: a downward-pointing arrow.",
                "The arrow is black and has a rectangular shaft with a triangular head.",
            ],
            keywords=["arrow", "down", "symbol", "icon", "direction"],
        )

        self.assertFalse(PackageStage()._should_export_asset_document(asset, "body", [asset]))


    def test_split_main_document_cleans_english_step_and_generic_table_headings(self):
        text = PackageStage()._render_split_main_document(
            source_md=(
                "## Step 4: Exemptions - See instructions for income limitations 10 a Enter the exemption amount\n"
                "- . See instructions. a .00\n\n"
                "### Step 5: Net Income and Tax 11 Residents: Net income: Subtract Line 10 from Line 9\n"
                "37If you have an amount on Line 32.\n"
                "aCheck if this applies.\n"
                "adirect deposit - Complete the information below.\n"
                "b  paper check.\n"
                "## Self  Spouse  Dependent(s)\n"
                "Refer to the 2025 instructions. DR. AP. RR DC IR ID\n"
                "St- k   l-V\n"
                ". DR     AP      RR      DC      IR      ID\n"
                "PrintReset\n\n"
                "## Table\n\n"
                "- 32 If Line 31 is greater than Line 24."
            ),
            source_title="Illinois Form",
            source_filename="il-1040.pdf",
            form_entries=[],
            semantic_output_language="en",
        )

        self.assertIn("## Step 4: Exemptions", text)
        self.assertIn("## Step 5: Net Income and Tax", text)
        self.assertIn("- See instructions. a .00", text)
        self.assertIn("37 If you have an amount on Line 32.", text)
        self.assertIn("a Check if this applies.", text)
        self.assertIn("a direct deposit - Complete the information below.", text)
        self.assertIn("b paper check.", text)
        self.assertIn("Self Spouse Dependent(s)", text)
        self.assertNotIn("## Self", text)
        self.assertNotIn("DR. AP. RR DC IR ID", text)
        self.assertNotIn("St- k", text)
        self.assertNotIn("DR      AP", text)
        self.assertNotIn("PrintReset", text)
        self.assertNotIn("## Table", text)
        self.assertNotIn("10 a Enter", text)
        self.assertNotIn("11 Residents", text)


    def test_split_main_document_cleans_chinese_toc_tables_and_figure_links(self):
        text = PackageStage()._render_split_main_document(
            source_md=(
                "壹、背景說明\n"
                "## 壹背景說明\n"
                "## 一、著作財產權授權書範本\n"
                "## 壹、背景說明.. ...2 貳、使用說明： ....3 一、適用於政府機關之委辦案件.. ....3 二、適用於政府機關邀請講座演講... ....9 三、適用於政府機關辦理活動涉及利用他人著作之授權\n"
                "表格名稱：壹、背景說明.. ...2 貳、使用說明： ....3 一、適用於政府機關之委辦案件.. ....3 二、適用於政府機關邀請講座演講... ....9 三、適用於政府機關辦理活動涉及利用他人著作之授權\n"
                "欄位：編、號 類、型\n"
                "## 二、著作權歸屬約定書範本\n"
                "## 內容\n"
                "- 第 1 列：編 號；類 型\n"
                "## 常見查詢主題\n\n"
                "## 詳細流程路徑\n"
                "- 廠商乙 > 不須交付 > 政府機關甲\n"
                "![Figure 2](asset://assets/figures/fig0001.jpg)\n"
                "2\n"
            ),
            source_title="政府機關著作權約定文件範本",
            source_filename="sample.pdf",
            form_entries=[],
            semantic_output_language="zh-TW",
        )

        self.assertIn("## 壹背景說明", text)
        self.assertIn("## 一、著作財產權授權書範本", text)
        self.assertIn("欄位：編、號 類、型", text)
        self.assertIn("## 二、著作權歸屬約定書範本", text)
        self.assertIn("- 第 1 列：編 號；類 型", text)
        self.assertIn("## 詳細流程路徑", text)
        self.assertIn("- 廠商乙 > 不須交付 > 政府機關甲", text)
        self.assertNotRegex(text, r"^壹、背景說明$", msg=text)
        self.assertNotIn("表格名稱：壹、背景說明", text)
        self.assertNotIn("## 壹、背景說明..", text)
        self.assertNotIn("## 內容", text)
        self.assertNotIn("## 常見查詢主題", text)
        self.assertNotIn("asset://", text)
        self.assertNotIn("Figure 2", text)
        self.assertNotRegex(text, r"^2$", msg=text)


    def test_flowchart_summary_keeps_ocr_nodes_when_vlm_path_is_partial(self):
        text = PackageStage()._render_visual_semantic_content(
            "長期照顧服務民眾抱怨申訴處理作業流程圖",
            {
                "image_type": "flowchart",
                "structured_content": [
                    "收到民眾申訴 > 書面提交：親自移交、郵寄、傳真或電子郵件 > 總收文統一收件，並掛文號 > 送承辦單位辦理"
                ],
                "all_text": [
                    "收到民眾申訴",
                    "口頭及電話提交",
                    "是否為 長照申訴案件",
                    "以公務機密維護密件之相關規定辦理",
                    "由主管負責接待，並填寫受理民眾申訴案件紀錄表",
                    "回覆申訴人",
                    "結案或製成處理",
                ],
            },
            semantic_output_language="zh-TW",
        )

        self.assertIn("## 詳細流程路徑", text)
        self.assertIn("## 圖中文字", text)
        self.assertIn("口頭及電話提交", text)
        self.assertIn("結案或製成處理", text)


    def test_split_flowchart_document_keeps_retrieval_text_image_text_section(self):
        asset = AssetEntry(
            type="figure_asset",
            asset_id="fig0000",
            doc_id="doc-flow",
            run_id="run-flow",
            title="長期照顧服務民眾、抱怨申訴處理作業流程圖",
            page_idx=0,
            asset_path="assets/figures/fig0000.jpg",
            block_id="image",
            retrieval_text=(
                "長期照顧服務民眾、抱怨申訴處理作業流程圖\n"
                "## 詳細流程路徑\n"
                "- 收到民眾申訴 > 送承辦單位辦理\n\n"
                "## 圖中文字\n"
                "- 口頭及電話提交\n"
                "- 回覆申訴人\n"
                "- 結案或製成處理"
            ),
            structured_content="收到民眾申訴 > 送承辦單位辦理",
            semantic_caption="This is a flowchart for complaint handling.",
        )

        text = PackageStage()._render_split_asset_document(
            asset=asset,
            source_title="長期照顧服務民眾、抱怨申訴處理作業流程圖",
            source_filename="flow.pdf",
            semantic_output_language="zh-TW",
        )

        self.assertIn("## 圖中文字", text)
        self.assertIn("口頭及電話提交", text)
        self.assertIn("結案或製成處理", text)


    def test_flowchart_output_backfills_same_page_ocr_text(self):
        document_ir = DocumentIR(
            doc_id="doc-flow",
            run_id="run-flow",
            source=SourceInfo(path="流程圖.pdf", ext="pdf", sha256="abc", size_bytes=1),
            engine=EngineInfo(backend="pipeline", method="ocr"),
            blocks=[
                Block(
                    block_id="title",
                    type=BlockType.TEXT,
                    page_idx=0,
                    reading_order=0,
                    payload={"text": "長期照顧服務民眾抱怨申訴處理作業流程圖"},
                ),
                Block(
                    block_id="image",
                    type=BlockType.IMAGE,
                    page_idx=0,
                    reading_order=1,
                    payload={},
                ),
                Block(
                    block_id="oral",
                    type=BlockType.TEXT,
                    page_idx=0,
                    reading_order=2,
                    payload={"text": "口頭及電話提交"},
                ),
                Block(
                    block_id="close",
                    type=BlockType.TEXT,
                    page_idx=0,
                    reading_order=3,
                    payload={"text": "結案或製成處理"},
                ),
                Block(
                    block_id="reply",
                    type=BlockType.TEXT,
                    page_idx=0,
                    reading_order=4,
                    payload={"text": "回覆申訴人"},
                ),
            ],
        )

        output = PackageStage()._augment_visual_output_from_page_text(
            {
                "image_type": "other",
                "structured_content": ["收到民眾申訴 > 送承辦單位辦理"],
            },
            document_ir,
            document_ir.blocks[1],
        )

        self.assertEqual(output["image_type"], "flowchart")
        self.assertIn("長期照顧服務民眾抱怨申訴處理作業流程圖", output["all_text"])
        self.assertIn("結案或製成處理", output["all_text"])


    def test_chinese_visual_summary_does_not_invent_workflow_steps(self):
        text = PackageStage()._render_visual_semantic_content(
            "Government Agency A (甲)",
            {
                "image_type": "flowchart",
                "structured_content": "Lecturer B (乙) > Authorize (授權) > Government Agency A (甲)",
                "keywords": ["政府機關", "講座", "授權"],
            },
            semantic_output_language="zh-TW",
        )

        self.assertIn("流程從「講座乙」開始", text)
        self.assertIn("最後可能連到「政府機關甲」", text)
        self.assertIn("- 講座乙 > 授權 > 政府機關甲", text)
        self.assertNotIn("Government Agency", text)
        self.assertNotIn("Lecturer B", text)
        self.assertNotIn("Authorize", text)
        self.assertNotIn("受理、補正、調查", text)
        self.assertNotIn("審議、通知、結案", text)


    def test_fragment_table_assets_are_not_split_documents(self):
        asset = AssetEntry(
            type="table_asset",
            asset_id="tbl0000",
            doc_id="doc",
            run_id="run",
            title="Step 1: Personal Information",
            page_idx=0,
            asset_path="",
            block_id="block",
            retrieval_text="Content type: form/table fragment\n### Step 1: Personal Information",
            structured_content="Content type: form/table fragment\n### Step 1: Personal Information",
        )

        self.assertFalse(PackageStage()._should_export_asset_document(asset, "body", [asset]))


    def test_english_table_title_uses_step_heading_from_body(self):
        html = (
            "<table>"
            "<tr><td>Step 8: Payments and Refundable Credit</td><td></td></tr>"
            "<tr><td>25 Illinois Income Tax withheld.</td><td>.00</td></tr>"
            "</table>"
        )

        title = PackageStage()._improve_table_asset_title_from_body(
            html,
            "*60012251W* Page 2 Table 2",
            "en",
        )

        self.assertEqual(title, "Step 8: Payments and Refundable Credit")


    def test_english_step_title_drops_trailing_instruction(self):
        html = (
            "<table>"
            "<tr><td>Step 1: Personal Information Enter personal information and Social Security numbers</td></tr>"
            "<tr><td>You must provide the entire SSN.</td></tr>"
            "</table>"
        )

        title = PackageStage()._improve_table_asset_title_from_body(
            html,
            "*60012251W* Page 1 Table 1",
            "en",
        )

        self.assertEqual(title, "Step 1: Personal Information")


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



    def test_low_confidence_table_ocr_does_not_invent_columns(self):
        html = (
            "<table>"
            "<tr><td>圖隊填表前精樣閱本表下方之填表注意事項</td><td>3</td><td>.7</td><td>驗</td><td></td></tr>"
            "<tr><td></td><td>1</td><td>□</td><td>、</td><td>2</td></tr>"
            "<tr><td>範，惟仍應事先向所屬單位主管報備。</td><td></td><td>4</td><td>.</td><td>5</td></tr>"
            "<tr><td></td><td>6</td><td>7</td><td>8</td><td>9</td></tr>"
            "</table>"
        )

        text = semantic_table_to_text(html, caption="表單 OCR 片段")

        self.assertIn("內容類型：低可信度表格 OCR", text)
        self.assertIn("表格 OCR 品質不足", text)
        self.assertNotIn("## 資料列", text)
        self.assertNotIn("欄位4", text)
        self.assertNotIn("### 驗", text)

    def test_low_confidence_table_assets_are_not_split_documents(self):
        asset = AssetEntry(
            type="table_asset",
            asset_id="tbl0000",
            doc_id="doc",
            run_id="run",
            title="表單 OCR 片段",
            page_idx=0,
            asset_path="",
            block_id="block",
            retrieval_text="內容類型：低可信度表格 OCR",
            structured_content="內容類型：低可信度表格 OCR\n表格 OCR 品質不足",
        )

        self.assertFalse(PackageStage()._should_export_asset_document(asset, "body", [asset]))

    def test_many_table_assets_are_exported_as_table_collection(self):
        stage = PackageStage()
        assets = []
        source_parts = []
        for idx in range(31):
            title = f"花蓮縣民政處檔案分類及保存年限區分表 第 {idx + 1} 頁 表格 {idx + 1}"
            body = "\n".join(
                [
                    f"## {title}",
                    f"表格名稱：{title}",
                    "欄位：分類號、項目、保存年限",
                    "",
                    "## 資料列",
                    "",
                    f"### A 01 01 {idx:02d}",
                    "- 項目：民政業務",
                    "- 保存年限：永久",
                ]
            )
            asset = AssetEntry(
                type="table_asset",
                asset_id=f"tbl{idx:04d}",
                doc_id="doc",
                run_id="run",
                title=title,
                page_idx=idx,
                asset_path="",
                block_id=f"block{idx:04d}",
                retrieval_text=body,
                structured_content=body,
            )
            assets.append(asset)
            source_parts.append(f"{body}\n\n[[asset:{asset.asset_id}]]")
        source_md = "\n\n".join(source_parts)
        document_ir = DocumentIR(
            doc_id="doc",
            run_id="run",
            source=SourceInfo(path="hualien.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=idx) for idx in range(31)],
            blocks=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = stage._write_document_exports(
                outputs_dir=Path(tmpdir),
                source_md=source_md,
                assets=assets,
                structured_paths={},
                document_ir=document_ir,
                semantic_output_language="zh-TW",
            )
            documents_dir = Path(paths["documents_dir"])
            index = json.loads(Path(paths["documents_index"]).read_text(encoding="utf-8"))
            main_text = Path(paths["main_document"]).read_text(encoding="utf-8")

            self.assertTrue((documents_dir / "table_collection_0000.md").exists())
            self.assertFalse((documents_dir / "tbl0000.md").exists())
            self.assertIn("table_collection", {item["kind"] for item in index})
            self.assertIn("## 關聯表格集合", main_text)
            self.assertIn("31 個表格區塊", main_text)
            self.assertNotIn("### A 01 01", main_text)

    def test_chinese_flowchart_summary_uses_terminal_nodes_and_localizes_branches(self):
        text = PackageStage()._render_visual_semantic_content(
            "長期照顧服務民眾抱怨申訴處理作業流程圖",
            {
                "image_type": "flowchart",
                "structured_content": [
                    "收到民眾申訴 > 是否為長照申訴案件 (No) > 結案",
                    "收到民眾申訴 > 是否為長照申訴案件 (Yes) > 派案處理 > 回覆申訴人",
                ],
                "keywords": ["申訴", "長照", "結案"],
            },
            semantic_output_language="zh-TW",
        )

        self.assertIn("可能結束於「結案、回覆申訴人」", text)
        self.assertIn("是否為長照申訴案件 （否）", text)
        self.assertIn("是否為長照申訴案件 （是）", text)
        self.assertNotIn("是否為長照申訴案件 (No)", text)

    def test_infer_source_title_prefers_complete_chinese_document_title(self):
        source_md = (
            "# 行政院所屬中央及地方各機關（構）\n\n"
            "適用對象說明。\n\n"
            "# 性騷擾案件申訴處理作業流程指引\n\n"
            "本指引提供申訴受理與處理流程。"
        )

        title = PackageStage()._infer_source_title(source_md, "sexual-harassment-guide.pdf")

        self.assertEqual(title, "性騷擾案件申訴處理作業流程指引")

    def test_legal_representative_fields_are_not_travel_fields(self):
        self.assertEqual(_infer_field_section("法定代理人姓名"), "申請/基本資料")
        self.assertEqual(_infer_field_section("委任代理人"), "申請/基本資料")
        self.assertEqual(_infer_field_section("職務代理人"), "出差/行程資訊")


    def test_split_main_document_cleans_isolated_cjk_ocr_noise_for_english(self):
        text = PackageStage()._render_split_main_document(
            source_md=(
                "## Specimen Instructions\n"
                "出 Relevant Immunization History\n"
                "Warning: JavaScript Window - CDC Specimen Submission Form 一 Specimen submissions require supplemental information.\n"
                "- Windows | 日\n"
                "- 口: Acanthamoeba Molecular Detection\n"
            ),
            source_title="Specimen Form",
            source_filename="specimen.pdf",
            form_entries=[],
            semantic_output_language="en",
        )

        self.assertIn("Relevant Immunization History", text)
        self.assertIn("Form - Specimen submissions", text)
        self.assertIn("- Windows", text)
        self.assertIn("- Acanthamoeba Molecular Detection", text)
        self.assertNotIn("出", text)
        self.assertNotIn(" 一 ", text)
        self.assertNotIn("日", text)
        self.assertNotIn("口", text)

    def test_low_value_form_asset_is_not_split_document(self):
        asset = AssetEntry(
            type="form_asset",
            asset_id="form0000",
            doc_id="doc",
            run_id="run",
            title="form0000",
            page_idx=0,
            asset_path="",
            block_id="form_page_0000",
            retrieval_text="The document appears to be a blank page with no visible text, fields, or content.",
            filling_guide="",
            field_schema=[],
        )

        self.assertFalse(PackageStage()._should_export_asset_document(asset, "", [asset]))

    def test_semantic_repair_triggered_by_readiness_issue(self):
        quality_gate = SimpleNamespace(
            issues=[SimpleNamespace(code="merged_field_detected", severity="warning")],
            stats={"semantic_quality": {"rag_readiness_score": 0.85, "recommended_repairs": ["split_merged_fields"]}},
        )

        self.assertTrue(PackageStage._quality_gate_needs_semantic_repair(quality_gate))


    def test_semantic_repair_rejects_chinese_template_for_english_output(self):
        repaired = (
            "# Authorization Form\n\n"
            "## 表單用途\n"
            "這是一段中文模板標題，對英文輸出而言不應接受。"
            "This extra text only makes the sample long enough for validation. " * 4
        )

        self.assertFalse(
            PackageStage._semantic_repair_markdown_is_usable(
                repaired,
                "# Authorization Form\n\n## Form Purpose\nCurrent body text." * 4,
                "en",
            )
        )


    def test_semantic_repair_updates_form_markdown_and_chunks(self):
        class FakeRepairAdapter:
            def __init__(self):
                self.calls = []

            async def enrich(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    success=True,
                    output={
                        "status": "repaired",
                        "repaired_markdown": (
                            "# Authorization Form\n\n"
                            "## Purpose\n"
                            "Use this form to authorize disclosure of records for a specific request.\n\n"
                            "## Fields\n"
                            "- Name\n"
                            "- SSN / Birthday\n"
                            "- Signature and date signed\n\n"
                            "## Approval And Notes\n"
                            "The requester must sign the form before records can be released."
                        ),
                        "summary": "Removed boilerplate and rebuilt useful field sections.",
                        "applied_repairs": ["rewrite_semantic_markdown", "split_merged_fields"],
                        "confidence": 0.91,
                    },
                    tokens_used=37,
                    duration_seconds=0.2,
                )

        stage = PackageStage()
        document_ir = DocumentIR(
            doc_id="doc",
            run_id="run",
            source=SourceInfo(path="authorization.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=0)],
            blocks=[
                Block(
                    block_id="b0",
                    type=BlockType.TEXT,
                    page_idx=0,
                    payload={"text": "Authorization Form Name SSN / Birthday Signature Date Signed"},
                )
            ],
        )
        quality_gate = SimpleNamespace(
            issues=[
                SimpleNamespace(
                    code="merged_field_detected",
                    severity="warning",
                    message="Field labels are merged.",
                    page_idx=None,
                    evidence={"fields": ["Name SSN Birthday Signature"]},
                )
            ],
            stats={"semantic_quality": {"rag_readiness_score": 0.8, "recommended_repairs": ["split_merged_fields"]}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            forms = outputs / "forms"
            docs = outputs / "documents"
            forms.mkdir()
            docs.mkdir()
            form_md = forms / "form_0000.md"
            form_doc = docs / "form_0000.md"
            form_md.write_text(
                "# Authorization Form\n\n## RAG Summary\nThis document can answer generic questions.\n",
                encoding="utf-8",
            )
            form_doc.write_text("# Authorization Form\n\nOld body", encoding="utf-8")
            (outputs / "forms_index.json").write_text(
                json.dumps(
                    [
                        {
                            "form_id": "form_0000",
                            "subdoc_id": "doc:form:0000",
                            "title": "Authorization Form",
                            "page_indices": [0],
                            "page_label": "Page 1",
                            "field_count": 1,
                            "files": {"markdown": str(form_md)},
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (outputs / "documents_index.json").write_text(
                json.dumps(
                    [
                        {
                            "document_id": "main",
                            "kind": "main",
                            "title": "Authorization Form",
                            "source_filename": "authorization.pdf",
                            "file": str(docs / "main.md"),
                        },
                        {
                            "document_id": "form_0000",
                            "kind": "form",
                            "title": "Authorization Form",
                            "source_filename": "authorization.pdf",
                            "file": str(form_doc),
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (outputs / "structured_chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "old",
                        "doc_id": "doc:form:0000",
                        "content": "old parser fallback",
                        "metadata": {"subdoc_id": "doc:form:0000"},
                    }
                ) + "\n",
                encoding="utf-8",
            )

            adapter = FakeRepairAdapter()
            stats = asyncio.run(
                stage._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Authorization Form",
                    structured_output=SimpleNamespace(plan=SimpleNamespace(document_type="form_collection")),
                    quality_gate=quality_gate,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=adapter,
                )
            )

            self.assertEqual(stats["applied_count"], 1)
            self.assertEqual(adapter.calls[0]["kind"], "semantic_repair")
            self.assertIn("SSN / Birthday", form_md.read_text(encoding="utf-8"))
            self.assertIn("Approval And Notes", form_doc.read_text(encoding="utf-8"))
            chunks_text = (outputs / "structured_chunks.jsonl").read_text(encoding="utf-8")
            self.assertIn('"view": "semantic_repair"', chunks_text)
            self.assertIn("split_merged_fields", chunks_text)
            self.assertNotIn("old parser fallback", chunks_text)
            repair_log = json.loads((outputs / "semantic_repair.json").read_text(encoding="utf-8"))
            self.assertEqual(repair_log["items"][0]["status"], "applied")



    def test_semantic_repair_structured_form_document_updates_main_document(self):
        class FakeRepairAdapter:
            async def enrich(self, **kwargs):
                return SimpleNamespace(
                    success=True,
                    output={
                        "status": "repaired",
                        "repaired_markdown": "# Visa Checklist\n\n## Required Documents\n- Passport copy.\n- Proof of financial resources of at least NT$100,000.\n\n## Applicant Signature\nThe applicant signs after confirming the checklist.",
                        "summary": "Rebuilt the whole checklist as one RAG-ready document.",
                        "applied_repairs": ["whole_document_rewrite"],
                        "confidence": 0.91,
                    },
                    tokens_used=80,
                    duration_seconds=0.2,
                )

        stage = PackageStage()
        document_ir = DocumentIR(
            doc_id="doc",
            run_id="run",
            source=SourceInfo(path="visa-checklist.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)],
            blocks=[
                Block(
                    block_id="b0",
                    type=BlockType.TEXT,
                    page_idx=0,
                    payload={"text": "Visa checklist passport proof of financial resources applicant signature"},
                )
            ],
        )
        quality_gate = SimpleNamespace(
            issues=[
                SimpleNamespace(
                    code="semantic_template_incomplete",
                    severity="high",
                    message="Missing final sections.",
                    page_idx=None,
                    evidence={},
                )
            ],
            stats={"semantic_quality": {"rag_readiness_score": 0.4, "recommended_repairs": ["whole_document_rewrite"]}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            docs = outputs / "documents"
            docs.mkdir()
            (docs / "main.md").write_text("# Visa Checklist\n\nOld parser body", encoding="utf-8")
            (outputs / "documents_index.json").write_text(
                json.dumps(
                    [
                        {
                            "document_id": "main",
                            "kind": "main",
                            "title": "Visa Checklist",
                            "source_filename": "visa-checklist.pdf",
                            "file": str(docs / "main.md"),
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (outputs / "structured_rag.md").write_text("# Visa Checklist\n\nOld structured body", encoding="utf-8")
            (outputs / "source.md").write_text("# Visa Checklist\n\nOld source body", encoding="utf-8")
            (outputs / "rag.md").write_text("# Visa Checklist\n\nOld rag body", encoding="utf-8")
            (outputs / "structured_chunks.jsonl").write_text(json.dumps({"chunk_id": "old", "content": "old fallback"}) + "\n", encoding="utf-8")

            stats = asyncio.run(
                stage._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Visa Checklist",
                    structured_output=SimpleNamespace(
                        plan=SimpleNamespace(document_type="form_document", title="Visa Checklist"),
                        rag_markdown="# Visa Checklist\n\nOld structured body",
                    ),
                    quality_gate=quality_gate,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=FakeRepairAdapter(),
                )
            )

            self.assertEqual(stats["applied_count"], 1)
            for filename in ("structured_rag.md", "source.md", "rag.md"):
                self.assertIn("NT$100,000", (outputs / filename).read_text(encoding="utf-8"))
            self.assertIn("NT$100,000", (docs / "main.md").read_text(encoding="utf-8"))
            chunks_text = (outputs / "structured_chunks.jsonl").read_text(encoding="utf-8")
            self.assertIn('"view": "semantic_repair"', chunks_text)
            self.assertNotIn("old fallback", chunks_text)


    def test_semantic_repair_retains_rejected_form_candidate_and_marks_chunks(self):
        class FakeRejectedRepairAdapter:
            async def enrich(self, **kwargs):
                return SimpleNamespace(
                    success=True,
                    output={
                        "status": "uncertain",
                        "repaired_markdown": "",
                        "summary": "Reviewer could not produce a grounded final rewrite.",
                        "applied_repairs": [],
                        "confidence": 0.0,
                    },
                    tokens_used=22,
                    duration_seconds=0.1,
                )

        stage = PackageStage()
        document_ir = DocumentIR(
            doc_id="doc",
            run_id="run",
            source=SourceInfo(path="visa-checklist.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)],
            blocks=[
                Block(
                    block_id="b0",
                    type=BlockType.TEXT,
                    page_idx=0,
                    payload={"text": "Employment-seeking visa checklist. Passport no. Professional sector. Conditions."},
                )
            ],
        )
        quality_gate = SimpleNamespace(
            issues=[
                SimpleNamespace(
                    code="vlm_enrichment_parse_failed",
                    severity="high",
                    message="VLM output could not be parsed.",
                    page_idx=0,
                    evidence={"error": "JSON_PARSE_FAILED"},
                )
            ],
            stats={"semantic_quality": {"rag_readiness_score": 0.2, "recommended_repairs": ["repair_unparsed_enrichment_output"]}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            forms = outputs / "forms"
            docs = outputs / "documents"
            forms.mkdir()
            docs.mkdir()
            form_md = forms / "form_0000.md"
            form_chunks = forms / "form_0000.chunks.jsonl"
            form_doc = docs / "form_0000.md"
            candidate_markdown = (
                "# Visa Checklist\n\n"
                "## Filling Focus\n"
                "This parser/VLM candidate still has issues, but it is the best available non-empty semantic output. " * 3
            )
            form_md.write_text(candidate_markdown, encoding="utf-8")
            form_chunks.write_text(json.dumps({"chunk_id": "old_form", "content": "candidate fallback"}) + "\n", encoding="utf-8")
            form_doc.write_text("# Visa Checklist\n\nOld body", encoding="utf-8")
            (outputs / "forms_index.json").write_text(
                json.dumps(
                    [
                        {
                            "form_id": "form_0000",
                            "subdoc_id": "doc:form:0000",
                            "logical_doc_id": "doc::form:0000",
                            "title": "Visa Checklist",
                            "page_indices": [0],
                            "page_label": "Page 1",
                            "field_count": 4,
                            "files": {"markdown": str(form_md), "chunks": str(form_chunks)},
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (outputs / "documents_index.json").write_text(
                json.dumps(
                    [
                        {
                            "document_id": "main",
                            "kind": "main",
                            "title": "Visa Checklist",
                            "source_filename": "visa-checklist.pdf",
                            "file": str(docs / "main.md"),
                        },
                        {
                            "document_id": "form_0000",
                            "kind": "form",
                            "title": "Visa Checklist",
                            "source_filename": "visa-checklist.pdf",
                            "file": str(form_doc),
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (outputs / "structured_chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "old_form",
                        "doc_id": "doc:form:0000",
                        "content": "candidate fallback",
                        "metadata": {"subdoc_id": "doc:form:0000"},
                    },
                    ensure_ascii=False,
                )
                + "\n"
                + json.dumps(
                    {
                        "chunk_id": "other",
                        "doc_id": "doc:form:0001",
                        "content": "other content",
                        "metadata": {"subdoc_id": "doc:form:0001"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (outputs / "structured_rag.md").write_text(candidate_markdown, encoding="utf-8")

            stats = asyncio.run(
                stage._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md="# Visa Checklist",
                    structured_output=SimpleNamespace(plan=SimpleNamespace(document_type="form_collection")),
                    quality_gate=quality_gate,
                    enrichments={},
                    semantic_output_language="zh-TW",
                    review_adapter=FakeRejectedRepairAdapter(),
                )
            )

            self.assertEqual(stats["applied_count"], 0)
            self.assertEqual(stats["blocked_count"], 0)
            self.assertEqual(stats["fallback_count"], 1)
            self.assertIn("best available non-empty semantic output", form_md.read_text(encoding="utf-8"))
            self.assertNotIn("已阻擋自動進入 RAG", form_md.read_text(encoding="utf-8"))
            self.assertIn("best available non-empty semantic output", form_doc.read_text(encoding="utf-8"))
            chunks_text = (outputs / "structured_chunks.jsonl").read_text(encoding="utf-8")
            self.assertIn("candidate fallback", chunks_text)
            self.assertIn("other content", chunks_text)
            self.assertIn('"semantic_repair_status": "fallback_retained"', chunks_text)
            self.assertIn('"auto_rag_ready": false', chunks_text)
            form_chunks_text = form_chunks.read_text(encoding="utf-8")
            self.assertIn("candidate fallback", form_chunks_text)
            self.assertIn("fallback_retained", form_chunks_text)
            self.assertIn("best available non-empty semantic output", (outputs / "structured_rag.md").read_text(encoding="utf-8"))
            forms_index = json.loads((outputs / "forms_index.json").read_text(encoding="utf-8"))
            self.assertEqual(forms_index[0]["semantic_repair"]["status"], "fallback_retained")
            self.assertFalse(forms_index[0]["semantic_repair"]["auto_rag_ready"])

    def test_semantic_repair_retains_structured_candidate_when_reviewer_output_unusable(self):
        class FakeRejectedRepairAdapter:
            async def enrich(self, **kwargs):
                return SimpleNamespace(
                    success=True,
                    output={
                        "status": "uncertain",
                        "repaired_markdown": "",
                        "summary": "Reviewer returned invalid final markdown.",
                        "applied_repairs": [],
                        "confidence": 0.0,
                    },
                    tokens_used=22,
                    duration_seconds=0.1,
                )

        stage = PackageStage()
        candidate_markdown = "# Visa Checklist\n\n## Required Documents\n- Passport copy.\n- Proof of financial resources.\n"
        document_ir = DocumentIR(
            doc_id="doc",
            run_id="run",
            source=SourceInfo(path="visa-checklist.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)],
            blocks=[
                Block(
                    block_id="b0",
                    type=BlockType.TEXT,
                    page_idx=0,
                    payload={"text": "Visa checklist passport proof of financial resources."},
                )
            ],
        )
        quality_gate = SimpleNamespace(
            issues=[
                SimpleNamespace(
                    code="semantic_template_incomplete",
                    severity="high",
                    message="Reviewer repair is required.",
                    page_idx=None,
                    evidence={},
                )
            ],
            stats={"semantic_quality": {"rag_readiness_score": 0.3, "recommended_repairs": ["whole_document_rewrite"]}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            docs = outputs / "documents"
            docs.mkdir()
            (docs / "main.md").write_text("# Visa Checklist\n\nOld body", encoding="utf-8")
            (outputs / "documents_index.json").write_text(
                json.dumps(
                    [
                        {
                            "document_id": "main",
                            "kind": "main",
                            "title": "Visa Checklist",
                            "source_filename": "visa-checklist.pdf",
                            "file": str(docs / "main.md"),
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            for filename in ("structured_rag.md", "source.md", "rag.md"):
                (outputs / filename).write_text(candidate_markdown, encoding="utf-8")
            (outputs / "structured_chunks.jsonl").write_text(
                json.dumps(
                    {
                        "chunk_id": "old",
                        "doc_id": "doc",
                        "run_id": "run",
                        "view": "structured_rag",
                        "content": "Passport copy and proof of financial resources.",
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            stats = asyncio.run(
                stage._apply_semantic_repair(
                    outputs_dir=outputs,
                    document_ir=document_ir,
                    source_md=candidate_markdown,
                    structured_output=SimpleNamespace(
                        plan=SimpleNamespace(document_type="form_document", title="Visa Checklist"),
                        rag_markdown=candidate_markdown,
                    ),
                    quality_gate=quality_gate,
                    enrichments={},
                    semantic_output_language="en",
                    review_adapter=FakeRejectedRepairAdapter(),
                )
            )

            self.assertEqual(stats["applied_count"], 0)
            self.assertEqual(stats["blocked_count"], 0)
            self.assertEqual(stats["fallback_count"], 1)
            self.assertIn("Passport copy", (outputs / "structured_rag.md").read_text(encoding="utf-8"))
            self.assertIn("Passport copy", (docs / "main.md").read_text(encoding="utf-8"))
            chunks_text = (outputs / "structured_chunks.jsonl").read_text(encoding="utf-8")
            self.assertIn("Passport copy and proof", chunks_text)
            self.assertIn('"semantic_repair_status": "fallback_retained"', chunks_text)
            self.assertIn('"auto_rag_ready": false', chunks_text)

    def test_form_document_does_not_export_form_asset_subdocuments(self):
        stage = PackageStage()
        document_ir = DocumentIR(
            doc_id="doc",
            run_id="run",
            source=SourceInfo(path="visa-checklist.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)],
            blocks=[],
        )
        asset = AssetEntry(
            type="form_asset",
            asset_id="form0002",
            doc_id="doc",
            run_id="run",
            title="Required Documents",
            page_idx=1,
            asset_path="assets/forms/form_p0001.png",
            block_id="b000004",
            retrieval_text="Required documents passport proof of financial resources",
            filling_guide="## Form Purpose\nRequired document checklist",
            field_schema=[{"name": "Passport", "type": "checkbox", "required": True}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir)
            plan_path = outputs / "document_plan.json"
            plan_path.write_text(json.dumps({"document_type": "form_document"}), encoding="utf-8")
            paths = stage._write_document_exports(
                outputs_dir=outputs,
                source_md="# Visa Checklist\n\nWhole-form final semantic content.",
                assets=[asset],
                structured_paths={"document_plan": str(plan_path)},
                document_ir=document_ir,
                semantic_output_language="en",
            )

            index = json.loads(Path(paths["documents_index"]).read_text(encoding="utf-8"))
            self.assertEqual([item["document_id"] for item in index], ["main"])
            self.assertFalse((Path(paths["documents_dir"]) / "form0002.md").exists())


    def test_low_value_forms_index_item_is_not_exported_as_document(self):
        stage = PackageStage()
        document_ir = DocumentIR(
            doc_id="doc",
            run_id="run",
            source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=100),
            engine=EngineInfo(backend="pipeline", method="auto"),
            pages=[PageInfo(page_idx=0)],
            blocks=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            forms_dir = tmp / "forms"
            forms_dir.mkdir()
            form_md = forms_dir / "form_0000.md"
            form_md.write_text(

                "# sample.pdf\n\n"
                "## sample\n"
                "Page: Page 1\n\n"
                "### Form Purpose\n"
                "\"sample\" is a form from source file \"sample.pdf\". Use it to capture, submit, authorize, verify, or record the listed information.\n\n"
                "### Use Cases\n"
                "Retrieve this document when users ask about the purpose, completion method, fields, approval flow, or notes for \"sample\".\n\n"
                "### Form Structure\n"
                "- Basic Information - Entry Details - Approval\n",
                encoding="utf-8",
            )
            forms_index = tmp / "forms_index.json"
            forms_index.write_text(
                json.dumps(
                    [
                        {
                            "form_id": "form_0000",
                            "title": "sample",
                            "page_indices": [0],
                            "page_label": "Page 1",
                            "field_count": 0,
                            "record_count": 4,
                            "files": {"markdown": str(form_md)},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            paths = stage._write_document_exports(
                outputs_dir=tmp,
                source_md="# Sample Source\n\nUseful body text.",
                assets=[],
                structured_paths={"forms_index": str(forms_index)},
                document_ir=document_ir,
                semantic_output_language="en",
            )
            documents_dir = Path(paths["documents_dir"])
            index = json.loads(Path(paths["documents_index"]).read_text(encoding="utf-8"))
            main_text = Path(paths["main_document"]).read_text(encoding="utf-8")

            self.assertFalse((documents_dir / "form_0000.md").exists())
            self.assertEqual({item["kind"] for item in index}, {"main"})
            self.assertNotIn("Related Forms and Attachments", main_text)


if __name__ == "__main__":
    unittest.main()
