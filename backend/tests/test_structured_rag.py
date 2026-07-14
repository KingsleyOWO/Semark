import asyncio
import json

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.quality_gate import (
    _build_vlm_audit_candidates,
    _check_assets_semantics,
    run_quality_gate,
)
from app.pipeline.stages.package import AssetEntry, PackageStage
from app.pipeline.structured_rag import (
    build_form_documents_rag,
    build_structured_rag,
    is_form_like_document,
    looks_like_reference_table,
    normalize_vlm_table_records,
    plan_document,
    record_to_rag_text,
    select_vlm_fallback_pages,
    write_structured_rag_outputs,
)


def test_travel_allowance_table_generates_row_level_records():
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TEXT,
                page_idx=0,
                payload={
                    "text": "中央政府各機關派赴國外各地區出差人員生活費日支數額表",
                    "text_level": 2,
                },
            ),
            Block(
                block_id="b000002",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "自115年1月1日生效", "text_level": 0},
            ),
            Block(
                block_id="b000003",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "單位：美元", "text_level": 0},
            ),
            Block(
                block_id="b000004",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr>
                        <td colspan="2">編號</td>
                        <td>名稱(地區、國家、城市或其他)</td>
                        <td>日支數額</td>
                      </tr>
                      <tr><td>地區、國家</td><td>城市或其他</td></tr>
                      <tr><td>A</td><td></td><td>亞太地區</td><td></td></tr>
                      <tr><td></td><td></td><td>日本(Japan)</td><td></td></tr>
                      <tr><td></td><td>6</td><td>東京(Tokyo)</td><td>299</td></tr>
                      <tr><td></td><td>70</td><td>德里國家首都區(NCT)</td><td></td></tr>
                      <tr><td></td><td></td><td>(09/01-03/31)</td><td>259</td></tr>
                      <tr><td></td><td>71</td><td>浦內(Pune)</td><td>186</td></tr>
                      <tr><td></td><td>8</td><td>其他(Other)</td><td>209</td></tr>
                    </table>
                    """,
                },
            ),
        ],
    )

    output = build_structured_rag(document_ir)

    assert output.plan.document_type == "travel_daily_allowance_table"
    assert output.plan.query_granularity == "one_record_per_location_rate"
    assert output.plan.currency == "USD"
    assert output.plan.effective_date == "115年1月1日"

    tokyo = next(record for record in output.records if record["city_zh"] == "東京")
    assert tokyo["region"] == "亞太地區"
    assert tokyo["country_zh"] == "日本"
    assert tokyo["country_en"] == "Japan"
    assert tokyo["city_en"] == "Tokyo"
    assert tokyo["rate_usd"] == 299
    assert tokyo["location_type"] == "city"

    delhi = next(record for record in output.records if record["condition"] == "(09/01-03/31)")
    assert delhi["city_zh"] == "德里國家首都區"
    assert delhi["rate_usd"] == 259
    assert not any(record["rate_usd"] == 70 for record in output.records)

    pune = next(record for record in output.records if record["city_en"] == "Pune")
    assert pune["country_zh"] == "日本"

    assert any("東京(Tokyo)" in chunk["content"] for chunk in output.chunks)
    assert len(output.records) == 4



def test_english_official_form_with_instruction_page_stays_single_document():
    document_ir = DocumentIR(
        doc_id="doc-ssa",
        run_id="run-ssa",
        source=SourceInfo(path="ssa-827.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[PageInfo(page_idx=0, page_image_path="p0.png"), PageInfo(page_idx=1, page_image_path="p1.png")],
        blocks=[
            Block(
                block_id="title",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "AUTHORIZATION TO DISCLOSE INFORMATION TO THE SOCIAL SECURITY ADMINISTRATION (SSA)"},
            ),
            Block(
                block_id="form-table",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": "<table><tr><td>Name</td><td>SSN</td><td>Signature</td></tr></table>",
                },
            ),
            Block(
                block_id="explanation-title",
                type=BlockType.TEXT,
                page_idx=1,
                payload={"text": "Explanation of Form SSA-827"},
            ),
            Block(
                block_id="privacy",
                type=BlockType.TEXT,
                page_idx=1,
                payload={"text": "Privacy Act Statement Collection and Use of Personal Information. We will use the information you provide to determine your eligibility or continuing eligibility of benefits."},
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form-table": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "AUTHORIZATION TO DISCLOSE INFORMATION TO THE SOCIAL SECURITY ADMINISTRATION (SSA)",
                    "document_type": "form",
                    "field_schema": [{"name": "Name", "type": "name", "required": True}],
                    "filling_guide": "## Form Purpose\nUse this form to authorize disclosure.",
                },
            }
        },
        semantic_output_language="en",
    )

    assert output.plan.document_type == "form_document"
    assert output.plan.rag_strategy == "single_form_document_semantic_chunks"
    assert output.plan.evidence["single_source_form_document"] is True
    assert output.plan.evidence["source_page_indices"] == [0, 1]
    assert output.stats["form_count"] == 1
    assert output.stats["supporting_page_count"] == 0
    assert not any(chunk["view"] == "structured_supporting_page" for chunk in output.chunks)
    assert all(record["page_indices"] == [0, 1] for record in output.records)



def test_english_text_form_fallback_does_not_emit_chinese_query_terms():
    from app.pipeline.structured_rag import _collect_form_like_text_page_outputs

    document_ir = DocumentIR(
        doc_id="doc-auth",
        run_id="run-auth",
        source=SourceInfo(path="authorization-form.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(block_id="b0", type=BlockType.TEXT, page_idx=0, payload={"text": "AUTHORIZATION FORM"}),
            Block(block_id="b1", type=BlockType.TEXT, page_idx=0, payload={"text": "Name"}),
            Block(block_id="b2", type=BlockType.TEXT, page_idx=0, payload={"text": "Date Signed"}),
            Block(block_id="b3", type=BlockType.TEXT, page_idx=0, payload={"text": "Signature"}),
            Block(block_id="b4", type=BlockType.TEXT, page_idx=0, payload={"text": "Phone Number"}),
            Block(block_id="b5", type=BlockType.TEXT, page_idx=0, payload={"text": "I authorize disclosure of the listed records."}),
        ],
    )

    outputs = _collect_form_like_text_page_outputs(document_ir, semantic_output_language="en")

    assert outputs
    retrieval_text = outputs[0]["output"]["retrieval_text"]
    assert "form" in retrieval_text
    assert not any(term in retrieval_text for term in ["表單", "申請", "填寫規則", "簽核流程"])


def test_compact_text_no_ellipsis_uses_english_period_for_english_text():
    from app.pipeline.structured_rag import _compact_text_no_ellipsis

    text = _compact_text_no_ellipsis("This is an English sentence with many words and no Chinese content", 28)

    assert text.endswith(".")
    assert not text.endswith("。")
    assert "wor." not in text

    dangling = _compact_text_no_ellipsis("All records including, and not limited to medical records", 27)
    assert not dangling.endswith("and not.")
    assert not dangling.endswith("including.")


def test_reference_table_detection_is_generic_not_form_specific():
    rate_table = """
    <table>
      <tr><td>職稱/職級別</td><td>交通費</td><td>宿費(平日)</td><td>宿費(假日)</td><td>雜費</td></tr>
      <tr><td>主管級</td><td>按實檢據報支</td><td>4,000</td><td>5,000</td><td>600</td></tr>
      <tr><td>研究員</td><td>按實檢據報支</td><td>3,500</td><td>4,500</td><td>500</td></tr>
      <tr><td>辦事員</td><td>按實檢據報支</td><td>3,500</td><td>4,500</td><td>450</td></tr>
    </table>
    """
    form_table = """
    <table>
      <tr><td>申請人：</td><td>__________</td><td>申請日期：   年   月   日</td></tr>
      <tr><td>出差地點</td><td colspan="2">____________________</td></tr>
      <tr><td>報支單位</td><td>□示範院 □其他</td><td>請勾選</td></tr>
      <tr><td>單位主管簽名</td><td></td><td>申請人簽章</td></tr>
    </table>
    """

    assert looks_like_reference_table(rate_table)
    assert not looks_like_reference_table(form_table)


def test_domestic_travel_expense_table_generates_role_level_records():
    document_ir = DocumentIR(
        doc_id="doc-domestic",
        run_id="run-domestic",
        source=SourceInfo(path="5-3表四國內出差旅費報支數額表113.10.09版.doc", ext="doc", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "表四 示範研究院國內出差旅費報支數額表"},
            ),
            Block(
                block_id="b000002",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td rowspan="2">職稱/職級別</td><td rowspan="2">交通費</td><td colspan="3">每日費用(新台幣元)</td></tr>
                      <tr><td>宿費(平日)</td><td>宿費(假日)</td><td>雜費</td></tr>
                      <tr><td>院長、副院長、主任秘書</td><td>按實檢據報支</td><td>4,500</td><td>5,500</td><td>700</td></tr>
                      <tr><td>正式編製及任務編組單位正(副)主管資深研究員</td><td>按實檢據報支</td><td>4,000</td><td>5,000</td><td>600</td></tr>
                      <tr><td>研究員、副研究員、助理研究員</td><td>按實檢據報支</td><td>3,500</td><td>4,500</td><td>500</td></tr>
                    </table>
                    """,
                },
            ),
            Block(
                block_id="b000003",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "備註："},
            ),
            Block(
                block_id="b000004",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "一、住宿費應檢據覈實報支。"},
            ),
            Block(
                block_id="b000005",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "二、本表自114.1.1正式生效"},
            ),
        ],
    )

    output = build_structured_rag(document_ir)

    assert output.plan.document_type == "travel_domestic_expense_rate_table"
    target = next(record for record in output.records if "正式編製" in record["role_title"])
    assert target["transport_fee_rule"] == "按實檢據報支"
    assert target["lodging_weekday_twd"] == 4000
    assert target["lodging_holiday_twd"] == 5000
    assert target["miscellaneous_twd"] == 600
    assert "宿費平日每日 4000 元" in record_to_rag_text(target)
    note = next(record for record in output.records if record["document_type"] == "table_note")
    assert "住宿費應檢據覈實報支" in note["note_text"]
    assert "114.1.1正式生效" in note["note_text"]
    assert "表格備註" in record_to_rag_text(note)
    assert len([record for record in output.records if record["document_type"] == "travel_domestic_expense_rate_table"]) == 3

    form_output = build_form_documents_rag(
        document_ir,
        {
            "b000002": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {"title": "正式編製及任務編組單位正(副)主管資深研究員"},
            }
        },
    )
    assert form_output.records == []


def test_selects_unknown_allowance_pages_for_vlm_fallback():
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[
            PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png"),
            PageInfo(page_idx=1, page_image_path="assets/pages/p0001.png"),
        ],
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "中央政府各機關派赴國外各地區出差人員生活費日支數額表"},
            ),
            Block(
                block_id="b000002",
                type=BlockType.UNKNOWN,
                page_idx=1,
                payload={"text": "亞太地區 國家 城市 日支數額 美元"},
            ),
        ],
    )

    assert select_vlm_fallback_pages(document_ir, records=[], max_pages=5) == [1]


def test_normalizes_vlm_table_records_for_rag_chunks():
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TEXT,
                page_idx=0,
                payload={
                    "text": (
                        "中央政府各機關派赴國外各地區出差人員生活費日支數額表 "
                        "自115年1月1日生效 單位：美元"
                    )
                },
            ),
        ],
    )
    plan = plan_document(document_ir)
    records = normalize_vlm_table_records(
        output={
            "records": [
                {
                    "region": "大陸地區、香港及澳門",
                    "country_zh": "大陸地區",
                    "country_en": None,
                    "city_zh": "北京",
                    "city_en": "Beijing",
                    "location_label": "北京(Beijing)",
                    "location_type": "city",
                    "rate_usd": 295,
                    "condition": None,
                    "confidence": 0.92,
                    "evidence_text": "北京(Beijing) 295",
                }
            ]
        },
        document_ir=document_ir,
        plan=plan,
        page_idx=28,
        seq_start=10,
    )

    assert records[0]["record_id"] == "rec000010"
    assert records[0]["block_id"] == "vlm-page-0028"
    assert records[0]["extraction_route"] == "vlm_page_image_unknown_table"
    assert records[0]["location_type"] == "city"
    assert records[0]["city_zh"] == "北京"
    assert records[0]["rate_usd"] == 295


def test_builds_form_subdocuments_from_form_enrichments(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="personnel.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[],
    )
    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0010": {
                "kind": "form_asset",
                "input": {"page_idx": 10},
                "output": {
                    "title": "聘任兼任人員新聘申請",
                    "document_type": "form",
                    "triggers": ["新聘申請", "兼任人員"],
                    "filling_guide": (
                        "## 填寫規則\n"
                        "填寫個人基本資料與聘任單位意見。\n"
                        "## 簽核流程\n"
                        "主任、單位主管、院長依序簽署。"
                    ),
                    "field_schema": [
                        {"name": "姓名", "type": "text", "required": True},
                        {"name": "姓名", "type": "text", "required": False},
                        {"name": "保險", "type": "checkbox", "required": True},
                    ],
                    "retrieval_text": "用於兼任人員僱傭、見習、實習的新聘申請。",
                },
            }
        },
    )

    assert output.plan.document_type == "form_collection"
    assert output.stats["form_count"] == 1
    assert any(record["content_type"] == "form_summary" for record in output.records)
    field = next(record for record in output.records if record.get("field_name") == "姓名")
    assert field["subdoc_id"].startswith("form:0000:")
    assert field["logical_doc_id"].startswith("doc-a::form:0000:")
    assert field["parent_doc_id"] == "doc-a"
    assert field["section"] == "申請/基本資料"
    assert field["required"] is True
    assert sum(1 for record in output.records if record.get("field_name") == "姓名") == 1
    assert "### 表單欄位" in output.rag_markdown
    field_chunk = next(
        chunk for chunk in output.chunks if chunk["metadata"]["content_type"] == "form_field"
    )
    assert field_chunk["doc_id"].startswith("doc-a::form:0000:")
    assert field_chunk["metadata"]["parent_doc_id"] == "doc-a"

    paths = write_structured_rag_outputs(output, tmp_path)
    assert "forms_index" in paths
    assert (tmp_path / "forms" / "form_0000.md").exists()
    assert (tmp_path / "forms" / "form_0000.fields.jsonl").exists()


def test_english_table_form_title_prefers_document_title_over_numbered_field():
    document_ir = DocumentIR(
        doc_id="doc-4506t",
        run_id="run-4506t",
        source=SourceInfo(path="f4506t.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="mineru", method="auto"),
        blocks=[
            Block(
                block_id="tbl",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td colspan="4">4506-T Request for Transcript of Tax Return Form Do not sign this form unless all applicable lines have been completed.</td></tr>
                      <tr><td>1a Name shown on tax return. If a joint return, enter the name shown first.</td><td></td><td>1b First social security number on tax return</td></tr>
                      <tr><td>Signature date.</td><td></td><td></td></tr>
                    </table>
                    """,
                },
            )
        ],
    )

    output = build_form_documents_rag(document_ir, {}, semantic_output_language="en")

    assert output.plan.document_type == "form_document"
    form_names = {record.get("form_name") for record in output.records if record.get("document_type") == "form_document"}
    assert "4506-T Request for Transcript of Tax Return" in form_names
    assert "1a Name shown on tax return" not in form_names


def test_multpage_source_form_is_single_document_without_form_subfiles(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-visa",
        run_id="run-visa",
        source=SourceInfo(path="20180122_尋職簽證申請檢核表-final.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="mineru", method="auto"),
        pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)],
        blocks=[
            Block(
                block_id="p0-title",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "尋職簽證申請檢核表 ROC EMPLOYMENT-SEEKING VISA APPLICATION CHECKLIST"},
            ),
            Block(
                block_id="p0-fields",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "姓名 Name 護照號碼 Passport No. 出生日期 Date of birth 申請人簽名 Signature"},
            ),
            Block(
                block_id="p1-docs",
                type=BlockType.TEXT,
                page_idx=1,
                payload={"text": "□外國護照正、影本 Passport (with a photocopy) □新臺幣 10 萬元以上或等值之財力證明 Proof of financial resources of at least NT$100,000 尋職計畫 Plan for seeking employment 備考 Remarks 主管 承辦人員 申請人簽名"},
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_p0": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "尋職簽證申請檢核表",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "姓名", "type": "name", "required": True},
                        {"name": "護照號碼", "type": "text", "required": True},
                        {"name": "申請人簽名", "type": "signature", "required": True},
                    ],
                    "filling_guide": "## 表單用途\n用於申請中華民國尋職簽證。\n## 填寫重點\n填寫申請人基本資料與專業領域。",
                    "all_text": ["姓名 Name", "護照號碼 Passport No.", "申請人簽名 Signature"],
                },
            },
            "form_p1": {
                "kind": "form_asset",
                "input": {"page_idx": 1},
                "output": {
                    "title": "ROC EMPLOYMENT-SEEKING VISA APPLICATION CHECKLIST",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "外國護照正、影本", "type": "checkbox", "required": True},
                        {"name": "新臺幣 10 萬元以上或等值之財力證明", "type": "checkbox", "required": True},
                        {"name": "尋職計畫", "type": "text", "required": True},
                    ],
                    "filling_guide": "## 附件/佐證資料\n應備文件包含外國護照正影本、新臺幣 10 萬元以上或等值之財力證明、尋職計畫。\n## 注意事項\n備考欄保留個案要求補件。",
                    "all_text": [
                        "□外國護照正、影本 Passport (with a photocopy)",
                        "□新臺幣 10 萬元以上或等值之財力證明 Proof of financial resources of at least NT$100,000",
                        "尋職計畫 Plan for seeking employment",
                    ],
                },
            },
        },
    )

    assert output.plan.document_type == "form_document"
    assert output.stats["form_count"] == 1
    subdoc_ids = {record["subdoc_id"] for record in output.records}
    assert len(subdoc_ids) == 1
    assert next(iter(subdoc_ids)).startswith("form:0000:")
    assert "尋職簽證申請檢核表" in next(iter(subdoc_ids))
    assert all(record["page_indices"] == [0, 1] for record in output.records)
    assert "NT$100,000" in output.rag_markdown

    paths = write_structured_rag_outputs(output, tmp_path)
    assert "forms_index" not in paths
    assert not (tmp_path / "forms_index.json").exists()
    assert not (tmp_path / "forms").exists()


def test_text_only_english_form_generates_form_subdocument():
    document_ir = DocumentIR(
        doc_id="doc-en-form",
        run_id="run-en-form",
        source=SourceInfo(path="internationaltravelform.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="mineru", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "INTERNATIONAL TRAVEL REIMBURSEMENT CLAIM FORM Form and receipts must be submitted within 45 days"},
            ),
            Block(
                block_id="b000002",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Date: Preparer: Dept.: Payee: Phone: Email:"},
            ),
            Block(
                block_id="b000003",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Business Purpose: Air Fare Amount: Rental Car Amount: Total Transportation Expenses:"},
            ),
            Block(
                block_id="b000004",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Payee Signature: Date Signed:"},
            ),
        ],
    )

    output = build_form_documents_rag(document_ir, {}, semantic_output_language="en")

    assert output.plan.document_type == "form_document"
    assert output.stats["form_count"] == 1
    assert "Form Purpose" in output.rag_markdown
    assert "INTERNATIONAL TRAVEL REIMBURSEMENT CLAIM FORM" in output.rag_markdown
    fields = {record.get("field_name") for record in output.records if record.get("content_type") == "form_field"}
    assert {"Date", "Preparer", "Payee", "Phone", "Email", "Business Purpose", "Air Fare Amount", "Payee Signature"}.issubset(fields)
    assert any(record.get("section") == "費用/報支資訊" for record in output.records if record.get("field_name") == "Air Fare Amount")


def test_uuid_form_title_is_replaced_by_local_document_title():
    document_ir = DocumentIR(
        doc_id="doc-tw-form",
        run_id="run-tw-form",
        source=SourceInfo(path="30376e94-6bc1-45e3-bcd8-537a11d1b18a.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="mineru", method="auto"),
        blocks=[
            Block(block_id="title", type=BlockType.TEXT, page_idx=0, payload={"text": "委託書"}),
            Block(block_id="field", type=BlockType.TEXT, page_idx=0, payload={"text": "委託人姓名： 受委託人姓名： 委託人簽章："}),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "title": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "e94-6bc1-45e3-bcd8-537a11d1b18a",
                    "document_type": "form",
                    "field_schema": [{"name": "委託人姓名", "type": "name", "required": True}],
                    "filling_guide": "## 表單用途\n用於委託他人代辦戶政事項。",
                    "retrieval_text": "委託書 委託人姓名 受委託人姓名",
                },
            }
        },
    )

    form_names = {record.get("form_name") for record in output.records if record.get("document_type") == "form_document"}
    assert "委託書" in form_names
    assert "表單：e94-6bc1-45e3-bcd8-537a11d1b18a" not in output.rag_markdown


def test_spreadsheet_form_output_augments_fields_from_ir_tables():
    document_ir = DocumentIR(
        doc_id="doc-xls",
        run_id="run-xls",
        source=SourceInfo(path="5-3表二國外出差旅費報支單114.09.26版.xls", ext="xls", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="native_spreadsheet", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png"), PageInfo(page_idx=1, page_image_path="assets/pages/p0001.png")],
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "表二 國外出差旅費報支單"},
            ),
            Block(
                block_id="b000002",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td colspan="2">姓名</td><td colspan="10"></td></tr>
                      <tr><td rowspan="2" colspan="2">年月日</td><td rowspan="2">起訖地點</td><td rowspan="2">工作紀要</td><td colspan="2">交通費</td><td rowspan="2">生活費</td><td colspan="2">辦公費</td><td rowspan="2">幣別</td><td rowspan="2">匯率</td><td rowspan="2">折合台幣</td></tr>
                      <tr><td>飛機</td><td>其他</td><td>雜費</td><td>其他</td></tr>
                      <tr><td colspan="11">小計</td><td>0</td></tr>
                      <tr><td colspan="12">上列出差旅費合計：零</td></tr>
                    </table>
                    """,
                },
            ),
            Block(
                block_id="b000003",
                type=BlockType.TEXT,
                page_idx=1,
                payload={"text": "單位主管:"},
            ),
            Block(
                block_id="b000004",
                type=BlockType.TEXT,
                page_idx=1,
                payload={"text": "出差人簽名："},
            ),
            Block(
                block_id="b000005",
                type=BlockType.TABLE,
                page_idx=1,
                payload={"table_body": "<table><tr><td>單據編號</td><td>備註</td></tr></table>"},
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "b000002": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "表二國外出差旅費報支單",
                    "document_type": "form",
                    "field_schema": [{"name": "姓名", "type": "name", "required": True}],
                    "filling_guide": "## 填寫規則\n填寫姓名。",
                    "retrieval_text": "國外出差旅費報支單 OCR_NOISE_TOKEN 交通費 出差人簽名 匯率 單位主管 填寫規則 姓名 小計 工作紀要 幣別",
                },
            }
        },
    )

    fields = {record.get("field_name") for record in output.records if record.get("content_type") == "form_field"}
    assert {"交通費", "生活費", "辦公費", "幣別", "匯率", "折合台幣", "單據編號", "備註", "單位主管", "出差人簽名"}.issubset(fields)
    first = output.records[0]
    assert first["page_indices"] == [0, 1]
    assert "主要欄位分組" in first["content"]
    assert "OCR_NOISE_TOKEN" not in first["content"]
    assert "### 表單欄位" in output.rag_markdown
    assert "匯率" in output.rag_markdown
    assert "單據編號" in output.rag_markdown


def test_main_rag_render_can_exclude_split_form_pages():
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="personnel.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="main-title",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "示範研究院人員留職停薪辦法", "text_level": 1},
            ),
            Block(
                block_id="main-body",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "所有需以書面提出之留職停薪申請，應填具留職停薪申請表。"},
            ),
            Block(
                block_id="form-title",
                type=BlockType.TEXT,
                page_idx=1,
                payload={"text": "表一 示範研究院人員留職停薪申請表", "text_level": 2},
            ),
            Block(
                block_id="form-table",
                type=BlockType.TABLE,
                page_idx=1,
                payload={
                    "table_body": "<table><tr><td>姓名/員工編號</td><td>職級</td></tr></table>",
                },
            ),
        ],
    )

    markdown, _ = PackageStage()._render_rag_md(
        document_ir=document_ir,
        asset_map={},
        enrichments={},
        suppress_form_enrichment=True,
        excluded_page_indices={1},
    )

    assert "留職停薪申請，應填具留職停薪申請表" in markdown
    assert "表一 示範研究院人員留職停薪申請表" not in markdown
    assert "姓名/員工編號" not in markdown


def test_collect_structured_form_page_indices_accepts_page_indices_list():
    records = [
        {"document_type": "form_document", "page_indices": [10, 11]},
        {"document_type": "form_document", "page_idx": 16},
        {"document_type": "main_document", "page_indices": [3]},
    ]

    assert PackageStage()._collect_structured_form_page_indices(records) == {10, 11, 16}


def test_render_rag_md_accepts_list_structured_content():
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="flow.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="fig-a",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "fig-a.png"},
            ),
        ],
    )

    markdown, _ = PackageStage()._render_rag_md(
        document_ir=document_ir,
        asset_map={},
        enrichments={
            "fig-a": {
                "kind": "figure_description",
                "output": {
                    "structured_content": [
                        "受理單位 > 正式受理申訴案",
                        "正式受理申訴案 > 通知地方主管機關",
                    ],
                },
            },
        },
    )

    assert "受理單位 > 正式受理申訴案" in markdown
    assert "正式受理申訴案 > 通知地方主管機關" in markdown


def test_render_rag_md_weaves_figure_all_text_and_drops_asset_link():
    """A figure whose information lives in ``all_text`` (e.g. a UI screenshot with an
    empty ``structured_content``) must be woven into the RAG body as searchable text,
    and the RAG markdown must not contain a broken internal ``asset://`` image link.
    """
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="guide.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="fig-a",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "fig-a.png", "caption": "下午12:03 9月7日 週三"},
            ),
        ],
    )
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0000",
        doc_id="doc-a",
        run_id="run-a",
        title="投影機控制畫面",
        page_idx=0,
        asset_path="assets/figures/fig0000.jpg",
        block_id="fig-a",
        retrieval_text="投影機控制畫面",
        semantic_caption="A screenshot of an iPad control panel.",
        structured_content="",
        keywords=["投影機", "HDMI"],
    )

    markdown, _ = PackageStage()._render_rag_md(
        document_ir=document_ir,
        asset_map={"fig-a": asset},
        enrichments={
            "fig-a": {
                "kind": "figure_description",
                "output": {
                    "semantic_caption": "A screenshot of an iPad control panel.",
                    "image_type": "screenshot",
                    "structured_content": "",
                    "all_text": [
                        "情境模式",
                        "音量控制",
                        "投影機",
                        "系統關閉",
                        "ON",
                        "OFF",
                        "HDMI",
                        "黑幕",
                        "升降",
                    ],
                    "facts": ["The interface is a control panel for room equipment."],
                    "keywords": ["投影機", "HDMI"],
                },
            },
        },
    )

    # The screenshot's visible Chinese UI labels become searchable body text.
    assert "投影機" in markdown
    assert "HDMI" in markdown
    assert "黑幕" in markdown
    assert "升降" in markdown
    # No broken internal image link should remain in the RAG markdown.
    assert "asset://" not in markdown


def test_render_rag_md_keeps_prose_steps_on_visual_structured_page():
    """Regression: authored prose (numbered steps / 提醒 paragraphs) on a page that
    also carries a figure with non-empty ``structured_content`` must NOT be dropped.

    A figure's ``structured_content`` marks its page "visually structured", which
    previously suppressed *every* non-image block on that page except the first
    reading-order heading — silently deleting the document's actual instructions.
    Observed live on the mailbox-archive guide: steps 5-7 and both 提醒 reminders
    vanished from every packaged output while remaining present in the IR.
    """
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="guide.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="title",
                type=BlockType.TEXT,
                page_idx=0,
                reading_order=0,
                payload={"text": "信箱封存操作說明", "text_level": 1},
            ),
            Block(
                block_id="fig",
                type=BlockType.IMAGE,
                page_idx=0,
                reading_order=1,
                payload={"img_path": "fig.png", "caption": "封存對話框"},
            ),
            Block(
                block_id="step6",
                type=BlockType.TEXT,
                page_idx=0,
                reading_order=2,
                payload={"text": "6. 封存時，outlook 右下角顯示「正在封存」"},
            ),
            Block(
                block_id="remind",
                type=BlockType.TEXT,
                page_idx=0,
                reading_order=3,
                payload={"text": "提醒：封存後郵件移至本機硬碟，電腦重灌時該檔案需一併備份。"},
            ),
        ],
    )

    markdown, _ = PackageStage()._render_rag_md(
        document_ir=document_ir,
        asset_map={},
        enrichments={
            "fig": {
                "kind": "figure_description",
                "output": {
                    # Non-empty structured_content marks page 0 "visual-structured".
                    "structured_content": "封存 > 封存此資料夾及所有的子資料夾",
                    "all_text": ["封存", "確定", "取消"],
                    "keywords": ["封存"],
                },
            },
        },
    )

    # The authored prose steps/reminders must survive alongside the figure content.
    assert "正在封存" in markdown, markdown
    assert "重灌" in markdown, markdown
    assert "一併備份" in markdown, markdown


def test_visual_semantic_falls_back_to_caption_when_no_chinese_text():
    """A photo whose only usable description lives in the VLM caption/facts (no
    Chinese OCR; English facts stripped by the zh-TW language filter) must still
    produce a searchable description — not collapse to the bare OCR fragment.

    Regression: the 204 電子白板 reboot photo rendered as just "JECTOR" while the
    VLM had captured the rear rocker power switch in semantic_caption + facts.
    """
    rendered = PackageStage()._render_visual_semantic_content(
        "Figure 7",
        {
            "semantic_caption": "The rear panel has a rocker power switch below the arrow sticker.",
            "image_type": "photo",
            "structured_content": "",
            "all_text": ["JECTOR"],
            "facts": ["The power switch is a rocker switch located below the rear arrow sticker."],
            "keywords": ["power switch", "restart"],
        },
        semantic_output_language="zh-TW",
    )
    assert "rocker" in rendered.lower(), rendered
    assert "power switch" in rendered.lower(), rendered
    # Must not collapse to the useless bare OCR token.
    assert rendered.strip() != "JECTOR", rendered


def test_visual_semantic_keeps_chinese_ocr_without_leaking_english_caption():
    """Guard: when a figure carries real Chinese OCR, the (English) caption must
    NOT be injected into the zh-TW corpus — the visible Chinese text is used as-is.
    """
    rendered = PackageStage()._render_visual_semantic_content(
        "Figure 1",
        {
            "semantic_caption": "A close-up photo of a control panel with buttons.",
            "image_type": "photo",
            "structured_content": "",
            "all_text": ["電源", "訊號切換／筆電:HDMI2"],
            "facts": ["The left label reads Power."],
            "keywords": ["電源"],
        },
        semantic_output_language="zh-TW",
    )
    assert "電源" in rendered, rendered
    assert "訊號切換" in rendered, rendered
    assert "control panel" not in rendered.lower(), rendered


def test_visual_semantic_screenshot_paths_render_plain_not_flow_template():
    """#4: PATH-notation lines in a SCREENSHOT are UI menus/folder trees the VLM
    transcribed — not a workflow. Flow-templating them fabricates 語意摘要/詳細流程路徑
    sections (observed: an Outlook folder tree rendered as a fake "flow" that repeated
    the account address dozens of times). Screenshots must render as plain lines.
    """
    rendered = PackageStage()._render_visual_semantic_content(
        "訊號源設定",
        {
            "semantic_caption": "設定介面截圖。",
            "image_type": "screenshot",
            "structured_content": "訊號源設定 > HDMI1(PC)\n訊號源設定 > HDMI2(SideInput)",
            "all_text": ["訊號源設定", "HDMI1(PC)", "HDMI2(SideInput)"],
            "facts": ["紅框特別標示了兩個常用的輸入來源選項，需要重點關注。"],
            "keywords": ["訊號源設定", "HDMI1"],
        },
        semantic_output_language="zh-TW",
    )
    assert "語意摘要" not in rendered, rendered
    assert "詳細流程路徑" not in rendered, rendered
    assert "常見查詢主題" not in rendered, rendered
    # Content must still be searchable as plain lines.
    assert "訊號源設定 > HDMI1(PC)" in rendered, rendered
    assert "重點關注" in rendered, rendered


def test_visual_semantic_filters_corpus_watermark_terms(monkeypatch):
    """#5: on-screen background watermarks get OCR'd by the VLM into
    all_text/facts/keywords, polluting retrieval text with identity strings.
    Terms come from the corpus ruleset (``document_markers.watermark_terms``);
    real terms live only in the local, non-committed ruleset.
    """
    from app.pipeline.corpus_rules import CorpusRules

    monkeypatch.setattr(
        "app.pipeline.stages.package.get_corpus_rules",
        lambda: CorpusRules(document_markers={"watermark_terms": ("示範40", "深耕示範")}),
    )
    stage = PackageStage()
    rendered = stage._render_visual_semantic_content(
        "訊號源設定",
        {
            "semantic_caption": "設定介面截圖。",
            "image_type": "screenshot",
            "structured_content": "",
            "all_text": ["訊號源設定", "示範40", "深耕示範·邁向國際"],
            "facts": ["背景有「示範40」的浮水印文字。", "紅框標示了常用的輸入來源選項。"],
            "keywords": ["訊號源設定", "示範40"],
        },
        semantic_output_language="zh-TW",
    )
    assert "示範40" not in rendered, rendered
    assert "深耕示範" not in rendered, rendered
    assert "訊號源設定" in rendered, rendered
    assert "輸入來源" in rendered, rendered

    # Flow path too: watermark keywords must not surface in 常見查詢主題.
    flow_rendered = stage._render_visual_semantic_content(
        "作業流程圖",
        {
            "semantic_caption": "",
            "image_type": "flowchart",
            "structured_content": "受理單位 > 正式受理申訴案",
            "all_text": ["示範40"],
            "facts": [],
            "keywords": ["申訴", "示範40"],
        },
        semantic_output_language="zh-TW",
    )
    assert "示範40" not in flow_rendered, flow_rendered
    assert "申訴" in flow_rendered, flow_rendered


def test_visual_semantic_flowchart_still_gets_flow_template():
    """Guard for #4: genuine flowcharts keep the semantic flow template."""
    rendered = PackageStage()._render_visual_semantic_content(
        "申訴流程圖",
        {
            "semantic_caption": "",
            "image_type": "flowchart",
            "structured_content": "受理單位 > 正式受理申訴案",
            "all_text": [],
            "facts": [],
            "keywords": ["申訴"],
        },
        semantic_output_language="zh-TW",
    )
    assert "語意摘要" in rendered, rendered
    assert "受理單位 > 正式受理申訴案" in rendered, rendered


def test_write_document_exports_ignores_missing_forms_index(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="source.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[],
    )

    paths = PackageStage()._write_document_exports(
        outputs_dir=tmp_path,
        source_md="# 主文件\n\n內容",
        assets=[],
        structured_paths={},
        document_ir=document_ir,
    )

    assert (tmp_path / "documents" / "main.md").is_file()
    assert (tmp_path / "documents_index.json").is_file()
    assert paths["main_document"].endswith("main.md")



def test_write_document_exports_skips_duplicate_single_visual_child(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="flow.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[],
    )
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0000",
        doc_id="doc-a",
        run_id="run-a",
        title="性騷擾申訴流程",
        page_idx=0,
        asset_path="assets/figures/fig0000.png",
        block_id="fig-a",
        retrieval_text="性騷擾申訴流程",
        structured_content="受理單位 > 正式受理申訴案\n正式受理申訴案 > 通知地方主管機關",
    )

    PackageStage()._write_document_exports(
        outputs_dir=tmp_path,
        source_md="# 性騷擾申訴流程\n\n受理單位 > 正式受理申訴案",
        assets=[asset],
        structured_paths={},
        document_ir=document_ir,
    )

    assert (tmp_path / "documents" / "main.md").is_file()
    assert not (tmp_path / "documents" / "fig0000.md").exists()
    index = (tmp_path / "documents_index.json").read_text(encoding="utf-8")
    assert '"document_id": "main"' in index
    assert '"document_id": "fig0000"' not in index


def test_render_split_asset_document_prefers_structured_content():
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0000",
        doc_id="doc-a",
        run_id="run-a",
        title="Figure 1",
        page_idx=0,
        asset_path="assets/figures/fig0000.png",
        block_id="fig-a",
        retrieval_text="English fallback caption",
        semantic_caption="English fallback caption",
        structured_content="性騷擾申訴流程\n受理單位 > 正式受理申訴案",
    )

    markdown = PackageStage()._render_split_asset_document(
        asset=asset,
        source_title="性騷擾防治作業流程圖",
        source_filename="flow.pdf",
    )

    assert markdown.startswith("# 性騷擾申訴流程")
    assert "受理單位 > 正式受理申訴案" in markdown
    assert "來源文件：性騷擾防治作業流程圖" not in markdown



def test_render_split_asset_document_salvages_jsonish_visual_caption():
    jsonish_caption = '''{
  "semantic_caption": "This is a flowchart for 個資安維事件 response.",
  "image_type": "flowchart",
  "structured_content": [
    "本院人員獲悉疑似個資安維事件 > 通報個資中心",
    "通報個資中心 > 個資中心執秘完成通報紀錄"
'''
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0000",
        doc_id="doc-a",
        run_id="run-a",
        title="Figure 1",
        page_idx=0,
        asset_path="assets/figures/fig0000.png",
        block_id="fig-a",
        retrieval_text="Figure 1",
        semantic_caption=jsonish_caption,
        structured_content="",
    )

    markdown = PackageStage()._render_split_asset_document(
        asset=asset,
        source_title="個資安維事件應變流程圖",
        source_filename="flow.pdf",
    )

    assert markdown.startswith("# 個資安維事件應變流程圖")
    assert "流程路徑" in markdown
    assert "本院人員獲悉疑似個資安維事件 > 通報個資中心" in markdown
    assert '"structured_content"' not in markdown


def test_render_rag_md_skips_noisy_ocr_on_structured_visual_page():
    document_ir = DocumentIR(
        doc_id="doc-a",
        run_id="run-a",
        source=SourceInfo(path="flow.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="title",
                type=BlockType.TEXT,
                page_idx=0,
                reading_order=0,
                payload={"text": "示範研究院性騷擾防治申訴及懲戒作業流程圖"},
            ),
            Block(
                block_id="noise-before",
                type=BlockType.TEXT,
                page_idx=0,
                reading_order=1,
                payload={"text": "通知他方行"},
            ),
            Block(
                block_id="fig-a",
                type=BlockType.IMAGE,
                page_idx=0,
                reading_order=2,
                payload={"img_path": "fig-a.png"},
            ),
            Block(
                block_id="noise-after",
                type=BlockType.TEXT,
                page_idx=0,
                reading_order=3,
                payload={"text": "處理單位討論並作成附理由之決議"},
            ),
        ],
    )

    markdown, _ = PackageStage()._render_rag_md(
        document_ir=document_ir,
        asset_map={},
        enrichments={
            "fig-a": {
                "kind": "figure_description",
                "output": {
                    "structured_content": "受理單位 > 正式受理申訴案\n正式受理申訴案 > 通知地方主管機關",
                },
            },
        },
    )

    assert "示範研究院性騷擾防治申訴及懲戒作業流程圖" in markdown
    assert "受理單位 > 正式受理申訴案" in markdown
    assert "通知他方行" not in markdown
    assert "處理單位討論並作成附理由之決議" not in markdown

def test_quality_gate_uses_structured_form_text_for_category_checks(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-form",
        run_id="run-form",
        source=SourceInfo(path="travel_form.xls", ext="xls", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="native_spreadsheet", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path=None)],
        blocks=[
            Block(
                block_id="b000000",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "工作表：國內(外)出差單"},
            ),
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>表三 國內（外）出差單</td></tr>
                      <tr><td>申請人：</td><td>單位主管</td><td>主任秘書</td><td>副院長</td><td>院長</td><td>董事長</td></tr>
                      <tr><td>註：1.本單應於出差前填寫。</td></tr>
                    </table>
                    """,
                    "table_caption": "國內(外)出差單 表格",
                },
            ),
        ],
    )
    structured_output = build_form_documents_rag(
        document_ir,
        {
            "b000001": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "表三國內（外）出差單",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "申請人", "type": "name", "required": True},
                        {"name": "單位主管", "type": "signature", "required": True},
                        {"name": "主任秘書", "type": "signature", "required": False},
                        {"name": "副院長", "type": "signature", "required": False},
                        {"name": "院長", "type": "signature", "required": False},
                        {"name": "董事長", "type": "signature", "required": False},
                    ],
                    "filling_guide": "## 簽核流程\n單位主管 → 主任秘書 → 副院長 → 院長 → 董事長",
                    "retrieval_text": "",
                },
            }
        },
    )
    weak_asset = AssetEntry(
        type="form_asset",
        asset_id="form0000",
        doc_id="doc-form",
        run_id="run-form",
        title="國內(外)出差單 表格",
        page_idx=0,
        asset_path="",
        block_id="b000001",
        retrieval_text="國內(外)出差單 表格",
    )

    result = asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md="",
            assets=[weak_asset],
            structured_output=structured_output,
            enrichments={},
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
        )
    )

    codes = {issue.code for issue in result.issues}
    assert "form_signature_fields_missing" not in codes
    assert "semantic_output_too_short" not in codes
    assert codes == {"source_preview_missing"}
    assert result.status == "pass"


def test_export_assets_does_not_duplicate_form_table_block(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-form",
        run_id="run-form",
        source=SourceInfo(path="travel_form.xls", ext="xls", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="native_spreadsheet", method="auto"),
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_caption": "國內(外)出差單 表格",
                    "table_body": "<table><tr><td>申請人：</td><td>單位主管</td></tr></table>",
                },
            )
        ],
    )

    assets, asset_map = asyncio.run(
        PackageStage()._export_assets(
            document_ir=document_ir,
            assets_dir=tmp_path / "assets",
            parse_cache_path=None,
            enrichments={
                "b000001": {
                    "kind": "form_asset",
                    "input": {"page_idx": 0},
                    "output": {"title": "", "retrieval_text": ""},
                    "quality": {"needs_review": False},
                    "evidence": {"page_idx": 0, "asset_path": None},
                }
            },
        )
    )

    assert len(assets) == 1
    assert assets[0].type == "form_asset"
    assert assets[0].asset_id == "form0000"
    assert asset_map["b000001"].asset_id == "form0000"



def test_pdf_form_with_complete_vlm_schema_does_not_merge_noisy_ir_fields():
    document_ir = DocumentIR(
        doc_id="doc-harassment",
        run_id="run-harassment",
        source=SourceInfo(path="表一-示範研究院性騷擾事件申訴書.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>姓名 性別 □男□女 出生年月日 年 月 日</td></tr>
                      <tr><td>下午時分本院受理人員</td></tr>
                      <tr><td>辦公座位調動提供相關醫療或心理諮商協助</td></tr>
                    </table>
                    """,
                },
            )
        ],
    )
    output = {
        "title": "表一 示範研究院性騷擾事件申訴書",
        "document_type": "form",
        "field_schema": [
            {"name": "申訴日期", "type": "date", "required": False},
            {"name": "姓名", "type": "name", "required": False},
            {"name": "性別", "type": "checkbox", "required": False},
            {"name": "出生年月日", "type": "date", "required": False},
            {"name": "身分證統一編號（或護照號碼）", "type": "id", "required": False},
            {"name": "連絡電話", "type": "text", "required": False},
            {"name": "發生時間", "type": "date", "required": False},
            {"name": "發生地點", "type": "text", "required": False},
            {"name": "發生過程", "type": "text", "required": False},
            {"name": "請求事項", "type": "checkbox", "required": False},
            {"name": "申訴人簽名", "type": "signature", "required": False},
            {"name": "本院受理人員", "type": "text", "required": False},
        ],
        "filling_guide": "## 表單用途\n用於提出性騷擾事件申訴。\n## 填寫重點\n填寫申訴人資料、事件內容與請求事項。",
        "retrieval_text": "性騷擾事件申訴書 申訴人資料 事件內容 請求事項",
        "triggers": ["性騷擾", "申訴書"],
    }

    structured_output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": output,
                "quality": {"needs_review": False},
            }
        },
    )

    text = structured_output.rag_markdown
    assert "申訴人簽名" in text
    assert "本院受理人員" in text
    assert text.count("### 表單用途") == 1
    assert "原始抽取補充" not in text
    assert "女 出生年月日" not in text
    assert "下午時分本院受理人員" not in text
    assert "辦公座位調動提供相關醫療或心理諮商協助" not in text


def test_spreadsheet_form_still_augments_vlm_schema_from_ir_rows():
    document_ir = DocumentIR(
        doc_id="doc-xls",
        run_id="run-xls",
        source=SourceInfo(path="5-3表三國內(外)出差單112.08.09版.xls", ext="xls", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="native_spreadsheet", method="auto"),
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>表三 國內（外）出差單</td></tr>
                      <tr><td>申請人</td><td>出差事由</td><td>單位主管</td><td>主任秘書</td></tr>
                      <tr><td>註：1.本單應於出差前填寫。</td></tr>
                    </table>
                    """,
                },
            )
        ],
    )

    structured_output = build_form_documents_rag(
        document_ir,
        {
            "b000001": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "表三 國內（外）出差單",
                    "document_type": "form",
                    "field_schema": [{"name": "申請人", "type": "name", "required": True}],
                    "filling_guide": "## 表單用途\n用於申請出差。",
                    "retrieval_text": "",
                },
                "quality": {"needs_review": False},
            }
        },
    )

    text = structured_output.rag_markdown
    assert "出差事由" in text
    assert "單位主管" in text
    assert "主任秘書" in text
    assert "本單應於出差前填寫" in text

def test_form_like_spreadsheet_table_builds_synthetic_form_document():
    document_ir = DocumentIR(
        doc_id="doc-taxi",
        run_id="run-taxi",
        source=SourceInfo(path="5-3表五大台北地區計程車資請領單97.09.01版.xls", ext="xls", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="native_spreadsheet", method="auto"),
        pages=[PageInfo(page_idx=0), PageInfo(page_idx=1), PageInfo(page_idx=2)],
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_caption": "大台北地區計程車資請領單",
                    "table_body": """
                    <table>
                      <tr><td>申請人</td><td>日</td><td>事由</td><td>起始地點</td><td>到達地點</td><td>金額</td></tr>
                      <tr><td></td><td></td><td></td><td></td><td></td><td></td></tr>
                      <tr><td>合計金額</td><td></td><td>單位主管核定</td><td></td><td></td><td></td></tr>
                    </table>
                    """,
                },
            ),
            Block(
                block_id="b000002",
                type=BlockType.TABLE,
                page_idx=2,
                payload={
                    "table_caption": "領款人簽章",
                    "table_body": "<table><tr><td>領款人簽章</td><td></td></tr></table>",
                },
            ),
        ],
    )

    output = build_form_documents_rag(document_ir, {})

    assert is_form_like_document(document_ir)
    assert output.plan.document_type == "form_collection"
    subdoc_ids = {record["subdoc_id"] for record in output.records if record["document_type"] == "form_document"}
    assert len(subdoc_ids) == 1
    assert {0, 1, 2} == set(next(record["page_indices"] for record in output.records if record["document_type"] == "form_document"))
    assert any(record.get("field_name") == "領款人簽章" for record in output.records)
    assert "請領單" in output.rag_markdown
    assert "# 表五大台北地區計程車資請領單" in output.rag_markdown
    assert "# 單位主管核定" not in output.rag_markdown


def test_quality_gate_flags_form_like_document_without_form_structure(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-taxi",
        run_id="run-taxi",
        source=SourceInfo(path="大台北地區計程車資請領單.xls", ext="xls", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="native_spreadsheet", method="auto"),
        pages=[PageInfo(page_idx=0)],
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": "<table><tr><td>申請人</td><td>事由</td><td>金額</td><td>單位主管核定</td><td>領款人簽章</td></tr></table>",
                },
            )
        ],
    )
    structured_output = build_structured_rag(document_ir)

    result = asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md="申請人 事由 金額 單位主管核定 領款人簽章",
            assets=[],
            structured_output=structured_output,
            enrichments={},
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
        )
    )

    assert "form_like_document_not_structured" in {issue.code for issue in result.issues}
    assert result.status == "needs_review"



def test_quality_gate_flags_empty_structured_output_for_form_document_when_source_exists(tmp_path):
    from types import SimpleNamespace

    document_ir = DocumentIR(
        doc_id="doc-flow",
        run_id="run-flow",
        source=SourceInfo(path="flowchart.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
        blocks=[
            Block(
                block_id="title",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "不同性騷擾申訴對象標準作業流程圖"},
            ),
            Block(
                block_id="fig",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "images/flowchart.jpg"},
            ),
        ],
    )
    structured_output = SimpleNamespace(
        plan=SimpleNamespace(document_type="form_document"),
        records=[],
        chunks=[],
        rag_markdown="",
    )

    result = asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md="被害人提出申訴。依事件場域與當事人身分判斷適用法律，並分流至性別平等工作法、性別平等教育法或性騷擾防治法。",
            assets=[],
            structured_output=structured_output,
            enrichments={
                # A form-type document with an enriched figure and empty structured
                # output is a real defect (its rows should have been extracted), so
                # it must be flagged. Generic/prose docs are exempt and covered
                # separately in test_quality_gate_triggers.py.
                "fig": {
                    "kind": "figure_caption",
                    "input": {"page_idx": 0},
                    "output": {"semantic_caption": "性騷擾申訴標準作業流程圖"},
                }
            },
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
        )
    )

    codes = {issue.code for issue in result.issues}
    assert "structured_output_empty" in codes
    assert result.status == "needs_review"
    assert "generate_structured_semantic_output" in result.stats["semantic_quality"]["recommended_repairs"]


def test_quality_gate_flags_vlm_json_parse_failure(tmp_path):
    from types import SimpleNamespace

    document_ir = DocumentIR(
        doc_id="doc-vlm",
        run_id="run-vlm",
        source=SourceInfo(path="flowchart.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
        blocks=[
            Block(
                block_id="fig",
                type=BlockType.IMAGE,
                page_idx=0,
                payload={"img_path": "images/flowchart.jpg"},
            )
        ],
    )
    structured_output = SimpleNamespace(
        plan=SimpleNamespace(document_type="generic_document"),
        records=[],
        chunks=[{"content": "## Flowchart\n- Victim files a complaint."}],
        rag_markdown="## Flowchart\n- Victim files a complaint.",
    )

    result = asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md="Victim files a complaint and the process branches by applicable law.",
            assets=[],
            structured_output=structured_output,
            enrichments={
                "fig": {
                    "kind": "figure_description",
                    "input": {"page_idx": 0},
                    "output": {"_error": "JSON_PARSE_FAILED: Unterminated string"},
                    "quality": {"tokens_used": 4096},
                }
            },
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
        )
    )

    codes = {issue.code for issue in result.issues}
    assert "vlm_enrichment_parse_failed" in codes
    assert result.status == "needs_review"
    assert "repair_unparsed_enrichment_output" in result.stats["semantic_quality"]["recommended_repairs"]


def test_quality_gate_flags_english_structured_output_with_chinese_template_terms(tmp_path):
    from types import SimpleNamespace

    document_ir = DocumentIR(
        doc_id="doc-en",
        run_id="run-en",
        source=SourceInfo(path="ssa-827.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
        blocks=[
            Block(
                block_id="title",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "Authorization to Disclose Information to SSA"},
            )
        ],
    )
    structured_output = SimpleNamespace(
        plan=SimpleNamespace(document_type="form_collection"),
        records=[],
        chunks=[{"content": "Common query keywords: 填寫規則, 申請, 簽核流程, 表單."}],
        rag_markdown="## Form Purpose\nUse this form to authorize disclosure.\n\nCommon query keywords: 填寫規則, 申請, 簽核流程, 表單.",
    )

    result = asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md="Authorization to Disclose Information to the Social Security Administration.",
            assets=[],
            structured_output=structured_output,
            enrichments={},
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
            semantic_output_language="en",
        )
    )

    codes = {issue.code for issue in result.issues}
    assert "target_language_mismatch" in codes
    assert result.stats["semantic_output_language"] == "en"
    assert "rewrite_in_target_language" in result.stats["semantic_quality"]["recommended_repairs"]


def test_quality_gate_reports_rag_readiness_metrics(tmp_path):
    from types import SimpleNamespace

    document_ir = DocumentIR(
        doc_id="doc-quality",
        run_id="run-quality",
        source=SourceInfo(path="sample.xls", ext="xls", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="native_spreadsheet", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path="p0000.png")],
        blocks=[],
    )
    structured_output = SimpleNamespace(
        plan=SimpleNamespace(document_type="form_collection"),
        records=[
            {
                "document_type": "form_document",
                "content_type": "form_field",
                "field_name": "□沖預借金額：□應補發金額: 受款人：應繳回金額",
                "section": "表單欄位",
            },
            *[
                {
                    "document_type": "form_document",
                    "content_type": "form_field",
                    "field_name": f"一般欄位{i}",
                    "section": "表單欄位",
                }
                for i in range(5)
            ],
        ],
        chunks=[],
        rag_markdown="## 注意事項\n- 97.09.01版\n摘要...",
    )

    result = asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md="## 注意事項\n- 97.09.01版\n摘要...",
            assets=[],
            structured_output=structured_output,
            enrichments={},
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
        )
    )

    codes = {issue.code for issue in result.issues}
    assert "merged_field_detected" in codes
    assert "too_many_generic_fields" in codes
    assert "version_misclassified_as_note" in codes
    assert "summary_contains_ellipsis" in codes
    semantic_quality = result.stats["semantic_quality"]
    assert semantic_quality["rag_readiness_score"] < 1.0
    assert "split_merged_fields" in semantic_quality["recommended_repairs"]




def test_quality_gate_serializes_english_issue_messages(tmp_path):
    from types import SimpleNamespace

    document_ir = DocumentIR(
        doc_id="doc-quality-en",
        run_id="run-quality-en",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path="p0000.png")],
        blocks=[],
    )
    structured_output = SimpleNamespace(
        plan=SimpleNamespace(document_type="form_collection"),
        records=[
            {
                "document_type": "form_document",
                "content_type": "form_field",
                "field_name": "Travel: Foreign (Countries) / United States (States)",
                "section": "表單欄位",
            },
            *[
                {
                    "document_type": "form_document",
                    "content_type": "form_field",
                    "field_name": f"Generic Field {idx}",
                    "section": "表單欄位",
                }
                for idx in range(5)
            ],
        ],
        chunks=[],
        rag_markdown="Summary...",
    )

    result = asyncio.run(
        run_quality_gate(
            document_ir=document_ir,
            source_md="Summary...",
            assets=[],
            structured_output=structured_output,
            enrichments={},
            run_path=tmp_path,
            vlm_adapter=None,
            max_vlm_audits=0,
            semantic_output_language="en",
        )
    )

    payload = result.to_dict()

    assert payload["stats"]["semantic_output_language"] == "en"
    assert payload["issues"]
    assert all(
        not any("\u4e00" <= char <= "\u9fff" for char in issue["message"])
        for issue in payload["issues"]
    )
    assert any(issue["code"] == "too_many_generic_fields" for issue in payload["issues"])


def test_package_title_gate_rejects_page_numbers_and_long_sentences():
    stage = PackageStage()
    assert stage._is_unreliable_export_title("1")
    assert stage._is_unreliable_export_title("事件編號：xxx(年)-xx（單位）-xx(序號)")
    assert stage._is_unreliable_export_title("申請日期：__年__月__日")
    assert stage._is_unreliable_export_title(
        "審查內容加以評核，並得同意申請人提出升等或駁回。申請駁回者，得於知悉結果翌日起七個工作日內申覆。"
    )
    assert stage._is_unreliable_export_title("正，並自同日起施行）")
    assert stage._is_unreliable_export_title("壹總則")
    assert stage._is_unreliable_export_title("(一)營運目標")
    assert stage._is_unreliable_export_title("組織營運應有效率與效果。")
    assert stage._is_unreliable_export_title("正，並自同日起施行） 第 2 頁 表格 1")
    assert stage._is_unreliable_export_title("2021 12 d1 1587")
    assert not stage._is_unreliable_export_title("示範研究院個資事件處理報告單")


def test_package_source_title_falls_back_when_source_md_starts_with_page_number():
    stage = PackageStage()
    source_md = "# 1\n\nTABLE: 示範研究院人員薪酬管理辦法\nROW: 第一條 | 內容"
    assert stage._infer_source_title(source_md, "2-2.pdf") == "示範研究院人員薪酬管理辦法"

def test_package_source_title_uses_body_policy_title_when_headings_are_noise():
    stage = PackageStage()
    source_md = (
        "## 正，並自同日起施行）\n\n"
        "## 壹總則\n\n"
        "為落實財團法人示範研究院（以下簡稱本院）内部控制，考量本院整體營運活動、"
        "組織業務、控制環境及管理需求，並為加強財務管理，以保障資產安全及經營成效，"
        "防杜不法情事，有效達成組織目標，特訂定本制度。"
    )
    assert stage._infer_source_title(source_md, "10-3.pdf") == "財團法人示範研究院內部控制制度"




def test_package_skips_empty_generic_figure_asset_documents():
    stage = PackageStage()
    empty = AssetEntry(
        type="figure_asset",
        asset_id="fig0000",
        doc_id="doc",
        run_id="run",
        title="Figure 1",
        page_idx=0,
        asset_path="",
        block_id="fig-block-0",
        retrieval_text="Figure 1",
    )
    useful = AssetEntry(
        type="figure_asset",
        asset_id="fig0001",
        doc_id="doc",
        run_id="run",
        title="個資安維事件應變流程圖",
        page_idx=0,
        asset_path="assets/figures/fig0001.png",
        block_id="fig-block-1",
        retrieval_text="本院人員獲悉疑似個資安維事件 > 通報個資中心 > 結案",
        structured_content="本院人員獲悉疑似個資安維事件 > 通報個資中心 > 結案",
    )
    assert not stage._should_export_asset_document(empty, "", [empty])
    assert stage._should_export_asset_document(useful, "", [useful])


def test_quality_gate_ignores_empty_generic_figure_short_output():
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0000",
        doc_id="doc",
        run_id="run",
        title="Figure 1",
        page_idx=0,
        asset_path="",
        block_id="fig-block-0",
        retrieval_text="Figure 1",
    )

    issues = _check_assets_semantics([asset], set(), "")

    assert "semantic_output_too_short" not in {issue.code for issue in issues}


def test_quality_gate_uses_figure_semantic_caption_for_short_output_check():
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0015",
        doc_id="doc",
        run_id="run",
        title="Figure 15",
        page_idx=0,
        asset_path="",
        block_id="fig-block-15",
        retrieval_text="Figure 15",
        semantic_caption=(
            "This image displays a certificate used as reimbursement evidence, "
            "including issue date, entry and exit dates, and document purpose. "
            "The caption is sufficient semantic context for retrieval."
        ),
    )

    issues = _check_assets_semantics([asset], set(), "")

    assert "semantic_output_too_short" not in {issue.code for issue in issues}


def test_quality_gate_still_flags_short_flowchart_figure_output():
    asset = AssetEntry(
        type="figure_asset",
        asset_id="fig0000",
        doc_id="doc",
        run_id="run",
        title="Figure 1",
        page_idx=0,
        asset_path="",
        block_id="fig-block-0",
        retrieval_text="Flowchart",
    )

    issues = _check_assets_semantics([asset], set(), "")

    assert "semantic_output_too_short" in {issue.code for issue in issues}


def test_quality_gate_does_not_vlm_audit_table_note_misses():
    document_ir = DocumentIR(
        doc_id="doc-table",
        run_id="run-table",
        source=SourceInfo(path="table.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
    )
    issues = [
        type("Issue", (), {"page_idx": 0, "code": "table_notes_missing"})(),
        type("Issue", (), {"page_idx": 0, "code": "possible_wrong_asset_kind"})(),
    ]

    candidates = _build_vlm_audit_candidates(document_ir, issues, max_candidates=2)

    assert candidates == [
        {
            "page_idx": 0,
            "page_image_path": "assets/pages/p0000.png",
            "reasons": ["possible_wrong_asset_kind"],
        }
    ]



def test_quality_gate_vlm_audits_doc_level_empty_structured_output():
    document_ir = DocumentIR(
        doc_id="doc-flow",
        run_id="run-flow",
        source=SourceInfo(path="flowchart.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[
            PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png"),
            PageInfo(page_idx=1, page_image_path="assets/pages/p0001.png"),
        ],
    )
    issues = [type("Issue", (), {"page_idx": None, "code": "structured_output_empty"})()]

    candidates = _build_vlm_audit_candidates(document_ir, issues, max_candidates=1)

    assert candidates == [
        {
            "page_idx": 0,
            "page_image_path": "assets/pages/p0000.png",
            "reasons": ["structured_output_empty"],
        }
    ]


def test_form_title_prefers_source_name_over_field_like_header():
    document_ir = DocumentIR(
        doc_id="doc-property",
        run_id="run-property",
        source=SourceInfo(path="6-1表二財產增加單108.10.15版.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="b000001",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>請購單位</td><td>使用／保管單位</td></tr>
                      <tr><td>申請日期</td><td>主管</td></tr>
                      <tr><td>金額合計</td><td>副主管</td></tr>
                    </table>
                    """,
                },
            )
        ],
    )

    output = build_form_documents_rag(document_ir, {})

    assert output.records
    assert output.records[0]["form_name"] == "表二財產增加單"


def test_form_title_prefers_attachment_title_over_source_level_vlm_title(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-attachment-form",
        run_id="run-attachment-form",
        source=SourceInfo(path="專案核銷申請單懶人包.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="form_page_0000",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>附件1.領據</td></tr>
                      <tr><td>申請日期</td><td>領款人</td><td>金額</td><td>單位主管</td></tr>
                      <tr><td>年月日</td><td></td><td>新臺幣</td><td></td></tr>
                    </table>
                    """,
                },
            )
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "專案核銷申請單懶人包",
                    "document_type": "form",
                    "field_schema": [],
                    "filling_guide": "",
                    "retrieval_text": "",
                },
            }
        },
    )

    assert output.records
    assert output.records[0]["form_name"] == "附件1.領據"

    write_structured_rag_outputs(output, tmp_path)
    forms_index = json.loads((tmp_path / "forms_index.json").read_text(encoding="utf-8"))
    assert forms_index[0]["title"] == "附件1.領據"


def test_form_title_does_not_use_policy_sentence_with_checkbox_as_title():
    document_ir = DocumentIR(
        doc_id="doc-training",
        run_id="run-training",
        source=SourceInfo(path="示範研究院人員訓練辦法.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="b000101",
                type=BlockType.TABLE,
                page_idx=6,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>範，惟仍應事先向所屬單位主管報備。□依本院「人員訓練辦法」第六條第一項第一款辦理。</td></tr>
                      <tr><td>申請單位</td><td>申請日期</td><td>姓名（員工編號）</td></tr>
                      <tr><td>職級</td><td>職稱</td><td>單位主管</td></tr>
                      <tr><td>□依本院「人員訓練辦法」第六條第一項第二款辦理</td><td>□依本院「人員訓練辦法」第八條辦理</td></tr>
                    </table>
                    """,
                },
            )
        ],
    )

    output = build_form_documents_rag(document_ir, {})

    assert output.records
    assert "範，惟仍應事先" not in output.records[0]["form_name"]
    assert output.records[0]["form_name"] == "示範研究院人員訓練辦法"
    field_names = [record.get("field_name", "") for record in output.records if record.get("content_type") == "form_field"]
    assert all("範，惟仍應事先" not in field_name for field_name in field_names)
    assert "欄位：範，惟仍應事先" not in output.rag_markdown
    assert "範，惟仍應事先向所屬單位主管報備。(" not in output.rag_markdown


def test_quality_gate_does_not_vlm_audit_short_semantic_outputs():
    document_ir = DocumentIR(
        doc_id="doc-short",
        run_id="run-short",
        source=SourceInfo(path="short.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
    )
    issues = [type("Issue", (), {"page_idx": 0, "code": "semantic_output_too_short"})()]

    assert _build_vlm_audit_candidates(document_ir, issues, max_candidates=2) == []


def test_form_notes_ignore_bare_receipt_amount_numbers():
    document_ir = DocumentIR(
        doc_id="doc-expense",
        run_id="run-expense",
        source=SourceInfo(path="附件二經費報銷暨付款申請單.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="form-table",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>憑證編號</td><td>預算科目</td><td>報支金額</td><td>用途說明</td></tr>
                      <tr><td>沖預借款</td><td colspan="3">原預借： 元；沖銷： 元</td></tr>
                      <tr><td>付款方式：□現金□支票，抬頭： □匯款，戶名 銀行 帳號</td></tr>
                    </table>
                    """,
                },
            ),
            Block(
                block_id="amount-table",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>單據金額</td></tr>
                      <tr><td>1.</td><td></td></tr>
                      <tr><td>2.</td><td></td></tr>
                      <tr><td>3.</td><td></td></tr>
                    </table>
                    """,
                },
            ),
            Block(
                block_id="note1",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "1.報支一般行政費用時，本表單之預算科目及憑證編號兩欄由會計填寫。"},
            ),
            Block(
                block_id="note2",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "2.單據報銷之核定，依本院權責劃分辦法辦理。"},
            ),
        ],
    )

    output = build_form_documents_rag(document_ir, {})
    note_record = next(record for record in output.records if record.get("section") == "注意事項")

    assert "報支一般行政費用" in note_record["content"]
    assert "單據報銷之核定" in note_record["content"]
    assert "- 1. - 2." not in note_record["content"]


def test_vlm_all_text_notes_override_same_number_parser_ocr_notes():
    document_ir = DocumentIR(
        doc_id="doc-expense-vlm",
        run_id="run-expense-vlm",
        source=SourceInfo(path="附件二經費報銷暨付款申請單.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="form-table",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>經費報銷暨付款申請單</td></tr>
                      <tr><td>申請單位</td><td>申請日期</td><td>經辦人</td><td>單位主管</td></tr>
                    </table>
                    """,
                },
            ),
            Block(
                block_id="note6",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": "6.採購案金額30萬元以上者，請檢附訂講單（或合約），並應於支付尾款時檢附驗收單。"},
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "經費報銷暨付款申請單",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "申請單位", "type": "text", "evidence_text": "申請單位："},
                        {"name": "申請日期", "type": "date", "evidence_text": "申請日期："},
                        {"name": "報支計畫名稱", "type": "text", "evidence_text": "報支計畫名稱："},
                        {"name": "憑證編號", "type": "text", "evidence_text": "憑證編號"},
                        {"name": "預算科目", "type": "text", "evidence_text": "預算科目"},
                        {"name": "報支金額", "type": "number", "evidence_text": "報支金額"},
                        {"name": "用途說明", "type": "text", "evidence_text": "用途說明"},
                        {"name": "付款方式", "type": "checkbox", "evidence_text": "付款方式"},
                        {"name": "經辦人", "type": "signature", "evidence_text": "經辦人："},
                        {"name": "單位主管", "type": "signature", "evidence_text": "單位主管："},
                        {"name": "會計", "type": "signature", "evidence_text": "會計："},
                        {"name": "核定", "type": "signature", "evidence_text": "核定："},
                        {"name": "2. 單據報銷之核定，依本院權責劃分辦法辦理", "type": "signature"},
                        {"name": "經費報銷暨付款申請單", "type": "text"},
                    ],
                    "filling_guide": "## 表單用途\n用於經費報銷與付款申請。\n## 填寫重點\n請依憑證、付款資訊與簽核流程填寫。",
                    "all_text": [
                        "6.採購案金額30萬元以上者，請檢附訂購單（或合約），並應於支付尾款時檢附驗收單。"
                    ],
                },
            }
        },
    )

    markdown = output.rag_markdown

    assert "訂購單" in markdown
    assert "訂講單" not in markdown
    assert "欄位：2. 單據報銷之核定" not in markdown
    assert "欄位：經費報銷暨付款申請單" not in markdown

def test_render_form_documents_markdown_does_not_emit_ellipsis():
    from app.pipeline.structured_rag import DocumentPlan, render_form_documents_markdown

    plan = DocumentPlan(document_type="form_collection", title="表單集合")
    long_text = "這是一段很長的表單說明" * 120
    records = [
        {
            "subdoc_id": "form:0000:test",
            "form_name": "測試申請單",
            "page_label": "第 1 頁",
            "content_type": "form_summary",
            "content": long_text,
        },
        {
            "subdoc_id": "form:0000:test",
            "form_name": "測試申請單",
            "page_label": "第 1 頁",
            "content_type": "form_section",
            "section": "填寫重點",
            "content": long_text,
        },
    ]

    markdown = render_form_documents_markdown(plan, records)

    assert "..." not in markdown
    assert markdown.rstrip().endswith("。")


def test_quality_gate_table_notes_require_explicit_note_heading():
    from app.pipeline.quality_gate import _collect_notes_after_block

    document_ir = DocumentIR(
        doc_id="doc-reg",
        run_id="run-reg",
        source=SourceInfo(path="rule.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(block_id="tbl", type=BlockType.TABLE, page_idx=0, payload={"table_body": "<table></table>"}),
            Block(block_id="txt1", type=BlockType.TEXT, page_idx=0, payload={"text": "一、本院人員應遵守本院一切規章與主管指示忠誠服務。"}),
            Block(block_id="txt2", type=BlockType.TEXT, page_idx=0, payload={"text": "第二十條 本院人員奉派出差依相關辦法辦理。"}),
        ],
    )

    assert _collect_notes_after_block(document_ir, document_ir.blocks[0]) == []



def test_form_notes_split_and_dedupe_combined_vlm_and_parser_notes():
    document_ir = DocumentIR(
        doc_id="doc-training-notes",
        run_id="run-training-notes",
        source=SourceInfo(path="示範研究院人員訓練辦法.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="form-table",
                type=BlockType.TABLE,
                page_idx=6,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>示範研究院人員進修申請表</td></tr>
                      <tr><td>申請單位</td><td>申請日期</td><td>姓名(員工編號)</td><td>單位主管</td></tr>
                      <tr><td>註：1.申請進修或選修學分者需檢附相關證明文件。 2.本院人員不得同時進修選修學分短期研究及兼課。 3.本院人員因研究或業務需要，得依本辦法申請訓練。利用非上班時間參加訓練者，不受本辦法規</td></tr>
                    </table>
                    """,
                },
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0006": {
                "kind": "form_asset",
                "input": {"page_idx": 6},
                "output": {
                    "title": "示範研究院人員進修申請表",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "申請單位", "type": "text", "required": True},
                        {"name": "申請日期", "type": "date", "required": True},
                        {"name": "姓名(員工編號)", "type": "name", "required": True},
                        {"name": "職級", "type": "text", "required": True},
                        {"name": "職稱", "type": "text", "required": True},
                        {"name": "學校名稱", "type": "text", "required": False},
                        {"name": "系所名稱", "type": "text", "required": False},
                        {"name": "單位主管", "type": "signature", "required": True},
                    ],
                    "filling_guide": (
                        "## 表單用途\n用於人員進修申請。\n"
                        "## 注意事項\n"
                        "- 1.申請進修或選修學分者需檢附相關證明文件。\n"
                        "- 2.本院人員不得同時進修、選修學分、短期研究及兼課。\n"
                        "- 3.本院人員因研究或業務需要，得依本辦法申請訓練。利用非上班時間參加訓練者，不受本辦法規範，惟仍應事先向所屬單位主管報備。"
                    ),
                    "all_text": [
                        "1.申請進修或選修學分者需檢附相關證明文件。",
                        "2.本院人員不得同時進修、選修學分、短期研究及兼課。",
                        "3.本院人員因研究或業務需要，得依本辦法申請訓練。利用非上班時間參加訓練者，不受本辦法規範，惟仍應事先向所屬單位主管報備。",
                    ],
                },
            }
        },
    )

    markdown = output.rag_markdown

    assert "來源完整注意事項" not in markdown
    assert "不受本辦法規範，惟仍應事先向所屬單位主管報備" in markdown
    assert "不受本辦法規。" not in markdown


def test_form_title_rejects_numbered_clause_fragment_from_vlm_title():
    document_ir = DocumentIR(
        doc_id="doc-training-clause-title",
        run_id="run-training-clause-title",
        source=SourceInfo(path="示範研究院人員訓練辦法.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="form-table",
                type=BlockType.TABLE,
                page_idx=7,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>二、立保證規約人因申請(奉派)赴</td></tr>
                      <tr><td>身份證字號</td><td>進修(短期研究)，申請留職留薪期間自年</td></tr>
                      <tr><td>(簽名蓋章)</td><td>未服務期滿期間之天數比例</td></tr>
                    </table>
                    """,
                },
            )
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0007": {
                "kind": "form_asset",
                "input": {"page_idx": 7},
                "output": {
                    "title": "二、立保證規約人因申請(奉派)赴",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "身份證字號", "type": "id", "required": False},
                        {"name": "進修期間", "type": "date", "required": False},
                        {"name": "簽名蓋章", "type": "signature", "required": False},
                    ],
                    "filling_guide": "## 表單用途\n用於保證規約。",
                    "retrieval_text": "保證規約 身份證字號 簽名蓋章",
                },
            }
        },
    )

    assert output.records
    assert output.records[0]["form_name"] == "示範研究院人員訓練辦法"
    assert "二、立保證規約人因申請" not in output.records[0]["form_name"]



def test_form_notes_keep_parser_full_text_when_vlm_note_is_truncated():
    document_ir = DocumentIR(
        doc_id="doc-purchase-notes",
        run_id="run-purchase-notes",
        source=SourceInfo(path="附件一請購暨預付款申請單.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="ocr"),
        blocks=[
            Block(
                block_id="form-table",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>附件一請購暨預付款申請單</td></tr>
                      <tr><td>申請單位</td><td>申請日期</td><td>單位主管</td></tr>
                      <tr><td>註：1.報支計畫請購預估費用超過新台幣10萬元、或報支一般行政請購金額超過新台幣1萬元者，應事先填具請購單，並會簽專責採購單位(電腦中心:電腦相關設備；資服中心:書報雜誌及資料庫)。 2.預算審查權責：報支計畫者，由單位助理審查預算，單位主管核准；報支一般行政者，由行政處財務審查預算、權責主管核准。</td></tr>
                    </table>
                    """,
                },
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "附件一請購暨預付款申請單",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "申請單位", "type": "text", "required": True},
                        {"name": "申請日期", "type": "date", "required": True},
                        {"name": "單位主管", "type": "signature", "required": True},
                    ],
                    "filling_guide": "## 表單用途\n用於請購與預付款申請。",
                    "all_text": [
                        "1.報支計畫請購預估費用超過新台幣10萬元、或報支一般行政請購金額超過新台幣1萬元者，應事先填具請購單，並會簽專責",
                        "2.預算審查權責：報支計畫者，由單位助理審查預算，單位主管核准；報支一般行政者，由行政處財務審查預算、權責",
                    ],
                },
            }
        },
    )

    markdown = output.rag_markdown

    assert "會簽專責採購單位(電腦中心:電腦相關設備；資服中心:書報雜誌及資料庫)" in markdown
    assert "權責主管核准" in markdown


def test_form_documents_rag_outputs_english_semantic_template():
    document_ir = DocumentIR(
        doc_id="doc-en",
        run_id="run-en",
        source=SourceInfo(path="travel_reimbursement_form.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="form_page_0000",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>Travel Reimbursement Form</td></tr>
                      <tr><td>Applicant Name:</td><td>__________</td></tr>
                      <tr><td>Travel Date:</td><td>__________</td></tr>
                      <tr><td>Manager Signature:</td><td>__________</td></tr>
                    </table>
                    """,
                },
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "Travel Reimbursement Form",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "Applicant Name", "type": "name", "required": True},
                        {"name": "Travel Date", "type": "date", "required": True},
                        {"name": "Manager Signature", "type": "signature", "required": True},
                    ],
                    "filling_guide": "",
                    "retrieval_text": "Travel reimbursement form applicant travel date manager signature",
                },
            }
        },
        semantic_output_language="en",
    )

    assert output.stats["semantic_output_language"] == "en"
    assert "## Form Purpose" in output.rag_markdown
    assert "## RAG Query Summary" in output.rag_markdown
    assert "### Form Fields and Field Descriptions" in output.rag_markdown
    assert "## 表單用途" not in output.rag_markdown
    assert "用途與填寫重點" not in output.rag_markdown

def test_english_single_form_document_suppresses_source_text_dump_and_groups_fields():
    document_ir = DocumentIR(
        doc_id="doc-ssa-like",
        run_id="run-ssa-like",
        source=SourceInfo(path="authorization-form.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)],
        blocks=[
            Block(
                block_id="form_page_0000",
                type=BlockType.TABLE,
                page_idx=0,
                payload={"table_body": "Name SSN Birthday Authorization to disclose records Signature Date Signed"},
            ),
            Block(
                block_id="form_page_0001",
                type=BlockType.TEXT,
                page_idx=1,
                payload={"text": "Explanation of form, privacy act statement, paperwork reduction act statement."},
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "Authorization to Disclose Information",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "NAME (First, Middle, Last, Suffix)", "type": "text", "required": True},
                        {"name": "SSN", "type": "text", "required": True},
                        {"name": "Birthday (MM/DD/YYYY)", "type": "date", "required": True},
                        {"name": "PURPOSE", "type": "text", "required": False},
                        {"name": "Date Signed", "type": "date", "required": True},
                        {"name": "Witness Phone Number (or Address)", "type": "text", "required": False},
                    ],
                    "filling_guide": "## Form Purpose\nAuthorizes disclosure of records for a benefits request.\n\n## Filling Guidance\nComplete identity, authorization scope, and signature fields.",
                    "all_text": [
                        "Form SSA-827 (06-2024) UF",
                        "Source text line that should not become a rendered Source Extracted Text section.",
                    ],
                },
            },
            "form_page_0001": {
                "kind": "form_asset",
                "input": {"page_idx": 1},
                "output": {
                    "title": "Explanation of Authorization Form",
                    "document_type": "form",
                    "field_schema": [],
                    "filling_guide": "## Form Purpose\nExplains why authorization is needed.\n\n## Notes\nIncludes privacy act and paperwork reduction act notices.",
                    "all_text": ["Explanation text should remain evidence, not a formal dump section."],
                },
            },
        },
        semantic_output_language="en",
    )

    assert output.plan.document_type == "form_document"
    assert "Source Extracted Text" not in output.rag_markdown
    assert "formal dump section" not in output.rag_markdown
    assert output.rag_markdown.count("### Form Purpose") == 1
    field_sections = {
        record.get("section")
        for record in output.records
        if record.get("content_type") == "form_field"
    }
    assert "表單欄位" not in field_sections
    assert "授權/揭露範圍" in field_sections
    assert "申請/基本資料" in field_sections


def test_form_documents_rag_filters_low_value_template_form_subdocs(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-low-value-form",
        run_id="run-low-value-form",
        source=SourceInfo(path="specimen-guide.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0), PageInfo(page_idx=1)],
        blocks=[],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "specimen-guide",
                    "document_type": "form",
                    "field_schema": [],
                    "filling_guide": "",
                    "retrieval_text": "",
                    "needs_review": True,
                },
            },
            "form_page_0001": {
                "kind": "form_asset",
                "input": {"page_idx": 1},
                "output": {
                    "title": "Patient Intake Form",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "Patient Name", "type": "name", "required": True},
                        {"name": "Date of Birth", "type": "date", "required": True},
                    ],
                    "filling_guide": "",
                    "retrieval_text": "patient intake form patient name date of birth",
                },
            },
        },
        semantic_output_language="en",
    )

    assert output.stats["form_count"] == 1
    assert output.plan.evidence["form_page_count"] == 1
    assert "Patient Intake Form" in output.rag_markdown
    assert "specimen-guide. Section: Use Cases" not in output.rag_markdown

    paths = write_structured_rag_outputs(output, tmp_path)
    forms_index = json.loads((tmp_path / "forms_index.json").read_text(encoding="utf-8"))
    structured_records = (tmp_path / "structured_records.jsonl").read_text(encoding="utf-8")

    assert paths["forms_index"]
    assert len(forms_index) == 1
    assert forms_index[0]["title"] == "Patient Intake Form"
    assert "specimen-guide. Section: Use Cases" not in structured_records


def test_form_documents_rag_english_serialized_outputs_use_english_labels(tmp_path):
    document_ir = DocumentIR(
        doc_id="doc-en-jsonl",
        run_id="run-en-jsonl",
        source=SourceInfo(path="travel_reimbursement_form.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="form_page_0000",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>Travel Reimbursement Form</td></tr>
                      <tr><td>Applicant Name:</td><td>__________</td></tr>
                      <tr><td>Application Date:</td><td>__________</td></tr>
                      <tr><td>Manager Signature:</td><td>__________</td></tr>
                    </table>
                    """,
                },
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "Travel Reimbursement Form",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "Applicant Name", "type": "name", "required": True, "section": "申請/基本資料"},
                        {"name": "Application Date", "type": "date", "required": True, "section": "申請/基本資料"},
                        {"name": "Manager Signature", "type": "signature", "required": True, "section": "簽核/用印"},
                    ],
                    "filling_guide": "",
                    "retrieval_text": "Travel reimbursement form applicant date manager signature",
                },
            }
        },
        semantic_output_language="en",
    )

    field_chunk = next(chunk for chunk in output.chunks if chunk["metadata"]["content_type"] == "form_field")
    assert field_chunk["metadata"]["section"] == "Applicant / Basic Information"

    paths = write_structured_rag_outputs(output, tmp_path)
    fields_text = (tmp_path / "structured_records.jsonl").read_text(encoding="utf-8")
    chunks_text = (tmp_path / "structured_chunks.jsonl").read_text(encoding="utf-8")
    markdown_text = (tmp_path / "structured_rag.md").read_text(encoding="utf-8")

    assert output.plan.document_type == "form_document"
    assert "forms_index" not in paths
    assert "Applicant / Basic Information" in fields_text
    assert "Approval / Signature" in fields_text
    assert "申請/基本資料" not in fields_text
    assert "簽核/用印" not in fields_text
    assert "申請/基本資料" not in chunks_text
    assert "Page: Page 1" in markdown_text
    assert "Page：Page" not in markdown_text
    assert "apply, reimburse" not in markdown_text
    first_field_record = next(
        json.loads(line) for line in fields_text.splitlines()
        if json.loads(line).get("content_type") == "form_field"
    )
    assert first_field_record["section"] == "Applicant / Basic Information"


def test_form_documents_rag_auto_keeps_chinese_semantic_template():
    document_ir = DocumentIR(
        doc_id="doc-zh",
        run_id="run-zh",
        source=SourceInfo(path="請款申請單.pdf", ext="pdf", sha256="abc", size_bytes=123),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id="form_page_0000",
                type=BlockType.TABLE,
                page_idx=0,
                payload={
                    "table_body": """
                    <table>
                      <tr><td>請款申請單</td></tr>
                      <tr><td>申請人：</td><td>__________</td></tr>
                      <tr><td>申請日期：</td><td>年月日</td></tr>
                      <tr><td>單位主管：</td><td>__________</td></tr>
                    </table>
                    """,
                },
            ),
        ],
    )

    output = build_form_documents_rag(
        document_ir,
        {
            "form_page_0000": {
                "kind": "form_asset",
                "input": {"page_idx": 0},
                "output": {
                    "title": "請款申請單",
                    "document_type": "form",
                    "field_schema": [
                        {"name": "申請人", "type": "name", "required": True},
                        {"name": "申請日期", "type": "date", "required": True},
                        {"name": "單位主管", "type": "signature", "required": True},
                    ],
                    "filling_guide": "",
                    "retrieval_text": "請款申請單 申請人 申請日期 單位主管",
                },
            }
        },
        semantic_output_language="auto",
    )

    assert output.stats["semantic_output_language"] == "zh-TW"
    assert "## 表單用途" in output.rag_markdown
    assert "## RAG 查詢摘要" in output.rag_markdown
    assert "## Form Purpose" not in output.rag_markdown
