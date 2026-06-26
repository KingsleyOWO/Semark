from app.pipeline.semantic import (
    SemanticDocument,
    SemanticField,
    SemanticSource,
    clean_title_noise,
    evaluate_semantic_quality,
    extract_version,
    normalize_fields,
    normalize_notes,
    render_semantic_markdown,
)


def test_semantic_normalizer_cleans_title_and_extracts_version():
    assert clean_title_noise("表四昇示範研究院國內出差旅費報支數額表") == "表四示範研究院國內出差旅費報支數額表"
    version = extract_version("5-3表五大台北地區計程車資請領單97.09.01版.xls")
    assert version.raw == "97.09.01版"
    assert version.date == "97.09.01"


def test_semantic_normalizer_splits_merged_fields_and_classifies_sections():
    fields = normalize_fields([
        {"name": "□沖預借金額：□應補發金額: 受款人：應繳回金額"},
        {"name": "飛機"},
        {"name": "領款人簽章"},
    ])
    names = [field.normalized_name for field in fields]
    assert "沖預借金額" in names
    assert "應補發金額" in names
    assert "受款人" in names
    assert "應繳回金額" in names
    transport = next(field for field in fields if field.normalized_name == "飛機")
    assert transport.section == "交通工具"
    assert transport.type == "choice"
    signature = next(field for field in fields if field.normalized_name == "領款人簽章")
    assert signature.section == "簽核/用印"
    assert signature.type == "signature"


def test_semantic_notes_move_version_out_of_notes():
    notes, version = normalize_notes(["97.09.01版", "一、住宿費應檢據覈實報支。", "97.09.01版"])
    assert version.raw == "97.09.01版"
    assert notes == ["一、住宿費應檢據覈實報支。"]


def test_semantic_renderer_and_quality_report():
    document = SemanticDocument(
        document_type="form_document",
        title="表五大台北地區計程車資請領單",
        source=SemanticSource(file_name="5-3表五大台北地區計程車資請領單97.09.01版.xls", pages=[0, 1]),
        purpose="用於申請大台北地區計程車資報支。",
        usage_scenarios=["員工因業務需要搭乘計程車並申請費用報支時使用。"],
        fields=[
            SemanticField(name="申請人", normalized_name="申請人", type="name", required=True, requirement="required", section="申請/基本資料"),
            SemanticField(name="金額", normalized_name="金額", type="number", section="費用/報支資訊"),
            SemanticField(name="領款人簽章", normalized_name="領款人簽章", type="signature", section="簽核/用印"),
        ],
        approval_flow=["單位主管核定", "領款人簽章"],
    )
    markdown = render_semantic_markdown(document)
    assert "## 文件定位" in markdown
    assert "## 主要填寫內容" in markdown
    assert "申請/基本資料：申請人" in markdown
    report = evaluate_semantic_quality(document, markdown)
    assert report.rag_readiness_score == 1.0
    assert report.issues == []


def test_semantic_quality_flags_bad_rag_signals():
    fields = normalize_fields([{"name": "□沖預借金額：□應補發金額: 受款人：應繳回金額"}])
    document = SemanticDocument(
        document_type="form_document",
        title="表四昇台灣",
        source=SemanticSource(file_name="sample.xls"),
        fields=[SemanticField(name="很長的欄位名稱可能是多個欄位黏在一起超過三十六個字", normalized_name="很長的欄位名稱可能是多個欄位黏在一起超過三十六個字"), *fields],
        notes=["97.09.01版"],
    )
    report = evaluate_semantic_quality(document, "摘要...")
    codes = {issue.code for issue in report.issues}
    assert "ocr_title_noise" in codes
    assert "version_misclassified_as_note" in codes
    assert "field_name_too_long" in codes
    assert "summary_contains_ellipsis" in codes
    assert "split_merged_fields" in report.recommended_repairs


def test_semantic_normalizer_filters_instructional_noise_fields():
    fields = normalize_fields([
        {"name": "本申請書正本存查於人事單位（與當事人出勤紀錄併同存查5 年）。"},
        {"name": "（112/05/17核定版）"},
        {"name": "申請日期"},
    ])
    names = [field.normalized_name for field in fields]
    assert "申請日期" in names
    assert not any("存查" in name for name in names)
    assert not any("核定版" in name for name in names)


def test_semantic_notes_move核定_version_out_of_notes():
    notes, version = normalize_notes(["108.10.15 核定版", "註1：請檢附證明文件。"])
    assert version.raw == "108.10.15 核定版"
    assert notes == ["註1：請檢附證明文件。"]
