"""Regression tests from the 2026-07 screenshot-guide demo review.

Two public how-to guides (a zh-TW treasury payment guide and an English VA
portal walkthrough) exposed four defects in the delivered exports; each test
below pins one of them.
"""

import json

from app.pipeline.quality_gate import _check_semantic_template
from app.pipeline.stages.package import PackageStage


def test_infer_source_title_recognizes_operation_guide_title():
    # The real title is a plain first line; the only markdown heading in the
    # document is the generic section label 「【說明】」.
    source_md = (
        "列印國庫繳款書操作說明\n"
        "\n"
        "歡迎使用 VEB 系統\n"
        "\n"
        "## 【說明】\n"
        "\n"
        "請使用瀏覽器連結至「國庫收支書表條碼化 web 版」首頁。\n"
    )

    title = PackageStage()._infer_source_title(source_md, "treasury_guide.pdf")

    assert title == "列印國庫繳款書操作說明"


def test_bracket_section_labels_are_not_export_titles():
    assert PackageStage._is_unreliable_export_title("【說明】") is True
    assert PackageStage._is_unreliable_export_title("【注意】") is True
    # A genuine bracketed document title is longer than a section label.
    assert PackageStage._is_unreliable_export_title("【勞工保險投保薪資調整申報表】") is False


def test_split_main_body_drops_repeated_screenshot_lines():
    source_md = (
        "# 操作說明\n"
        "\n"
        "請輸入「12171002103」\n"
        "\n"
        "版權所有 © 中華民國 109 年，財政部國庫署，建議解析度 1024×768 以上。\n"
        "\n"
        "請輸入「12171002103」\n"
        "\n"
        "版權所有©中華民國109年，財政部國庫署，建議解析度1024×768以上。\n"
        "\n"
        "是\n"
        "\n"
        "是\n"
    )

    body = PackageStage()._clean_split_main_body(source_md, "操作說明", [])

    assert body.count("12171002103") == 1
    assert body.count("財政部國庫署") == 1
    # Short answer-like lines stay untouched.
    assert body.count("是") == 2


def test_repaired_main_export_updates_stub_index_title(tmp_path):
    outputs = tmp_path / "outputs"
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    main_path = documents_dir / "main.md"
    main_path.write_text("# demo_en_va_cep_login_flow_2pages\n\nstale\n", encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "main",
                    "kind": "main",
                    "title": "demo_en_va_cep_login_flow_2pages",
                    "source_filename": "demo_en_va_cep_login_flow_2pages.pdf",
                    "file": str(main_path),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repaired = (
        "# Welcome to the VA Customer Engagement Portal (CEP)\n"
        "\n"
        "After logging in through ID.me, the Customer Engagement Portal home page will appear.\n"
    )

    PackageStage._write_repaired_main_document_export(outputs, repaired)

    index = json.loads((outputs / "documents_index.json").read_text(encoding="utf-8"))
    assert index[0]["title"] == "Welcome to the VA Customer Engagement Portal (CEP)"
    assert main_path.read_text(encoding="utf-8").startswith("# Welcome")


def test_semantic_template_not_required_after_applied_semantic_repair():
    class Plan:
        document_type = "form_document"

    class Output:
        plan = Plan()
        semantic_repair_applied = True

    issues = _check_semantic_template(
        Output(),
        "## Login and Setup Steps\n1. Click I Agree.",
        "en",
    )

    assert issues == []


def test_semantic_template_still_required_without_repair():
    class Plan:
        document_type = "form_document"

    class Output:
        plan = Plan()

    issues = _check_semantic_template(Output(), "short field list only", "en")

    assert any(issue.code == "semantic_template_incomplete" for issue in issues)


def test_split_main_body_strips_trailing_orphan_open_quote():
    # Screenshot OCR crops can leave a dangling opening quote at line end
    # (「請輸入「國庫署」 「」); stripped, the line dedupes with the intact one.
    source_md = (
        "# 操作說明\n"
        "\n"
        "請輸入「國庫署」\n"
        "\n"
        "請輸入「國庫署」 「\n"
    )

    body = PackageStage()._clean_split_main_body(source_md, "操作說明", [])

    assert body.count("國庫署") == 1
    assert "」 「" not in body
