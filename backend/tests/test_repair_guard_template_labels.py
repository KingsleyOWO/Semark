"""Template scaffolding labels are renderer output, not document facts.

The reviewer prompt explicitly asks to remove repetitive template filler;
counting removed template section labels as "lost facts" made the guard
reject good rewrites (live: 2-12 訓練辦法 repairs rejected at 0.35-0.70
survival with missing tokens like 主要欄位分組/常見查詢關鍵字)."""

from app.pipeline.corpus_rules import get_rules
from app.pipeline.repair_guard import repair_preserves_facts

ORIGINAL = (
    "表單名稱：進修申請表\n"
    "主要欄位分組：申請/基本資料；簽核/用印。\n"
    "常見查詢關鍵字：進修、申請。\n"
    "用途與填寫重點：供人員申請進修。\n"
    "版本日期：114.12.11\n"
    "申請人：____\n"
)


def test_corpus_rules_expose_template_section_labels():
    labels = get_rules().marker_list("template_section_labels")
    assert "主要欄位分組" in labels
    assert "常見查詢關鍵字" in labels


def test_template_labels_do_not_count_as_lost_facts():
    repaired = (
        "# 進修申請表\n\n## 文件識別\n- 版本日期：114.12.11\n"
        "## 欄位\n- 申請人\n- 進修申請表\n"
    )
    ok, details = repair_preserves_facts(ORIGINAL, "", repaired)
    assert ok is True
    assert details["survival_ratio"] == 1.0


def test_real_fact_loss_is_still_rejected():
    repaired = "# 進修申請表\n\n## 欄位\n- 申請人\n"  # drops 114.12.11
    ok, details = repair_preserves_facts(ORIGINAL, "", repaired)
    assert ok is False
    assert any("114.12.11" in t for t in details["missing_tokens"])


def test_table_and_figure_weave_labels_do_not_count_as_lost_facts():
    # Live batch 2026-07: 6/8 reviewer repairs rejected as fact_loss with
    # missing tokens dominated by weave-template labels (表格名稱/內容類型/
    # 可用於查詢/欄位1) and asset-token fragments (0000 from [[asset:tbl0000]]).
    original = (
        "表格名稱：設備一覽表 第 3 頁 表格 1\n"
        "內容類型：表格片段或續接資料\n"
        "欄位：項目、說明\n"
        "分類或範圍：一般設備\n"
        "可用於查詢：白板、投影。\n"
        "- 欄位1：1\n"
        "[[asset:tbl0000]]\n"
        "申請日期：114.12.11\n"
    )
    repaired = "# 設備一覽表\n\n- 項目：白板\n- 說明：投影\n- 申請日期：114.12.11\n"

    ok, details = repair_preserves_facts(original, "", repaired)

    assert ok is True, details
    assert details["survival_ratio"] == 1.0, details


def test_bare_step_and_page_numbers_do_not_count_as_lost_facts():
    # Screenshot transcriptions carry step markers and page numbers (8, 9,
    # 10...) that a legitimate rewrite reorganizes away. Real values (three+
    # digits, decimals, percentages, dates) must still be preserved.
    original = (
        "步驟 8 之後接續步驟 9，再到步驟 10、11、12、13。\n"
        "配額為 2GB，版本 114.12.11，費用 3500 元，成長 0.19%。\n"
    )
    repaired = (
        "# 教學\n\n- 依序完成各步驟。\n"
        "- 配額為 2GB，版本 114.12.11，費用 3500 元，成長 0.19%。\n"
    )

    ok, details = repair_preserves_facts(original, "", repaired)

    assert ok is True, details

    dropped_real_value = "# 教學\n\n- 依序完成各步驟。\n- 配額為 2GB，版本 114.12.11，成長 0.19%。\n"
    ok2, details2 = repair_preserves_facts(original, "", dropped_real_value)

    assert ok2 is False
    assert any("3500" in t for t in details2["missing_tokens"])
