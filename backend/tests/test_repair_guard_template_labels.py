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
