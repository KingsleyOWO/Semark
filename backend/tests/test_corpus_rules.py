"""Tests for the data-driven corpus ruleset loader (app.pipeline.corpus_rules)."""

import json
import re

import pytest

from app.pipeline.corpus_rules import CorpusRules, get_rules, reset_rules_cache

ENV_VAR = "SEMARK_CORPUS_RULES_PATH"


@pytest.fixture(autouse=True)
def _fresh_rules_cache(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_rules_cache()
    yield
    reset_rules_cache()


def test_get_rules_loads_bundled_default():
    rules = get_rules()
    assert isinstance(rules, CorpusRules)
    assert rules.title_fixes
    assert rules.text_fixes
    assert rules.field_label_fixes
    assert rules.field_label_overrides
    assert rules.visual_label_map
    assert rules.flow_role_terms
    assert rules.document_markers
    assert rules.plan_keywords


def test_get_rules_caches_after_first_load():
    assert get_rules() is get_rules()


def test_env_override_loads_custom_ruleset(tmp_path, monkeypatch):
    custom_path = tmp_path / "custom.json"
    custom_path.write_text(
        json.dumps({"flow_role_terms": ["custom-role"]}), encoding="utf-8"
    )
    monkeypatch.setenv(ENV_VAR, str(custom_path))
    reset_rules_cache()
    rules = get_rules()
    assert rules.flow_role_terms == ("custom-role",)
    # Sections absent from the custom ruleset default to empty.
    assert rules.title_fixes == ()
    assert rules.visual_label_map == {}


def test_missing_override_path_falls_back_to_default(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "does-not-exist.json"))
    reset_rules_cache()
    with caplog.at_level("WARNING"):
        rules = get_rules()
    assert rules.flow_role_terms  # bundled defaults loaded
    assert any(
        record.levelname == "WARNING" and ENV_VAR in record.getMessage()
        for record in caplog.records
    )


def test_malformed_override_file_falls_back_to_default(tmp_path, monkeypatch, caplog):
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(broken_path))
    reset_rules_cache()
    with caplog.at_level("WARNING"):
        rules = get_rules()
    assert rules.flow_role_terms  # bundled defaults loaded
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_title_fixes_reproduce_irs_transcript_title():
    rules = get_rules()
    text = "Form 4506-T Request for Transcript of Tax Return Form Do not sign"
    for rule in rules.title_fixes:
        match = rule.pattern.search(text)
        if match:
            assert (
                match.expand(rule.replacement)
                == "Form 4506-T Request for Transcript of Tax Return"
            )
            break
    else:
        pytest.fail("no title fix matched the IRS 4506-T sample")


def test_text_fixes_repair_illinois_form_spacing():
    rules = get_rules()
    text = "IL-1040-VStaple your check here"
    for fix in rules.text_fixes:
        text = fix.sub(text)
    assert text.startswith("IL-1040-V Staple")

    text = "NRPart-year resident"
    for fix in rules.text_fixes:
        text = fix.sub(text)
    assert text.startswith("NR Part-year")


def test_field_label_fixes_strip_connexuc_prefix():
    rules = get_rules()
    text = "Bill attach ConnexUC Itinerary) Airfare Amount"
    for fix in rules.field_label_fixes:
        text = fix.sub(text)
    assert text == "Airfare Amount"


def test_field_label_overrides_canonicalize_ssn_label():
    rules = get_rules()
    text = "1b First social security number on tax return"
    for override in rules.field_label_overrides:
        if override.pattern.search(text):
            text = override.replacement
    assert text == "First social security number"


def test_visual_label_map_normalizes_bilingual_role_labels():
    rules = get_rules()
    text = "Government Agency A (甲) > Vendor B (乙)"
    for pattern, replacement in rules.visual_label_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    assert "政府機關甲" in text
    assert "廠商乙" in text


def test_flow_role_terms_include_complaint_workflow_roles():
    terms = get_rules().flow_role_terms
    assert "受理單位(人事)" in terms
    assert "被害人" in terms
    assert "調查小組" in terms


def test_document_markers_accessor():
    rules = get_rules()
    assert rules.marker_list("table_collection_title_markers") == (
        "檔案分類及保存年限區分表",
        "保存年限區分表",
    )
    assert "示範研究院" in rules.marker_list("noisy_header_exact")
    assert rules.marker_list("no-such-marker") == ()


def test_plan_keywords_accessor():
    rules = get_rules()
    assert rules.keyword_list("daily_allowance_amount_terms") == ("生活費", "日支")
    assert "職級別" in rules.keyword_list("domestic_rate_role_terms")
    assert rules.keyword_list("no-such-keywords") == ()
