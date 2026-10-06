"""Numbers inside a picture's delivered text stay in the documents; the page-number rule stays for prose.

Validation v20.5: the main document dropped every line of one to three digits as
OCR noise. In picture content those lines are values: Amazon review counts
153/178/155/62, a sticky-note count, Lighthouse scores. 57 such lines in 11
pictures left documents/ (the RAG delivery) while chunks kept them.
"""
from app.config import AgenticConfig
from app.pipeline.stages.package import PackageStage

PICTURE = "Buffalo Games - 海邊小屋\n4.200\n153\nUS$11.97"
SOURCE = f"# 報告\n\n一、計畫摘要\n7\n\n{PICTURE}\n\n二、執行成果"


def test_digit_lines_inside_picture_content_are_kept():
    body = PackageStage()._clean_split_main_body(SOURCE, "報告", [], picture_texts=[PICTURE])

    assert PICTURE in body


def test_a_bare_number_line_in_prose_is_still_removed():
    body = PackageStage()._clean_split_main_body(SOURCE, "報告", [], picture_texts=[PICTURE])

    assert "一、計畫摘要\n7" not in body and "一、計畫摘要" in body


def test_each_problem_gets_three_methods_and_the_run_keeps_its_time_and_call_limits():
    # Spot check v20.10: six actions for the whole document stopped repairs after 42 of
    # 104 minutes and 60 of 120 calls; 17 problems got no attempt. Each picture now tries
    # at most three different methods (patch, targeted edit, full re-read) while the
    # document's time and call limits stay what they were (decided 2026-10-06).
    from app.pipeline.review_budget import document_budget

    limits = AgenticConfig()
    assert limits.max_recovery_attempts_per_issue == 3
    assert limits.max_recovery_actions == 30  # a safety ceiling; time and calls bind first
    budget = document_budget(limits, 47)
    assert (budget["max_model_calls"], budget["max_seconds"]) == (120, 6225)
