"""Numbers inside a picture's delivered text stay in the documents; the page-number rule stays for prose.

Lumio v20.5: the main document dropped every line of one to three digits as
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


def test_recovery_may_take_twelve_actions_by_default():
    assert AgenticConfig().max_recovery_actions == 12
