"""A long line with an ellipsis or a doubled full stop is prose, not a table of contents.

Spot check 2026-10-06: main.md dropped a whole table row because its author typed
「4..產業變動快速」, and across the 44 stored reports three body paragraphs with
「…」 or 「….等」 were in no delivered file at all. A contents line pairs a dot
leader with a page number; dots alone do not make one.
"""
from app.pipeline.stages.package import PackageStage

ROW = ("- 第 4 列：第 1 欄：遭遇問題；第 2–3 欄：1. 核心技術方案的品牌識別度未能充分展現，影響海外市場開拓。"
       " 2. 行銷資源有限，缺乏系統化溝通。 3. 缺乏有經驗的品牌顧問輔導 4..產業變動快速，品牌策略須加快應變速度")
PROSE = ("進行品牌識別系統規範手冊之基本設計(初版)，應用設計之事務用品（名片、信封、信紙、牛皮袋、電子簽章、"
         "簡報格式….等的規劃設計，於每月之月會進行討論修正，並將於下年度進行品牌內化工作項目。")
SURVEY = ("在選擇教具時，您最在意的條件是什麼？請排序提出 (音色、音準、耐用性、安全性、外觀、價格...)；"
          "受訪者表示第一個一定是耐用度，第二是價錢，第三才是外觀與顏色，家長通常比較重視安全性。")
CONTENTS = "四、品牌發展需求及遭遇問題…………………………………………………………….13"
COLLAPSED = ("貳、品牌輔導架構與內容規劃……………………………16；伍、實際執行進度…………………………67；"
             "二、管顧核心能量簡介……………………………………72")


def body(*lines):
    return PackageStage()._clean_split_main_body("\n\n".join(lines), "示範公司結案報告", [])


def test_long_lines_with_dots_but_no_page_number_stay_in_main_document():
    text = body(ROW, PROSE, SURVEY)

    assert ROW in text
    assert PROSE in text
    assert SURVEY in text


def test_contents_lines_with_page_numbers_are_still_removed():
    text = body("## 目錄", CONTENTS, COLLAPSED, ROW)

    assert "13" not in text.replace(ROW, "")
    assert "管顧核心能量簡介" not in text
    assert ROW in text
