"""A document title must be a title, not the first sentence that says 「表」.

plan_document() picked the document title with `next(text for text in texts if
"表" in text)`. That substring also matches 表現, 代表, 發表, 圖表, 表示 and
外表, so any abstract or body paragraph using one of those words won the title
slot: 43 of the 167 documents in the 2026-08 corpus ended up with a whole
paragraph as `plan.title` (longest: 310 characters).

The damage is not cosmetic. plan.title is handed to infer_table_asset_title()
as `source_title`, which is what names a table when the table itself has no
caption — 52 of 128 tables — and the name is then truncated at 100 characters.
Tables were therefore titled with half a sentence of prose.
"""

import unittest

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, SourceInfo
from app.pipeline.structured_rag import plan_document


def _document(texts: list[tuple[str, int]]) -> DocumentIR:
    return DocumentIR(
        doc_id="doc-title",
        run_id="run-title",
        source=SourceInfo(path="sample.pdf", ext="pdf", sha256="abc", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        blocks=[
            Block(
                block_id=f"b{idx:06d}",
                type=BlockType.TEXT,
                page_idx=0,
                payload={"text": text, "text_level": level},
            )
            for idx, (text, level) in enumerate(texts)
        ],
    )


class PlanTitleSelectionTests(unittest.TestCase):
    def test_prose_containing_表現_does_not_become_the_title(self):
        document = _document(
            [
                ("示範經濟研究月刊 2026年3月", 0),
                ("示範島輸出產品替代彈性分析與政策意涵", 1),
                (
                    "隨著關稅政策持續升溫，示範島對甲國的出口表現遂成為評估其衝擊影響的核心，"
                    "本文以進口替代彈性衡量產品受關稅衝擊的敏感程度。",
                    0,
                ),
            ]
        )

        plan = plan_document(document)

        self.assertEqual(plan.title, "示範島輸出產品替代彈性分析與政策意涵")

    def test_prose_containing_代表_or_發表_does_not_become_the_title(self):
        for prose in (
            "上述數字代表產業結構的長期轉變，並不代表短期景氣已見底。",
            "本研究成果已於國際研討會發表，並將持續追蹤後續政策變化。",
            "圖表資料來源為公開統計，經作者整理後彙製而成。",
        ):
            with self.subTest(prose=prose[:8]):
                document = _document([("示範島產業結構轉型觀察", 1), (prose, 0)])
                self.assertEqual(plan_document(document).title, "示範島產業結構轉型觀察")

    def test_a_real_table_caption_still_wins_the_title(self):
        # The tightening must not cost us the case it was written for: a
        # standalone rate table whose leading text really is 「表 N …」.
        document = _document(
            [
                ("附件三", 0),
                ("表四 示範研究院國內出差旅費報支數額表", 0),
                ("單位：新臺幣元", 0),
            ]
        )

        self.assertEqual(
            plan_document(document).title, "表四 示範研究院國內出差旅費報支數額表"
        )

    def test_a_table_caption_beats_a_heading_block(self):
        document = _document(
            [
                ("示範島差旅制度說明", 1),
                ("表 3 示範島各職級出差日支數額", 0),
            ]
        )

        self.assertEqual(plan_document(document).title, "表 3 示範島各職級出差日支數額")

    def test_a_sentence_that_starts_with_表_is_not_a_caption(self):
        document = _document(
            [
                ("示範島產業健康度分析", 1),
                ("表1顯示，五大產業的供需失衡程度差異甚大，其中電池產業最為嚴重。", 0),
            ]
        )

        self.assertEqual(plan_document(document).title, "示範島產業健康度分析")

    def test_a_bare_table_number_is_not_a_title(self):
        # Live: one article's real H1 was displaced by a lone 「表1」 caption
        # marker sitting in the first twelve blocks.
        document = _document(
            [
                ("新興科技驅動之 示範農業關鍵應用模式探析", 1),
                ("在氣候變遷與勞動人口減少的雙重壓力下，農業轉型已成全球重要課題。", 0),
                ("表1", 0),
            ]
        )

        self.assertEqual(
            plan_document(document).title, "新興科技驅動之 示範農業關鍵應用模式探析"
        )

    def test_falls_back_to_the_first_text_when_nothing_looks_like_a_title(self):
        document = _document(
            [
                ("示範島總體經濟情勢回顧", 0),
                ("本文回顧近年總體經濟情勢。", 0),
            ]
        )

        self.assertEqual(plan_document(document).title, "示範島總體經濟情勢回顧")


if __name__ == "__main__":
    unittest.main()
