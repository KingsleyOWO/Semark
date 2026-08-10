"""Block.get_text() is typed `-> str`, but MinerU emits img_caption /
chart_caption as *lists* of strings and normalize copies them into the IMAGE
payload verbatim. A non-empty caption therefore made get_text() return a list,
and every str-consuming caller blew up.

Observed in production (2026-08-07): 6 of 65 runs died in the enrich stage with
`TypeError: sequence item N: expected str instance, list found` — 5 via
_page_has_form_cues, 1 via _is_structured_rate_table_document. Documents whose
figures carried no caption were unaffected (an empty list is falsy, so `or ""`
masked the contract violation).
"""

import unittest

from app.config import EnrichConfig, PipelineConfig
from app.models.document_ir import (
    Block,
    BlockType,
    DocumentIR,
    EngineInfo,
    PageInfo,
    SourceInfo,
)
from app.pipeline.stages.enrich import EnrichStage
from app.pipeline.structured_rag import _page_context

REFERENCE_TABLE_HTML = (
    "<table>"
    "<tr><td>地區</td><td>日支數額</td><td>幣別</td></tr>"
    "<tr><td>亞洲</td><td>120</td><td>美元</td></tr>"
    "<tr><td>歐洲</td><td>160</td><td>美元</td></tr>"
    "<tr><td>美洲</td><td>180</td><td>美元</td></tr>"
    "</table>"
)


def _image_block(block_id: str, caption, page_idx: int = 0, reading_order: int = 1) -> Block:
    return Block(
        block_id=block_id,
        type=BlockType.IMAGE,
        page_idx=page_idx,
        bbox_norm=[10, 10, 900, 400],
        reading_order=reading_order,
        payload={"img_path": "images/fig.jpg", "caption": caption, "footnote": []},
    )


def _document(blocks: list[Block], pages: int = 1) -> DocumentIR:
    return DocumentIR(
        doc_id="doc-caption",
        run_id="run-caption",
        source=SourceInfo(path="研究報告.pdf", ext="pdf", sha256="abc", size_bytes=100),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=i) for i in range(pages)],
        blocks=blocks,
    )


def _stage() -> EnrichStage:
    return EnrichStage(
        db=None,
        config=PipelineConfig(enrich=EnrichConfig(enable_vlm=True, vlm_enrich_figures=True)),
    )


class BlockGetTextContractTest(unittest.TestCase):
    """get_text() must honour its `-> str` annotation for every payload shape."""

    def test_image_list_caption_is_joined_into_text(self):
        block = _image_block("b1", ["2026年Demandsage新創企業統計"])
        self.assertEqual(block.get_text(), "2026年Demandsage新創企業統計")

    def test_image_multi_part_caption_keeps_every_part(self):
        block = _image_block("b1", ["有／無對策下風險差異分布", "資料來源:本院整理"])
        self.assertEqual(
            block.get_text(),
            "有／無對策下風險差異分布 資料來源:本院整理",
        )

    def test_image_empty_and_missing_captions_are_empty_string(self):
        self.assertEqual(_image_block("b1", []).get_text(), "")
        self.assertEqual(_image_block("b2", None).get_text(), "")
        self.assertEqual(_image_block("b3", "").get_text(), "")

    def test_image_string_caption_still_passes_through(self):
        self.assertEqual(_image_block("b1", "圖1 組織架構").get_text(), "圖1 組織架構")

    def test_text_table_and_list_blocks_are_unchanged(self):
        text = Block(block_id="t", type=BlockType.TEXT, page_idx=0, payload={"text": "第一條"})
        table = Block(
            block_id="tb", type=BlockType.TABLE, page_idx=0, payload={"table_body": "<table></table>"}
        )
        listing = Block(
            block_id="l", type=BlockType.LIST, page_idx=0, payload={"items": ["一", "二"]}
        )
        self.assertEqual(text.get_text(), "第一條")
        self.assertEqual(table.get_text(), "<table></table>")
        self.assertEqual(listing.get_text(), "一\n二")

    def test_every_block_type_returns_str(self):
        blocks = [
            _image_block("b1", ["圖說"]),
            Block(block_id="t", type=BlockType.TEXT, page_idx=0, payload={"text": "文"}),
            Block(block_id="tb", type=BlockType.TABLE, page_idx=0, payload={"table_body": "x"}),
            Block(block_id="eq", type=BlockType.EQUATION, page_idx=0, payload={"latex": "x=1"}),
            Block(block_id="c", type=BlockType.CODE, page_idx=0, payload={"code": "pass"}),
            Block(block_id="l", type=BlockType.LIST, page_idx=0, payload={"items": ["a"]}),
        ]
        for block in blocks:
            with self.subTest(block=block.block_id):
                self.assertIsInstance(block.get_text(), str)


class FormPageDetectionSurvivesCaptionedFiguresTest(unittest.TestCase):
    """Reproduces both production crash paths inside _detect_form_pages."""

    def test_page_form_cue_scan_survives_captioned_figure(self):
        """5 of the 6 production failures: enrich.py _page_has_form_cues."""
        document_ir = _document(
            [
                Block(
                    block_id="b0",
                    type=BlockType.TEXT,
                    page_idx=0,
                    payload={"text": "我國高齡化趨勢分析", "text_level": 1},
                ),
                _image_block("b1", ["我國與APEC經濟體出生時預期壽命趨勢"]),
            ]
        )

        self.assertEqual(_stage()._detect_form_pages(document_ir), [])

    def test_reference_table_scan_survives_captioned_figure(self):
        """The 6th production failure: _is_structured_rate_table_document."""
        document_ir = _document(
            [
                Block(
                    block_id="b0",
                    type=BlockType.TABLE,
                    page_idx=0,
                    payload={"table_body": REFERENCE_TABLE_HTML},
                ),
                _image_block("b1", ["2026年Demandsage新創企業統計"]),
            ]
        )

        self.assertEqual(_stage()._detect_form_pages(document_ir), [])

    def test_caption_form_cues_are_still_detected(self):
        """Coercion must not blind the detector to real cues inside a caption."""
        document_ir = _document(
            [
                _image_block("b0", ["申請人:  簽名蓋章欄位"]),
            ]
        )

        self.assertEqual(_stage()._detect_form_pages(document_ir), [0])


class StructuredRagPageContextTest(unittest.TestCase):
    """_page_context feeds the structured-table VLM fallback and reads every
    block on the page, IMAGE included — same latent crash."""

    def test_page_context_survives_captioned_figure(self):
        document_ir = _document(
            [
                Block(
                    block_id="b0",
                    type=BlockType.TEXT,
                    page_idx=0,
                    payload={"text": "表1 各年齡層分布"},
                ),
                _image_block("b1", ["各年齡層分布圖"]),
            ]
        )

        context = _page_context(document_ir, 0)

        self.assertIn("表1 各年齡層分布", context)
        self.assertIn("各年齡層分布圖", context)


if __name__ == "__main__":
    unittest.main()
