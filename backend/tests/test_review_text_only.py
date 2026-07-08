"""Text-only reviewer models (DeepSeek-V4) reject image input; the
SEMARK_REVIEW_SEND_PAGE_IMAGE setting must suppress the reviewer page image."""

import unittest
from unittest import mock

from app.models.document_ir import Block, BlockType, DocumentIR, EngineInfo, PageInfo, SourceInfo
from app.pipeline.stages.package import PackageStage


def _doc_with_page_image():
    return DocumentIR(
        doc_id="d",
        run_id="r",
        source=SourceInfo(path="x.pdf", ext=".pdf", sha256="x", size_bytes=1),
        engine=EngineInfo(backend="pipeline", method="auto"),
        pages=[PageInfo(page_idx=0, page_image_path="assets/pages/p0000.png")],
        blocks=[Block(block_id="t", type=BlockType.TEXT, page_idx=0, bbox_norm=[0, 0, 1000, 100], payload={"text": "x"})],
    )


class ReviewImageToggleTest(unittest.TestCase):
    def test_page_image_suppressed_when_setting_false(self):
        stage = PackageStage()
        with mock.patch("app.config.settings.review_send_page_image", False):
            self.assertIsNone(stage._semantic_repair_page_image(_doc_with_page_image(), [0], None))

    def test_setting_true_still_resolves_page_image(self):
        # When enabled the guard is bypassed and the normal page-resolution runs
        # (proven by it consulting the page-image path); False must skip that.
        stage = PackageStage()
        doc = _doc_with_page_image()
        with mock.patch("app.config.settings.review_send_page_image", True), \
                mock.patch.object(stage, "_document_page_image_path", return_value="assets/pages/p0000.png") as spy:
            stage._semantic_repair_page_image(doc, [0], None)
            self.assertTrue(spy.called)
        with mock.patch("app.config.settings.review_send_page_image", False), \
                mock.patch.object(stage, "_document_page_image_path") as spy_off:
            stage._semantic_repair_page_image(doc, [0], None)
            self.assertFalse(spy_off.called)


if __name__ == "__main__":
    unittest.main()
