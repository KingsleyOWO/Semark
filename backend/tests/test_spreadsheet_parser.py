import shutil
from pathlib import Path

import pytest
from openpyxl import Workbook

from app.adapters.spreadsheet import parse_spreadsheet
from app.config import settings
from app.models.document_ir import SourceInfo
from app.pipeline.stages.normalize import HAS_PYMUPDF, NormalizeStage
from app.pipeline.stages.package import PackageStage


def _make_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "表四國內出差旅費"
    sheet.append(["職稱/職級別", "交通費", "宿費平日", "宿費假日", "雜費"])
    sheet.append(["院長、副院長、主任秘書", "按實檢據報支", 4500, 5500, 700])
    sheet.append(["研究員、副研究員、助理研究員", "按實檢據報支", 3500, 4500, 500])
    workbook.save(path)


def test_parse_spreadsheet_emits_structured_table(tmp_path):
    source = tmp_path / "travel.xlsx"
    _make_workbook(source)

    result = parse_spreadsheet(source, tmp_path / "out")

    assert result.success
    assert result.content_list_path is not None
    content = result.content_list_path.read_text(encoding="utf-8")
    assert "表四國內出差旅費" in content
    assert "<th>職稱/職級別</th>" in content
    assert "<td>研究員、副研究員、助理研究員</td>" in content
    assert result.stats["parser"] == "native_spreadsheet"
    assert result.stats["table_count"] == 1


async def test_spreadsheet_content_flows_to_package_markdown(tmp_path):
    source = tmp_path / "travel.xlsx"
    _make_workbook(source)
    parse_result = parse_spreadsheet(source, tmp_path / "out")
    assert parse_result.content_list_path is not None

    normalize_result = await NormalizeStage().run(
        doc_id="doc-xlsx",
        run_id="run-xlsx",
        content_list_path=parse_result.content_list_path,
        source_info=SourceInfo(
            path=str(source),
            ext=".xlsx",
            sha256="abc",
            size_bytes=source.stat().st_size,
        ),
        render_pages=False,
        mineru_version="native_spreadsheet",
    )
    assert normalize_result.success
    assert normalize_result.document_ir is not None

    package = PackageStage()
    source_md, _source_map = package._render_rag_md(
        document_ir=normalize_result.document_ir,
        asset_map={},
        enrichments={},
    )

    assert "表格名稱：表四國內出差旅費 表格" in source_md
    assert "欄位：職稱/職級別、交通費、宿費平日、宿費假日、雜費" in source_md
    assert "### 研究員、副研究員、助理研究員" in source_md
    assert "- 宿費平日：3500" in source_md
    assert "- 宿費假日：4500" in source_md
    assert "TABLE:" not in source_md


async def test_spreadsheet_normalize_renders_preview_image(tmp_path):
    if not HAS_PYMUPDF:
        pytest.skip("PyMuPDF is not installed")
    if not shutil.which("libreoffice"):
        pytest.skip("LibreOffice is not installed")

    source = tmp_path / "travel-preview.xlsx"
    _make_workbook(source)
    parse_result = parse_spreadsheet(source, tmp_path / "out-preview")
    assert parse_result.content_list_path is not None

    doc_id = "doc-xlsx-preview"
    run_id = "run-xlsx-preview"
    normalize_result = await NormalizeStage().run(
        doc_id=doc_id,
        run_id=run_id,
        content_list_path=parse_result.content_list_path,
        source_info=SourceInfo(
            path=str(source),
            ext="xlsx",
            sha256="abc",
            size_bytes=source.stat().st_size,
        ),
        render_pages=True,
        mineru_version="native_spreadsheet",
    )

    assert normalize_result.success
    assert normalize_result.document_ir is not None
    assert normalize_result.stats["pages_with_images"] >= 1

    first_page = normalize_result.document_ir.pages[0]
    assert first_page.page_image_path == "assets/pages/p0000.png"
    assert first_page.width_px and first_page.height_px
    image_path = settings.get_run_path(doc_id, run_id) / first_page.page_image_path
    assert image_path.exists()
