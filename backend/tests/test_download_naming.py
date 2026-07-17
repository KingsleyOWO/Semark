import json

from app.api.routes.converters import md_to_txt
from app.api.routes.download import (
    FileType,
    OutputFormat,
    _get_document_files,
    _get_file_content,
)


def test_document_zip_paths_keep_run_id_out_of_file_names(tmp_path):
    outputs = tmp_path / "outputs"
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "main.md").write_text("# Main", encoding="utf-8")
    (documents_dir / "form_0000.md").write_text("# Form", encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "main",
                    "kind": "main",
                    "title": "主文",
                    "file": str(documents_dir / "main.md"),
                },
                {
                    "document_id": "form_0000",
                    "kind": "form",
                    "title": "表單",
                    "file": str(documents_dir / "form_0000.md"),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files = _get_document_files(
        outputs,
        source_name="人員管理辦法",
        format=OutputFormat.MD,
        archive_folder_name="人員管理辦法_01KT0FRA7KE1",
    )

    assert [name for name, _content in files] == [
        "人員管理辦法_01KT0FRA7KE1_documents/人員管理辦法_main.md",
        "人員管理辦法_01KT0FRA7KE1_documents/人員管理辦法_form01.md",
    ]


def test_document_zip_paths_respect_selected_document_ids(tmp_path):
    outputs = tmp_path / "outputs"
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "main.md").write_text("# Main", encoding="utf-8")
    (documents_dir / "form_0000.md").write_text("# Form", encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {"document_id": "main", "kind": "main", "file": str(documents_dir / "main.md")},
                {
                    "document_id": "form_0000",
                    "kind": "form",
                    "file": str(documents_dir / "form_0000.md"),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files = _get_document_files(
        outputs,
        source_name="人員管理辦法",
        format=OutputFormat.TXT,
        document_ids=["form_0000"],
    )

    assert len(files) == 1
    assert files[0][0] == "人員管理辦法_documents/人員管理辦法_form01.txt"
    assert files[0][1].decode("utf-8").strip() == "Form"

def _write_viewer_main_document(outputs):
    """Create the split main document exactly as the viewer serves it."""
    documents_dir = outputs / "documents"
    documents_dir.mkdir(parents=True)
    viewer_main_md = (
        "# 使用說明\n\n"
        "## 一、登入系統\n\n"
        "(1)開啟瀏覽器後進入員工專區。\n\n"
        "圖片顯示登入畫面的介面。\n"
    )
    (documents_dir / "main.md").write_text(viewer_main_md, encoding="utf-8")
    (outputs / "documents_index.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "main",
                    "kind": "main",
                    "title": "使用說明",
                    "file": str(documents_dir / "main.md"),
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return viewer_main_md


def test_main_text_download_serves_viewer_main_document(tmp_path):
    outputs = tmp_path / "outputs"
    viewer_main_md = _write_viewer_main_document(outputs)
    # Legacy divergent render must be ignored when the viewer document exists.
    (outputs / "main_text.md").write_text("# 使用說明\n\n舊的純文字版。\n", encoding="utf-8")

    result = _get_file_content(
        outputs,
        FileType.MAIN_TEXT,
        OutputFormat.MD,
        run_id="01TESTRUN",
        source_name="使用說明",
    )

    assert result is not None
    content, filename = result
    assert content.decode("utf-8") == viewer_main_md
    assert filename == "使用說明_main.md"


def test_main_text_download_converts_viewer_main_document(tmp_path):
    outputs = tmp_path / "outputs"
    viewer_main_md = _write_viewer_main_document(outputs)

    result = _get_file_content(
        outputs,
        FileType.MAIN_TEXT,
        OutputFormat.TXT,
        run_id="01TESTRUN",
        source_name="使用說明",
    )

    assert result is not None
    content, filename = result
    assert content.decode("utf-8") == md_to_txt(viewer_main_md)
    assert filename == "使用說明_main.txt"


def test_main_text_download_falls_back_for_legacy_runs(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "main_text.md").write_text("# 使用說明\n\n舊版主文。\n", encoding="utf-8")

    result = _get_file_content(
        outputs,
        FileType.MAIN_TEXT,
        OutputFormat.MD,
        run_id="01TESTRUN",
        source_name="使用說明",
    )

    assert result is not None
    content, filename = result
    assert content.decode("utf-8") == "# 使用說明\n\n舊版主文。\n"
    assert filename == "使用說明_main_text.md"


def test_batch_zip_names_do_not_collide_across_runs_of_same_source():
    from app.api.routes.download import _dedupe_zip_entry_name

    used: set[str] = set()
    first = _dedupe_zip_entry_name("使用說明_main.md", "01KXMSEA64P5WFHRVPG1Z7F1J4", used)
    second = _dedupe_zip_entry_name("使用說明_main.md", "01KXMXHAJWM176238VVFG5TYWM", used)
    third = _dedupe_zip_entry_name("使用說明_main.md", "01KXMXHAJWM176238VVFG5TYWM", used)

    assert first == "使用說明_main.md"
    assert second == "使用說明_main_01KXMXHAJWM1.md"
    assert third == "使用說明_main_01KXMXHAJWM1_2.md"
    assert len({first, second, third}) == 3


def test_source_name_strips_edge_whitespace():
    from app.api.routes.download import _source_name_for

    assert _source_name_for("/data/uploads/201電子白板 .pdf", "01RUNABCDEFGHIJK") == "201電子白板"
    assert _source_name_for(None, "01RUNABCDEFGHIJK") == "01RUNABCDEFG"
