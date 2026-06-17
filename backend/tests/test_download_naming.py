import json

from app.api.routes.download import OutputFormat, _get_document_files


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
