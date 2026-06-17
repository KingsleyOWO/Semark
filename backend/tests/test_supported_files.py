from app.supported_files import is_supported_input


def test_supported_input_extensions():
    supported = [
        "sample.pdf",
        "sample.docx",
        "sample.doc",
        "sample.pptx",
        "sample.ppt",
        "sample.xlsx",
        "sample.xls",
        "sample.odt",
        "sample.odp",
        "sample.ods",
        "sample.html",
        "sample.htm",
        "sample.png",
        "sample.jpg",
        "sample.jpeg",
    ]

    for filename in supported:
        assert is_supported_input(filename)


def test_unsupported_input_extensions():
    for filename in ["sample.exe", "sample.zip", "sample.txt", "sample.md"]:
        assert not is_supported_input(filename)
