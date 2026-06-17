"""Supported input file formats for ingestion and parsing."""

from pathlib import Path

PDF_EXTENSIONS = {".pdf"}
SPREADSHEET_NATIVE_EXTENSIONS = {".xls", ".xlsx"}
WORD_NATIVE_EXTENSIONS = {".docx"}
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".odp", ".ods"}
HTML_EXTENSIONS = {".html", ".htm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CONVERT_TO_PDF_EXTENSIONS = (
    OFFICE_EXTENSIONS - SPREADSHEET_NATIVE_EXTENSIONS - WORD_NATIVE_EXTENSIONS
) | HTML_EXTENSIONS
DIRECT_MINERU_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS
SUPPORTED_INPUT_EXTENSIONS = (
    DIRECT_MINERU_EXTENSIONS
    | CONVERT_TO_PDF_EXTENSIONS
    | SPREADSHEET_NATIVE_EXTENSIONS
    | WORD_NATIVE_EXTENSIONS
)
SUPPORTED_INPUT_EXTENSIONS_LABEL = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))


def normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_supported_input(filename: str) -> bool:
    return normalize_extension(filename) in SUPPORTED_INPUT_EXTENSIONS
