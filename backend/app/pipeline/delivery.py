"""Atomic writes for delivered pipeline artifacts.

Every delivered file (source.md, dataset.md, rag.md, documents/*.md +
documents_index.json, chunks.jsonl, quality_gate.json, ...) must go through
here rather than a bare write_text/open("w"): a crash mid-write must never
leave a truncated file that a download endpoint then serves as valid, or an
index that references a file which was never finished. Each write lands via
a temp file in the SAME directory followed by os.replace, which is atomic on
a single filesystem; fsync is not required for this use case.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "atomic_write_text",
    "atomic_write_json",
    "atomic_write_jsonl",
]


@contextmanager
def _atomic_target(path: Path, encoding: str = "utf-8") -> Iterator[TextIO]:
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            yield handle
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    with _atomic_target(path, encoding=encoding) as handle:
        handle.write(text)


def atomic_write_json(path: Path, obj: Any, **dumps_kwargs: Any) -> None:
    atomic_write_text(path, json.dumps(obj, **dumps_kwargs), encoding="utf-8")


def atomic_write_jsonl(path: Path, lines_iter: Iterable[Any]) -> None:
    """Write one JSON-serialized object per line (ensure_ascii=False, matching
    every existing jsonl writer in this pipeline)."""
    with _atomic_target(path, encoding="utf-8") as handle:
        for line in lines_iter:
            handle.write(json.dumps(line, ensure_ascii=False))
            handle.write("\n")
