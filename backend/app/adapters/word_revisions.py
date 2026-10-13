"""Word tracked changes, read the way the document reads once every change is accepted.

python-docx only reads runs that sit directly in a paragraph, so text typed while
change tracking was on (inside w:ins) was dropped: a total corrected from 600,000 to
300,000 by inserting "3" and deleting "6" came out as "00,000", and inserted
paragraphs, cells and whole tables vanished. Deleted text (w:delText) was already
skipped, but deleted pictures and rows were still read.
"""

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_KEEP = (W + "ins", W + "moveTo")
_DROP = (W + "del", W + "moveFrom")
REVISED_PARTS = ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml")


def accept_tracked_changes(root) -> int:
    """Accept every insertion and deletion under ``root`` in place; returns how many changes were applied."""
    count = 0
    for element in list(root.iter(*_KEEP, *_DROP)):
        parent = element.getparent()
        if parent is None:
            continue
        count += 1
        if parent.tag == W + "trPr":
            # A row marker: a deleted row goes, an inserted row just loses the marker.
            row = parent.getparent()
            if element.tag in _DROP and row is not None and row.getparent() is not None:
                row.getparent().remove(row)
            else:
                parent.remove(element)
            continue
        if parent.tag == W + "rPr":
            # Paragraph-mark and run-property markers carry no text.
            parent.remove(element)
            continue
        if element.tag in _KEEP:
            index = parent.index(element)
            for child in list(element):
                parent.insert(index, child)
                index += 1
        parent.remove(element)
    return count


def is_revised_part(name: str) -> bool:
    return name in REVISED_PARTS or (
        name.startswith(("word/header", "word/footer")) and name.endswith(".xml"))
