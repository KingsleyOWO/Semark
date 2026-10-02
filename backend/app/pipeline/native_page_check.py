"""Source check for native-Word pages without a model call.

The Word->PDF conversion that the source review already makes for its page
inventory has a born-digital text layer. For a page that holds only native
Word text and tables, string comparison against that layer proves both sides
of the page audit: every printed line reached the parse, and every block
reached the documents and chunks that own it, in source order.

The check only ever certifies. Any page it cannot prove — pictures, gaps,
unmatched lines, text that differs from the print, unattributed output — is
handed back to the model audit unchanged.
"""

import html
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import fitz

from app.adapters.word import GENERATED_TABLE_CAPTION
from app.models.document_ir import Block, BlockType
from app.pipeline.content_coverage import normalized_text, plain_text
from app.pipeline.final_delivery import SUPPLEMENT_HEADINGS
from app.pipeline.package_utils import _table_label

# A line printed on this many pages is a running header/footer, not content.
FURNITURE_MIN_PAGES = 3
# A shorter piece split across a page break could match by coincidence.
SPLIT_MIN_CHARS = 4


def _norm(text: str) -> str:
    return normalized_text(plain_text(text))


# Text the package writes around source content: table labels, the supplement
# heading, the placeholder name of an unnamed Word table.
_LABELS = tuple(sorted({_norm(_table_label(key, language))
                        for key in ("table_name", "columns", "context", "content_type")
                        for language in ("zh-TW", "en")}, key=len, reverse=True))
_SCAFFOLD_LINES = {_norm(heading) for heading in SUPPLEMENT_HEADINGS.values()}
_PLACEHOLDER_CAPTION = re.compile(re.escape(_norm(GENERATED_TABLE_CAPTION.format(""))) + r"\d+$")


def _is_scaffold(line: str, source: str) -> bool:
    """A package label line whose remaining text is this page's own source."""
    if line in _SCAFFOLD_LINES:
        return True
    for label in _LABELS:
        if line.startswith(label):
            rest = line[len(label):]
            return not rest or rest in source or bool(_PLACEHOLDER_CAPTION.match(rest))
    return False


def document_source_text(blocks: Sequence[Block]) -> str:
    """Normalized source of every block, for tracing package labels anywhere in the document."""
    return "\n".join(piece for block in blocks for piece in block_pieces(block))


def block_pieces(block: Block) -> list[str]:
    """Normalized source text of a block; one piece per table cell."""
    if block.type == BlockType.TABLE:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", block.payload.get("table_body", ""), flags=re.S)
        return [p for p in (_norm(html.unescape(re.sub(r"<[^>]+>", " ", c))) for c in cells) if p]
    text = _norm(block.get_text())
    return [text] if text else []


def _is_plain_native(block: Block) -> bool:
    return ((block.payload.get("word_source") or {}).get("source") == "native_docx"
            and block.type in {BlockType.TEXT, BlockType.TABLE}
            and not any(key in block.payload for key in ("agentic_gap", "agentic_source_region")))


def _merge_overlap(left: str, right: str) -> str:
    for size in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return left + right


class NativePageText:
    """Normalized printed lines and drawn pictures of every page of a Word-rendered PDF."""

    def __init__(self, pdf_path: Path):
        with fitz.open(pdf_path) as document:
            self.lines = [[n for n in (_norm(line) for line in page.get_text().splitlines()) if n]
                          for page in document]
            pictures = [[info.get("xref", 0) for info in page.get_image_info(xrefs=True)] for page in document]
        self._line_pages = Counter(line for lines in self.lines for line in set(lines))
        picture_pages = Counter(x for xrefs in pictures for x in set(xrefs) if x)
        # A logo drawn on many pages is a header; an inline image (xref 0) never is.
        self.unparsed_pictures = [[x for x in xrefs if not x or picture_pages[x] < FURNITURE_MIN_PAGES]
                                  for xrefs in pictures]

    def _furniture(self, line: str) -> bool:
        return line.isdigit() or self._line_pages[line] >= FURNITURE_MIN_PAGES

    def _body(self, page: int) -> str:
        return "".join(line for line in self.lines[page] if not self._furniture(line))

    def _printed(self, piece: str, pages: list[int]) -> bool:
        """On the block's pages or a neighbour, whole or split at a page break."""
        window = [p for p in range(pages[0] - 1, pages[-1] + 2) if 0 <= p < len(self.lines)]
        if any(piece in "".join(self.lines[p]) for p in window):
            return True
        if len(piece) < 2 * SPLIT_MIN_CHARS:
            return False
        return any(piece[:k] in self._body(a) and piece[k:] in self._body(b)
                   for a, b in zip(window, window[1:])
                   for k in range(SPLIT_MIN_CHARS, len(piece) - SPLIT_MIN_CHARS + 1))

    def check(self, page: int, blocks: list[Block], mapping: dict[str, list[int]], *,
              documents: list[dict], chunks: list[dict], chunks_required: bool,
              nearby: Sequence[Block] = (), document_text: str = "") -> str | None:
        """None when the page is proven; otherwise why the model audit must decide.

        ``document_text`` is the whole document's normalized source; a package
        label around text from any page of it is scaffolding, not invented output.
        """
        if not blocks:
            return "page_has_no_blocks"
        if any(b.type == BlockType.IMAGE for b in blocks):
            return "page_has_picture_blocks"
        if bad := next((b for b in blocks if not _is_plain_native(b)), None):
            return f"{bad.block_id}: not a plain native Word block"
        if self.unparsed_pictures[page]:
            return f"{len(self.unparsed_pictures[page])} drawn picture(s) on the page are not in the parse"
        for block in blocks:
            pages = mapping.get(block.block_id) or []
            if not pages or pages != list(range(pages[0], pages[-1] + 1)):
                return f"{block.block_id}: no contiguous source pages"

        # Source side: each block is printed where it is mapped, in reading order.
        text = "".join(self.lines[page])
        cursor = 0
        for block in sorted(blocks, key=lambda b: (b.reading_order, b.block_id)):
            pieces = block_pieces(block)
            for piece in pieces:
                if not self._printed(piece, mapping[block.block_id]):
                    return f"{block.block_id}: not printed on its source page(s): {piece[:40]}"
            if pieces and len(mapping[block.block_id]) == 1:
                found = text.find(pieces[0], cursor)
                if found < 0:
                    return f"{block.block_id}: printed out of reading order"
                cursor = found

        # Every printed line belongs to this page's blocks (or a neighbour's, at a boundary).
        owned = [p for b in [*blocks, *nearby] for p in block_pieces(b)]
        joined = "".join(owned)
        for line in self.lines[page]:
            if self._furniture(line) or line in joined or any(line in piece for piece in owned):
                continue
            return f"printed line not in the parse: {line[:40]}"

        # Delivery side: each block reaches the documents and chunks that own it.
        ids = {b.block_id for b in blocks}
        source = document_text or "".join(owned)
        unbound = [d for d in documents if d.get("binding", "").startswith("unbound")
                   and ids.intersection(d.get("block_ids", []))
                   and not _is_scaffold(_norm(d.get("text", "")), source)]
        if unbound:
            return "delivered output not traceable to a source block: " + _norm(unbound[0].get("text", ""))[:40]
        files: dict[str, list[dict]] = {}
        for paragraph in documents:
            files.setdefault(paragraph.get("file", ""), []).append(paragraph)
        for block in blocks:
            owned_files = ["".join(_norm(p.get("text", "")) for p in paragraphs
                                   if block.block_id in p.get("block_ids", []))
                           for paragraphs in files.values()]
            for piece in block_pieces(block):
                if not any(piece in delivered for delivered in owned_files):
                    return f"{block.block_id}: not delivered in its documents: {piece[:40]}"
            if chunks_required:
                own = [_norm(c.get("content", "")) for c in chunks if block.block_id in c.get("block_ids", [])]
                spans = own + [_merge_overlap(a, b) for a, b in zip(own, own[1:])]
                for piece in block_pieces(block):
                    if not any(piece in span for span in spans):
                        return f"{block.block_id}: not delivered in its chunks: {piece[:40]}"
        order = {b.block_id: b.reading_order for b in blocks}
        for paragraphs in files.values():
            seen = [p["block_ids"][0] for p in paragraphs
                    if len(p.get("block_ids", [])) == 1 and p["block_ids"][0] in order]
            firsts = list(dict.fromkeys(seen))
            if [order[b] for b in firsts] != sorted(order[b] for b in firsts):
                return "delivered out of reading order"
        return None
