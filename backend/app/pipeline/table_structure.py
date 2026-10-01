"""Source-addressed table cells shared by parsing, recovery and rendering.

The stored grid describes extraction, not certification. Source review remains
in review_coverage, keyed by the owning block; no model can set a verified flag.
"""
import hashlib
import html
from html.parser import HTMLParser

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TableCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    text: str
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    header: bool = False
    source_bbox: list[float] | None = None

    @property
    def id(self):
        return f"r{self.row}c{self.column}"


class TableStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    table_id: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    cells: list[TableCell]
    source: dict = Field(default_factory=dict)
    body_sha256: str = ""

    @model_validator(mode="after")
    def unique_positions(self):
        occupied = set()
        for cell in self.cells:
            for row in range(cell.row, cell.row + cell.rowspan):
                for column in range(cell.column, cell.column + cell.colspan):
                    if row > self.row_count or column > self.column_count or (row, column) in occupied:
                        raise ValueError("table_cells_overlap_or_exceed_grid")
                    occupied.add((row, column))
        return self

    def to_html(self):
        lines = ['<table data-source-grid="1">']
        for row in range(1, self.row_count + 1):
            lines.append("<tr>")
            for cell in sorted((c for c in self.cells if c.row == row), key=lambda c: c.column):
                tag = "th" if cell.header else "td"
                spans = ''.join(f' {name}="{getattr(cell, name)}"' for name in ("rowspan", "colspan")
                                if getattr(cell, name) > 1)
                text = html.escape(cell.text).replace('\n', '<br>')
                lines.append(f'<{tag}{spans}>{text}</{tag}>')
            lines.append("</tr>")
        return '\n'.join(lines + ['</table>'])


class TableCells(HTMLParser):
    """Physical cells with logical grid columns, including rowspan occupancy."""
    def __init__(self, body):
        super().__init__()
        self.cells, self.row, self.column, self.current = [], 0, 1, None
        self.occupied = set()
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row += 1
            self.column = 1
        elif tag in {'td', 'th'} and self.row:
            while (self.row, self.column) in self.occupied:
                self.column += 1
            attributes = dict(attrs)
            def span(name):
                try:
                    return max(1, min(1000, int(attributes.get(name, 1))))
                except (TypeError, ValueError):
                    return 1
            self.current = {'id': f'r{self.row}c{self.column}', 'row': self.row,
                            'column': self.column, 'text': '', 'header': tag == 'th',
                            'rowspan': span('rowspan'), 'colspan': span('colspan')}
        elif self.current is not None and tag in {'br', 'p'}:
            self.current['text'] += ' '

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'] += data

    def handle_endtag(self, tag):
        if tag in {'td', 'th'} and self.current is not None:
            cell = self.current
            cell['text'] = ' '.join(cell['text'].split())
            self.cells.append(cell)
            for row in range(cell['row'], cell['row'] + cell['rowspan']):
                for column in range(cell['column'], cell['column'] + cell['colspan']):
                    self.occupied.add((row, column))
            self.column += cell['colspan']
            self.current = None


def table_structure(block, *, store=False):
    body = str(block.payload.get('table_body') or '')
    sha = hashlib.sha256(body.encode()).hexdigest()
    cached = block.payload.get('table_structure')
    if cached and cached.get('body_sha256') == sha:
        return TableStructure.model_validate(cached)
    parsed = TableCells(body)
    if not parsed.cells:
        return None
    cells = [TableCell.model_validate({k: v for k, v in c.items() if k != 'id'}) for c in parsed.cells]
    word = block.payload.get('word_source', {})
    structure = TableStructure(table_id=block.block_id,
        row_count=max(parsed.row, max(c.row + c.rowspan - 1 for c in cells)),
        column_count=max(c.column + c.colspan - 1 for c in cells), cells=cells, body_sha256=sha,
        source={'block_id': block.block_id, 'word_block_index': word.get('block_index'),
                'page_idx': None if word else block.page_idx, 'bbox': block.bbox_norm,
                'source_sha256': block.payload.get('agentic_source_sha256'),
                'source_pages': block.payload.get('agentic_source_pages', [])})
    if store:
        block.payload['table_structure'] = structure.model_dump()
    return structure


def attach_structure(block, structure):
    """One rendering path; grid and body cannot diverge when committing a table."""
    body = structure.to_html()
    structure.body_sha256 = hashlib.sha256(body.encode()).hexdigest()
    block.payload.update(table_body=body, table_structure=structure.model_dump())
    return block


def canonical_table_html(block):
    structure = table_structure(block)
    return structure.to_html() if structure is not None else str(block.payload.get("table_body") or "")


def materialize_table_structures(ir):
    """Normalize derived grids before review; persistence must not change its input."""
    from app.models.document_ir import BlockType

    for block in ir.blocks:
        if block.type == BlockType.TABLE:
            table_structure(block, store=True)
