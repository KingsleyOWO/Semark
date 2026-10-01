"""Source-row partitions, bounded continuation reads and lossless assembly."""
import json
import shutil

import fitz

from app.models.document_ir import Block, BlockType
from app.pipeline.agentic_evidence import table_cells
from app.pipeline.content_coverage import TextCoverage, normalized_text
from app.pipeline.delivery import atomic_write_json
from app.pipeline.image_reading import ReadingFormatError, read_contract
from app.pipeline.recovery_models import (
    MAX_REGION_CONTEXT_READS,
    MAX_REGION_FORMAT_SPLITS,
    RegionReading,
)
from app.pipeline.table_structure import TableCell, TableStructure, attach_structure


def plan_regions(inventory, ir, page_idx, layout):
    from app.pipeline.document_recovery import regions
    with fitz.open(inventory.pdf) as pdf:
        page = pdf[page_idx]
        height, width = page.rect.height, page.rect.width
        tables = page.find_tables().tables
        rows, protected = [], []
        native = [b for b in inventory.blocks(page_idx, ir) if b.type == BlockType.TABLE
                  and inventory.mapping.get(b.block_id) == [page_idx]]
        for index, table in enumerate(tables):
            extracted = normalized_text(' '.join(' '.join(c or '' for c in row) for row in table.extract()))
            matches = []
            for block in native:
                cells = table_cells(block)
                labels = [normalized_text(c['text']) for c in cells if len(normalized_text(c['text'])) >= 4]
                if (cells and max(c['row'] for c in cells) == table.row_count and labels
                        and sum(label in extracted for label in labels) / len(labels) >= .8):
                    matches.append((block.block_id, cells))
            block_id, cells = matches[0] if len(matches) == 1 else (None, [])
            seen_cells = set()
            x_edges = sorted({c[x] for c in table.cells if c for x in (0, 2)})
            y_edges = sorted({r.bbox[1] for r in table.rows} | {table.bbox[3]})
            for n, row in enumerate(table.rows):
                box = [row.bbox[0] * 1000 / width, row.bbox[1] * 1000 / height,
                       row.bbox[2] * 1000 / width, row.bbox[3] * 1000 / height]
                source_cells = []
                for column, rect in enumerate(row.cells, 1):
                    if not rect or tuple(rect) in seen_cells:
                        continue
                    seen_cells.add(tuple(rect))
                    source_cells.append({'column': column,
                        'colspan': max(1, sum(rect[0] <= x < rect[2] for x in x_edges)),
                        'rowspan': max(1, sum(rect[1] <= y < rect[3] for y in y_edges)),
                        'bbox': [rect[0] * 1000 / width, rect[1] * 1000 / height,
                                 rect[2] * 1000 / width, rect[3] * 1000 / height]})
                rows.append({'source_id': f'{block_id or "pdf-table-" + str(index)}:r{n + 1}',
                             'table_id': block_id or f'pdf-table-{index}', 'row': n + 1, 'bbox': box,
                             'row_count': table.row_count, 'column_count': table.col_count,
                             'cells': source_cells,
                             'native_cells': [c for c in cells if c['row'] == n + 1]})
            # Merged cells are indivisible even when a neighboring column has a row edge.
            protected.extend((c[1] * 1000 / height, c[3] * 1000 / height) for c in table.cells if c)
        for block in page.get_text('dict')['blocks']:
            if block['type'] == 0:
                protected.extend((line['bbox'][1] * 1000 / height, line['bbox'][3] * 1000 / height)
                                 for line in block['lines'])
            elif block['type'] == 1:
                protected.append((block['bbox'][1] * 1000 / height, block['bbox'][3] * 1000 / height))
        if not rows:
            cores = regions(layout)
        else:
            # Tables remain full width, even if the planner requested quadrants.
            count = 2 if layout == 'horizontal2' else 4
            edges = {round(r['bbox'][1]) for r in rows} | {round(r['bbox'][3]) for r in rows}
            edges |= set(range(50, 951, 10))
            safe = [y for y in edges if all(not lo + 1 < y < hi - 1 for lo, hi in protected)]
            cuts = [0]
            for index in range(1, count):
                candidates = [y for y in safe if cuts[-1] + 40 <= y <= 1000 - 40 * (count - index)]
                if candidates:
                    cuts.append(min(candidates, key=lambda y: abs(y - index * 1000 / count)))
            cuts.append(1000)
            cores = [[0, lo, 1000, hi] for lo, hi in zip(cuts, cuts[1:])]
        return [{'core': core, 'owned_rows': [r for r in rows if core[1] <= round(r['bbox'][1]) < core[3]],
                 'context_rows': [r for r in rows if r['row'] == 1 and round(r['bbox'][1]) < core[1]]}
                for core in cores]


def expand_context(core, direction, plan):
    """Adjacent complete regions provide context; source ownership never changes."""
    box = list(core)
    if direction in {'above', 'both', 'all'}:
        box[1] = max((p['core'][1] for p in plan if p['core'][1] < core[1]), default=0)
    if direction in {'below', 'both', 'all'}:
        box[3] = min((p['core'][3] for p in plan if p['core'][3] > core[3]), default=1000)
    if direction in {'left', 'all'}:
        box[0] = max((p['core'][0] for p in plan if p['core'][0] < core[0]), default=0)
    if direction in {'right', 'all'}:
        box[2] = min((p['core'][2] for p in plan if p['core'][2] > core[2]), default=1000)
    return box


async def read_regions(runner, ir, problem, decision, folder, skill):
    from app.pipeline.document_recovery import crop_region
    page = problem['page_idx']
    folder.mkdir(parents=True, exist_ok=True)
    selected = runner.layout_store.selected(page, decision.region_ids, runner.state['layout_pages'].get(str(page)))
    details = []
    for item in selected:
        if item.get('provider') == 'word_native':
            image = folder / (item['region_id'] + '.png')
            shutil.copy2(runner.inventory.word_images[item['word_block_id']], image)
            crop = item['bbox']
        else:
            image, crop = crop_region(runner.inventory, page, item['bbox'], folder, item['region_id'])
        details.append({**item, 'crop_bbox': crop, 'image': image})
    atomic_write_json(folder / 'selected-layout-regions.json', {
        'source_sha256': runner.source_sha, 'page_idx': page,
        'regions': [{**item, 'image': item['image'].name} for item in details]}, ensure_ascii=False, indent=2)
    plan = plan_regions(runner.inventory, ir, page, decision.layout)
    atomic_write_json(folder / 'region-plan.json', plan, ensure_ascii=False, indent=2)
    blocks, observations, completed = [], [], []
    continuations, splits = 0, 0

    async def read_part(region, index):
        nonlocal continuations, splits
        core = region['core']
        extra = [item for item in details if item['bbox'][0] < core[2] and item['bbox'][2] > core[0]
                 and item['bbox'][1] < core[3] and item['bbox'][3] > core[1]]
        requested = list(core)
        for attempt in range(2):
            image, crop = crop_region(runner.inventory, page, requested, folder, f'{index}-{attempt}')
            payload = {'skill': skill, 'page_idx': page, 'source_sha256': runner.source_sha,
                       'core_bbox': core, 'crop_bbox': crop,
                       'owned_rows': region['owned_rows'], 'context_rows': region['context_rows'],
                       'coordinate_system': '0..1000 relative to the entire source page',
                       'output_limits': {'max_rows': 40, 'max_prose': 24,
                                         'max_output_tokens': runner.owner.limits.max_output_tokens},
                       'detail_images': [{'image_number': n + 3, **{k: v for k, v in item.items() if k != 'image'}}
                                         for n, item in enumerate(extra)]}
            if len(json.dumps(payload, ensure_ascii=False)) > 42000:
                raise ValueError('recovery_context_limit')
            try:
                reading = await read_contract(runner.owner,
                    'read_region_alternate' if decision.model_role == 'review' else 'read_region', payload,
                    [image, runner.inventory.image(page)] + [item['image'] for item in extra], RegionReading,
                    max_attempts=1)
            except ReadingFormatError as exc:
                children = split_region(region)
                observations.append({'core': core, 'failure_kind': 'response_format_error', 'error': str(exc)})
                atomic_write_json(folder / 'region-observations.json', observations, ensure_ascii=False, indent=2)
                if splits >= MAX_REGION_FORMAT_SPLITS or not children:
                    raise
                splits += 1
                for child_index, child in enumerate(children):
                    await read_part(child, f'{index}-split{child_index}')
                return
            observations.append({'core': core, 'crop': crop, 'reading': reading.model_dump()})
            atomic_write_json(folder / 'region-observations.json', observations, ensure_ascii=False, indent=2)
            if reading.complete:
                break
            if reading.status != 'needs_context':
                raise ValueError('region_unreadable: ' + reading.observation)
            expanded = expand_context(core, reading.context_request, plan)
            if attempt or continuations >= MAX_REGION_CONTEXT_READS or expanded == requested:
                raise ValueError('region_context_exhausted: ' + reading.observation)
            continuations += 1
            requested = expanded
        validate_region_rows(region, reading)
        completed.append((region, reading))

    for index, region in enumerate(plan):
        await read_part(region, str(index))
    tables = {}
    with fitz.open(runner.inventory.pdf) as pdf:
        source = pdf[page]
        native_prose = [(normalized_text(b[4]), [b[0] * 1000 / source.rect.width,
            b[1] * 1000 / source.rect.height, b[2] * 1000 / source.rect.width,
            b[3] * 1000 / source.rect.height]) for b in source.get_text('blocks') if b[6] == 0]
    source_cells = [c['bbox'] for part in plan for row in part['owned_rows'] for c in row['cells']]
    for region, reading in completed:
        row_lookup = {row.source_id: row for row in reading.rows}
        for owned in region['owned_rows']:
            row = row_lookup[owned['source_id']]
            table = tables.setdefault(owned['table_id'], {'rows': [], 'cells': []})
            table['rows'].append(owned)
            values = {c.column: c.text for c in row.cells}
            for cell in owned['cells']:
                table['cells'].append(TableCell(row=owned['row'], column=cell['column'],
                    text=values[cell['column']], rowspan=cell['rowspan'], colspan=cell['colspan'],
                    header=row.role == 'header', source_bbox=cell['bbox']))
        if reading.text and any(r['owned_rows'] for r, _ in completed):
            blocks.append(Block(block_id=f'recovery_{folder.name}_text{len(blocks)}', type=BlockType.TEXT,
                page_idx=page, bbox_norm=region['core'], payload={'text': reading.text, **runner.source_payload(page),
                    'agentic_source_region': region['core'], 'agentic_recovery_tool': 'read_regions'}))
        for prose in reading.prose:
            bbox, origin = locate_prose(prose, region['core'], native_prose, source_cells)
            blocks.append(Block(block_id=f'recovery_{folder.name}_text{len(blocks)}', type=BlockType.TEXT,
                page_idx=page, bbox_norm=bbox, payload={'text': prose.text, **runner.source_payload(page),
                    'agentic_source_region': bbox, 'agentic_geometry_origin': origin,
                    'agentic_recovery_tool': 'read_regions'}))
    if not tables:
        # Backward-compatible prose path. Tables never pass through this string join.
        readings = [reading.text for _, reading in completed if reading.text]
        reject_overlap_duplicates(readings, runner.inventory.pdf, page)
        if readings:
            blocks.insert(0, Block(block_id=f'recovery_{folder.name}_page', type=BlockType.TEXT,
                page_idx=page, bbox_norm=[0, 0, 1000, 1000], payload={
                    'text': '\n\n'.join(readings), **runner.source_payload(page),
                    'agentic_source_region': [0, 0, 1000, 1000], 'agentic_recovery_tool': 'read_regions'}))
    else:
        for table_id, table in tables.items():
            rows = sorted(table['rows'], key=lambda row: row['row'])
            if [row['row'] for row in rows] != list(range(1, rows[0]['row_count'] + 1)):
                raise ValueError('region_table_missing_or_duplicate_rows')
            bbox = [round(min(r['bbox'][0] for r in rows)), round(min(r['bbox'][1] for r in rows)),
                    round(max(r['bbox'][2] for r in rows)), round(max(r['bbox'][3] for r in rows))]
            bid = f'recovery_{folder.name}_table{len(blocks)}'
            structure = TableStructure(table_id=table_id, row_count=rows[0]['row_count'],
                column_count=rows[0]['column_count'], cells=table['cells'], source={
                    'block_id': bid, 'source_table_id': table_id, 'page_idx': page,
                    'source_sha256': runner.source_sha, 'source_rows': [r['source_id'] for r in rows]})
            block = Block(block_id=bid, type=BlockType.TABLE, page_idx=page, bbox_norm=bbox,
                payload={**runner.source_payload(page), 'agentic_source_region': bbox,
                         'agentic_recovery_tool': 'read_regions',
                         'agentic_source_row_ids': [r['source_id'] for r in rows]})
            blocks.append(attach_structure(block, structure))
    for block in blocks:
        block.payload['agentic_layout_region_ids'] = list(decision.region_ids)
    blocks.sort(key=lambda block: (block.bbox_norm[1], block.bbox_norm[0]))
    atomic_write_json(folder / 'region-readings.json', [b.model_dump() for b in blocks], ensure_ascii=False)
    return runner.replace_page(ir, page, blocks), [page]


def locate_prose(prose, core, native_prose, source_cells):
    """Use a unique native text anchor when available; tables own their cells."""
    text = normalized_text(prose.text)
    matches = [box for native, box in native_prose if text and text == native]
    bbox = matches[0] if len(matches) == 1 else prose.bbox
    origin = 'native_text' if len(matches) == 1 else 'model_estimate'
    x0, y0, x1, y1 = bbox
    if not (core[0] <= x0 < core[2] and core[1] <= y0 < core[3]):
        raise ValueError('region_prose_outside_owned_core')
    area = (x1 - x0) * (y1 - y0)
    for cell in source_cells:
        overlap = max(0, min(x1, cell[2]) - max(x0, cell[0])) * max(0, min(y1, cell[3]) - max(y0, cell[1]))
        if overlap >= area * .5:
            raise ValueError('region_prose_overlaps_table_cell: table content must use source rows')
    return [max(0, min(1000, round(value))) for value in bbox], origin


def validate_region_rows(region, reading):
    expected = [r['source_id'] for r in region['owned_rows']]
    if [r.source_id for r in reading.rows] != expected:
        raise ValueError('region_omitted_native_cell_or_source_row: missing/duplicate/unowned/reordered rows')
    if expected and reading.text.strip():
        raise ValueError('region_unlocated_prose: tables use rows; surrounding text needs source bbox')
    for source, row in zip(region['owned_rows'], reading.rows, strict=True):
        columns = [cell['column'] for cell in source['cells']]
        if [c.column for c in row.cells] != columns:
            raise ValueError('region_missing_duplicate_or_wrong_column: ' + row.source_id)
        values = {cell.column: cell.text for cell in row.cells}
        for cell in source['native_cells']:
            if cell['text'] and not TextCoverage([values.get(cell['column'], '')]).covers(cell['text']):
                raise ValueError('region_omitted_native_cell: ' + row.source_id + ':' + cell['id'])
        if any(cell['header'] for cell in source['native_cells']) and row.role != 'header':
            raise ValueError('region_source_header_changed')


def split_region(region):
    core, rows = region['core'], region['owned_rows']
    if rows:
        if len(rows) < 2:
            return []  # A source row cannot be split into unrelated text halves.
        cut = round(rows[len(rows) // 2]['bbox'][1])
        if any(r['bbox'][1] < cut < r['bbox'][3] for r in rows):
            return []
        if any(c['bbox'][1] < cut < c['bbox'][3] for r in rows for c in r['cells']):
            return []
    else:
        if core[3] - core[1] < 160:
            return []
        cut = (core[1] + core[3]) // 2
    if not core[1] < cut < core[3]:
        return []
    children = []
    for lo, hi in ((core[1], cut), (cut, core[3])):
        children.append({'core': [core[0], lo, core[2], hi],
            'owned_rows': [r for r in rows if lo <= round(r['bbox'][1]) < hi],
            'context_rows': region['context_rows'] + [r for r in rows if round(r['bbox'][1]) < lo]})
    return children


def reject_overlap_duplicates(readings, pdf_path, page):
    with fitz.open(pdf_path) as pdf:
        original = normalized_text(pdf[page].get_text())
    fragments = [normalized_text(t) for t in readings]
    for first in range(len(fragments)):
        for second in range(first + 1, len(fragments)):
            for offset in range(0, max(0, len(fragments[first]) - 39), 10):
                phrase = fragments[first][offset:offset + 40]
                if phrase in fragments[second] and original.count(phrase) == 1:
                    raise ValueError('region_assembly_retains_overlap_duplicate')
