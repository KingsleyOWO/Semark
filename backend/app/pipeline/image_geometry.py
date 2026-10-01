"""Measured axes + source-grounded labels; geometry is an observation, not approval.

Line extraction uses OpenCV morphology (horizontal/vertical structuring elements).
Only an unambiguous crossing pair is supported; photos, grids and other charts keep
using the ordinary image tool. Every proposed relationship still needs pixel audit.
"""
from pydantic import Field, model_validator

from app.pipeline.image_source_review import ImageSourceRelation, WordPictureItem


class LocatedRelation(ImageSourceRelation):
    label: str = Field(default='', max_length=180)

    @model_validator(mode='after')
    def entity_label(self):
        if self.kind == 'spatial_position' and not self.label.strip():
            raise ValueError('A positioned object needs its exact visible label')
        return self


class LocatedReading(WordPictureItem):
    relationships: list[LocatedRelation] = Field(max_length=16)


def measured_axes(source):
    """Original-pixel straight-line measurements, or None when ambiguous."""
    try:
        import cv2
    except ImportError:
        return None
    from PIL import Image
    with Image.open(source) as picture:
        if picture.width * picture.height > 20_000_000:
            return None
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    height, width = gray.shape
    binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    found = []
    for horizontal in (True, False):
        kernel = (max(10, width // 5), 1) if horizontal else (1, max(10, height // 5))
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = [b for b in boxes if b[2] > width * .35 and b[3] < height * .02] if horizontal else [
            b for b in boxes if b[3] > height * .35 and b[2] < width * .02]
        if len(boxes) != 1:
            return None
        x, y, w, h = boxes[0]
        found.append([x * 1000 / width, y * 1000 / height,
                      (x + w) * 1000 / width, (y + h) * 1000 / height])
    horizontal, vertical = found
    x = (vertical[0] + vertical[2]) / 2
    y = (horizontal[1] + horizontal[3]) / 2
    if not (horizontal[0] < x < horizontal[2] and vertical[1] < y < vertical[3]):
        return None
    return {'horizontal_line': horizontal, 'vertical_line': vertical,
            'intersection': [x, y], 'coordinate_system': '0..1000 original image',
            'provider': 'opencv_morphology', 'opencv_version': cv2.__version__}


def geometry_eligible(prior):
    kinds = {r.get('kind') for r in prior.get('relationships', [])}
    return {'axis_mapping', 'spatial_position'}.issubset(kinds) and kinds <= {
        'axis_mapping', 'spatial_position', 'arrow'}


def measured_side(low, high, axis, *, vertical=False):
    # Do not force a logo which straddles a line into a quadrant. Bboxes are
    # source observations and can be corrected by the subsequent pixel audit.
    if low <= axis <= high:
        return 'spans_axis'
    return ('above' if vertical else 'left') if high < axis else ('below' if vertical else 'right')


def join_located_labels(text, labels):
    """Join line wraps only for a separately located label with identical letters.

    Keep the source characters/order and every occurrence. A label observation
    cannot introduce text; its full whitespace-free string must already exist
    as consecutive whole source lines. Raw observations remain in the evidence.
    """
    names = {''.join(label.split()) for label in labels if label.strip()}
    lines, output, index = text.splitlines(), [], 0
    while index < len(lines):
        end, value = index, ''
        while end < len(lines) and lines[end].strip():
            value += ''.join(lines[end].split())
            end += 1
            if value in names:
                break
            if not any(name.startswith(value) for name in names):
                end = index + 1
                break
        if value not in names:
            end = index + 1
        output.append(' '.join(lines[index:end]))
        index = end
    return '\n'.join(output)


def canonical_reading(reading, axes, prior, *, language):
    if reading.source_id != prior['source_id'] or reading.image_type not in {'chart', 'diagram'}:
        raise ValueError('Geometry proposal does not describe the supported source diagram')
    required = {r['relation_id']: r['kind'] for r in prior['relationships']}
    got = {r.relation_id: r.kind for r in reading.relationships}
    if any(got.get(key) != kind for key, kind in required.items()):
        raise ValueError('Geometry repair omitted or repurposed a prior relationship')
    if not reading.complete or not reading.relationships_complete or any(not r.certain for r in reading.relationships):
        raise ValueError('Geometry observations incomplete; retain original candidate')
    x, y = axes['intersection']
    relations = []
    zh = language != 'en'
    for relation in reading.relationships:
        value = relation.model_dump(exclude={'label'})
        if relation.kind == 'spatial_position':
            x0, y0, x1, y1 = relation.bbox
            horizontal = measured_side(x0, x1, x)
            vertical = measured_side(y0, y1, y, vertical=True)
            h = {'left': '位於垂直軸左側', 'right': '位於垂直軸右側', 'spans_axis': '橫跨垂直軸'}
            v = {'above': '位於水平軸上方', 'below': '位於水平軸下方', 'spans_axis': '跨越水平軸'}
            fact = (f'圖中「{relation.label}」標記的可見範圍{h[horizontal]}，{v[vertical]}。' if zh else
                    f'The visible extent of the {relation.label} mark is {horizontal} relative to the vertical axis '
                    f'and {vertical} relative to the horizontal axis.')
            value.update(source_fact=fact, horizontal=horizontal, vertical=vertical)
        relations.append(value)
    return WordPictureItem.model_validate({**reading.model_dump(exclude={'relationships'}),
        'text': join_located_labels(reading.text, [r.label for r in reading.relationships]),
        'relationships': relations})


async def repair_axis_image(runner, ir, block, page, region, source, folder, prior, axes, *, model_role):
    """A bounded alternative inside repair_image; old observations remain parents."""
    from app.pipeline.delivery import atomic_write_json
    from app.pipeline.image_evidence import source_region_views
    from app.pipeline.image_reading import read_contract
    from app.pipeline.image_recovery import build_image_parts_candidate
    from app.pipeline.semantic.language import resolve_semantic_output_language

    language = resolve_semantic_output_language(runner.owner.config.package.semantic_output_language.value, ir)
    old_id = runner.evidence.add(provider='prior_vlm', kind='image_relationship_reading', page=page,
                                block_id=block.block_id, source=source, content=prior)
    geometry_id = runner.evidence.add(provider='opencv', kind='image_axis_measurement', page=page,
                                     block_id=block.block_id, source=source, content=axes)
    x, y = axes['intersection']
    crop = [max(0, int(x) - 200), max(0, int(y) - 220), min(1000, int(x) + 220), min(1000, int(y) + 220)]
    views, regions = source_region_views(source, [crop])
    reading = await read_contract(runner.owner,
        'read_image_relations_alternate' if model_role == 'review' else 'read_image_relations', {
            'source_id': prior['source_id'], 'output_language': language,
            'measured_axes': axes, 'source_detail_regions': regions,
            'image_order': ['original image', 'original-pixel crop; coordinates always refer to original image'],
            'prior_observations_untrusted': prior['relationships'],
            'instruction': 'Re-read all labels and relationships from pixels. Prior IDs identify obligations, NOT correct '
                'positions. Keep each prior relation ID/kind; correct its content and bbox, add IDs for omitted content '
                '(including faded duplicate marks). For spatial_position, label is the exact visible object name '
                '(qualify faded/solid when distinct); bbox must enclose the ENTIRE logo/label mark in original-image '
                'coordinates, not only its center or a text fragment. Axes measurements locate lines only; verify that '
                'they are the diagram axes. Do not infer numeric data or intentions. Arrow facts must name both '
                'visible endpoints (e.g. faded to solid), not just its direction. Use source language for text and '
                'requested output language for facts. This is a repair proposal for independent audit, never approval.',
        }, views, LocatedReading, max_attempts=2)
    revised = canonical_reading(reading, axes, prior, language=language)
    parts = {'visual_facts': [r.source_fact for r in revised.relationships],
             'text_regions': [{'text': revised.text, 'bbox': [0, 0, 1000, 1000]}], 'gaps': [],
             'information_status': 'informative', 'image_type': revised.image_type,
             'text_dependent': False, 'partial': False}
    atomic_write_json(folder / 'image-geometry.json', {'measurement_evidence_id': geometry_id,
        'prior_evidence_id': old_id, 'proposal': reading.model_dump(), 'candidate_reading': revised.model_dump(),
        'status': 'candidate_requires_independent_source_and_delivery_audit'}, ensure_ascii=False, indent=2)
    candidate = build_image_parts_candidate(runner, ir, block, page, region, source, parts,
        [old_id, geometry_id], folder, relationship_reading=revised.model_dump(), relationship_parents=[old_id, geometry_id])
    selected = [b for b in candidate.blocks if b.payload.get('agentic_image_origin') == block.block_id]
    axis_ids = [selected[i].block_id for i, r in enumerate(revised.relationships) if r.kind == 'axis_mapping']
    for item in selected:
        item.payload['agentic_image_repair_method'] = 'axis_geometry'
        item.payload['agentic_depends_on_block_ids'] = [bid for bid in axis_ids if bid != item.block_id]
    return candidate
