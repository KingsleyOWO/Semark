"""Blind image relationship evidence and strict accounting in the existing page audit.

Source readings remain fallible observations, never proof by model agreement.
No proposed caption enters the blind read. A separate source/delivery audit must
account for each relationship, including uncertainty and missing arrows.
"""
from typing import Literal

from pydantic import Field, model_validator

from app.pipeline.content_coverage import normalized_text
from app.pipeline.recovery_models import Contract, NativeItem
from app.pipeline.zh_text import to_taiwan_traditional

WORD_PICTURE_BATCH_SIZE = 2

Direction = Literal['none', 'up', 'down', 'left', 'right', 'up_left', 'up_right',
                    'down_left', 'down_right', 'bidirectional', 'uncertain']
Horizontal = Literal['unspecified', 'left', 'right', 'on_axis', 'spans_axis', 'uncertain']
Vertical = Literal['unspecified', 'above', 'below', 'on_axis', 'spans_axis', 'uncertain']


class ImageSourceRelation(Contract):
    relation_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,24}$')
    kind: Literal['axis_mapping', 'spatial_position', 'arrow', 'value_pair', 'table_relation', 'other']
    source_fact: str = Field(min_length=1, max_length=320)
    bbox: list[int] = Field(min_length=4, max_length=4)
    direction: Direction = 'none'
    horizontal: Horizontal = 'unspecified'
    vertical: Vertical = 'unspecified'
    certain: bool

    @model_validator(mode='after')
    def valid_relation(self):
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError('Source relationship bbox must be a nonempty original-image 0..1000 rectangle')
        if self.kind == 'arrow' and self.direction == 'none':
            raise ValueError('Arrows require a direction, or explicit uncertain direction')
        if self.certain and 'uncertain' in (self.direction, self.horizontal, self.vertical):
            raise ValueError('An uncertain direction/position cannot be certain')
        return self


class WordPictureItem(NativeItem):
    image_type: Literal['photo', 'logo', 'chart', 'table', 'diagram', 'document', 'other']
    relationships: list[ImageSourceRelation] = Field(max_length=16)
    relationships_complete: bool

    @model_validator(mode='after')
    def sideless_positions(self):
        """A side is only meaningful against stated axes.

        On a chart with axes, a position given without a side stays uncertain and
        can never certify; on a photo or slide it is a description ("people around
        a long table") and is checked like any other stated relationship.
        """
        axes = any(r.kind == 'axis_mapping' for r in self.relationships)
        for relation in self.relationships:
            if relation.kind == 'spatial_position' and relation.horizontal == relation.vertical == 'unspecified':
                if axes:
                    relation.horizontal, relation.vertical, relation.certain = 'uncertain', 'uncertain', False
                else:
                    relation.kind = 'other'
        return self

    @model_validator(mode='after')
    def unique_relationships(self):
        ids = [r.relation_id for r in self.relationships]
        if len(ids) != len(set(ids)):
            raise ValueError('Source relationship IDs must be unique within an image')
        if self.image_type in {'chart', 'table', 'diagram'} and self.relationships_complete and not ids:
            raise ValueError('A chart/table/diagram cannot claim complete relationships with an empty label-only reading')
        return self


class WordPictureReading(Contract):
    items: list[WordPictureItem] = Field(min_length=1, max_length=4)


class ImageRelationshipCheck(Contract):
    relation_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,24}$')
    status: Literal['matched', 'contradicted', 'missing', 'uncertain']
    delivered_quote: str = Field(default='', max_length=500)
    observation: str = Field(min_length=1, max_length=240)
    direction: Direction = 'none'
    horizontal: Horizontal = 'unspecified'
    vertical: Vertical = 'unspecified'


SOURCE_RELATION_INSTRUCTIONS = """In addition to exact lettering, read the image's substantive RELATIONSHIPS independently.
No proposed caption, expected answer or neighboring prose is supplied. Classify image_type from pixels.
For charts/diagrams/tables, labels alone are incomplete evidence. In relationships record axis endpoints
and meaning; each entity's actual side relative to the named axes; arrows from tail to head including
movement between repeated/faded marks; value-to-year/series/unit links; table row/header links.
Use short source_fact and a bounding box in original-image 0..1000 coordinates per relation, local IDs r0,r1,...
For spatial_position use horizontal/vertical only relative to the named visible axes/reference, not the
whole-image midpoint. Marks crossing an axis must be spans_axis, not forced into a quadrant. An ambiguous
side is uncertain. For arrows direction is tail-to-head in image coordinates (up is toward image top).
Names alone and an arrow's presence do not capture its meaning. Never infer sequence from mere alignment.
Omit direction/side fields where irrelevant. certain=false retains an ambiguous relationship explicitly.
Keep each source_fact <=320 characters; maximum 16 relationships. If more substantive relationships exist,
set relationships_complete=false and explain the omitted scope; never silently truncate or claim completeness.
Photos/logos/documents without substantive relationships may have an empty list with relationships_complete=true.
Lettering completeness and relationship completeness are separate. Put literal wording in text, never only
in observation. This reading is candidate evidence for an independent pixel/delivery audit, not certification."""

RELATION_AUDIT_INSTRUCTIONS = """independent_image_readings are BLIND source observations, not ground truth.
Check them again against the original pixels and the actual delivered documents/chunks, never by agreement.
image_reading_scopes maps each original image to its current required block IDs (including split text blocks).
For every listed relation return exactly one image_relationship_checks entry across that original image's
required blocks, on the block that delivers it. Do not repeat the relation on every child or borrow another
image's relation IDs. Check BOTH source correctness and delivery preservation.
status=matched requires a verbatim delivered_quote from that block's supplied delivery, including the actual
relationship (a list of labels is insufficient). Report direction/horizontal/vertical as expressed by the
DELIVERY, not copied from the source reading. If the source relation is uncertain or unsupported, use uncertain.
If delivery asserts the opposite side/direction use contradicted; omitted arrow/axis/value linkage is missing.
Bounded or crossing-axis positions must not be forced into quadrants. A static map/matrix is not a temporal
flow: check start/end and causal language against actual arrows. Preserve explicit uncertainty. Source text
can match exactly while the relationships are wrong. Keep observations <=240 characters, quote <=500.
A failed/missing/uncertain relationship means needs_review for that image block, even if all labels match."""


async def read_word_picture_batch(owner, payload, images, *, source_ids):
    """Reuse persisted format handling; at most four images, split only bad contracts."""
    from app.pipeline.image_reading import ReadingFormatError, read_contract
    request = {**payload, 'source_relation_contract': SOURCE_RELATION_INSTRUCTIONS}
    try:
        return await read_contract(owner, 'verify_native_reading', request, images, WordPictureReading,
                                   source_ids=source_ids, max_attempts=1 if len(source_ids) > 1 else 2)
    except ReadingFormatError:
        if len(source_ids) <= 1:
            raise
    midpoint = len(source_ids) // 2
    items = []
    for lo, hi in ((0, midpoint), (midpoint, len(source_ids))):
        subset = source_ids[lo:hi]
        child = {**payload, 'sources': payload['sources'][lo:hi], 'image_order': subset}
        result = await read_word_picture_batch(owner, child, images[lo:hi], source_ids=subset)
        items.extend(result.items)
    return WordPictureReading(items=items)


def appears_in(piece, text):
    """Model-written wording found in delivery as written or after zh-TW conversion of both.

    Conversion is gated per text, so it can rewrite a delivery that keeps an
    author's simplified glyph; the unconverted comparison still applies.
    """
    return (normalized_text(piece) in normalized_text(text)
            or normalized_text(to_taiwan_traditional(piece)) in normalized_text(to_taiwan_traditional(text)))


QUOTE_LINE_GAP = 120  # normalized characters a quote may skip between the delivered lines it cites


def quote_found(quote, text):
    """Each cited line verbatim and in order, skipping at most a line or two between them.

    Lines far apart are not one relationship: a label and an unrelated number
    later on must not pass as a value pair.
    """
    if appears_in(quote, text):
        return True
    lines = [line for line in quote.splitlines() if normalized_text(line)]
    if not lines:
        return False
    for convert in (str, to_taiwan_traditional):
        target = normalized_text(convert(text))
        pieces = [normalized_text(convert(line)) for line in lines]
        start = target.find(pieces[0])
        while start >= 0:  # the first cited line may also occur earlier, e.g. in a title
            position = start + len(pieces[0])
            for piece in pieces[1:]:
                found = target.find(piece, position)
                if found < 0 or found - position > QUOTE_LINE_GAP:
                    break
                position = found + len(piece)
            else:
                return True
            start = target.find(pieces[0], start + 1)
    return False


def relationship_issue(reading, finding, delivered, *, documents=None, chunks=None):
    """Cannot certify absent coverage, unsupported quotations or opposite relations.

    A matched status alone is insufficient. Direction fields describe delivery;
    disagreeing with the independent source observation leaves uncertainty for a
    source repair, never promotes either model's words to truth.
    """
    if not reading.relationships_complete:
        return 'Independent image relationships are incomplete; labels alone cannot verify visual meaning.'
    relations = {r.relation_id: r for r in reading.relationships}
    # A check for a relation the reading never listed certifies nothing, so it is dropped.
    checks = [c for c in finding.image_relationship_checks if c.relation_id in relations]
    ids = [c.relation_id for c in checks]
    if len(ids) != len(set(ids)) or set(ids) != set(relations):
        return 'Image relationship audit omitted or duplicated source relation IDs.'
    for check in checks:
        relation = relations[check.relation_id]
        if not relation.certain or check.status != 'matched':
            return f'Image relationship {check.relation_id} remains {check.status if relation.certain else "uncertain"}: {relation.source_fact}'
        quote = check.delivered_quote
        if not normalized_text(quote) or not quote_found(quote, delivered):
            return f'Image relationship {check.relation_id} has no grounded quote in its own delivered content.'
        for label, excerpts in [('documents', documents), ('chunks', chunks)]:
            owned = [e.get('text', e.get('content', '')) for e in excerpts or []
                     if finding.block_id in e.get('block_ids', [e.get('block_id')])]
            if excerpts is not None and not quote_found(quote, '\n'.join(owned)):
                return f'Image relationship {check.relation_id} has no owned quote in actual {label}.'
        if relation.kind == 'arrow' and check.direction != relation.direction:
            return f'Image relationship {check.relation_id} direction differs between source and delivery.'
        if relation.kind == 'spatial_position' and (check.horizontal, check.vertical) != (relation.horizontal, relation.vertical):
            return f'Image relationship {check.relation_id} position differs between source and delivery.'
    return ''


def inherited_image_readings(blocks, root):
    """Recover one immutable blind reading per image origin after block splitting."""
    import hashlib

    from app.pipeline.evidence import EvidenceStore

    result = {}
    for block in blocks:
        payload = block.payload
        if not (eid := payload.get('agentic_relationship_evidence_id')):
            continue
        origin = payload.get('agentic_image_origin')
        source = root / payload.get('agentic_source_image', '')
        if not origin or not source.resolve().is_relative_to((root / 'agentic/evidence').resolve()) or not source.is_file():
            raise ValueError('Inherited image relationship source is unavailable')
        store = EvidenceStore(root, payload.get('agentic_source_sha256'))
        evidence = store.get(eid, block_id=origin)
        if (evidence['kind'] != 'image_relationship_reading'
                or evidence['page_idx'] not in payload.get('agentic_source_pages', [])
                or evidence.get('source', {}).get('sha256') != hashlib.sha256(source.read_bytes()).hexdigest()):
            raise ValueError('Inherited image relationship evidence belongs to a different source')
        reading = WordPictureItem.model_validate(evidence['content'])
        if origin in result and result[origin] != reading:
            raise ValueError('Conflicting inherited image relationship evidence')
        result[origin] = reading
    return result


def relationship_group_issues(reading, findings, delivered, *, documents, chunks):
    """A split image must account for each relation exactly once across its children.

    Each check's quotation still belongs to the child claiming it. Renaming the
    old image to text blocks cannot make a missing/opposite relationship pass.
    """
    expected = {r.relation_id: r for r in reading.relationships}
    checks = [(finding, check) for finding in findings for check in finding.image_relationship_checks
              if check.relation_id in expected]
    ids = [check.relation_id for _, check in checks]
    if not reading.relationships_complete or len(ids) != len(set(ids)) or set(ids) != set(expected):
        detail = 'Split image relationship coverage is incomplete, duplicated or has unknown IDs.'
        return {finding.block_id: detail for finding in findings}
    issues = {}
    for finding, check in checks:
        subset = reading.model_copy(update={'relationships': [expected[check.relation_id]]})
        checked = finding.model_copy(update={'image_relationship_checks': [check]})
        issue = relationship_issue(subset, checked, delivered.get(finding.block_id, ''),
                                   documents=documents, chunks=chunks)
        if issue:
            issues[finding.block_id] = issue
    return issues


def apply_group_relationship_review(review, readings, scopes, delivered, *, documents, chunks):
    """Add conservative relationship failures without erasing the pixel audit."""
    issues = {}
    for origin, reading in readings.items():
        if origin not in scopes:
            continue
        findings = [f for f in review.blocks if f.block_id in scopes[origin]]
        issues.update(relationship_group_issues(reading, findings, delivered, documents=documents, chunks=chunks))
    for finding in review.blocks:
        if finding.block_id in issues:
            finding.status = 'needs_review'
            finding.context_complete = False
            finding.observation = issues[finding.block_id][:700]
    if issues:
        review.source_complete = False
        review.declared_gaps_only = False
        review.issues = (review.issues + [bid + ': ' + issue for bid, issue in issues.items()])[:30]
    return issues


def required_relationship_origins(owner, blocks, pages=None):
    """Existing typed obligations survive a tool change, even when IDs disappear."""
    relevant = {b.block_id for b in blocks} | {b.payload.get('agentic_image_origin') for b in blocks}
    required = {b.payload['agentic_image_origin'] for b in blocks
                if b.payload.get('agentic_relationship_evidence_id') and b.payload.get('agentic_image_origin')}
    for page, record in owner.state.get('pages', {}).items():
        if pages is not None and int(page) not in pages:
            continue
        required.update(origin for origin, reading in record.get('word_image_readings', {}).items()
                        if origin in relevant and 'relationships_complete' in reading
                        and (reading.get('relationships') or not reading['relationships_complete']))
    return required


def validate_relationship_preservation(owner, before, candidate, root, pages):
    """Whole-page fallback must not erase a known image relationship obligation."""
    required = required_relationship_origins(owner, before.blocks, pages)
    if not required:
        return
    inherited = inherited_image_readings(candidate.blocks, root)
    for origin in required:
        old, new = before.get_block(origin), candidate.get_block(origin)
        unchanged_source = (old is not None and new is not None and old.type.value == new.type.value == 'image'
            and old.payload.get('word_source') == new.payload.get('word_source')
            and old.payload.get('img_path') == new.payload.get('img_path'))
        if not unchanged_source and origin not in inherited:
            raise ValueError('image_relationship_provenance_lost: preserve original image or use repair_image with inherited source evidence')
