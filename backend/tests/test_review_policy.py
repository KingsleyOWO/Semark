"""The same partial evidence must yield the same stop and acceptance decision."""
import copy

import pytest

from app.pipeline.review_policy import dependency_closed_subset, partial_review_accepted
from app.pipeline.review_sources import needs_content_recovery


def partial_record():
    return {'status': 'needs_review', 'reason': 'source_delivery_mismatch',
            'block_ids': ['scene', 'gap'], 'declared_gap_ids': ['gap'],
            'review': {'source_complete': False, 'reading_order_correct': True, 'declared_gaps_only': True,
                       'blocks': [{'block_id': 'scene', 'status': 'verified', 'context_complete': True},
                                  {'block_id': 'gap', 'status': 'needs_review', 'context_complete': False}]}}


@pytest.mark.parametrize('damage', ['context', 'missing', 'duplicate', 'order', 'undeclared'])
def test_incomplete_partial_evidence_cannot_be_accepted_or_marked_terminal(damage):
    record = partial_record()
    review = record['review']
    assert partial_review_accepted(review, ['gap'], record['block_ids'])
    assert not needs_content_recovery(record)
    if damage == 'context':
        review['blocks'][0]['context_complete'] = False
    elif damage == 'missing':
        review['blocks'].pop(0)
    elif damage == 'duplicate':
        review['blocks'].append(copy.deepcopy(review['blocks'][0]))
    elif damage == 'order':
        review['reading_order_correct'] = False
    else:
        review['declared_gaps_only'] = False
    assert not partial_review_accepted(review, ['gap'], record['block_ids'])
    assert needs_content_recovery(record)


def test_only_a_declared_whole_image_gap_can_finish_without_usable_facts():
    record = partial_record()
    record['review']['blocks'].pop(0)
    assert partial_review_accepted(record['review'], ['gap'], ['gap'])
    assert not partial_review_accepted(record['review'], [], ['gap'])


def test_dependency_pruning_handles_chains_and_cycles_without_mutating_input():
    candidates = {'photo', 'diagram', 'label', 'caption'}
    deps = {'diagram': {'label'}, 'label': {'diagram', 'missing_unit'}, 'caption': {'diagram'}}
    assert dependency_closed_subset(candidates, deps) == {'photo'}
    assert len(candidates) == 4
