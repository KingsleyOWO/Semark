"""Response-contract slips that are not source evidence must not burn a model call.

Lumio v20: 19 of 113 review calls were thrown away for an echoed request
field, an over-long explanation, or a spatial relation without a side.
"""
import pytest
from pydantic import ValidationError
from test_document_recovery import RecoveryModel, setup

from app.pipeline.image_reading import ImageParts, ImageText, read_contract
from app.pipeline.image_source_review import (
    WordPictureItem,
    relationship_issue,
)
from app.pipeline.recovery_models import ImageEvidenceBatchSelection

PARTS = {'image_type': 'photo', 'visual_facts': ['店內有陳列貨架。'], 'visual_independent_of_text': True,
         'text_regions': [{'region_id': 'sign', 'bbox': [0, 0, 1000, 200], 'readable': True, 'text': '示範書店'}],
         'complete': True, 'observation': 'A shop with a sign.'}


@pytest.mark.asyncio
async def test_request_fields_copied_into_the_answer_are_ignored(tmp_path):
    class Echo(RecoveryModel):
        async def query(self, task, payload, image, schema):
            return {**PARTS, 'source_id': payload['source_id'], 'output_language': payload['output_language']}, 10
    _, owner, runner = await setup(tmp_path, Echo())

    parts = await read_contract(owner, 'verify_image_parts', {'source_id': 'original-image', 'output_language': 'zh-TW'},
                                runner.inventory.image(0), ImageParts, max_attempts=1)

    assert parts.text_regions[0].text == '示範書店'
    assert owner.state['usage']['model_calls'] == 1


@pytest.mark.asyncio
async def test_the_production_model_client_ignores_copied_request_fields():
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.config import VLMApiMode
    from app.pipeline.agentic import RegionModel

    request = {'source_id': 'original-image', 'output_language': 'zh-TW'}
    answer = json.dumps({**PARTS, **request}, ensure_ascii=False)
    adapter = SimpleNamespace(
        config=SimpleNamespace(model='test', api_mode=VLMApiMode.OLLAMA),
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=object(), finish_reason='stop')], usage=SimpleNamespace(total_tokens=10)))))),
        _message_text=lambda _: answer, _extract_json_payload=lambda v: v,
        _encode_image_as_data_uri=lambda _: 'data:image/png;base64,')

    data, _ = await RegionModel(adapter, 4096).query('verify_image_parts', request, None, ImageParts)

    assert data['text_regions'][0]['text'] == '示範書店' and 'source_id' not in data


def test_an_over_long_explanation_is_shortened_but_content_never_is():
    selection = ImageEvidenceBatchSelection.model_validate(
        {'source_ref': 'e1', 'decision': 'supported', 'reason': '可見文字。' * 60})

    assert len(selection.reason) <= 120
    with pytest.raises(ValidationError):
        ImageText.model_validate({'region_id': 'r', 'bbox': [0, 0, 1000, 1000], 'readable': True, 'text': '字' * 1601})


def test_a_position_without_a_side_does_not_reject_the_reading_nor_certify_a_chart():
    # The sideless position is kept instead of failing the whole reading; on a chart
    # with axes it stays uncertain (descriptions elsewhere: test_image_review_false_failures).
    reading = WordPictureItem.model_validate(
        {'source_id': 'image0', 'text': 'LUMIO', 'complete': True, 'observation': 'Map.', 'image_type': 'diagram',
         'relationships': [
             {'relation_id': 'r0', 'kind': 'axis_mapping', 'source_fact': 'Horizontal axis: simple to bold.',
              'bbox': [0, 400, 1000, 600], 'certain': True},
             {'relation_id': 'r1', 'kind': 'spatial_position', 'source_fact': 'LUMIO sits near the axis.',
              'bbox': [0, 0, 1000, 500], 'certain': True}],
         'relationships_complete': True})

    relation = reading.relationships[1]
    assert (relation.horizontal, relation.vertical, relation.certain) == ('uncertain', 'uncertain', False)

    class Finding:
        block_id = 'b0'
        image_relationship_checks = [type('Check', (), {'relation_id': rid, 'status': 'matched', 'delivered_quote': 'LUMIO',
                                                         'direction': 'none', 'horizontal': 'unspecified',
                                                         'vertical': 'unspecified'})() for rid in ('r0', 'r1')]
    assert 'uncertain' in relationship_issue(reading, Finding(), 'LUMIO')
