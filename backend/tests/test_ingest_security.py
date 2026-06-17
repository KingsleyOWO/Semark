from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes.ingest import ingest_document
from app.config import settings


@pytest.mark.asyncio
async def test_local_path_ingest_is_guarded_by_setting(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_path_ingest", False)

    with pytest.raises(HTTPException) as exc:
        await ingest_document(SimpleNamespace(path="/tmp/example.pdf"), db=object())

    assert exc.value.status_code == 403
    assert "Local path ingestion is disabled" in exc.value.detail
