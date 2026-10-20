import pytest
from fastapi import HTTPException

from api.chat import require_admin


def test_admin_api_requires_configured_secret(monkeypatch):
    monkeypatch.delenv("EVOLVEAI_ADMIN_TOKEN", raising=False)
    with pytest.raises(HTTPException) as exc:
        require_admin("anything")
    assert exc.value.status_code == 503


def test_admin_api_rejects_wrong_secret(monkeypatch):
    monkeypatch.setenv("EVOLVEAI_ADMIN_TOKEN", "expected-secret")
    with pytest.raises(HTTPException) as exc:
        require_admin("wrong-secret")
    assert exc.value.status_code == 401


def test_admin_api_accepts_matching_secret(monkeypatch):
    monkeypatch.setenv("EVOLVEAI_ADMIN_TOKEN", "expected-secret")
    assert require_admin("expected-secret") is None
