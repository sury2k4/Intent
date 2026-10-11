from __future__ import annotations

import pytest

from model_intent.db import Database, require_database_url, validate_database_target


VALID_URL = "postgresql://model_intent:secret@127.0.0.1:5433/model_intent_unified"


def test_database_url_must_be_explicit(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="explicit DATABASE_URL"):
        Database()


@pytest.mark.parametrize("url", [
    "postgresql://model_intent:secret@127.0.0.1:5432/model_intent",
    "postgresql://model_intent:secret@localhost:5433/model_intent_unified",
    "postgresql://model_intent:secret@127.0.0.1:5433/model_intent",
    "postgresql://model_intent:secret@127.0.0.1:5433/model_intent_unified?hostaddr=127.0.0.2",
    "postgresql://model_intent:secret@127.0.0.1:5433/model_intent_unified?port=5432",
    "postgresql://model_intent:secret@127.0.0.1:5433/model_intent_unified?dbname=model_intent",
    "postgresql://127.0.0.1,127.0.0.2:5433/model_intent_unified",
    "host=127.0.0.1 port=5433 dbname=model_intent_unified",
    "postgresql://model_intent:secret@127.0.0.1:notaport/model_intent_unified",
])
def test_database_url_rejects_non_unified_target_without_leaking_credentials(url):
    with pytest.raises(RuntimeError, match="isolated unified database") as error:
        Database(database_url=url)
    assert "secret" not in str(error.value)


def test_configured_db_backends_require_unified_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for config in ({"retrieval_backend": "pgvector"}, {"require_explicit_database_url": True}):
        with pytest.raises(RuntimeError, match="explicit DATABASE_URL"):
            validate_database_target(config)
    assert validate_database_target({"retrieval_backend": "numpy"}) is None


def test_unified_url_is_accepted_but_explicit_invalid_url_never_falls_back(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", VALID_URL)
    assert require_database_url() == VALID_URL
    assert validate_database_target({"retrieval_backend": "pgvector"}) == VALID_URL
    with pytest.raises(RuntimeError, match="explicit DATABASE_URL"):
        require_database_url("")
