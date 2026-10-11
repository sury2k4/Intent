from pathlib import Path
from types import SimpleNamespace

import pytest

from model_intent import cli
from model_intent.cli import build_parser, main


def test_admin_cli_defaults_pin_intent_serving_resources():
    parser = build_parser()
    category = parser.parse_args(["embeddings", "sync"])
    assert Path(category.config).name == "benchmark.poi.pg.v3.yaml"
    assert Path(category.taxonomy).name == "place_types.poc.v3.json"
    semantic = parser.parse_args(["properties", "sync"])
    assert Path(semantic.semantic).name == "property_semantic_profiles.poc.v2.json"
    assert Path(semantic.config).name == "benchmark.poi.pg.v3.yaml"


def test_cli_only_exposes_generation_administration():
    parser = build_parser()
    for command in ("query", "search", "pois", "benchmark", "taxonomy"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])
    promote = parser.parse_args(["embeddings", "promote", "21", "--rollback-record", "rollback.json"])
    assert promote.generation_id == 21
    assert promote.rollback_record == "rollback.json"
    assert parser.parse_args(["embeddings", "rollback", "--record", "rollback.json"]).record == "rollback.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["embeddings", "sync", "--activate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["db", "migrate"])


class OfflineDatabase:
    def __init__(self, **_):
        pass

    def open(self):
        pass

    def close(self):
        pass


def test_category_promotion_cli_dispatches_to_admin_module(monkeypatch, capsys, tmp_path):
    from model_intent import category_promotion

    recorded = {}

    def promote(_, generation_id, **kwargs):
        recorded.update(kwargs, generation_id=generation_id)
        return {"promoted_generation_id": generation_id}

    monkeypatch.setattr(cli, "Database", OfflineDatabase)
    monkeypatch.setattr(category_promotion, "promote_category_generation", promote)
    assert main(["embeddings", "promote", "21", "--rollback-record", str(tmp_path / "rollback.json")]) == 0
    assert recorded["generation_id"] == 21
    assert Path(recorded["config_path"]).name == "benchmark.poi.pg.v3.yaml"
    assert '"promoted_generation_id": 21' in capsys.readouterr().out


def test_category_sync_stages_without_activating(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "Database", OfflineDatabase)
    monkeypatch.setattr(cli, "sync_embeddings", lambda *args, **kwargs: (
        seen.append(kwargs["activate"]) or SimpleNamespace(active=False)
    ))
    assert main(["embeddings", "sync"]) == 0
    assert seen == [False]
    assert '"active": false' in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ["db", "status"], ["embeddings", "status"], ["embeddings", "sync"],
    ["embeddings", "promote", "21", "--rollback-record", "unused.json"],
    ["embeddings", "rollback", "--record", "unused.json"],
    ["properties", "status"], ["properties", "migrate"], ["properties", "sync"],
])
def test_database_admin_commands_refuse_legacy_target(monkeypatch, capsys, args):
    monkeypatch.setenv("DATABASE_URL", "postgresql://model_intent@127.0.0.1:5432/model_intent")
    assert main(args) == 2
    assert "isolated unified database" in capsys.readouterr().err
