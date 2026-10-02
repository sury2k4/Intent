import pytest

from model_intent.cli import main


def test_cli_taxonomy_validate(capsys):
    assert main(["taxonomy", "validate"]) == 0
    assert '"status": "ok"' in capsys.readouterr().out


def test_cli_fake_query(capsys):
    assert main(["query", "--model", "fake", "--config", "configs/benchmark.fake.yaml", "--taxonomy", "data/taxonomies/place_types.poc.v1.json", "--query", "đau đầu cần mua thuốc"]) == 0
    output = capsys.readouterr().out
    assert '"prediction"' in output
    assert '"decision"' in output


def test_cli_validates_poi_contracts(capsys):
    assert main(["pois", "validate"]) == 0
    output = capsys.readouterr().out
    assert '"status": "ok"' in output
    assert '"complete": true' in output


def test_cli_search_parser_has_separate_poi_limit():
    from model_intent.cli import build_parser

    args = build_parser().parse_args(["search", "--query", "cafe", "--poi-top-k", "7", "--model", "fake"])
    assert args.poi_top_k == 7
    assert args.category_top_k == 5


def test_serving_cli_defaults_to_unified_e5_taxonomy():
    from pathlib import Path

    from model_intent.cli import build_parser

    parser = build_parser()
    for command in ("query", "search"):
        args = parser.parse_args([command, "--query", "cafe"])
        assert Path(args.config).name == "benchmark.poi.pg.v3.yaml"
        assert Path(args.taxonomy).name == "place_types.poc.v3.json"
        assert args.model == "e5-base-multilingual"
    args = parser.parse_args(["embeddings", "sync"])
    assert Path(args.config).name == "benchmark.poi.pg.v3.yaml"
    assert Path(args.taxonomy).name == "place_types.poc.v3.json"


def test_promotion_cli_parses_explicit_pair_and_staging_flags():
    from model_intent.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["embeddings", "sync", "--no-activate"])
    assert args.no_activate is True
    assert parser.parse_args(["embeddings", "sync"]).no_activate is False
    with pytest.raises(SystemExit):
        parser.parse_args(["embeddings", "sync", "--activate"])
    poi_args = parser.parse_args(["pois", "sync", "--manifest", "manifest.json", "--no-activate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["pois", "sync", "--manifest", "manifest.json", "--activate"])
    assert poi_args.no_activate is True
    promote_args = parser.parse_args([
        "pois", "promote", "--embedding-generation-id", "21", "--poi-generation-id", "22",
        "--config", "configs/benchmark.poi.pg.v3.yaml", "--taxonomy", "taxonomy.json",
        "--rollback-record", "rollback.json",
    ])
    assert promote_args.embedding_generation_id == 21
    assert promote_args.poi_generation_id == 22
    assert promote_args.rollback_record == "rollback.json"
    assert parser.parse_args(["pois", "rollback", "--record", "rollback.json"]).record == "rollback.json"


def test_promotion_cli_dispatches_config_provenance(monkeypatch, capsys, tmp_path):
    from model_intent import cli

    class OfflineDatabase:
        def __init__(self, **kwargs):
            pass

        def open(self):
            pass

        def close(self):
            pass

    recorded = {}

    def promote(_, embedding_generation_id, poi_generation_id, **kwargs):
        recorded.update(kwargs, embedding_generation_id=embedding_generation_id,
                        poi_generation_id=poi_generation_id)
        return {"ok": True}

    monkeypatch.setattr(cli, "Database", OfflineDatabase)
    monkeypatch.setattr(cli, "_promotion_provenance", lambda *args: {
        "taxonomy_fingerprint": "a" * 64, "mapping_sha256": "b" * 64,
        "property_taxonomy_sha256": "c" * 64, "model_config_hash": "d" * 64,
        "source_fingerprint": "e" * 64, "embedding_content_set_hash": "f" * 64,
        "renderer_version": "test-renderer", "profile_schema_version": 3,
        "expected_profile_count": 2, "expected_category_count": 1,
    })
    monkeypatch.setattr(cli, "promote_generation_pair", promote)
    assert main([
        "pois", "promote", "--embedding-generation-id", "21", "--poi-generation-id", "22",
        "--rollback-record", str(tmp_path / "rollback.json"),
    ]) == 0
    assert recorded["embedding_generation_id"] == 21
    assert recorded["poi_generation_id"] == 22
    assert recorded["mapping_sha256"] == "b" * 64
    assert '"ok": true' in capsys.readouterr().out


def test_promotion_provenance_rejects_mismatched_pinned_taxonomy(monkeypatch, tmp_path):
    import pytest

    from model_intent import cli

    pinned = tmp_path / "pinned.json"
    other = tmp_path / "other.json"
    config_path = tmp_path / "configs" / "serving.yaml"
    monkeypatch.setattr(cli, "load_yaml", lambda *args: {
        "retrieval_backend": "pgvector", "taxonomy_path": str(pinned),
        "poi_search": {"backend": "postgres", "dataset_key": "hcmc-osm", "mapping_path": "mapping.json",
                       "property_path": "property.json"},
    })
    with pytest.raises(ValueError, match="taxonomy_path"):
        cli._promotion_provenance(str(config_path), str(other), "hcmc-osm")


def test_unilateral_activation_and_cleanup_cli_fail_before_database_construction(monkeypatch, capsys):
    from model_intent import cli

    def forbidden_database(**_):
        raise AssertionError("must not construct Database for unilateral activate/cleanup")

    monkeypatch.setattr(cli, "Database", forbidden_database)
    for args in (
        ["embeddings", "activate", "17"], ["embeddings", "cleanup"],
        ["pois", "activate", "17"],
    ):
        assert main(args) == 2
        assert "disabled" in capsys.readouterr().err


def test_sync_cli_stages_by_default(monkeypatch, capsys):
    from model_intent import cli
    from types import SimpleNamespace

    class OfflineDatabase:
        def __init__(self, **_):
            pass

        def open(self):
            pass

        def close(self):
            pass

    seen = []
    monkeypatch.setattr(cli, "Database", OfflineDatabase)
    monkeypatch.setattr(cli, "sync_embeddings", lambda *args, **kwargs: (
        seen.append(kwargs["activate"]) or SimpleNamespace(active=False)
    ))
    monkeypatch.setattr(cli, "sync_poi_artifact", lambda *args, **kwargs: (
        seen.append(kwargs["activate"]) or {"activated": False}
    ))
    for args in (
        ["embeddings", "sync"], ["embeddings", "sync", "--no-activate"],
        ["pois", "sync", "--manifest", "unused.json"],
        ["pois", "sync", "--manifest", "unused.json", "--no-activate"],
    ):
        assert main(args) == 0
        capsys.readouterr()
    assert seen == [False] * 4


def test_all_database_cli_commands_reject_legacy_target_before_connection(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://model_intent@127.0.0.1:5432/model_intent")
    for args in (
        ["db", "status"], ["db", "migrate"],
        ["embeddings", "status"], ["embeddings", "sync"],
        ["pois", "status"], ["pois", "sync", "--manifest", "ignored.json"],
        ["pois", "promote", "--embedding-generation-id", "1", "--poi-generation-id", "2", "--rollback-record", "unused.json"],
        ["pois", "rollback", "--record", "unused.json"],
        ["query", "--query", "cafe"], ["search", "--query", "cafe"],
    ):
        assert main(args) == 2
        assert "isolated unified database" in capsys.readouterr().err
