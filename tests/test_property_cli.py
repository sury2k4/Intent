from model_intent.cli import build_parser
from model_intent.property_embedding_sync import PROPERTY_MIGRATIONS


def test_property_commands_stage_by_default_with_scoped_migration():
    parser = build_parser()
    migrate = parser.parse_args(["properties", "migrate"])
    assert migrate.command == "properties" and migrate.properties_command == "migrate"
    assert PROPERTY_MIGRATIONS.name == "property"
    status = parser.parse_args(["properties", "status"])
    assert status.properties_command == "status"
    staged = parser.parse_args(["properties", "sync"])
    assert staged.properties_command == "sync"
    assert staged.batch_size == 16
    assert "property_semantic_profiles.poc.v2.json" in staged.semantic
    assert "benchmark.poi.pg.v3.yaml" in staged.config
