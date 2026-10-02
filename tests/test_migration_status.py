from __future__ import annotations

from contextlib import contextmanager

from model_intent.migrations import discover_migrations, migration_status


class ReadOnlyConnection:
    def __init__(self, ledger=None, applied=()):
        self.ledger = ledger
        self.applied = applied
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        assert statement.startswith("SELECT ")
        return self

    def fetchone(self):
        return (self.ledger,)

    def fetchall(self):
        return self.applied


class StubDatabase:
    def __init__(self, connection):
        self.stub = connection

    @contextmanager
    def connection(self):
        yield self.stub, None


def test_missing_migration_ledger_is_observed_without_creation(tmp_path):
    (tmp_path / "001_initial.sql").write_text("SELECT 1;", encoding="utf-8")
    connection = ReadOnlyConnection()
    status = migration_status(StubDatabase(connection), tmp_path)
    assert len(status) == 1
    assert status[0]["applied"] is False
    assert status[0]["checksum_matches"] is True
    assert connection.statements == ["SELECT to_regclass('schema_migrations')"]


def test_migration_status_flags_changed_applied_checksum_without_writing(tmp_path):
    (tmp_path / "001_initial.sql").write_text("SELECT 1;", encoding="utf-8")
    migration = discover_migrations(tmp_path)[0]
    connection = ReadOnlyConnection("schema_migrations", [(migration.version, "0" * 64)])
    status = migration_status(StubDatabase(connection), tmp_path)
    assert status[0]["applied"] is True
    assert status[0]["checksum_matches"] is False
    assert connection.statements == [
        "SELECT to_regclass('schema_migrations')",
        "SELECT version, checksum FROM schema_migrations",
    ]


def test_cli_status_exits_nonzero_for_checksum_mismatch(monkeypatch, capsys):
    from model_intent import cli

    class Database:
        def __init__(self, **kwargs):
            pass

        def open(self):
            pass

        def check(self):
            return {"database": "model_intent_unified"}

        def close(self):
            pass

    monkeypatch.setattr(cli, "Database", Database)
    monkeypatch.setattr(cli, "migration_status", lambda *args: [{"checksum_matches": False}])
    assert cli.main(["db", "status"]) == 2
    assert '"checksum_matches": false' in capsys.readouterr().out
