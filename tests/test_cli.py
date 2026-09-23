from model_intent.cli import main


def test_cli_taxonomy_validate(capsys):
    assert main(["taxonomy", "validate"]) == 0
    assert '"status": "ok"' in capsys.readouterr().out


def test_cli_fake_query(capsys):
    assert main(["query", "--model", "fake", "--config", "configs/benchmark.fake.yaml", "--query", "đau đầu cần mua thuốc"]) == 0
    output = capsys.readouterr().out
    assert '"prediction"' in output
    assert '"decision"' in output
