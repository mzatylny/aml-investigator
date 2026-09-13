import json
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from aml.__main__ import GNN_OUTPUT_FILES, main, write_experiment
from aml.data import generate_transactions
from aml.gnn import load_graphsage
from aml.model import run_experiment, validate_model_pair


@pytest.fixture(scope="module")
def experiment():
    return run_experiment(generate_transactions(seed=3, days=10, accounts=30, daily_transactions=20),
                          seed=3, with_gnn=True, gnn_epochs=1)


def test_baseline_export_removes_old_gnn_files_and_preserves_user_files(experiment, tmp_path):
    write_experiment(experiment, tmp_path)
    assert all((tmp_path / name).is_file() for name in GNN_OUTPUT_FILES)
    (tmp_path / "notes.txt").write_text("Keep this note.")
    replacement = run_experiment(generate_transactions(seed=4, days=10, accounts=30, daily_transactions=20), seed=4)
    write_experiment(replacement, tmp_path)
    assert not any((tmp_path / name).exists() for name in GNN_OUTPUT_FILES)
    assert (tmp_path / "notes.txt").read_text() == "Keep this note."
    forest = joblib.load(tmp_path / "model.joblib")
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert forest["run_id"] == replacement.report["run_id"] == report["run_id"]
    assert forest["run_id"] != experiment.report["run_id"]
    assert "graphsage" not in report["test"]


@pytest.mark.parametrize("run_id", [None, "different-experiment"])
def test_missing_or_mismatched_model_identity_is_rejected(experiment, run_id):
    with pytest.raises(ValueError, match="same identified experiment"):
        validate_model_pair({"run_id": run_id}, experiment.gnn)


def test_cli_rejects_mismatched_checkpoints_before_scoring(experiment, tmp_path, monkeypatch, capsys):
    write_experiment(experiment, tmp_path)
    forest = joblib.load(tmp_path / "model.joblib")
    forest["run_id"] = "another-run"
    joblib.dump(forest, tmp_path / "model.joblib")
    monkeypatch.setattr(sys, "argv", ["aml", "score", "--input", str(tmp_path / "not-read.csv"),
        "--model", str(tmp_path / "model.joblib"), "--gnn-model", str(tmp_path / "graphsage.pt"),
        "--output", str(tmp_path / "scores.csv")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "same identified experiment" in capsys.readouterr().err
    assert not (tmp_path / "scores.csv").exists()


def test_matching_saved_pair_scores_identically_through_cli(experiment, tmp_path, monkeypatch):
    write_experiment(experiment, tmp_path)
    forest = joblib.load(tmp_path / "model.joblib")
    gnn = load_graphsage(tmp_path / "graphsage.pt")
    assert forest["run_id"] == gnn.run_id == experiment.report["run_id"]
    validate_model_pair(forest, gnn)
    columns = ["transaction_id", "timestamp", "sender", "receiver", "amount", "currency"]
    tx = experiment.transactions
    tx.iloc[:-20][columns].to_csv(tmp_path / "history.csv", index=False)
    tx.iloc[-20:][columns].to_csv(tmp_path / "new.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["aml", "score", "--input", str(tmp_path / "new.csv"),
        "--history", str(tmp_path / "history.csv"), "--model", str(tmp_path / "model.joblib"),
        "--gnn-model", str(tmp_path / "graphsage.pt"), "--output", str(tmp_path / "scores.csv")])
    main()
    scored = pd.read_csv(tmp_path / "scores.csv")
    np.testing.assert_allclose(scored.gnn_score, tx.gnn_score.iloc[-20:], atol=1e-6)
    np.testing.assert_allclose(scored.risk_score, tx.risk_score.iloc[-20:], atol=1e-6)
    assert scored.gnn_alert.tolist() == tx.gnn_alert.iloc[-20:].tolist()
