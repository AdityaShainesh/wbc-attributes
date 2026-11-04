"""Tests for src/train.py.

The leakage guards run first and need no data, so they are fast. The end-to-end smoke
run is marked slow: it is the 500-image, 2-epoch run CLAUDE.md section 7 requires before
any full run, and it downloads pretrained weights on first use.

Run everything:      pytest
Skip the slow run:   pytest -m "not slow"
"""
import json

import pytest
import yaml

from src.data.pbc import PBC_DIR
from src.train import LeakageError, check_no_target_leakage, main, run_from_config
from src.utils.config import ConfigError

SMOKE = "configs/smoke.yaml"


def smoke_config(**overrides):
    config = yaml.safe_load(open(SMOKE, encoding="utf-8"))
    config.update(overrides)
    return config


# -------------------------------------------------------------- leakage guards

def test_training_on_a_target_domain_is_refused():
    config = smoke_config()
    config["data"]["dataset"] = "lisc"
    config["data"]["split"] = "Train"
    with pytest.raises(LeakageError, match="lisc"):
        check_no_target_leakage(config)


def test_selecting_the_checkpoint_on_a_target_domain_is_refused():
    config = smoke_config()
    config["train"]["selection"] = {"dataset": "lisc", "split": "Test"}
    with pytest.raises(LeakageError, match="lisc"):
        check_no_target_leakage(config)


def test_evaluating_on_a_target_domain_during_training_is_refused():
    config = smoke_config()
    config["eval"]["datasets"] = [{"dataset": "lisc", "split": "Test"}]
    with pytest.raises(LeakageError, match="lisc"):
        check_no_target_leakage(config)


def test_a_source_only_config_passes_the_guard():
    assert check_no_target_leakage(smoke_config()) is True


def test_guard_runs_before_any_data_is_loaded(monkeypatch):
    # If the guard ran after dataset construction, a leaking config would read target
    # images before failing. Prove it fails without touching the data layer at all.
    import src.data

    def explode(*_args, **_kwargs):
        raise AssertionError("dataset was built before the leakage guard ran")

    monkeypatch.setattr(src.data, "build_dataset", explode)
    config = smoke_config()
    config["data"]["dataset"] = "lisc"
    with pytest.raises(LeakageError):
        run_from_config(config)


# ------------------------------------------------------------- config contract

def test_missing_required_key_is_reported_not_defaulted():
    config = smoke_config()
    del config["train"]
    with pytest.raises(ConfigError, match="train"):
        run_from_config(config)


def test_seeds_must_be_a_non_empty_list():
    with pytest.raises(ConfigError, match="seeds"):
        run_from_config(smoke_config(seeds=[]))


# ------------------------------------------------------------------- smoke run

@pytest.mark.slow
@pytest.mark.skipif(not PBC_DIR.exists(), reason="data/raw/ not populated")
def test_smoke_run_writes_a_complete_results_file(tmp_path):
    config = smoke_config(run_id="pytest_smoke")
    record = run_from_config(config, results_dir=tmp_path)

    path = tmp_path / "pytest_smoke.json"
    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == record

    # CLAUDE.md section 5: the resolved config is embedded, not referenced.
    assert written["config"]["run_id"] == "pytest_smoke"
    assert written["config"]["data"]["subset_size"] == 500
    assert "commit" in written["git"] and "dirty" in written["git"]
    assert written["hardware"]["torch"]
    assert written["wall_clock_s"] > 0

    # One entry per seed, plus an aggregate with mean and std.
    assert list(written["seeds"]) == [str(s) for s in config["seeds"]]
    for seed_record in written["seeds"].values():
        assert len(seed_record["history"]) == config["train"]["epochs"]
        assert seed_record["best_epoch"] >= 1
        for split_metrics in seed_record["metrics"].values():
            assert "macro_f1" in split_metrics
            assert set(split_metrics["per_class"]) == set(record["classes"])
    assert "macro_f1" in written["aggregate"]
    assert set(written["aggregate"]["macro_f1"]) == {"mean", "std", "values"}


@pytest.mark.slow
@pytest.mark.skipif(not PBC_DIR.exists(), reason="data/raw/ not populated")
def test_main_runs_from_the_command_line_contract(tmp_path, monkeypatch):
    # CLAUDE.md section 5: python -m src.train --config <x> and nothing else.
    monkeypatch.setattr("sys.argv", ["src.train", "--config", SMOKE])
    monkeypatch.setenv("WBC_RESULTS_DIR", str(tmp_path))
    main()
    assert (tmp_path / "smoke.json").exists()
