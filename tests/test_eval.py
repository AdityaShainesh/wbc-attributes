"""Tests for src/eval.py.

The guards here are the mirror image of the ones in test_train_smoke.py. Training must
refuse target data; evaluation is the one place target data is allowed -- but it must
still refuse the splits reserved for Model 4's domain head, and it must never restrict
what the model is allowed to predict.
"""
import pytest
import yaml

from src.eval import (
    EvalLeakageError,
    adaptation_splits,
    check_domains,
    present_classes_for,
    run_from_config,
)
from src.utils.config import ConfigError

EVAL_CONFIG = "configs/eval.yaml"


def config(**overrides):
    base = yaml.safe_load(open(EVAL_CONFIG, encoding="utf-8"))
    base.update(overrides)
    return base


# ------------------------------------------------------- adaptation splits

def test_raabin_train_is_recognised_as_adaptation_only():
    assert ("raabin", "Train") in adaptation_splits()
    assert ("lisc", "Train") in adaptation_splits()


def test_excluded_splits_are_also_refused():
    assert ("lisc", "Train_Aug") in adaptation_splits()


def test_evaluating_an_adaptation_split_is_refused():
    broken = config(domains=[{"dataset": "raabin", "split": "Train"}])
    with pytest.raises(EvalLeakageError, match="raabin/Train"):
        check_domains(broken)


def test_evaluating_an_excluded_split_is_refused():
    broken = config(domains=[{"dataset": "lisc", "split": "Train_Aug"}])
    with pytest.raises(EvalLeakageError, match="Train_Aug"):
        check_domains(broken)


def test_the_shipped_config_passes_the_guard():
    assert check_domains(config()) is True


def test_the_guard_names_every_offender_not_just_the_first():
    broken = config(domains=[{"dataset": "raabin", "split": "Train"},
                             {"dataset": "lisc", "split": "Train"}])
    with pytest.raises(EvalLeakageError) as exc:
        check_domains(broken)
    assert "raabin/Train" in str(exc.value) and "lisc/Train" in str(exc.value)


def test_guard_runs_before_any_checkpoint_is_read():
    broken = config(domains=[{"dataset": "raabin", "split": "Train"}],
                    checkpoints_dir="/nonexistent")
    # A missing checkpoints dir would also fail, so reaching EvalLeakageError proves
    # the leakage check came first.
    with pytest.raises(EvalLeakageError):
        run_from_config(broken)


# ------------------------------------------------------ class coverage

def test_test_b_declares_two_present_classes():
    assert present_classes_for("raabin", "Test-B") == ["lymphocyte", "neutrophil"]


def test_full_coverage_splits_declare_nothing_to_restrict():
    assert present_classes_for("raabin", "TestA") is None
    assert present_classes_for("lisc", "Test") is None


def test_source_split_has_no_restriction():
    assert present_classes_for("pbc", "test") is None


# ------------------------------------------------------------ config contract

def test_missing_runs_key_is_reported():
    broken = config()
    del broken["runs"]
    with pytest.raises(ConfigError, match="runs"):
        run_from_config(broken)


def test_shipped_config_evaluates_source_and_all_three_targets():
    domains = {(d["dataset"], d["split"]) for d in config()["domains"]}
    assert ("pbc", "test") in domains
    assert ("lisc", "Test") in domains
    assert ("raabin", "TestA") in domains
    assert ("raabin", "Test-B") in domains


def test_abstention_thresholds_are_shared_across_domains():
    """One grid for every domain; a per-domain grid would be tuning on target data."""
    thresholds = config()["abstention"]["thresholds"]
    assert thresholds == sorted(thresholds)
    assert thresholds[0] == 0.0          # includes full coverage as the reference point
