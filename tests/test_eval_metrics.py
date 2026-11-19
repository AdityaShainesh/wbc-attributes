"""Tests for the evaluation metrics: ECE, abstention curves, restricted label spaces.

These are the numbers the project reports, so each test pins a property that would
otherwise be easy to get subtly wrong and impossible to notice: ECE is 0 for a perfectly
calibrated model and 1 for a confidently wrong one, coverage falls as the threshold
rises, and a split that contains only some of the five classes is scored over the
classes it actually has -- without ever restricting what the model is allowed to predict.
"""
import numpy as np
import pytest

from src.metrics import (
    abstention_curve,
    classification_metrics,
    expected_calibration_error,
    restrict_to_present_classes,
)

CLASSES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]


def one_hot(index, n=5, confidence=1.0):
    row = np.full(n, (1.0 - confidence) / (n - 1))
    row[index] = confidence
    return row


# ----------------------------------------------------------------------- ECE

def test_perfectly_calibrated_certain_predictions_have_zero_ece():
    probs = np.array([one_hot(i) for i in range(5)])
    assert expected_calibration_error(probs, list(range(5)), bins=15) == pytest.approx(0.0)


def test_confidently_wrong_predictions_have_ece_near_one():
    probs = np.array([one_hot(0) for _ in range(10)])
    ece = expected_calibration_error(probs, [1] * 10, bins=15)
    assert ece == pytest.approx(1.0, abs=1e-6)


def test_ece_matches_a_hand_computed_case():
    # Two samples at confidence 0.6: one right, one wrong -> accuracy 0.5 in that bin,
    # mean confidence 0.6, so ECE = |0.5 - 0.6| = 0.1
    probs = np.array([one_hot(0, confidence=0.6), one_hot(0, confidence=0.6)])
    assert expected_calibration_error(probs, [0, 1], bins=15) == pytest.approx(0.1, abs=1e-9)


def test_ece_is_bounded_and_finite_for_random_input():
    rng = np.random.RandomState(0)
    logits = rng.rand(200, 5)
    probs = logits / logits.sum(axis=1, keepdims=True)
    ece = expected_calibration_error(probs, rng.randint(0, 5, 200), bins=15)
    assert 0.0 <= ece <= 1.0


def test_empty_bins_do_not_contribute():
    # All confidences land in one bin; the other 14 are empty and must be skipped,
    # not counted as zero-error mass.
    probs = np.array([one_hot(0, confidence=0.95) for _ in range(20)])
    assert expected_calibration_error(probs, [0] * 20, bins=15) == pytest.approx(0.05, abs=1e-9)


# ---------------------------------------------------------------- abstention

def test_coverage_falls_as_the_threshold_rises():
    rng = np.random.RandomState(0)
    probs = rng.dirichlet(np.ones(5), size=100)
    curve = abstention_curve(probs, rng.randint(0, 5, 100), CLASSES,
                             thresholds=[0.0, 0.3, 0.6, 0.9])
    coverages = [point["coverage"] for point in curve]
    assert coverages == sorted(coverages, reverse=True)
    assert coverages[0] == pytest.approx(1.0)


def test_accuracy_rises_with_the_threshold_when_confidence_is_informative():
    # Confident predictions are correct, unconfident ones are not.
    probs = np.array([one_hot(0, confidence=0.99) for _ in range(50)]
                     + [one_hot(0, confidence=0.3) for _ in range(50)])
    targets = [0] * 50 + [1] * 50
    curve = {p["threshold"]: p for p in
             abstention_curve(probs, targets, CLASSES, thresholds=[0.0, 0.5])}
    assert curve[0.0]["accuracy"] == pytest.approx(0.5)
    assert curve[0.5]["accuracy"] == pytest.approx(1.0)
    assert curve[0.5]["coverage"] == pytest.approx(0.5)


def test_a_threshold_that_abstains_on_everything_reports_zero_coverage():
    probs = np.array([one_hot(0, confidence=0.5) for _ in range(10)])
    point = abstention_curve(probs, [0] * 10, CLASSES, thresholds=[0.99])[0]
    assert point["coverage"] == 0.0
    assert point["accuracy"] is None      # undefined, not silently 0.0
    assert point["n_covered"] == 0


# ------------------------------------------------- restricted label spaces

def test_restriction_scores_only_the_classes_the_split_contains():
    # Raabin Test-B: lymphocyte and neutrophil only.
    present = ["lymphocyte", "neutrophil"]
    targets = [0, 0, 1, 1]           # neutrophil, neutrophil, lymphocyte, lymphocyte
    predictions = [0, 0, 1, 1]
    metrics = restrict_to_present_classes(
        classification_metrics(targets, predictions, CLASSES), present)
    assert set(metrics["per_class"]) == set(present)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["n_classes"] == 2


def test_predictions_outside_the_present_classes_still_count_as_errors():
    """The model may predict all five; it is not told the target's label space.

    Restricting the model's outputs would use knowledge of the target domain, which
    CLAUDE.md section 3 forbids. So an absent-class prediction is simply wrong.
    """
    present = ["lymphocyte", "neutrophil"]
    targets = [1, 1, 0]               # lymphocyte, lymphocyte, neutrophil
    predictions = [1, 2, 0]           # the middle one is monocyte -- impossible here
    metrics = restrict_to_present_classes(
        classification_metrics(targets, predictions, CLASSES), present)
    assert metrics["per_class"]["lymphocyte"]["recall"] == pytest.approx(0.5)
    assert metrics["per_class"]["neutrophil"]["recall"] == pytest.approx(1.0)
    assert metrics["out_of_space_predictions"] == 1


def test_restriction_refuses_an_unknown_class_name():
    metrics = classification_metrics([0], [0], CLASSES)
    with pytest.raises(ValueError, match="platelet"):
        restrict_to_present_classes(metrics, ["platelet"])


def test_restriction_refuses_a_declared_class_with_no_support():
    """Catches configs/class_mapping.yaml disagreeing with the data about a split.

    Every class appears in per_class with zero support, so a declared-but-absent class
    would otherwise average in as a silent zero and drag the macro score down.
    """
    metrics = classification_metrics([0], [0], CLASSES)   # neutrophil only
    with pytest.raises(ValueError, match="basophil.*zero support"):
        restrict_to_present_classes(metrics, ["neutrophil", "basophil"])


def test_restriction_is_a_no_op_when_every_class_is_present():
    metrics = classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], CLASSES)
    restricted = restrict_to_present_classes(metrics, CLASSES)
    assert restricted["macro_f1"] == metrics["macro_f1"]
    assert restricted["n_classes"] == 5
