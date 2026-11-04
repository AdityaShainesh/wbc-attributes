"""Tests for src/metrics.py.

CLAUDE.md section 6: macro-F1 and per-class recall are primary, accuracy alone is never
an acceptable summary, and basophil/eosinophil recall must be separable. These tests pin
the orientation of the confusion matrix and the zero-support behaviour, because getting
either wrong produces numbers that look plausible and are wrong.
"""
import numpy as np
import pytest

from src.metrics import classification_metrics, confusion_matrix

CLASSES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]


def test_confusion_matrix_rows_are_true_classes():
    # one true 0 predicted as 1, nothing else
    cm = confusion_matrix([0], [1], n_classes=2)
    assert cm[0, 1] == 1
    assert cm[1, 0] == 0


def test_confusion_matrix_counts_every_sample():
    targets = [0, 1, 2, 2, 4]
    predictions = [0, 1, 2, 0, 4]
    cm = confusion_matrix(targets, predictions, n_classes=5)
    assert cm.sum() == len(targets)
    assert np.trace(cm) == 4


def test_perfect_predictions_give_one_everywhere():
    targets = [0, 1, 2, 3, 4]
    m = classification_metrics(targets, targets, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert all(p["recall"] == 1.0 for p in m["per_class"].values())


def test_macro_f1_is_unweighted_so_imbalance_hurts_it():
    # 90 neutrophils, 10 basophils, model always says neutrophil.
    targets = [0] * 90 + [4] * 10
    predictions = [0] * 100
    m = classification_metrics(targets, predictions, CLASSES)
    assert m["accuracy"] == pytest.approx(0.90)
    # Three classes have no support, one is perfect, one is zero -> macro-F1 well below
    # accuracy. This gap is the whole reason accuracy is banned as a summary.
    assert m["macro_f1"] < 0.25
    assert m["per_class"]["basophil"]["recall"] == 0.0
    assert m["per_class"]["neutrophil"]["recall"] == 1.0


def test_rare_class_recall_is_reported_separately():
    targets = [0] * 10 + [3] * 2 + [4] * 2
    predictions = [0] * 10 + [3, 0] + [4, 0]
    m = classification_metrics(targets, predictions, CLASSES)
    assert m["per_class"]["eosinophil"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["basophil"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["eosinophil"]["support"] == 2


def test_class_with_no_support_has_zero_recall_and_is_flagged():
    targets = [0, 0, 1]
    predictions = [0, 0, 1]
    m = classification_metrics(targets, predictions, CLASSES)
    for absent in ("monocyte", "eosinophil", "basophil"):
        assert m["per_class"][absent]["support"] == 0
        assert m["per_class"][absent]["recall"] == 0.0


def test_class_never_predicted_has_zero_precision_not_a_crash():
    targets = [0, 1]
    predictions = [0, 0]
    m = classification_metrics(targets, predictions, CLASSES)
    assert m["per_class"]["lymphocyte"]["precision"] == 0.0
    assert m["per_class"]["lymphocyte"]["f1"] == 0.0


def test_macro_recall_is_the_mean_of_per_class_recalls():
    targets = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    m = classification_metrics(targets, predictions, CLASSES)
    recalls = [p["recall"] for p in m["per_class"].values()]
    assert m["macro_recall"] == pytest.approx(sum(recalls) / len(recalls))


def test_metrics_accept_tensors_and_arrays():
    import torch
    targets = torch.tensor([0, 1, 2])
    predictions = np.array([0, 1, 2])
    m = classification_metrics(targets, predictions, CLASSES)
    # Every present class is perfect; the two absent ones still contribute 0 to the
    # macro average, so macro_f1 is 3/5. Same convention as the zero-support test.
    assert [m["per_class"][c]["f1"] for c in CLASSES[:3]] == [1.0, 1.0, 1.0]
    assert m["macro_f1"] == pytest.approx(0.6)
    assert m["accuracy"] == 1.0


def test_confusion_matrix_is_json_serializable_in_the_metrics_dict():
    import json
    m = classification_metrics([0, 1], [0, 1], CLASSES)
    json.dumps(m)  # raises if numpy types leaked through
