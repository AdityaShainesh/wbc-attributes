"""Classification metrics.

CLAUDE.md section 6: macro-F1 and per-class recall are primary and accuracy alone is
never an acceptable summary, because a model that always answers "neutrophil" scores 32%
accuracy on PBC and 0% recall on four classes. Every function here therefore returns
per-class numbers alongside any average.

Calibration (ECE) and accuracy-vs-coverage under abstention belong to the eval harness,
not here: they need probabilities rather than predictions, and they are reported once,
not every epoch.

Zero-support and never-predicted classes score 0.0 rather than raising or returning NaN,
matching scikit-learn's zero_division=0, and `support` is always reported so a 0.0 that
means "absent" can be told apart from one that means "wrong".
"""
import numpy as np


def _as_array(values):
    if hasattr(values, "detach"):          # torch tensor
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1).astype(int)


def confusion_matrix(targets, predictions, n_classes):
    """Rows are true classes, columns are predictions."""
    targets, predictions = _as_array(targets), _as_array(predictions)
    if targets.shape != predictions.shape:
        raise ValueError(f"shape mismatch: {targets.shape} targets vs {predictions.shape} predictions")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def classification_metrics(targets, predictions, classes):
    """Per-class recall, precision, F1 and support, plus macro averages and accuracy."""
    matrix = confusion_matrix(targets, predictions, len(classes))
    correct = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)

    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(correct), where=denominator > 0)
    total = float(matrix.sum())

    return {
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "per_class": {
            name: {
                "recall": float(recall[i]),
                "precision": float(precision[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
        "confusion_matrix": matrix.tolist(),
        "n": int(total),
    }


def aggregate_over_seeds(records, keys=("macro_f1", "macro_recall", "accuracy")):
    """mean / std / raw values for each metric across seeds.

    CLAUDE.md section 5: every reported number is mean +- std over 3 seeds. Population
    std (ddof=0) over exactly the seeds that ran -- with three of them, the distinction
    from the sample std matters enough to state.
    """
    out = {}
    for key in keys:
        values = [float(record[key]) for record in records if key in record]
        if not values:
            continue
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
    return out


# --------------------------------------------------------------- calibration

def _as_probabilities(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"expected a (n, classes) probability array, got shape {probs.shape}")
    return probs


def expected_calibration_error(probabilities, targets, bins=15):
    """ECE: average gap between confidence and accuracy, weighted by bin population.

    CLAUDE.md section 6 asks for this alongside macro-F1, because a model that is wrong
    about a target domain AND confident about it is far more dangerous in a clinical
    workflow than one that is wrong and uncertain -- the second can be sent to a human.

    Equal-width bins over [0, 1] on the max-softmax confidence. Empty bins contribute
    nothing: counting them as zero error would quietly shrink the reported ECE.
    """
    probabilities = _as_probabilities(probabilities)
    targets = _as_array(targets)
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == targets).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        # Upper edge inclusive in the last bin so confidence == 1.0 is not dropped.
        in_bin = (confidence > low) & (confidence <= high) if high < 1.0 else \
                 (confidence > low) & (confidence <= 1.0)
        count = int(in_bin.sum())
        if count == 0:
            continue
        error += (count / total) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def reliability_bins(probabilities, targets, bins=15):
    """Per-bin confidence, accuracy and count -- the data behind the ECE number."""
    probabilities = _as_probabilities(probabilities)
    targets = _as_array(targets)
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == targets).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high) if high < 1.0 else \
                 (confidence > low) & (confidence <= 1.0)
        count = int(in_bin.sum())
        out.append({
            "low": float(low), "high": float(high), "count": count,
            "confidence": float(confidence[in_bin].mean()) if count else None,
            "accuracy": float(correct[in_bin].mean()) if count else None,
        })
    return out


# ---------------------------------------------------------------- abstention

def abstention_curve(probabilities, targets, classes, thresholds):
    """Accuracy and macro-F1 against coverage, abstaining below each max-softmax threshold.

    The threshold grid must be fixed on the source domain. Choosing it to flatter a
    target domain is threshold selection on target data, which CLAUDE.md section 3
    forbids outright.
    """
    probabilities = _as_probabilities(probabilities)
    targets = _as_array(targets)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)

    curve = []
    for threshold in thresholds:
        keep = confidence >= threshold
        covered = int(keep.sum())
        point = {
            "threshold": float(threshold),
            "coverage": covered / len(targets) if len(targets) else 0.0,
            "n_covered": covered,
            "n_abstained": int(len(targets) - covered),
        }
        if covered:
            kept = classification_metrics(targets[keep], predictions[keep], classes)
            point.update(accuracy=kept["accuracy"], macro_f1=kept["macro_f1"])
        else:
            # Undefined rather than zero: nothing was predicted, so nothing was right.
            point.update(accuracy=None, macro_f1=None)
        curve.append(point)
    return curve


# ------------------------------------------------- restricted label spaces

def restrict_to_present_classes(metrics, present_classes):
    """Recompute averages over only the classes a split actually contains.

    Raabin Test-B holds just lymphocyte and neutrophil, so five-class macro-F1 there is
    capped at 0.4 by three absent classes and says nothing about the model.

    The model is NOT restricted -- it still predicts over all five classes, and a
    prediction of an absent class counts as an error. Masking its outputs would use
    knowledge of the target label space, which is exactly the kind of data-driven
    decision CLAUDE.md section 3 rules out. `out_of_space_predictions` records how often
    it happened, since a model that frequently predicts impossible classes on a target
    domain is itself a finding.
    """
    unknown = [c for c in present_classes if c not in metrics["per_class"]]
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)} not in this split's metrics; present_classes must be "
            f"a subset of {', '.join(metrics['per_class'])}"
        )
    # A declared class with no support means configs/class_mapping.yaml disagrees with
    # the data on what this split contains. Averaging over it would silently drag the
    # macro score towards zero, so it fails instead.
    empty = [c for c in present_classes if metrics["per_class"][c]["support"] == 0]
    if empty:
        raise ValueError(
            f"{', '.join(empty)} declared present for this split but has zero support; "
            f"target_splits in class_mapping.yaml does not match the data"
        )
    kept = {c: metrics["per_class"][c] for c in present_classes}
    absent = [c for c in metrics["per_class"] if c not in present_classes]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    order = list(metrics["per_class"])
    absent_columns = [order.index(c) for c in absent]

    return {
        **metrics,
        "per_class": kept,
        "n_classes": len(kept),
        "macro_f1": float(np.mean([m["f1"] for m in kept.values()])),
        "macro_recall": float(np.mean([m["recall"] for m in kept.values()])),
        "macro_precision": float(np.mean([m["precision"] for m in kept.values()])),
        "restricted_to": list(present_classes),
        "out_of_space_predictions": int(matrix[:, absent_columns].sum()) if absent_columns else 0,
    }
