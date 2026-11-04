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
