"""Zero-shot evaluation across domains: the project's actual measurement.

    python -m src.eval --config configs/eval.yaml

One config, no other arguments (CLAUDE.md section 5). Writes results/<run_id>.json.

THIS IS THE ONCE-ONLY TOUCH OF TARGET DATA. Everything before it -- training, model
selection, thresholds, stain references -- was decided on PBC alone. Running this
reveals target performance, and from that point on any change made to a model is a
change made with knowledge of the target domains, which weakens the zero-shot claim for
whatever comes after. Arms should be frozen before they are evaluated here.

What this does NOT do, deliberately:
  - no tuning, no threshold search on target, no checkpoint selection. Checkpoints come
    from training runs that chose them on PBC val; this only reads them.
  - no restriction of model outputs to a target's label space. Raabin Test-B contains
    two of five classes; the model still predicts all five there and an impossible
    prediction counts as an error (see metrics.restrict_to_present_classes).
  - no adaptation splits. raabin/Train and LISC/Train_Aug exist for Model 4's domain
    head and are refused here.

Preprocessing for each arm is read from the config embedded in its training results,
not restated here: a Macenko-trained model must be evaluated on Macenko-normalised
images, and re-deriving that by hand is how arms silently stop being comparable.
"""
import argparse
import time
from pathlib import Path

import torch
import yaml

from src.data import TARGET_DATASETS, build_dataloader, load_attribute_space, load_label_space
from src.data.mapping import MAPPING_PATH
from src.metrics import (
    abstention_curve,
    aggregate_over_seeds,
    classification_metrics,
    expected_calibration_error,
    reliability_bins,
    restrict_to_present_classes,
)
from src.models import build_model
from src.utils.config import ConfigError, git_state, load_config, require
from src.utils.results import hardware, results_dir, write_record
from src.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]


class EvalLeakageError(Exception):
    """An eval config would have read data reserved for adaptation."""


def adaptation_splits(path=MAPPING_PATH):
    """Splits declared as DANN-adaptation-only in configs/class_mapping.yaml."""
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = set()
    for dataset, splits in (mapping.get("target_splits") or {}).items():
        for split, policy in splits.items():
            if isinstance(policy, dict) and policy.get("role") in ("dann_adaptation", "excluded"):
                out.add((dataset, split))
    return out


def check_domains(config):
    """Refuse any domain reserved for adaptation, or excluded outright."""
    reserved = adaptation_splits()
    offenders = [f"{d['dataset']}/{d['split']}" for d in config["domains"]
                 if (d["dataset"], d["split"]) in reserved]
    if offenders:
        raise EvalLeakageError(
            "these splits are reserved for Model 4's domain head or excluded entirely, "
            "and must never be evaluated: " + ", ".join(offenders)
        )
    return True


def present_classes_for(dataset, split, path=MAPPING_PATH):
    """Declared class coverage, or None when the split carries all five."""
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    policy = (mapping.get("target_splits") or {}).get(dataset, {}).get(split)
    return policy.get("present_classes") if isinstance(policy, dict) else None


def load_training_record(run_id, directory=None):
    path = Path(directory or results_dir()) / f"{run_id}.json"
    if not path.exists():
        raise ConfigError(
            f"no training results at {path}; evaluation reads the config and seeds from "
            f"the run that produced the checkpoints"
        )
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def predict(model, loader, device):
    """Softmax probabilities and targets over a loader. No gradients, no augmentation."""
    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for images, labels, _attributes in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            targets.append(labels)
    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def evaluate_checkpoint(checkpoint, training_config, domains, classes, labels, attributes,
                        device, bins, thresholds):
    model = build_model(training_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    out = {}
    for domain in domains:
        name = f"{domain['dataset']}_{domain['split']}"
        # Preprocessing comes from the arm's own training config; only the split changes.
        spec = {k: training_config["data"][k]
                for k in ("image_size", "normalize", "stain_norm", "batch_size", "num_workers")
                if k in training_config["data"]}
        spec.update(domain, augment=False)
        probabilities, targets = predict(
            model, build_dataloader(spec, labels=labels, attributes=attributes), device)

        metrics = classification_metrics(targets, probabilities.argmax(axis=1), classes)
        present = present_classes_for(domain["dataset"], domain["split"])
        if present:
            metrics = restrict_to_present_classes(metrics, present)
        metrics["ece"] = expected_calibration_error(probabilities, targets, bins)
        metrics["reliability"] = reliability_bins(probabilities, targets, bins)
        metrics["abstention"] = abstention_curve(probabilities, targets, classes, thresholds)
        out[name] = metrics
        covered = "all 5 classes" if not present else f"{len(present)} of 5 classes"
        print(f"    {name}: macro_f1={metrics['macro_f1']:.4f} ece={metrics['ece']:.4f} "
              f"({covered}, n={metrics['n']})", flush=True)
    return out


def run_from_config(config, results_directory=None):
    require(config, "run_id", "runs", "domains")
    check_domains(config)

    bins = config.get("calibration", {}).get("bins", 15)
    thresholds = config.get("abstention", {}).get("thresholds", [0.0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, attributes = load_label_space(), load_attribute_space()
    checkpoints = Path(config.get("checkpoints_dir") or (results_dir() / "checkpoints"))
    started = time.time()

    evaluated = {}
    for run_id in config["runs"]:
        print(f"\n{run_id}", flush=True)
        training = load_training_record(run_id, config.get("results_dir"))
        seed_everything(training["config"].get("seeds", [0])[0],
                        training["config"].get("deterministic", True))

        per_seed = {}
        for seed in training["seeds"]:
            path = checkpoints / f"{run_id}_seed{seed}.pt"
            if not path.exists():
                raise ConfigError(f"missing checkpoint {path}")
            print(f"  seed {seed}", flush=True)
            per_seed[seed] = evaluate_checkpoint(
                torch.load(path, map_location=device, weights_only=False),
                training["config"], config["domains"], labels.classes,
                labels, attributes, device, bins, thresholds)

        domain_names = sorted({n for s in per_seed.values() for n in s})
        evaluated[run_id] = {
            "training_commit": training["git"]["commit"],
            "training_run": run_id,
            "seeds": per_seed,
            "aggregate": {name: aggregate_over_seeds(
                [s[name] for s in per_seed.values()],
                keys=("macro_f1", "macro_recall", "accuracy", "ece"))
                for name in domain_names},
        }

    record = {
        "run_id": config["run_id"],
        "config": config,
        "git": git_state(),
        "hardware": {**hardware(), "device": str(device)},
        "classes": labels.classes,
        "runs": evaluated,
        "wall_clock_s": round(time.time() - started, 1),
    }
    path = write_record(record, config["run_id"], results_directory)
    print(f"\nwrote {path}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to the eval YAML config")
    args = parser.parse_args()
    run_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
