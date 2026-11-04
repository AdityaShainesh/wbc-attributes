"""Baseline training loop.

    python -m src.train --config configs/<x>.yaml

That is the whole contract (CLAUDE.md section 5): one config, no other arguments, no
manual steps, no edits to source files. Everything the run did ends up in
results/<run_id>.json, including the config itself.

The config lists its seeds and this script runs all of them, so "mean +- std over 3
seeds" is one command rather than three invocations someone has to remember to average.

Leakage guard: check_no_target_leakage runs before any data is loaded. Target domains
(LISC, and Raabin when it arrives) may not be trained on, selected on, or evaluated
during training -- they are touched once, by the eval harness, with the checkpoint
already chosen. If that guard ever seems to be in the way, the answer is to stop and ask,
not to weaken it.
"""
import argparse
import time

import torch
import torch.nn as nn

from src.data import TARGET_DATASETS, build_dataloader, load_attribute_space, load_label_space
from src.metrics import aggregate_over_seeds, classification_metrics
from src.models import build_model
from src.utils.config import ConfigError, git_state, load_config, require
from src.utils.results import checkpoint_path, hardware, write_record
from src.utils.seed import seed_everything

OPTIMIZERS = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


class LeakageError(Exception):
    """A config would have let a target domain influence training or selection."""


def check_no_target_leakage(config):
    """Raise unless every dataset this run touches is a source domain."""
    offenders = []

    training = config.get("data", {}).get("dataset")
    if training in TARGET_DATASETS:
        offenders.append(f"data.dataset={training} (training set)")

    selection = config.get("train", {}).get("selection", {}).get("dataset")
    if selection in TARGET_DATASETS:
        offenders.append(f"train.selection.dataset={selection} (checkpoint selection)")

    for i, spec in enumerate(config.get("eval", {}).get("datasets", [])):
        if spec.get("dataset") in TARGET_DATASETS:
            offenders.append(f"eval.datasets[{i}].dataset={spec['dataset']}")

    if offenders:
        raise LeakageError(
            "target-domain data cannot be used during training (CLAUDE.md section 3): "
            + "; ".join(offenders)
            + ". Target domains are evaluated once, afterwards, with the checkpoint "
              "already chosen."
        )
    return True


def evaluate(model, loader, classes, device):
    """Predictions over a loader, as metrics. No gradients, no augmentation."""
    model.eval()
    targets, predictions, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels, _attributes in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, labels))
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, classes)
    metrics["loss"] = total_loss / max(len(targets), 1)
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, seen = 0.0, 0
    for images, labels, _attributes in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * labels.size(0)
        seen += labels.size(0)
    return total_loss / max(seen, 1)


def run_one_seed(config, seed, labels, attributes, device, results_directory=None):
    seed_everything(seed, config.get("deterministic", True))
    train_spec = config["train"]
    classes = labels.classes

    train_loader = build_dataloader(
        dict(config["data"], seed=seed, shuffle=True),
        labels=labels, attributes=attributes, train=True,
    )
    selection_spec = dict(train_spec.get("selection", {"dataset": "pbc", "split": "val"}))
    selection_spec.setdefault("image_size", config["data"].get("image_size", 224))
    selection_loader = build_dataloader(
        dict(selection_spec, batch_size=config["data"].get("batch_size", 32)),
        labels=labels, attributes=attributes,
    )

    model = build_model(config).to(device)
    optimizer_name = train_spec.get("optimizer", "adamw").lower()
    if optimizer_name not in OPTIMIZERS:
        raise ConfigError(f"unknown optimizer {optimizer_name!r}; known: {', '.join(OPTIMIZERS)}")
    optimizer = OPTIMIZERS[optimizer_name](
        model.parameters(),
        lr=float(train_spec.get("lr", 1e-4)),
        weight_decay=float(train_spec.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    early = train_spec.get("early_stopping", {}) or {}
    watch = early.get("metric", "macro_f1")
    patience = early.get("patience")

    history, best = [], {"score": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, int(train_spec["epochs"]) + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        selection_metrics = evaluate(model, selection_loader, classes, device)
        score = selection_metrics[watch]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_metric": watch,
            "selection_score": score,
            "selection_loss": selection_metrics["loss"],
            "seconds": round(time.time() - started, 1),
        })
        print(f"  seed {seed} epoch {epoch}: train_loss={train_loss:.4f} "
              f"{watch}={score:.4f} ({history[-1]['seconds']}s)", flush=True)

        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif patience and epoch - best["epoch"] >= int(patience):
            print(f"  seed {seed}: early stop at epoch {epoch} "
                  f"(no improvement since {best['epoch']})", flush=True)
            break

    # Report the selected checkpoint, not the last epoch.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        path = checkpoint_path(config["run_id"], seed, results_directory)
        torch.save({"config": config, "seed": seed, "epoch": best["epoch"],
                    "state_dict": best["state"]}, path)

    metrics = {}
    for spec in config["eval"]["datasets"]:
        name = f"{spec['dataset']}_{spec['split']}"
        loader = build_dataloader(
            dict(spec, batch_size=config["data"].get("batch_size", 32),
                 image_size=config["data"].get("image_size", 224)),
            labels=labels, attributes=attributes,
        )
        metrics[name] = evaluate(model, loader, classes, device)

    return {
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_selection_score": best["score"],
        "epochs_run": len(history),
        "history": history,
        "metrics": metrics,
        "model": model.describe(),
    }


def run_from_config(config, results_dir=None):
    """Run every seed in the config and write one results file. Returns the record."""
    check_no_target_leakage(config)
    require(config, "run_id", "seeds", "model", "data", "train", "eval")
    seeds = config["seeds"]
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ConfigError("`seeds` must be a non-empty list, e.g. seeds: [0, 1, 2]")
    require(config["train"], "epochs")
    require(config["eval"], "datasets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_label_space()
    attributes = load_attribute_space()
    started = time.time()

    per_seed = {}
    for seed in seeds:
        print(f"seed {seed}:", flush=True)
        per_seed[str(seed)] = run_one_seed(config, seed, labels, attributes, device, results_dir)

    # Aggregate over seeds for each evaluated split, per CLAUDE.md section 5.
    aggregate = {}
    split_names = {name for record in per_seed.values() for name in record["metrics"]}
    for name in sorted(split_names):
        aggregate[name] = aggregate_over_seeds(
            [record["metrics"][name] for record in per_seed.values()]
        )
    primary = config["eval"]["datasets"][0]
    primary_name = f"{primary['dataset']}_{primary['split']}"

    record = {
        "run_id": config["run_id"],
        "config": config,
        "git": git_state(),
        "hardware": {**hardware(), "device": str(device)},
        "classes": labels.classes,
        "seeds": per_seed,
        "per_split_aggregate": aggregate,
        "aggregate": aggregate.get(primary_name, {}),
        "primary_split": primary_name,
        "wall_clock_s": round(time.time() - started, 1),
    }
    path = write_record(record, config["run_id"], results_dir)
    print(f"\nwrote {path}", flush=True)
    for metric, stats in record["aggregate"].items():
        print(f"  {primary_name} {metric}: {stats['mean']:.4f} +- {stats['std']:.4f}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to the run's YAML config")
    args = parser.parse_args()
    run_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
