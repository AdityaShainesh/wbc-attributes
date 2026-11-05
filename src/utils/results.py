"""Writing results/<run_id>.json.

CLAUDE.md section 5 fixes the contents: the full resolved config embedded rather than
referenced, the git commit and dirty flag, all metrics per domain and per class,
wall-clock time and hardware. Embedding the config is the point -- a results file that
says "see configs/foo.yaml" is worthless once configs/foo.yaml has moved on.

RESULTS_DIR can be overridden with WBC_RESULTS_DIR, which exists so tests can write to a
temporary directory. It does not change what a run computes.
"""
import json
import os
import platform
import socket
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "results"


def results_dir(override=None):
    if override:
        return Path(override)
    return Path(os.environ.get("WBC_RESULTS_DIR", DEFAULT_RESULTS_DIR))


def hardware():
    """What the run actually executed on. Matters for reading wall-clock times."""
    info = {
        # Infrastructure overrides that do not change what a run computes, but do change
        # its wall-clock time; recorded so timings are readable after the fact.
        "num_workers_override": os.environ.get("WBC_NUM_WORKERS"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hostname": socket.gethostname(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
    }
    if torch.cuda.is_available():
        info.update(
            device="cuda",
            gpu=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
            cuda=torch.version.cuda,
        )
    return info


def write_record(record, run_id, directory=None):
    """Write the record to <results>/<run_id>.json and return the path."""
    target = results_dir(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


def checkpoint_path(run_id, seed, directory=None):
    target = results_dir(directory) / "checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    return target / f"{run_id}_seed{seed}.pt"
