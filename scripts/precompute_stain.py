"""Build the stain cache for one config.

    python -m scripts.precompute_stain --config configs/stain_reinhard.yaml

Writes a mirror of the source image tree holding each image after stain normalisation
and resize, under <results_dir>/stain_cache/<method>_<size>_<reference fingerprint>/.
See src/data/stain_cache.py for why the cache exists and how it is keyed.

Which images get cached is read from the config -- the three source splits plus every
domain in eval.domains -- so the cache cannot drift from the run that uses it. A config
that adds a domain without a rebuild fails at loader construction with the command to
fix it, rather than part-way through a run.

Idempotent: an image whose cached copy exists is skipped, so an interrupted build
resumes. Each file is written to a temporary name and renamed, so a half-written image
is never mistaken for a finished one.

LEAKAGE. This reads target images, which is legitimate and unavoidable: the same fixed,
source-derived transform has to be applied to a target domain before the model sees it,
and doing it ahead of time changes nothing about what is computed. No statistic is
measured from any image here, and nothing read influences any decision (CLAUDE.md sec 3).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config                                    # noqa: E402
from src.data.datasets import PbcWbcAttDataset, RaabinFolderDataset   # noqa: E402
from src.data.domains import DATASETS, Domain, domain_of              # noqa: E402
from src.data.stain import build_normalizer                           # noqa: E402
from src.data.stain_cache import (                                    # noqa: E402
    cache_root,
    is_enabled,
    mirror_path,
    write_manifest,
)

SOURCE_SPLITS = ("pbc_train", "pbc_val", "pbc_test")


def dataset_for(cfg, dataset_id: str):
    """The dataset, with no transform: this script applies its own fixed steps."""
    mapping, label_space = cfg.data.class_mapping, cfg.data.label_space
    if domain_of(dataset_id) is Domain.SOURCE:
        return PbcWbcAttDataset(
            dataset_id=dataset_id,
            pbc_images=cfg.paths.pbc_images,
            wbcatt_dir=cfg.paths.wbcatt_dir,
            mapping=mapping,
            label_space=label_space,
            image_size=cfg.data.image_size,
            transform=None,
        )
    return RaabinFolderDataset(
        dataset_id=dataset_id,
        root=cfg.paths.raabin_root,
        mapping=mapping,
        label_space=label_space,
        image_size=cfg.data.image_size,
        transform=None,
    )


def wanted_datasets(cfg) -> list[str]:
    out = list(SOURCE_SPLITS)
    for dataset_id in getattr(cfg.eval, "domains", ()) or ():
        if dataset_id in DATASETS and dataset_id not in out:
            out.append(dataset_id)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not is_enabled(cfg):
        print(f"{args.config}: data.stain_norm is none or cache is off; nothing to do")
        return 0

    # The same deterministic prefix the loaders would otherwise run every epoch.
    normalize = build_normalizer(cfg)
    resize = transforms.Resize((cfg.data.image_size, cfg.data.image_size))
    root = cache_root(cfg)
    print(f"{cfg.data.stain_norm.method} at {cfg.data.image_size}px -> {root}", flush=True)

    started, counts = time.time(), {}
    for dataset_id in wanted_datasets(cfg):
        dataset = dataset_for(cfg, dataset_id)
        written = skipped = 0
        for i, source in enumerate(dataset.paths, 1):
            destination = mirror_path(cfg, source)
            if destination.exists():
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                processed = resize(normalize(image.convert("RGB")))
            # PNG bytes under the source filename: lossless, and keeping the name is
            # what lets the loaders swap roots without touching the datasets.
            temporary = destination.with_name(destination.name + ".tmp")
            processed.save(temporary, format="PNG", optimize=False)
            temporary.replace(destination)
            written += 1
            if i % 500 == 0:
                print(f"  {dataset_id}: {i}/{len(dataset.paths)}", flush=True)
        counts[dataset_id] = len(dataset.paths)
        print(f"  {dataset_id}: {written} written, {skipped} already cached", flush=True)

    manifest = write_manifest(cfg, counts)
    total = sum(counts.values())
    print(f"\n{total} images in {time.time() - started:.0f}s; manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
