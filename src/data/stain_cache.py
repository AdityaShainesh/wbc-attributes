"""Precomputed stain-normalised images, as a mirror of the source image tree.

Stain normalisation is deterministic and costs 43 ms (Macenko) to 86 ms (Reinhard) per
image. This protocol trains 30 epochs with no early stopping, so normalising on the fly
would add roughly four CPU-hours per seed and leave the GPU waiting on the loader for
most of a run. Paying it once per image instead is the difference between a stain arm
costing the same as the baseline and costing several times more.

HOW IT WORKS. The cache is a parallel tree with the same layout and the same filenames
as the source, holding each image after stain normalisation and resize. When it is in
use the loaders are handed a config whose image roots point into the cache and whose
stain method is "none" -- so datasets.py and the dataset construction in loaders.py are
untouched, and a reviewer has one swap to check rather than a new code path through the
data layer.

WHAT IS AND IS NOT CACHED. Only the deterministic prefix: normalise, then resize.
Augmentation, ToTensor and the ImageNet normalisation still run every epoch, so nothing
random is frozen into the cache and training sees the distribution it would see without
one.

FILE FORMAT. Cached files keep the source filename, including its .jpg extension, but
hold PNG bytes -- lossless, so the cache cannot quietly degrade an arm through repeated
JPEG compression. Both PIL and this repo's _load_rgb identify an image by its header
rather than its extension, and the PBC CSVs name .jpg files, so keeping the name is what
lets the root swap work at all.

KEYING. The key covers the method, the image size AND a fingerprint of the reference
image, because all three change the pixels. Changing the reference therefore misses the
cache rather than silently reusing images normalised against the old one. A cache that
is absent or incomplete is an error, never a silent fall back to the slow path: a run
that was meant to use it and did not would cost hours and nobody would notice.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

MANIFEST_NAME = "manifest.json"


class StainCacheError(Exception):
    """The cache is missing, incomplete, or built for a different reference."""


def is_enabled(cfg) -> bool:
    spec = getattr(cfg.data, "stain_norm", None)
    return bool(spec and spec.method != "none" and spec.cache)


def reference_fingerprint(cfg) -> str:
    """First 8 hex of the reference image's sha256, resolved through the config."""
    from src.data.stain import load_reference

    reference = load_reference(cfg.data.stain_norm.reference)
    image = reference.resolve(cfg.paths.pbc_images)
    return hashlib.sha256(image.read_bytes()).hexdigest()[:8]


def cache_key(cfg) -> str:
    return (f"{cfg.data.stain_norm.method}_{cfg.data.image_size}"
            f"_{reference_fingerprint(cfg)}")


def cache_root(cfg) -> Path:
    return Path(cfg.paths.results_dir) / "stain_cache" / cache_key(cfg)


def pbc_anchor(cfg) -> Path:
    """PBC paths are mirrored relative to this.

    _candidate_paths resolves a CSV row against pbc_images.parent first, so mirroring
    from the parent is what makes the swapped root resolve the same rows.
    """
    return Path(cfg.paths.pbc_images).parent


def mirror_path(cfg, source: Path) -> Path:
    """Where a source image's normalised copy lives."""
    source = Path(source)
    root = cache_root(cfg)
    for anchor, bucket in ((pbc_anchor(cfg), "pbc"), (Path(cfg.paths.raabin_root), "raabin")):
        try:
            return root / bucket / source.resolve().relative_to(Path(anchor).resolve())
        except ValueError:
            continue
    raise StainCacheError(
        f"{source} is under neither {pbc_anchor(cfg)} nor {cfg.paths.raabin_root}; "
        f"the stain cache cannot work out where to mirror it"
    )


def manifest_path(cfg) -> Path:
    return cache_root(cfg) / MANIFEST_NAME


def write_manifest(cfg, counts: dict) -> Path:
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "key": cache_key(cfg),
        "method": cfg.data.stain_norm.method,
        "image_size": cfg.data.image_size,
        "reference": cfg.data.stain_norm.reference,
        "reference_fingerprint": reference_fingerprint(cfg),
        "counts": counts,
    }, indent=1), encoding="utf-8")
    return path


def read_manifest(cfg) -> dict:
    path = manifest_path(cfg)
    if not path.exists():
        raise StainCacheError(
            f"no stain cache at {cache_root(cfg)}. Build it with\n"
            f"  python -m scripts.precompute_stain --config <config>\n"
            f"or set data.stain_norm.cache: false to normalise on the fly (hours slower)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def require_dataset(cfg, dataset_id: str) -> None:
    """Raise unless this dataset was included when the cache was built."""
    manifest = read_manifest(cfg)
    if dataset_id not in manifest.get("counts", {}):
        raise StainCacheError(
            f"the stain cache at {cache_root(cfg)} does not contain {dataset_id!r} "
            f"(it has {', '.join(sorted(manifest.get('counts', {}))) or 'nothing'}). "
            f"Rebuild it with scripts/precompute_stain.py, which reads the domains from "
            f"the config."
        )


def cached_cfg(cfg, dataset_id: str):
    """A config whose image roots point into the cache and whose stain step is off.

    Returns cfg unchanged when the cache is not in use. The ORIGINAL config is what gets
    recorded in the checkpoint, so provenance still names the real stain method: this
    swap exists only for the duration of loader construction.
    """
    if not is_enabled(cfg):
        return cfg
    require_dataset(cfg, dataset_id)
    root = cache_root(cfg)
    paths = dataclasses.replace(
        cfg.paths,
        pbc_images=root / "pbc" / Path(cfg.paths.pbc_images).name,
        raabin_root=root / "raabin",
    )
    data = dataclasses.replace(
        cfg.data,
        stain_norm=dataclasses.replace(cfg.data.stain_norm, method="none"),
    )
    return dataclasses.replace(cfg, paths=paths, data=data)
