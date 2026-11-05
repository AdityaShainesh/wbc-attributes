"""Precomputed stain-normalised images.

Stain normalisation costs 43 ms (Macenko) to 86 ms (Reinhard) per image and is fully
deterministic, so paying it once per image beats paying it once per image *per epoch per
seed*. On a GPU box the uncached pipeline leaves the card idle waiting on the loader.

What is cached is the image after stain normalisation, padding and resizing -- i.e.
everything deterministic. Augmentation, tensor conversion and mean/std normalisation
still happen per epoch, so nothing random is frozen into the cache and training sees the
same distribution it would without it.

The cache is keyed by method and image size, because a cache built at 224 is silently
wrong for a run at 360. A missing cache is an error, never a quiet fallback to the slow
path: a run that was meant to use the cache and didn't would take hours longer and
nobody would notice until the bill arrived.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "data" / "processed" / "stain"
RAW = ROOT / "data" / "raw"


def cache_dir(method, image_size):
    return CACHE_ROOT / f"{method}_{image_size}"


def relative_to_raw(raw_path):
    """Path of a raw image relative to data/raw, whatever symlinks are in the way.

    Three layouts all occur in practice and each breaks a different naive version:

      data/raw is a real directory          -- plain relative_to works
      data/raw is a symlink                 -- Kaggle, one dataset: resolving only the
                                               image path lands it in /kaggle/input
                                               while RAW stays under /kaggle/working
      data/raw holds symlinked children     -- Kaggle, several datasets: now RAW itself
                                               is local and real, but each child points
                                               off into a different input tree

    So: try the path as given, then with both sides resolved, then the resolved path
    against the unresolved root. The relative path is identical in every case, which is
    what keeps one cache valid across all of them.
    """
    path = Path(raw_path)
    for candidate, base in ((path, RAW), (path.resolve(), RAW.resolve()), (path.resolve(), RAW)):
        try:
            return candidate.relative_to(base)
        except ValueError:
            continue
    raise ValueError(
        f"{raw_path} is not under {RAW} (resolved: {RAW.resolve()}); the stain cache "
        f"cannot work out where to store it"
    )


def cached_path(method, image_size, raw_path):
    """Where the normalised copy of a raw image lives. PNG: lossless, no re-compression."""
    return cache_dir(method, image_size) / relative_to_raw(raw_path).with_suffix(".png")


class CachedStainDataset(Dataset):
    """Serves a base dataset's labels with images read from the stain cache.

    Wraps rather than modifies the datasets: PbcDataset and LiscDataset stay responsible
    for labels, attributes and exclusions, and this only changes where pixels come from.
    """

    def __init__(self, base, method, image_size, transform):
        self.base = base
        self.method = method
        self.image_size = image_size
        self.transform = transform

        # base may be a Subset, in which case only the selected records matter.
        inner = getattr(base, "dataset", base)
        self.records = ([inner.records[i] for i in base.indices]
                        if isinstance(base, Subset) else base.records)
        self.labels = inner.labels
        self.attributes = getattr(inner, "attributes", None)

        missing = self._first_missing()
        if missing is not None:
            raise FileNotFoundError(
                f"stain cache miss for {missing}. Build it with\n"
                f"  python scripts/precompute_stain.py\n"
                f"or set stain_norm.cache: false to normalise on the fly (far slower)."
            )

    def _first_missing(self):
        for record in self.records:
            if not cached_path(self.method, self.image_size, record.path).exists():
                return record.img_name
        return None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(cached_path(self.method, self.image_size, record.path)) as image:
            tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(record.label, dtype=torch.long)
        attributes = getattr(record, "attributes", None)
        return (tensor, label,
                torch.tensor(attributes, dtype=torch.long) if attributes is not None else None)

    def __repr__(self):
        return (f"{type(self).__name__}({self.base!r}, method={self.method!r}, "
                f"size={self.image_size})")
