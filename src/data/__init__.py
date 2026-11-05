"""Dataset construction from a resolved config dict.

build_dataset / build_dataloader are the only entry points training and eval should use,
so which dataset a run touched is always a line in its config rather than a call site
somewhere in the code.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.cache import CachedStainDataset
from src.data.lisc import LiscDataset
from src.data.mapping import MappingError, load_attribute_space, load_label_space
from src.data.pbc import PbcDataset
from src.data.transforms import build_cached_transform, build_train_transform, build_transform
from src.utils.seed import seed_worker


def resolve_num_workers(config):
    """Loader workers, overridable with WBC_NUM_WORKERS.

    Infrastructure, not an experimental setting: it changes how fast a run goes and
    nothing about what it computes. The override exists so the same unmodified config
    can run on Windows (0 workers) and on a Kaggle GPU box (4) without editing a file,
    which CLAUDE.md section 5 forbids. The effective value is recorded in the results.
    """
    override = os.environ.get("WBC_NUM_WORKERS")
    if override is None:
        return int(config.get("num_workers", 0))
    try:
        return max(0, int(override))
    except ValueError:
        raise ValueError(f"WBC_NUM_WORKERS must be an integer, got {override!r}") from None

# Target domains are listed here so an attempt to load one is a deliberate, visible act.
SOURCE_DATASETS = ("pbc",)
TARGET_DATASETS = ("lisc",)


def build_dataset(config, labels=None, attributes=None, train=False):
    """Build the dataset named by config["dataset"] / config["split"].

    train=True swaps in the augmenting transform. Augmentation is never applied to an
    evaluation split, so this is a caller's decision rather than a config key that could
    silently leave augmentation on at test time.
    """
    name = config["dataset"]
    image_size = config.get("image_size", 224)
    normalize = config.get("normalize", "imagenet")
    # The stain setting must be identical for training and evaluation, so it is read
    # from the same config key in both branches rather than passed in by the caller.
    stain = config.get("stain_norm")
    augment = train and config.get("augment", True)
    cached = bool(stain) and stain.get("cache", True)

    # With a cache the deterministic steps are already baked into the stored image, so
    # the live transform must not repeat them.
    transform = (build_cached_transform(normalize, augment) if cached else
                 build_train_transform(image_size, normalize, augment, stain) if train else
                 build_transform(image_size, normalize, stain))

    if name == "pbc":
        dataset = PbcDataset(config["split"], labels=labels, attributes=attributes,
                             transform=transform)
    elif name == "lisc":
        dataset = LiscDataset(config["split"], labels=labels, transform=transform)
    else:
        raise MappingError(
            f"unknown dataset {name!r}; known source datasets {SOURCE_DATASETS}, "
            f"target datasets {TARGET_DATASETS}"
        )
    # Subset first: there is no point validating or reading cached images for records
    # this run will never touch.
    dataset = maybe_subset(dataset, config)
    if cached:
        dataset = CachedStainDataset(dataset, stain["method"], image_size, transform)
    return dataset


def maybe_subset(dataset, config):
    """Take a fixed random subset if config["subset_size"] is set (smoke runs only).

    The subset is drawn with its own seed, not the run seed, so all three seeds of a
    smoke run see the same images and differ only in initialisation and shuffling.
    """
    size = config.get("subset_size")
    if not size or size >= len(dataset):
        return dataset
    rng = np.random.RandomState(config.get("subset_seed", 0))
    indices = sorted(rng.choice(len(dataset), size=size, replace=False).tolist())
    return Subset(dataset, indices)


def collate(batch):
    """Default collation, except that a batch of attribute-less items keeps None.

    Mixing attribute-labelled and attribute-free items in one batch is a bug, not a case
    to paper over, so it raises.
    """
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    attributes = [item[2] for item in batch]
    if all(a is None for a in attributes):
        return images, targets, None
    if any(a is None for a in attributes):
        raise ValueError("batch mixes items with and without attribute labels")
    return images, targets, torch.stack(attributes)


def build_dataloader(config, dataset=None, labels=None, attributes=None, train=False):
    """DataLoader over build_dataset(config), with shuffling seeded from the config."""
    if dataset is None:
        dataset = build_dataset(config, labels=labels, attributes=attributes, train=train)
    generator = None
    if config.get("shuffle", False):
        generator = torch.Generator().manual_seed(config.get("seed", 0))
    workers = resolve_num_workers(config)
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=config.get("shuffle", False),
        num_workers=workers,
        collate_fn=collate,
        generator=generator,
        worker_init_fn=seed_worker if workers else None,
        persistent_workers=workers > 0,
        drop_last=config.get("drop_last", False),
    )


__all__ = [
    "build_dataset", "build_dataloader", "collate", "maybe_subset",
    "PbcDataset", "LiscDataset", "MappingError",
    "load_label_space", "load_attribute_space",
    "SOURCE_DATASETS", "TARGET_DATASETS",
]
