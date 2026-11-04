"""Dataset construction from a resolved config dict.

build_dataset / build_dataloader are the only entry points training and eval should use,
so which dataset a run touched is always a line in its config rather than a call site
somewhere in the code.
"""
import torch
from torch.utils.data import DataLoader

from src.data.lisc import LiscDataset
from src.data.mapping import MappingError, load_attribute_space, load_label_space
from src.data.pbc import PbcDataset

# Target domains are listed here so an attempt to load one is a deliberate, visible act.
SOURCE_DATASETS = ("pbc",)
TARGET_DATASETS = ("lisc",)


def build_dataset(config, labels=None, attributes=None):
    """Build the dataset named by config["dataset"] / config["split"]."""
    name = config["dataset"]
    image_size = config.get("image_size", 224)
    normalize = config.get("normalize", "imagenet")
    if name == "pbc":
        return PbcDataset(config["split"], image_size=image_size, normalize=normalize,
                          labels=labels, attributes=attributes)
    if name == "lisc":
        return LiscDataset(config["split"], image_size=image_size, normalize=normalize,
                           labels=labels)
    raise MappingError(
        f"unknown dataset {name!r}; known source datasets {SOURCE_DATASETS}, "
        f"target datasets {TARGET_DATASETS}"
    )


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


def build_dataloader(config, dataset=None, labels=None, attributes=None):
    """DataLoader over build_dataset(config), with shuffling seeded from the config."""
    dataset = dataset or build_dataset(config, labels=labels, attributes=attributes)
    generator = None
    if config.get("shuffle", False):
        generator = torch.Generator().manual_seed(config.get("seed", 0))
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=config.get("shuffle", False),
        num_workers=config.get("num_workers", 0),
        collate_fn=collate,
        generator=generator,
        drop_last=config.get("drop_last", False),
    )


__all__ = [
    "build_dataset", "build_dataloader", "collate",
    "PbcDataset", "LiscDataset", "MappingError",
    "load_label_space", "load_attribute_space",
    "SOURCE_DATASETS", "TARGET_DATASETS",
]
