"""Image transforms. Every parameter comes from the config; nothing is measured here.

Resize and the normalisation constants are read from the config, which pins the latter
to the fixed ImageNet values (src/config.py). No statistic is computed from any dataset.

Augmentation is applied to training data only, which is source data only, and is
identical for the baseline, bottleneck and control - otherwise the three would be
compared across different protocols rather than different representations.
"""
from __future__ import annotations

from torchvision import transforms

from src.data.stain import build_normalizer


def _normalize(cfg):
    return transforms.Normalize(mean=cfg.data.normalization["mean"],
                                std=cfg.data.normalization["std"])


def _resize(cfg):
    """Plain resize to a square.

    Not shortest-side-plus-centre-crop: that would crop the edge of the cell, and
    cell_size is one of the 11 attributes being predicted. Blood cell crops are close to
    square already (PBC is 360x363), so the aspect distortion is slight.
    """
    size = cfg.data.image_size
    return transforms.Resize((size, size))


def _stain(cfg):
    """Stain normalisation, first, on the original pixels.

    Before the resize so the method sees the image it was designed for, and in BOTH
    transforms: an arm trained on normalised images and evaluated on raw ones would
    report a domain drop that is an artefact of the pipeline (CLAUDE.md sec 3).
    """
    normalizer = build_normalizer(cfg)
    return [normalizer] if normalizer is not None else []


def build_eval_transform(cfg):
    """Deterministic. Used for val, source test and every target domain."""
    return transforms.Compose(_stain(cfg) + [
        _resize(cfg),
        transforms.ToTensor(),
        _normalize(cfg),
    ])


def build_train_transform(cfg):
    """Eval transform plus augmentation. Training data only, which is source only."""
    aug = cfg.train.augmentation
    steps = _stain(cfg) + [_resize(cfg)]

    # Blood smears have no canonical orientation, so flips and full rotation are
    # label-preserving for all five classes and all 11 attributes.
    if aug["hflip"] > 0:
        steps.append(transforms.RandomHorizontalFlip(p=aug["hflip"]))
    if aug["vflip"] > 0:
        steps.append(transforms.RandomVerticalFlip(p=aug["vflip"]))
    if aug["rotation_degrees"] > 0:
        steps.append(transforms.RandomRotation(aug["rotation_degrees"]))
    if aug["color_jitter"] is not None:
        steps.append(transforms.ColorJitter(**aug["color_jitter"]))

    steps += [transforms.ToTensor(), _normalize(cfg)]
    return transforms.Compose(steps)
