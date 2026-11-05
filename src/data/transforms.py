"""The single image transform, applied identically to source and target domains.

Geometry is fixed from PBC alone: PBC images are 360x363 or 366x369, so padding to
square costs 3 pixels of replicated edge and resizing to 224 is a mild downscale. LISC
(72x78 up to 265x252) is then put through the exact same transform. That ordering is
deliberate -- choosing a resize to suit LISC's size distribution would be tuning on a
target domain, which CLAUDE.md section 3 forbids.

Padding replicates the edge rather than filling with a constant, so no artificial border
colour is introduced into a stain-sensitive task, and aspect ratio is preserved for the
small, variably shaped LISC crops.

Normalization currently uses ImageNet statistics. These are constants baked into the
pretrained backbones, not statistics measured from any dataset here, so they are safe.
Switching to PBC-train statistics is a legitimate alternative; it must be computed from
the PBC train split only and recorded in the config.
"""
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORMALIZATIONS = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD)}


def _normalization(name):
    try:
        return NORMALIZATIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown normalization {name!r}; known: {', '.join(NORMALIZATIONS)}"
        ) from None


class PadToSquare:
    """Pad the short side to match the long one, replicating the edge pixels."""

    def __call__(self, image):
        width, height = image.size
        if width == height:
            return image
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return transforms.functional.pad(
            image, [left, top, side - width - left, side - height - top], padding_mode="edge"
        )

    def __repr__(self):
        return f"{type(self).__name__}()"


class RandomRot90:
    """Rotate by a random multiple of 90 degrees. Lossless on a square image."""

    def __call__(self, image):
        turns = int(torch.randint(0, 4, (1,)).item())
        return image if turns == 0 else image.rotate(90 * turns, expand=False)

    def __repr__(self):
        return f"{type(self).__name__}()"


def _stain_steps(stain):
    """Stain normalisation, applied first so it sees the original pixels."""
    if not stain:
        return []
    from src.data.stain import build_normalizer  # imported lazily; needs scikit-image
    return [build_normalizer(stain["method"] if isinstance(stain, dict) else stain)]


def _hed_steps(hed_jitter):
    """HED colour augmentation. Train only -- never reachable from build_transform."""
    from src.data.hed import build_hed_jitter
    jitter = build_hed_jitter(hed_jitter)
    return [jitter] if jitter else []


def build_train_transform(image_size=224, normalize="imagenet", augment=True, stain=None,
                          hed_jitter=None):
    """Training transform: geometric augmentation only, deliberately no colour jitter.

    Blood cells have no canonical orientation, so flips and quarter turns are free
    label-preserving variation.

    Colour and stain augmentation is left out on purpose. It is itself a
    domain-generalization intervention, so putting it in the baseline would confound the
    experiment: if the attribute bottleneck then generalized better, its contribution
    could not be separated from the augmentation's. If colour augmentation is tested, it
    has to be an explicit arm applied identically to both models, recorded in the config.
    """
    mean, std = _normalization(normalize)
    steps = _stain_steps(stain)
    steps += [PadToSquare(), transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR)]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
        ]
        # After the resize, so the HED round trip runs on 224x224 rather than 360x363.
        steps += _hed_steps(hed_jitter)
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def build_cached_transform(normalize="imagenet", augment=False, hed_jitter=None):
    """For images already stain-normalised, padded and resized by the cache.

    Deliberately excludes PadToSquare and Resize: applying them twice would be a silent
    no-op today and a silent bug the moment the cache size and the run size diverge.
    """
    mean, std = _normalization(normalize)
    steps = ([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), RandomRot90()]
             + _hed_steps(hed_jitter)) if augment else []
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def build_deterministic_steps(image_size=224, stain=None):
    """Exactly the steps the cache bakes in: stain normalisation, pad, resize."""
    return transforms.Compose(_stain_steps(stain) + [
        PadToSquare(),
        transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
    ])


def build_transform(image_size=224, normalize="imagenet", stain=None):
    """Deterministic eval-style transform. No augmentation: that belongs to training.

    `stain` must match the training transform's. A model trained on Macenko-normalised
    images and evaluated on raw ones would show a spurious domain drop.
    """
    mean, std = _normalization(normalize)
    return transforms.Compose(_stain_steps(stain) + [
        PadToSquare(),
        transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
