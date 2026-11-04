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
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORMALIZATIONS = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD)}


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


def build_transform(image_size=224, normalize="imagenet"):
    """Deterministic eval-style transform. No augmentation: that belongs to training."""
    try:
        mean, std = NORMALIZATIONS[normalize]
    except KeyError:
        raise ValueError(
            f"unknown normalization {normalize!r}; known: {', '.join(NORMALIZATIONS)}"
        ) from None
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
