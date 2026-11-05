"""HED colour augmentation (Model 3).

Decompose the image into Haematoxylin / Eosin / DAB stain channels, scale and shift each
one by a small random amount, and recompose. The network then sees many simulated
staining protocols during training and, in principle, stops relying on the exact colours
of the lab it was trained in.

    x'_HED = alpha * x_HED + beta,  alpha_c ~ U(1-sigma, 1+sigma),  beta_c ~ U(-sigma, sigma)

Each of the three channels draws its own alpha and beta. Sharing one factor across all
three would be a brightness change, not a stain change.

TRAIN ONLY. This is augmentation: applying it to a validation or test split would be
measuring the model on data it will never face, and would quietly flatter the score.
build_train_transform is the only caller, and only when augment is on.

Why this is the one model allowed colour augmentation: it IS the intervention under test
(Objective 3 of the Phase II deck). Models 1, 2 and 5 deliberately get geometric
augmentation only, so that any difference in the cross-lab drop is attributable to the
method rather than to the augmentation policy.

Randomness comes from torch, so src/utils/seed.py seeds it along with everything else
and a given seed reproduces the same jitter sequence.
"""
import numpy as np
import torch
from PIL import Image
from skimage.color import hed_from_rgb, rgb_from_hed

DEFAULT_SIGMA = 0.05
EPS = 1e-6

# Deliberately NOT skimage's rgb2hed/hed2rgb. Those clamp stain concentrations at zero,
# which destroys near-white pixels: on a real PBC image the round trip alone moves
# bright pixels by 21.9 levels on average (max 135) before any jitter is applied, and
# 42% of a cell crop is bright background. Since augmentation is train-only, that would
# have put a systematic distortion between training and evaluation images and confounded
# Model 3's result. Using the same stain matrices directly in optical-density space
# round-trips exactly (measured max error 0.000).


class HedJitter:
    """Random per-channel scale and shift in HED stain space."""

    def __init__(self, sigma=DEFAULT_SIGMA):
        if sigma < 0:
            raise ValueError(f"hed_jitter sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)

    def _draw(self):
        """One (alpha, beta) pair per stain channel, from the seeded torch RNG."""
        alpha = (1.0 - self.sigma + 2 * self.sigma * torch.rand(3)).numpy().astype(np.float64)
        beta = (-self.sigma + 2 * self.sigma * torch.rand(3)).numpy().astype(np.float64)
        return alpha, beta

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        optical_density = -np.log10(np.clip(rgb, EPS, 1.0))
        concentrations = optical_density @ hed_from_rgb

        alpha, beta = self._draw()
        concentrations = concentrations * alpha + beta

        out = 10.0 ** (-(concentrations @ rgb_from_hed))
        return Image.fromarray(
            np.clip(out * 255.0, 0, 255).round().astype(np.uint8), mode="RGB"
        )

    def describe(self):
        return {"augmentation": "hed_jitter", "sigma": self.sigma}

    def __repr__(self):
        return f"{type(self).__name__}(sigma={self.sigma})"


def build_hed_jitter(spec):
    """From a config value: a dict, a bare number, or None."""
    if not spec:
        return None
    if isinstance(spec, dict):
        return HedJitter(spec.get("sigma", DEFAULT_SIGMA))
    return HedJitter(spec)
