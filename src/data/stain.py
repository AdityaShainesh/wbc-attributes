"""Stain normalisation: Macenko and Reinhard (Model 2).

Both map an image onto the colour statistics of a single reference image, so that a
smear stained in one lab looks like one stained in the reference lab. The CNN then sees
a narrower colour distribution and, in principle, transfers better.

THE REFERENCE MUST COME FROM PBC TRAIN. Choosing it from Raabin or LISC would hand the
model target-domain information while every number still looked clean -- precisely the
failure CLAUDE.md section 3 exists to prevent. configs/stain_reference.yaml records the
chosen file, the rule that chose it, and the precomputed statistics;
scripts/choose_stain_reference.py regenerates it from the PBC train split alone.

Reference statistics are computed once at construction, never per image: recomputing
per call would be slow and would silently drift if the reference file changed mid-run.

Macenko (2009): convert to optical density, drop near-white pixels, take the two
principal stain directions by SVD, read off robust per-stain concentrations, then
re-express those concentrations in the reference's stain basis.

Reinhard (2001): match per-channel mean and standard deviation in LAB space, which is
cheaper and assumes the shift is a global colour transfer.
"""
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

from src.data.mapping import MappingError

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
REFERENCE_PATH = CONFIGS / "stain_reference.yaml"

# Macenko hyperparameters, from the paper's defaults.
OD_WHITE_THRESHOLD = 0.15   # optical density below this is background, not stain
PERCENTILE = 99             # robust max concentration, instead of the true max
ALPHA = 1                   # angular percentile when picking the two stain vectors
EPS = 1e-6


def _to_array(image):
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def _to_image(array):
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def optical_density(rgb):
    """OD = -log10(I / 255). Clipped so pure black does not become infinite."""
    return -np.log10(np.clip(rgb, 1.0, 255.0) / 255.0)


class StainNormalizer:
    """Shared plumbing. Subclasses implement fit_reference and apply."""

    name = "base"

    def __init__(self, reference):
        self.reference_path = reference["path"]
        self.fit_reference(_to_array(Image.open(reference["absolute_path"])))

    def fit_reference(self, rgb):
        raise NotImplementedError

    def apply(self, rgb):
        raise NotImplementedError

    def __call__(self, image):
        out = self.apply(_to_array(image))
        if not np.isfinite(out).all():
            # Degenerate input (a flat patch has no stain directions to find). Returning
            # the original is honest; silently emitting NaNs would poison a whole run.
            return image.convert("RGB")
        return _to_image(out)

    def describe(self):
        return {"method": self.name, "reference": self.reference_path}


class MacenkoNormalizer(StainNormalizer):
    """Macenko (2009), delegated to torchstain so the arm is the literature method.

    KNOWN LIMITATION, measured over 40 PBC train images on 2026-11-05
    (reports/stain_normalisation.md). Macenko assumes H&E histology: a near-white
    background with the whole field stained, and exactly two stains. PBC crops are
    Wright-Giemsa single cells on a tinted pink background, so few pixels pass the
    tissue threshold and both estimated stain vectors carry almost no blue absorbance.
    The reference image does not map to itself: mean RGB shifts by 46.8 in one channel,
    against 0.5 for Reinhard. It still reduces a synthetic tint gap (32.9 -> 8.2 on
    average) but far less than Reinhard (-> 0.3), and worst on eosinophils (-> 25.2),
    whose red granules act as a third stain the model cannot represent.

    This is kept as-is rather than patched, because Model 2a's job is to measure what
    the literature method does on this data. A from-scratch implementation of the paper
    (commit history, removed once it had served its purpose) reproduced torchstain's
    behaviour including this failure, which is the evidence that the problem lies in the
    method's assumptions rather than in either implementation.
    """

    name = "macenko"

    def fit_reference(self, rgb):
        import torchstain
        self._impl = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        self._impl.fit(rgb.astype(np.uint8))
        # Exposed for inspection and for the results file; torchstain's own fitted
        # reference basis, not a second estimate of it.
        self.reference_stain_matrix = np.asarray(self._impl.HERef, dtype=float)
        self.reference_max_concentration = np.asarray(self._impl.maxCRef, dtype=float)

    def apply(self, rgb):
        out = self._impl.normalize(I=rgb.astype(np.uint8), stains=False)
        return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)

    def describe(self):
        import torchstain
        return {**super().describe(), "implementation": f"torchstain {torchstain.__version__}"}


class ReinhardNormalizer(StainNormalizer):
    name = "reinhard"

    def fit_reference(self, rgb):
        lab = rgb2lab(rgb / 255.0)
        self.reference_mean = lab.reshape(-1, 3).mean(axis=0)
        self.reference_std = lab.reshape(-1, 3).std(axis=0)

    def apply(self, rgb):
        lab = rgb2lab(rgb / 255.0)
        flat = lab.reshape(-1, 3)
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        matched = (flat - mean) / (std + EPS) * self.reference_std + self.reference_mean
        return lab2rgb(matched.reshape(lab.shape)) * 255.0


NORMALIZERS = {"macenko": MacenkoNormalizer, "reinhard": ReinhardNormalizer}


def load_reference(path=REFERENCE_PATH):
    """Read configs/stain_reference.yaml and resolve the image path."""
    path = Path(path)
    if not path.exists():
        raise MappingError(
            f"missing {path.name}; generate it with "
            f"`python scripts/choose_stain_reference.py` (PBC train only)"
        )
    reference = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    reference["absolute_path"] = raw / reference["path"]
    if reference.get("dataset") != "pbc" or reference.get("split") != "train":
        raise MappingError(
            f"the stain reference must come from the PBC train split, not "
            f"{reference.get('dataset')}/{reference.get('split')} (CLAUDE.md section 3)"
        )
    return reference


def build_normalizer(method, reference=None):
    if method not in NORMALIZERS:
        raise MappingError(
            f"unknown stain normalisation method {method!r}; known: {', '.join(NORMALIZERS)}"
        )
    return NORMALIZERS[method](reference or load_reference())
