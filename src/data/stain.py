"""Stain normalisation: Macenko (2009) and Reinhard (2001).

Two of the four literature arms in the comparative study (Phase II deck, objective 3).
Both map every image onto the colour statistics of ONE reference image, so that a smear
stained in one lab looks like one stained in the reference lab, and the network sees a
narrower colour distribution.

THE REFERENCE COMES FROM THE PBC TRAIN SPLIT, never from Raabin or LISC. Choosing it
from a target domain would hand the model target-domain information while leaving every
reported number looking clean, which is the exact failure CLAUDE.md sec 3 exists to
prevent. configs/stain_reference.yaml records the chosen file, the rule that chose it
and the size of the candidate pool; scripts/choose_stain_reference.py regenerates it
from pbc_train alone.

Applied to EVERY split -- train, val, source test and every target domain. A model
trained on normalised images and evaluated on raw ones would show a domain drop that is
an artefact of the pipeline rather than of the domain.

MEASURED LIMITATION OF MACENKO ON THIS DATA. Macenko assumes H&E histology: a near-white
background with the whole field stained, and exactly two stains. PBC is Wright-Giemsa
single cells on a tinted background, so few pixels pass the tissue threshold and both
estimated stain vectors carry almost no blue absorbance. Over 40 PBC train images the
reference does not map onto itself (mean RGB shifts by 46.8 in one channel, against 0.5
for Reinhard), and a synthetic tint gap of 32.9 closes only to 8.2 -- worst on
eosinophils (25.2), whose red granules act as a third stain the model cannot represent.
It is run UNMODIFIED anyway: this arm's job is to measure what the literature method
does on blood smears, so a poor result is a finding about the method, not a tuning
failure. A from-scratch implementation of the 2009 paper reproduced torchstain's
behaviour including this failure, which is what established the cause as the method's
assumptions rather than either implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

METHODS = ("none", "macenko", "reinhard")
EPS = 1e-6


class StainError(Exception):
    """The stain reference is missing, malformed, or not from the source domain."""


@dataclass(frozen=True)
class StainReference:
    """The one image both methods are defined by, plus how it was chosen."""

    path: str                 # relative to paths.pbc_images
    dataset: str
    split: str
    selection_rule: str
    candidates: int

    def resolve(self, pbc_images: Path) -> Path:
        """Locate the reference under paths.pbc_images.

        Tries the same order datasets._candidate_paths does -- parent first, then the
        image root itself -- because the PBC CSV rows are written relative to the parent
        ("PBC_dataset_normal_DIB/neutrophil/...") and the reference is recorded the same
        way. Resolving only one of the two would work on one layout and not the other.
        """
        root = Path(pbc_images)
        for candidate in (root.parent / self.path, root / self.path):
            if candidate.exists():
                return candidate
        raise StainError(
            f"stain reference image not found: tried {root.parent / self.path} and "
            f"{root / self.path}. Regenerate it with scripts/choose_stain_reference.py"
        )


def load_reference(path: str | Path) -> StainReference:
    path = Path(path)
    if not path.exists():
        raise StainError(
            f"missing stain reference {path}; generate it with "
            f"`python -m scripts.choose_stain_reference --config <config>` (pbc_train only)"
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = {"path", "dataset", "split", "selection_rule", "candidates"} - set(doc)
    if missing:
        raise StainError(f"{path}: missing {', '.join(sorted(missing))}")
    if doc["dataset"] != "pbc" or doc["split"] != "train":
        raise StainError(
            f"{path}: the stain reference must come from the PBC train split, not "
            f"{doc['dataset']}/{doc['split']} (CLAUDE.md sec 3)"
        )
    return StainReference(path=doc["path"], dataset=doc["dataset"], split=doc["split"],
                          selection_rule=doc["selection_rule"], candidates=int(doc["candidates"]))


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def _as_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


class _Normalizer:
    """Reference statistics are fitted once, never per image."""

    name = "base"

    def __init__(self, reference_image: Path):
        self.reference_path = str(reference_image)
        with Image.open(reference_image) as im:
            self.fit(_as_array(im))

    def fit(self, rgb: np.ndarray) -> None:
        raise NotImplementedError

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, image: Image.Image) -> Image.Image:
        out = self.apply(_as_array(image))
        if not np.isfinite(out).all():
            # A degenerate patch has no stain directions to find. Returning the original
            # is honest; emitting NaNs would poison a whole run silently.
            return image.convert("RGB")
        return _as_image(out)

    def describe(self) -> dict:
        return {"method": self.name, "reference": self.reference_path}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reference={self.reference_path!r})"


class MacenkoNormalizer(_Normalizer):
    """Delegated to torchstain, so this arm is the literature method and not our reading
    of it. See the module docstring for its measured limitation on Wright-Giemsa data."""

    name = "macenko"

    def fit(self, rgb: np.ndarray) -> None:
        import torchstain

        self._impl = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        self._impl.fit(rgb.astype(np.uint8))
        self.reference_stain_matrix = np.asarray(self._impl.HERef, dtype=float)
        self.reference_max_concentration = np.asarray(self._impl.maxCRef, dtype=float)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        out = self._impl.normalize(I=rgb.astype(np.uint8), stains=False)
        return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)

    def describe(self) -> dict:
        import torchstain

        return {**super().describe(), "implementation": f"torchstain {torchstain.__version__}"}


class ReinhardNormalizer(_Normalizer):
    """Match per-channel mean and standard deviation in LAB space, assuming the shift
    between labs is a global colour transfer. Maps its own reference onto itself to
    within 0.5 mean-RGB, so unlike Macenko its assumptions hold on this data."""

    name = "reinhard"

    def fit(self, rgb: np.ndarray) -> None:
        lab = rgb2lab(rgb / 255.0).reshape(-1, 3)
        self.reference_mean = lab.mean(axis=0)
        self.reference_std = lab.std(axis=0)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        lab = rgb2lab(rgb / 255.0)
        flat = lab.reshape(-1, 3)
        matched = ((flat - flat.mean(axis=0)) / (flat.std(axis=0) + EPS)
                   * self.reference_std + self.reference_mean)
        return lab2rgb(matched.reshape(lab.shape)) * 255.0


NORMALIZERS = {"macenko": MacenkoNormalizer, "reinhard": ReinhardNormalizer}


def build_normalizer(cfg):
    """The stain step for this config, or None when the arm does not normalise."""
    spec = getattr(cfg.data, "stain_norm", None)
    if spec is None or spec.method == "none":
        return None
    if spec.method not in NORMALIZERS:
        raise StainError(f"unknown stain method {spec.method!r}; known: {', '.join(METHODS)}")
    reference = load_reference(spec.reference)
    return NORMALIZERS[spec.method](reference.resolve(cfg.paths.pbc_images))
