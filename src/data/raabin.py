"""Raabin-WBC: a target domain. Class labels only, no attribute labels.

EVALUATION ONLY, with one exception. CLAUDE.md section 3 forbids using this data for
training, validation, early stopping, model or checkpoint selection, hyperparameter
tuning, normalisation statistics or threshold selection. The sole exception is Model 4
(DANN), which may consume the `Train` split WITHOUT LABELS for its domain head; that
split is never evaluated, and no other model may touch it.

Classes come from the folder name, capitalised on disk (`Neutrophil`, not `neutrophil`),
matched case-sensitively against configs/class_mapping.yaml.

Splits, from the double-labeled cropped release (Kouzehkanan et al. 2022):

    Train   10,175 images, 5 classes  -- DANN adaptation only
    TestA    4,339 images, 5 classes  -- zero-shot evaluation
    Test-B   2,119 images, 2 classes  -- zero-shot evaluation, SECOND microscope/camera

Test-B carries the largest domain shift in the project and only two of the five classes
(lymphocyte 148, neutrophil 1,971). Five-class macro-F1 on it is capped at 0.4 by the
three absent classes, so it is reported as a two-class evaluation in its own labelled
row. `present_classes` below exposes that to the eval harness so the limitation travels
with the data instead of living in someone's memory. See configs/class_mapping.yaml,
`target_splits`.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from src.data.mapping import MAPPING_PATH, MappingError, load_label_space
from src.data.transforms import build_transform

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RAABIN_DIR = RAW / "raabin"
SPLITS = ("Train", "TestA", "Test-B")
ADAPTATION_SPLITS = ("Train",)


@dataclass(frozen=True)
class RaabinRecord:
    img_name: str
    path: Path
    label: int


def split_policy(split, path=MAPPING_PATH):
    """What class_mapping.yaml says this split is for and which classes it covers."""
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    policy = (mapping.get("target_splits") or {}).get("raabin", {}).get(split)
    if policy is None:
        raise MappingError(
            f"raabin split {split!r} is not declared under target_splits in "
            f"{Path(path).name}; declare its role and class coverage before using it"
        )
    return policy


class RaabinDataset(Dataset):
    def __init__(self, split="TestA", image_size=224, normalize="imagenet",
                 labels=None, transform=None, allow_adaptation_split=False):
        if split not in SPLITS:
            raise MappingError(f"unknown Raabin split {split!r}; expected one of {SPLITS}")
        # The Train split exists for Model 4's domain head and nothing else. Loading it
        # from anywhere that has not said so explicitly is a leak, so it is refused by
        # default rather than guarded by a comment.
        if split in ADAPTATION_SPLITS and not allow_adaptation_split:
            raise MappingError(
                f"raabin/{split} is the DANN adaptation split (CLAUDE.md section 3): "
                f"unlabeled use by Model 4 only. Pass allow_adaptation_split=True if "
                f"this really is the DANN domain head; never evaluate on it."
            )

        directory = RAABIN_DIR / split
        if not directory.is_dir():
            raise FileNotFoundError(
                f"missing {directory}; extract it with python scripts/extract_raw.py"
            )

        self.split = split
        self.policy = split_policy(split)
        self.present_classes = self.policy.get("present_classes")
        self.labels = labels or load_label_space()
        self.transform = transform or build_transform(image_size, normalize)

        folders = sorted(d for d in directory.iterdir() if d.is_dir())
        self.labels.validate_observed({f"raabin:{d.name}" for d in folders})

        self.records = []
        for folder in folders:
            label = self.labels.index_for(f"raabin:{folder.name}")
            for path in sorted(p for p in folder.iterdir()
                               if p.is_file() and not p.name.startswith(".")):
                self.records.append(RaabinRecord(img_name=path.name, path=path, label=label))

    @property
    def class_count(self):
        """Classes actually present, which is 2 for Test-B and 5 elsewhere."""
        return len({record.label for record in self.records})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        # None, not zeros: Raabin has no attribute labels, and a zero vector would
        # silently become a training signal if it reached an attribute loss.
        return tensor, torch.tensor(record.label, dtype=torch.long), None

    def __repr__(self):
        return (f"{type(self).__name__}(split={self.split!r}, n={len(self)}, "
                f"classes={self.class_count})")
