"""LISC: a target domain. Class labels only, no attribute labels.

EVALUATION ONLY. CLAUDE.md section 3 forbids using this data for training, validation,
early stopping, model or checkpoint selection, hyperparameter tuning, normalization
statistics or threshold selection. Nothing in this module computes a statistic over the
images; it reads filenames and returns tensors through the transform fixed by PBC.

Classes come from the filename prefix (neut, lymp, mono, eosi, Baso), matched
case-sensitively against the mapping. The folder-level exclusion of Train_Aug is enforced
here: those 2,006 files are augmented copies of Train, 180 of them byte-identical to it.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.mapping import MappingError, load_label_space
from src.data.transforms import build_transform

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LISC_DIR = RAW / "LISC"


@dataclass(frozen=True)
class LiscRecord:
    img_name: str
    path: Path
    label: int


class LiscDataset(Dataset):
    def __init__(self, folder="Test", image_size=224, normalize="imagenet",
                 labels=None, transform=None):
        self.labels = labels or load_label_space()
        excluded_folders = self.labels.excluded_folders("lisc")
        if folder in excluded_folders:
            raise MappingError(
                f"{folder!r} is excluded in class_mapping.yaml "
                f"({self.labels.exclusion_reason('lisc', folder)}); it cannot be loaded"
            )
        directory = LISC_DIR / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"missing {directory}")

        self.folder = folder
        self.transform = transform or build_transform(image_size, normalize)

        excluded_files = self.labels.excluded_files("lisc")
        paths = [p for p in sorted(directory.iterdir())
                 if p.is_file() and not p.name.startswith(".")
                 and f"{folder}/{p.name}" not in excluded_files]

        self.labels.validate_observed({f"lisc:{p.name.split('_')[0]}" for p in paths})
        self.records = [
            LiscRecord(img_name=p.name, path=p,
                       label=self.labels.index_for(f"lisc:{p.name.split('_')[0]}"))
            for p in paths
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        # None, not a zero vector: LISC has no attribute labels, and a zero vector would
        # silently become a training signal if this ever reached an attribute loss.
        return tensor, torch.tensor(record.label, dtype=torch.long), None

    def __repr__(self):
        return f"{type(self).__name__}(folder={self.folder!r}, n={len(self)})"
