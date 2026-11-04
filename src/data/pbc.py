"""PBC + WBCAtt: the source domain, with 5-class labels and 11 attribute labels.

The split is WBCAtt's, read straight from data/raw/pbc_attr_v1_{split}.csv. Rows whose
filename appears in the mapping's excluded_files are dropped, which is what takes the
splits from 6,169 / 1,030 / 3,099 to 6,165 / 1,028 / 3,098 (see
reports/audit_report.md section 3 for why each file is excluded).
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.mapping import MappingError, load_attribute_space, load_label_space
from src.data.transforms import build_transform

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PBC_DIR = RAW / "PBC_dataset_normal_DIB"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PbcRecord:
    img_name: str
    path: Path
    label: int
    attributes: tuple


class PbcDataset(Dataset):
    def __init__(self, split, image_size=224, normalize="imagenet",
                 labels=None, attributes=None, transform=None):
        if split not in SPLITS:
            raise MappingError(f"unknown PBC split {split!r}; expected one of {SPLITS}")
        self.split = split
        self.labels = labels or load_label_space()
        self.attributes = attributes or load_attribute_space()
        self.transform = transform or build_transform(image_size, normalize)

        csv_path = RAW / f"pbc_attr_v1_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        # Fail on an unmapped label before building anything, so the error names the
        # class rather than surfacing as a missing key somewhere downstream.
        self.labels.validate_observed({f"wbcatt:{row['label']}" for row in rows})

        dropped = self.labels.excluded_image_names()
        self.records = []
        for row in rows:
            if row["img_name"] in dropped:
                continue
            self.records.append(PbcRecord(
                img_name=row["img_name"],
                path=RAW / row["path"],
                label=self.labels.index_for(f"wbcatt:{row['label']}"),
                attributes=self.attributes.vector_for(row),
            ))
        self.excluded_count = len(rows) - len(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return (tensor,
                torch.tensor(record.label, dtype=torch.long),
                torch.tensor(record.attributes, dtype=torch.long))

    def __repr__(self):
        return (f"{type(self).__name__}(split={self.split!r}, n={len(self)}, "
                f"excluded={self.excluded_count})")
