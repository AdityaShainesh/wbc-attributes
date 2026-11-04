"""Tests for the PBC and LISC datasets.

These read the real data under data/raw/, so they are skipped when it is absent. The
counts asserted here are the post-exclusion split sizes verified by the first audit
(reports/audit_report.md section 3): 6,165 / 1,028 / 3,098.
"""
import pytest

from src.data import build_dataset
from src.data.lisc import LISC_DIR, LiscDataset
from src.data.mapping import MappingError
from src.data.pbc import PBC_DIR, PbcDataset

pytestmark = pytest.mark.skipif(not PBC_DIR.exists(), reason="data/raw/ not populated")

EXPECTED_SIZES = {"train": 6165, "val": 1028, "test": 3098}


@pytest.fixture(scope="module")
def pbc_splits():
    return {split: PbcDataset(split) for split in EXPECTED_SIZES}


# ------------------------------------------------------------------ split sizes

@pytest.mark.parametrize("split,expected", EXPECTED_SIZES.items())
def test_split_size_matches_the_audit(pbc_splits, split, expected):
    assert len(pbc_splits[split]) == expected


def test_excluded_files_are_absent_from_every_split(pbc_splits):
    excluded = pbc_splits["train"].labels.excluded_image_names()
    for split, dataset in pbc_splits.items():
        assert not excluded & {record.img_name for record in dataset.records}, split


def test_splits_are_disjoint_by_filename(pbc_splits):
    names = {s: {r.img_name for r in d.records} for s, d in pbc_splits.items()}
    assert not names["train"] & names["val"]
    assert not names["train"] & names["test"]
    assert not names["val"] & names["test"]


def test_every_record_resolves_to_a_file_on_disk(pbc_splits):
    for split, dataset in pbc_splits.items():
        missing = [r.img_name for r in dataset.records if not r.path.exists()]
        assert not missing, f"{split}: {missing[:5]}"


# ----------------------------------------------------------------- label space

def test_class_indices_are_in_range_and_cover_all_five(pbc_splits):
    for dataset in pbc_splits.values():
        indices = {r.label for r in dataset.records}
        assert indices == {0, 1, 2, 3, 4}


def test_class_counts_match_the_audit(pbc_splits):
    counts = {}
    for r in pbc_splits["val"].records:
        counts[pbc_splits["val"].labels.classes[r.label]] = counts.get(
            pbc_splits["val"].labels.classes[r.label], 0) + 1
    # val after exclusions: one basophil and one neutrophil dropped
    assert counts == {"neutrophil": 333, "lymphocyte": 111, "monocyte": 146,
                      "eosinophil": 314, "basophil": 124}


# ------------------------------------------------------------------- attributes

def test_attribute_vector_has_one_index_per_head(pbc_splits):
    dataset = pbc_splits["val"]
    record = dataset.records[0]
    assert len(record.attributes) == len(dataset.attributes.heads)
    for value, size in zip(record.attributes, dataset.attributes.head_sizes):
        assert 0 <= value < size


def test_every_attribute_value_in_the_data_is_in_the_vocabulary(pbc_splits):
    # PbcDataset raises on an unknown value, so reaching here at all is the assertion;
    # this makes the intent explicit for all three splits.
    for dataset in pbc_splits.values():
        assert all(len(r.attributes) == 11 for r in dataset.records)


def test_attribute_indices_are_stable_across_reloads():
    first = PbcDataset("val")
    second = PbcDataset("val")
    assert [r.attributes for r in first.records[:50]] == [r.attributes for r in second.records[:50]]


# ------------------------------------------------------------------------- items

def test_item_returns_image_tensor_label_and_attributes(pbc_splits):
    image, label, attributes = pbc_splits["val"][0]
    assert image.shape == (3, 224, 224)
    assert 0 <= int(label) < 5
    assert attributes.shape == (11,)


def test_image_tensor_is_float_and_normalized(pbc_splits):
    image, _, _ = pbc_splits["val"][0]
    assert image.dtype.is_floating_point
    assert -5 < float(image.min()) and float(image.max()) < 5


# -------------------------------------------------------------------------- LISC

@pytest.mark.skipif(not LISC_DIR.exists(), reason="LISC not extracted")
class TestLisc:
    def test_train_and_test_sizes(self):
        assert len(LiscDataset("Train")) == 180
        assert len(LiscDataset("Test")) == 77

    def test_train_aug_is_refused_because_the_mapping_excludes_it(self):
        with pytest.raises(MappingError, match="Train_Aug"):
            LiscDataset("Train_Aug")

    def test_classes_map_onto_the_same_five_indices(self):
        dataset = LiscDataset("Test")
        assert {r.label for r in dataset.records} == {0, 1, 2, 3, 4}
        assert dataset.labels.classes == PbcDataset("val").labels.classes

    def test_items_have_the_same_shape_as_pbc(self):
        image, label, attributes = LiscDataset("Test")[0]
        assert image.shape == (3, 224, 224)
        assert 0 <= int(label) < 5
        assert attributes is None  # LISC has no attribute labels

    def test_lisc_paths_never_point_into_pbc(self):
        for record in LiscDataset("Train").records:
            assert PBC_DIR not in record.path.parents


# -------------------------------------------------------- the config entry point

def test_build_dataset_from_a_config_dict():
    config = {"dataset": "pbc", "split": "val", "image_size": 224}
    dataset = build_dataset(config)
    assert len(dataset) == EXPECTED_SIZES["val"]


def test_build_dataset_rejects_an_unknown_dataset_name():
    with pytest.raises(MappingError, match="raabin"):
        build_dataset({"dataset": "raabin", "split": "test"})
