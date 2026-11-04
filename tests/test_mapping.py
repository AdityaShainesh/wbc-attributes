"""Tests for src/data/mapping.py -- the config-to-label-space layer.

The point of these is the hard-error behaviour CLAUDE.md section 4 demands: an unmapped
class name must stop the run, not be quietly dropped. Each test builds a small mapping
dict in memory rather than editing configs/class_mapping.yaml, so the real config is
only exercised by the tests at the bottom.
"""
import pytest
import yaml

from src.data.mapping import (
    ATTRIBUTES_PATH,
    MAPPING_PATH,
    AttributeSpace,
    LabelSpace,
    MappingError,
)

MINIMAL = {
    "neutrophil": ["pbc:neutrophil", "wbcatt:Neutrophil", "lisc:neut"],
    "lymphocyte": ["pbc:lymphocyte", "wbcatt:Lymphocyte", "lisc:lymp"],
    "monocyte": ["pbc:monocyte", "wbcatt:Monocyte", "lisc:mono"],
    "eosinophil": ["pbc:eosinophil", "wbcatt:Eosinophil", "lisc:eosi"],
    "basophil": ["pbc:basophil", "wbcatt:Basophil", "lisc:Baso"],
    "excluded": {"pbc:platelet": "not a white blood cell"},
    "excluded_files": {},
    "secondary_subsets": {},
}


def space(**overrides):
    return LabelSpace(dict(MINIMAL, **overrides))


# --------------------------------------------------------------- the hard errors

def test_unmapped_class_name_raises_and_names_the_offender():
    labels = space()
    with pytest.raises(MappingError) as exc:
        labels.index_for("pbc:erythroblast")
    assert "pbc:erythroblast" in str(exc.value)


def test_unmapped_name_in_a_batch_of_observed_names_raises():
    labels = space()
    with pytest.raises(MappingError) as exc:
        labels.validate_observed({"pbc:neutrophil", "pbc:erythroblast", "lisc:junk"})
    message = str(exc.value)
    assert "pbc:erythroblast" in message and "lisc:junk" in message
    assert "pbc:neutrophil" not in message


def test_name_listed_under_two_classes_raises_at_construction():
    broken = dict(MINIMAL)
    broken["lymphocyte"] = MINIMAL["lymphocyte"] + ["pbc:neutrophil"]
    with pytest.raises(MappingError, match="pbc:neutrophil"):
        LabelSpace(broken)


def test_name_both_mapped_and_excluded_raises_at_construction():
    with pytest.raises(MappingError, match="pbc:neutrophil"):
        LabelSpace(dict(MINIMAL, excluded={"pbc:neutrophil": "contradiction"}))


def test_excluded_name_is_not_a_label_but_is_recognised():
    labels = space()
    assert labels.is_excluded("pbc:platelet")
    with pytest.raises(MappingError):
        labels.index_for("pbc:platelet")


def test_wrong_number_of_target_classes_raises():
    four = {k: v for k, v in MINIMAL.items() if k != "basophil"}
    with pytest.raises(MappingError, match="5"):
        LabelSpace(four)


# ------------------------------------------------------------------ index layout

def test_class_indices_follow_config_order_and_are_contiguous():
    labels = space()
    assert labels.classes == ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]
    assert [labels.index_for(f"pbc:{c}") for c in labels.classes] == [0, 1, 2, 3, 4]


def test_every_dataset_name_for_a_class_maps_to_the_same_index():
    labels = space()
    assert (labels.index_for("pbc:neutrophil")
            == labels.index_for("wbcatt:Neutrophil")
            == labels.index_for("lisc:neut"))


def test_matching_is_case_sensitive():
    labels = space()
    with pytest.raises(MappingError):
        labels.index_for("wbcatt:neutrophil")  # real label is "Neutrophil"


# -------------------------------------------------------------------- exclusions

def test_excluded_files_are_reported_per_dataset():
    labels = space(excluded_files={
        "pbc:neutrophil/BNE_1.jpg": "reason",
        "lisc:Train_Aug/": "folder",
    })
    assert labels.excluded_files("pbc") == {"neutrophil/BNE_1.jpg"}
    assert labels.excluded_folders("lisc") == {"Train_Aug"}
    assert labels.excluded_files("lisc") == set()


def test_excluded_image_names_are_exposed_for_filename_matching():
    labels = space(excluded_files={"pbc:basophil/BA_225540.jpg": "duplicate"})
    assert labels.excluded_image_names() == {"BA_225540.jpg"}


# ------------------------------------------------------------ attribute vocabulary

def test_attribute_space_indexes_values_in_declared_order():
    attrs = AttributeSpace({"attributes": {"granule_type": ["nil", "small", "round", "coarse"]}})
    assert attrs.heads == ["granule_type"]
    assert attrs.head_sizes == [4]
    assert attrs.index_for("granule_type", "nil") == 0
    assert attrs.index_for("granule_type", "coarse") == 3


def test_unknown_attribute_value_raises():
    attrs = AttributeSpace({"attributes": {"granularity": ["no", "yes"]}})
    with pytest.raises(MappingError) as exc:
        attrs.index_for("granularity", "maybe")
    assert "granularity" in str(exc.value) and "maybe" in str(exc.value)


def test_unknown_attribute_name_raises():
    attrs = AttributeSpace({"attributes": {"granularity": ["no", "yes"]}})
    with pytest.raises(MappingError, match="nucleus_shape"):
        attrs.index_for("nucleus_shape", "irregular")


def test_duplicate_value_in_a_vocabulary_raises():
    with pytest.raises(MappingError, match="granularity"):
        AttributeSpace({"attributes": {"granularity": ["no", "yes", "no"]}})


# ----------------------------------------------------- the real configs on disk

def test_real_mapping_loads_and_has_five_classes():
    labels = LabelSpace(yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8")))
    assert labels.classes == ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]


def test_real_mapping_excludes_the_eight_audited_files():
    labels = LabelSpace(yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8")))
    assert labels.excluded_image_names() == {
        ".DS_169665.jpg",      # macOS metadata with a .jpg name
        "BA_225540.jpg",       # train-side copies of test images
        "BA_381452.jpg",
        "EO_918881.jpg",
        "MO_997225.jpg",
        "BA_435917.jpg",       # val-side copy of a test image
        "BNE_191112.jpg",      # the conflicting-label pair, both halves
        "EO_225902.jpg",
    }
    assert labels.excluded_folders("lisc") == {"Train_Aug"}


def test_real_attribute_config_has_eleven_heads():
    attrs = AttributeSpace(yaml.safe_load(ATTRIBUTES_PATH.read_text(encoding="utf-8")))
    assert len(attrs.heads) == 11
    assert all(size >= 2 for size in attrs.head_sizes)
