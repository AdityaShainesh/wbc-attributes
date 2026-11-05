"""Tests for src/data/stain.py -- Macenko and Reinhard stain normalisation (Model 2).

The leakage tests matter most. Both methods are defined by statistics of a reference
image, and if that reference were ever drawn from Raabin or LISC the zero-shot claim
would be void while every number still looked fine (CLAUDE.md section 3).
"""
import numpy as np
import pytest
import yaml
from PIL import Image

from src.data.stain import (
    REFERENCE_PATH,
    MacenkoNormalizer,
    ReinhardNormalizer,
    build_normalizer,
    load_reference,
)
from src.data.mapping import MappingError


def synthetic(seed=0, tint=(1.0, 1.0, 1.0), size=64):
    """A cell-like image: pale background, darker blob, tinted by `tint`."""
    rng = np.random.RandomState(seed)
    image = np.full((size, size, 3), 235.0)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 4) ** 2
    image[blob] = (120.0, 90.0, 170.0)
    image += rng.normal(0, 3, image.shape)
    image *= np.asarray(tint)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


@pytest.fixture(scope="module")
def reference():
    return load_reference()


@pytest.fixture(params=["macenko", "reinhard"])
def normalizer(request, reference):
    return build_normalizer(request.param, reference)


# ------------------------------------------------------------------- leakage

def test_reference_image_comes_from_the_pbc_train_split(reference):
    assert reference["dataset"] == "pbc"
    assert reference["split"] == "train"
    assert reference["path"].startswith("PBC_dataset_normal_DIB/")


def test_reference_file_records_how_it_was_chosen(reference):
    # A reference picked by eye is unauditable; the rule and the candidate pool size
    # must be recorded alongside the filename.
    assert reference["selection_rule"]
    assert reference["candidates"] > 1000


def test_reference_config_mentions_no_target_dataset():
    text = REFERENCE_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("lisc", "raabin"):
        assert f"{forbidden}/" not in text and f"{forbidden}_" not in text


# ------------------------------------------------------------------ behaviour

def test_output_is_an_image_of_the_same_size_and_mode(normalizer):
    source = synthetic(tint=(1.1, 0.9, 0.95))
    out = normalizer(source)
    assert isinstance(out, Image.Image)
    assert out.size == source.size
    assert out.mode == "RGB"


def test_normalisation_is_deterministic(normalizer):
    source = synthetic(tint=(0.8, 1.0, 1.2))
    first = np.asarray(normalizer(source), dtype=float)
    second = np.asarray(normalizer(source), dtype=float)
    assert np.array_equal(first, second)


def real_image(reference):
    """A real PBC image. Synthetic two-colour patches are too degenerate for Macenko."""
    path = reference["absolute_path"].parent.parent / "lymphocyte" / "LY_883318.jpg"
    if not path.exists():
        pytest.skip("data/raw/ not populated")
    return Image.open(path).convert("RGB")


def tinted(image, tint):
    array = np.asarray(image, dtype=float) * np.asarray(tint)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def test_two_differently_tinted_copies_end_up_closer_together(normalizer, reference):
    """The whole point: remove the colour difference, keep the structure."""
    source = real_image(reference)
    warm, cool = tinted(source, (1.15, 0.95, 0.90)), tinted(source, (0.90, 0.98, 1.20))

    before = abs(np.asarray(warm, float).mean(axis=(0, 1))
                 - np.asarray(cool, float).mean(axis=(0, 1))).mean()
    after = abs(np.asarray(normalizer(warm), float).mean(axis=(0, 1))
                - np.asarray(normalizer(cool), float).mean(axis=(0, 1))).mean()
    assert after < before


def test_structure_survives_normalisation(normalizer, reference):
    """The cell must still be darker than the background afterwards."""
    source = real_image(reference)
    before = np.asarray(source, dtype=float).mean(axis=2)
    after = np.asarray(normalizer(source), dtype=float).mean(axis=2)
    # The pixels that were darkest before must still be darker than the rest.
    dark = before < np.percentile(before, 10)
    assert after[dark].mean() < after[~dark].mean() - 10


def test_reinhard_maps_the_reference_onto_itself(reference):
    """A normaliser that cannot reproduce its own reference is miscalibrated.

    Macenko is deliberately excluded: on Wright-Giemsa crops it fails this, which is
    the documented limitation in reports/stain_normalisation.md, not a regression.
    """
    normalizer = build_normalizer("reinhard", reference)
    source = Image.open(reference["absolute_path"]).convert("RGB")
    before = np.asarray(source, float).mean(axis=(0, 1))
    after = np.asarray(normalizer(source), float).mean(axis=(0, 1))
    assert abs(before - after).max() < 2.0


def test_a_flat_image_does_not_crash_or_produce_nan(normalizer):
    """Macenko's SVD is undefined with no stain variation; it must degrade, not explode."""
    flat = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
    out = np.asarray(normalizer(flat), dtype=float)
    assert np.isfinite(out).all()


# -------------------------------------------------------------- construction

def test_unknown_method_is_rejected(reference):
    with pytest.raises(MappingError, match="vahadane"):
        build_normalizer("vahadane", reference)


def test_each_method_builds_its_own_class(reference):
    assert isinstance(build_normalizer("macenko", reference), MacenkoNormalizer)
    assert isinstance(build_normalizer("reinhard", reference), ReinhardNormalizer)


def test_normalizers_describe_themselves_for_the_results_file(normalizer):
    description = normalizer.describe()
    assert description["method"] in ("macenko", "reinhard")
    assert description["reference"].startswith("PBC_dataset_normal_DIB/")


def test_reference_statistics_are_precomputed_not_recomputed_per_image(reference):
    # Recomputing the reference per call would be both slow and a silent source of
    # drift if the reference file changed mid-run.
    macenko = build_normalizer("macenko", reference)
    assert macenko.reference_stain_matrix.shape == (3, 2)
    reinhard = build_normalizer("reinhard", reference)
    assert len(reinhard.reference_mean) == 3 and len(reinhard.reference_std) == 3
