"""Tests for HED colour augmentation (Model 3).

Two properties matter beyond "it runs". It must be off at evaluation time -- augmenting
a test set would be silent cheating -- and it must be driven by the seeded torch RNG, so
that a run with seed 0 sees the same jitter every time it is repeated.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from src.data import build_dataset
from src.data.hed import HedJitter
from src.data.pbc import PBC_DIR
from src.data.transforms import build_train_transform, build_transform


def sample(seed=0):
    rng = np.random.RandomState(seed)
    base = np.full((48, 48, 3), 220.0)
    base[12:36, 12:36] = (130, 95, 175)
    return Image.fromarray(np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8))


def as_array(image):
    return np.asarray(image, dtype=float)


# ---------------------------------------------------------------------- basics

def test_output_is_rgb_of_the_same_size():
    out = HedJitter(sigma=0.05)(sample())
    assert isinstance(out, Image.Image) and out.mode == "RGB" and out.size == (48, 48)


def test_values_stay_in_range():
    out = as_array(HedJitter(sigma=0.3)(sample()))
    assert out.min() >= 0 and out.max() <= 255


def test_zero_sigma_is_an_exact_identity():
    """Only uint8 rounding may differ.

    skimage's rgb2hed/hed2rgb clamp concentrations at zero and fail this badly (max
    error 135 on a real image), which is why src/data/hed.py does the transform in
    optical-density space instead. Augmentation is train-only, so a lossy round trip
    would put a systematic difference between train and eval images.
    """
    source = sample()
    out = as_array(HedJitter(sigma=0.0)(source))
    assert abs(out - as_array(source)).max() <= 1.0


def test_larger_sigma_perturbs_more():
    source = sample()
    torch.manual_seed(0)
    small = abs(as_array(HedJitter(sigma=0.02)(source)) - as_array(source)).mean()
    torch.manual_seed(0)
    large = abs(as_array(HedJitter(sigma=0.20)(source)) - as_array(source)).mean()
    assert large > small * 2


def test_it_actually_changes_the_image():
    torch.manual_seed(0)
    assert not np.array_equal(as_array(HedJitter(sigma=0.05)(sample())), as_array(sample()))


# ------------------------------------------------------------------ randomness

def test_same_torch_seed_gives_the_same_jitter():
    jitter, source = HedJitter(sigma=0.1), sample()
    torch.manual_seed(7)
    first = as_array(jitter(source))
    torch.manual_seed(7)
    second = as_array(jitter(source))
    assert np.array_equal(first, second)


def test_successive_calls_differ():
    jitter, source = HedJitter(sigma=0.1), sample()
    torch.manual_seed(7)
    first = as_array(jitter(source))
    second = as_array(jitter(source))
    assert not np.array_equal(first, second)


def test_each_hed_channel_gets_its_own_factors():
    """A single shared factor would be a brightness change, not a stain change."""
    torch.manual_seed(0)
    jitter = HedJitter(sigma=0.1)
    alpha, beta = jitter._draw()
    assert len(set(np.round(alpha, 6))) == 3
    assert len(set(np.round(beta, 6))) == 3


# ------------------------------------------------------- never at evaluation

def test_eval_transform_never_contains_jitter():
    transform = build_transform(224, "imagenet")
    assert not any(isinstance(step, HedJitter) for step in transform.transforms)


def test_train_transform_includes_jitter_only_when_configured():
    with_jitter = build_train_transform(224, "imagenet", True, None, hed_jitter={"sigma": 0.05})
    without = build_train_transform(224, "imagenet", True, None)
    assert any(isinstance(s, HedJitter) for s in with_jitter.transforms)
    assert not any(isinstance(s, HedJitter) for s in without.transforms)


def test_augment_false_disables_jitter_too():
    transform = build_train_transform(224, "imagenet", False, None, hed_jitter={"sigma": 0.05})
    assert not any(isinstance(s, HedJitter) for s in transform.transforms)


@pytest.mark.skipif(not PBC_DIR.exists(), reason="data/raw/ not populated")
def test_a_val_split_built_from_config_has_no_jitter():
    spec = {"dataset": "pbc", "split": "val", "hed_jitter": {"sigma": 0.05}, "subset_size": 2}
    dataset = build_dataset(spec, train=False)
    inner = getattr(dataset, "dataset", dataset)   # unwrap Subset
    assert not any(isinstance(s, HedJitter) for s in inner.transform.transforms)
