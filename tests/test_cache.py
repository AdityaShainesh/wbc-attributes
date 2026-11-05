"""Tests for the stain cache and the WBC_NUM_WORKERS override.

The equivalence test is the important one: a cache that quietly returns something other
than the live pipeline would change every Model 2 number while looking like a speedup.
"""
import numpy as np
import pytest
import torch

from src.data import build_dataset, resolve_num_workers
from src.data.pbc import PBC_DIR, PbcDataset
from src.data.transforms import build_deterministic_steps

pytestmark = pytest.mark.skipif(not PBC_DIR.exists(), reason="data/raw/ not populated")

METHOD = "reinhard"   # exact and fast enough to cache a handful of images in a test
SIZE = 224


@pytest.fixture
def tiny_cache(tmp_path, monkeypatch):
    """Cache the first few val images into a temporary directory."""
    from src.data import cache as cache_module

    monkeypatch.setattr(cache_module, "CACHE_ROOT", tmp_path)
    steps = build_deterministic_steps(SIZE, {"method": METHOD})
    base = PbcDataset("val")
    # Cache exactly the records maybe_subset will pick, using its own rule.
    indices = sorted(np.random.RandomState(0).choice(len(base.records), 4, replace=False))
    for record in [base.records[i] for i in indices]:
        destination = cache_module.cached_path(METHOD, SIZE, record.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        with Image.open(record.path) as image:
            steps(image.convert("RGB")).save(destination, format="PNG")
    return tmp_path


def spec(**overrides):
    base = {"dataset": "pbc", "split": "val", "image_size": SIZE,
            "stain_norm": {"method": METHOD, "cache": True}, "subset_size": 4}
    base.update(overrides)
    return base


# ------------------------------------------------------------------ equivalence

def test_cached_images_match_the_live_pipeline_exactly(tiny_cache):
    cached = build_dataset(spec())
    live = build_dataset(spec(stain_norm={"method": METHOD, "cache": False}))
    for index in range(len(cached)):
        from_cache, label_a, _ = cached[index]
        from_live, label_b, _ = live[index]
        assert torch.equal(label_a, label_b)
        # PNG is lossless, so this is exact rather than approximate.
        assert torch.allclose(from_cache, from_live, atol=1e-6), index


def test_cached_dataset_keeps_labels_and_attributes(tiny_cache):
    cached = build_dataset(spec())
    image, label, attributes = cached[0]
    assert image.shape == (3, SIZE, SIZE)
    assert 0 <= int(label) < 5
    assert attributes.shape == (11,)


# ----------------------------------------------------------------- cache misses

def test_a_missing_cache_is_an_error_not_a_silent_slow_path(tmp_path, monkeypatch):
    from src.data import cache as cache_module

    monkeypatch.setattr(cache_module, "CACHE_ROOT", tmp_path)   # empty
    with pytest.raises(FileNotFoundError, match="precompute_stain"):
        build_dataset(spec())


def test_cache_is_keyed_by_method_and_size(tmp_path, monkeypatch):
    from src.data import cache as cache_module

    monkeypatch.setattr(cache_module, "CACHE_ROOT", tmp_path)
    raw = PBC_DIR / "lymphocyte" / "LY_883318.jpg"
    paths = {cache_module.cached_path(m, s, raw)
             for m in ("macenko", "reinhard") for s in (224, 360)}
    assert len(paths) == 4   # no two combinations collide


def test_cache_can_be_disabled_per_config(tiny_cache):
    # subset of 4 is cached; asking for more with cache on would miss, with cache off works
    dataset = build_dataset(spec(subset_size=8, stain_norm={"method": METHOD, "cache": False}))
    assert len(dataset) == 8


# -------------------------------------------------------- num_workers override

def test_num_workers_comes_from_the_config_by_default(monkeypatch):
    monkeypatch.delenv("WBC_NUM_WORKERS", raising=False)
    assert resolve_num_workers({"num_workers": 3}) == 3
    assert resolve_num_workers({}) == 0


def test_env_override_wins_over_the_config(monkeypatch):
    monkeypatch.setenv("WBC_NUM_WORKERS", "4")
    assert resolve_num_workers({"num_workers": 0}) == 4


def test_negative_override_is_clamped_not_passed_through(monkeypatch):
    monkeypatch.setenv("WBC_NUM_WORKERS", "-1")
    assert resolve_num_workers({"num_workers": 2}) == 0


def test_a_nonsense_override_raises_rather_than_being_ignored(monkeypatch):
    monkeypatch.setenv("WBC_NUM_WORKERS", "lots")
    with pytest.raises(ValueError, match="WBC_NUM_WORKERS"):
        resolve_num_workers({"num_workers": 0})


def test_the_override_is_recorded_in_the_results_hardware(monkeypatch):
    from src.utils.results import hardware

    monkeypatch.setenv("WBC_NUM_WORKERS", "4")
    assert hardware()["num_workers_override"] == "4"
