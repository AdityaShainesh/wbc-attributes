"""The like-for-like secondary gap, computed in the evaluation rather than afterwards.

CLAUDE.md sec 6 declares a secondary gap whose source side is
pbc_test_no_band_neutrophils, restricted to the classes present in the target domain
exactly as the primary gap's source side is. It has to be computed here because the
record carries no per-image predictions, so nothing downstream could reconstruct it; and
substituting the unrestricted secondary_subsets score would be a different quantity
wearing the same label.

Synthetic trees only. Nothing here reads, decodes or counts a real Raabin image.
"""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.eval import evaluate_checkpoint

GOLDEN = Path(__file__).resolve().parent / "golden" / "eval_record_before_secondary_gaps.json"
SUBSET = "pbc_test_no_band_neutrophils"
TARGETS = ("raabin_train", "raabin_testA", "raabin_testB")

# Paths, clocks and hardware differ per run and per machine; the config and the hashes
# derived from it carry tmp_path. What is left is the arithmetic.
VOLATILE = {"checkpoint", "checkpoint_sha256", "config", "config_sha256", "git",
            "hardware", "results_json", "wall_clock_seconds"}


def normalised(record: dict) -> dict:
    """Everything the change must not move, through a JSON round trip so that float
    dict keys compare as the strings they become on disk."""
    kept = {k: v for k, v in record.items() if k not in VOLATILE}
    return json.loads(json.dumps(kept, default=str))


@pytest.fixture
def record(experiment_config, tiny_raw_loadable, mapping, label_space, tmp_path):
    """One full evaluation of a tiny checkpoint over pbc_test and all three Raabin
    domains, from the synthetic tree."""
    from src.config import load_config, set_seed
    from src.train import save_checkpoint

    def run(**overrides):
        eval_kwargs = overrides.pop("_eval_kwargs", {})
        base = {"data.image_size": 32, "train.batch_size": 2, "train.epochs": 1,
                "bootstrap.n": 25}      # the CI shape is what matters here, not its width
        base.update(overrides)
        cfg = load_config(experiment_config(**base))
        path = tmp_path / f"last_{len(list(tmp_path.glob('last_*.pt')))}.pt"
        # save_checkpoint builds the model to get real parameter names, so its weights
        # come from the global RNG. Without this the golden records whatever RNG state
        # the surrounding test session happened to leave behind.
        set_seed(cfg.seed)
        save_checkpoint(path, cfg, mapping, label_space,
                        history=[{"epoch": 0, "val_macro_f1": 0.5}])
        return evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl", **eval_kwargs)

    return run


# --- capture ---------------------------------------------------------------------------

def test_capture_the_golden(record):
    """Run with CAPTURE_EVAL_GOLDEN=1 before the change to write the reference, then
    never again. Skipped otherwise."""
    if not os.environ.get("CAPTURE_EVAL_GOLDEN"):
        pytest.skip("set CAPTURE_EVAL_GOLDEN=1 to rewrite the golden")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(normalised(record()), indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")


# --- nothing that already existed may move ----------------------------------------------

def test_every_pre_existing_key_and_value_is_unchanged(record):
    """The guard the whole change is made under: secondary_gaps is additive."""
    golden = json.loads(GOLDEN.read_text())
    now = normalised(record())
    assert "secondary_gaps" in now, "the new block is missing"
    del now["secondary_gaps"]
    assert set(now) == set(golden), (
        f"top-level keys moved: added {sorted(set(now) - set(golden))}, "
        f"removed {sorted(set(golden) - set(now))}")
    for key in sorted(golden):
        assert now[key] == golden[key], f"{key} changed"


def test_the_existing_secondary_subsets_block_is_byte_identical(record):
    """The subset's own score is hoisted out of its comprehension to be reused by the
    gap; hoisting must not reshape or renumber it."""
    golden = json.loads(GOLDEN.read_text())
    assert normalised(record())["secondary_subsets"] == golden["secondary_subsets"]


def test_the_primary_gaps_are_untouched(record):
    golden = json.loads(GOLDEN.read_text())
    assert normalised(record())["gaps"] == golden["gaps"]


# --- the new block ------------------------------------------------------------------------

def test_a_secondary_gap_exists_for_every_target_domain(record):
    out = record()
    assert set(out["secondary_gaps"]) == {SUBSET}
    assert set(out["secondary_gaps"][SUBSET]) == set(TARGETS)


def test_no_secondary_gap_for_a_source_domain(record):
    """pbc_test is the source side; a gap of it against itself is not a quantity."""
    out = record()
    assert "pbc_test" not in out["secondary_gaps"][SUBSET]


def test_the_source_side_is_named_and_is_the_subset(record):
    out = record()
    for domain in TARGETS:
        assert out["secondary_gaps"][SUBSET][domain]["source_domain"] == SUBSET


def test_it_is_restricted_to_the_same_classes_as_the_primary_gap(record):
    """Same restriction, same classes - otherwise the two gaps are not comparable and
    the secondary one answers a question nobody asked."""
    out = record()
    for domain in TARGETS:
        assert (out["secondary_gaps"][SUBSET][domain]["source_classes"]
                == out["gaps"][domain]["source_classes"]), domain


def test_every_attribute_mode_gets_one(record):
    out = record()
    for domain in TARGETS:
        assert (set(out["secondary_gaps"][SUBSET][domain]["modes"])
                == set(out["attribute_modes"]))


def test_it_carries_a_bootstrap_interval_like_the_primary(record):
    out = record()
    for domain in TARGETS:
        for mode, entry in out["secondary_gaps"][SUBSET][domain]["modes"].items():
            assert set(entry) == set(out["gaps"][domain]["modes"][mode]), (domain, mode)
            assert entry["resampled_independently"] is True
            assert entry["lower"] <= entry["upper"]
            assert entry["gap"] == pytest.approx(
                entry["source_point"] - entry["target_point"])


def test_the_target_side_is_the_same_as_the_primary_gaps(record):
    """Only the source side changes. A secondary gap that also moved the target side
    would confound two differences at once."""
    out = record()
    for domain in TARGETS:
        for mode, entry in out["secondary_gaps"][SUBSET][domain]["modes"].items():
            assert (entry["target_point"]
                    == out["gaps"][domain]["modes"][mode]["target_point"])


def test_the_source_side_is_not_the_unrestricted_subset_score(record):
    """The failure this block exists to prevent downstream: using
    secondary_subsets[...]['macro_f1'], which is over all five classes, as the source
    side of a two-class gap."""
    out = record()
    entry = out["secondary_gaps"][SUBSET]["raabin_testB"]["modes"]["soft"]
    unrestricted = out["secondary_subsets"][SUBSET]["modes"]["soft"]
    kept = out["secondary_gaps"][SUBSET]["raabin_testB"]["source_classes"]
    if len(kept) < len(unrestricted["macro_f1_classes"]):
        assert entry["source_point"] != unrestricted["macro_f1"], (
            "the restricted source side equals the unrestricted one")


def test_a_dry_run_has_no_secondary_gaps(record):
    """A dry run reads pbc_val only: no target, no source test split, no gap."""
    out = record(_eval_kwargs={"dry_run": True})
    assert out["secondary_gaps"] == {}
