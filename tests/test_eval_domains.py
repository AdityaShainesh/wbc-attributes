"""Every configured eval domain must be buildable, or the run stops before the ledger.

The failure this file exists to prevent: make_eval_loader returns None when a target
root is not on disk, and evaluate_checkpoint turned that into skipped_domains and
carried on. On the one-time Raabin evaluation that is the worst possible behaviour -
a dataset attached at the wrong path would evaluate and ledger pbc_test, skip all three
Raabin domains, and report success. pbc_test is held out (CLAUDE.md sec 5), so the look
is spent and cannot be retaken without an override.

Synthetic trees only: tiny_raw builds its own PBC and Raabin folders under tmp_path.
Nothing here reads, decodes or counts a real Raabin image.
"""
import json

import pytest

pytest.importorskip("torch")

from src.eval import EvalError, evaluate_checkpoint

DOMAINS = ["pbc_test", "raabin_train", "raabin_testA", "raabin_testB"]


@pytest.fixture
def checkpoint(experiment_config, tiny_raw_loadable, mapping, label_space, tmp_path):
    """write(**overrides) -> (cfg, checkpoint path), saved as src.train saves one."""
    from src.config import load_config
    from src.train import save_checkpoint

    def write(**overrides):
        base = {"data.image_size": 32, "train.batch_size": 2, "train.epochs": 1,
                "eval.domains": list(DOMAINS)}
        base.update(overrides)
        cfg = load_config(experiment_config(**base))
        path = tmp_path / f"last_{len(list(tmp_path.glob('last_*.pt')))}.pt"
        save_checkpoint(path, cfg, mapping, label_space,
                        history=[{"epoch": 0, "val_macro_f1": 0.5}])
        return cfg, path

    return write


def _missing(tmp_path):
    return str(tmp_path / "no-such-raabin-root")


# --- the error ------------------------------------------------------------------------

def test_an_unbuildable_domain_is_an_error_not_a_skip(checkpoint, tmp_path):
    cfg, path = checkpoint(**{"paths.local.raabin_root": _missing(tmp_path)})
    with pytest.raises(EvalError) as excinfo:
        evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert "raabin_root" in str(excinfo.value)


def test_the_error_names_every_domain_that_cannot_be_built(checkpoint, tmp_path):
    """One domain named out of three would send someone looking for a problem that is
    not specific to it."""
    cfg, path = checkpoint(**{"paths.local.raabin_root": _missing(tmp_path)})
    with pytest.raises(EvalError) as excinfo:
        evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    message = str(excinfo.value)
    for domain in ("raabin_train", "raabin_testA", "raabin_testB"):
        assert domain in message, message


def test_the_error_names_the_path_that_is_missing(checkpoint, tmp_path):
    """The whole point is to make a path mismatch diagnosable without a debugger."""
    root = _missing(tmp_path)
    cfg, path = checkpoint(**{"paths.local.raabin_root": root})
    with pytest.raises(EvalError) as excinfo:
        evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert root in str(excinfo.value)


def test_a_configured_lisc_domain_is_refused(checkpoint, tmp_path):
    """LISC was never obtained and is out of scope (CLAUDE.md sec 2). It has no loader,
    so a config naming it is a config error, not a domain to quietly drop."""
    cfg, path = checkpoint(**{"eval.domains": DOMAINS + ["lisc"]})
    with pytest.raises(EvalError) as excinfo:
        evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert "lisc" in str(excinfo.value)


# --- nothing is spent before it fires --------------------------------------------------

def test_no_ledger_line_is_written_when_a_domain_cannot_be_built(checkpoint, tmp_path):
    """The regression in full: pbc_test is buildable and would otherwise be evaluated
    and ledgered while the three target domains were silently dropped."""
    ledger = tmp_path / "l.jsonl"
    cfg, path = checkpoint(**{"paths.local.raabin_root": _missing(tmp_path)})
    with pytest.raises(EvalError):
        evaluate_checkpoint(path, ledger_path=ledger)
    assert not ledger.exists() or ledger.read_text().strip() == ""


def test_no_results_record_is_written_when_a_domain_cannot_be_built(checkpoint,
                                                                    tmp_path):
    from src.eval import eval_results_path

    cfg, path = checkpoint(**{"paths.local.raabin_root": _missing(tmp_path)})
    with pytest.raises(EvalError):
        evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert not eval_results_path(cfg, dry_run=False).exists()


def test_it_fires_before_the_duplicate_check(checkpoint, tmp_path):
    """Ordering, observable: with a prior entry for pbc_test AND a missing root, the
    unbuildable-domain error is the one raised. An override must not be able to carry a
    run past a path mismatch."""
    ledger = tmp_path / "l.jsonl"
    cfg, path = checkpoint(**{"paths.local.raabin_root": _missing(tmp_path)})
    with pytest.raises(EvalError) as excinfo:
        evaluate_checkpoint(path, ledger_path=ledger,
                            override_reason="rerun after a fix")
    assert "raabin" in str(excinfo.value)


# --- the invariant ---------------------------------------------------------------------

def test_skipped_domains_is_empty_on_a_run_that_completes(checkpoint, tmp_path):
    cfg, path = checkpoint()
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    assert out["skipped_domains"] == {}
    assert sorted(out["domains"]) == sorted(DOMAINS)


def test_the_record_on_disk_also_shows_no_skipped_domain(checkpoint, tmp_path):
    from src.eval import eval_results_path

    cfg, path = checkpoint()
    evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl")
    record = json.loads(eval_results_path(cfg, dry_run=False).read_text())
    assert record["skipped_domains"] == {}


def test_a_dry_run_is_unaffected_by_a_missing_target_root(checkpoint, tmp_path):
    """A dry run reads pbc_val and nothing else, so a target path it will never touch
    must not stop it - that is what makes it usable as a pre-flight check."""
    cfg, path = checkpoint(**{"paths.local.raabin_root": _missing(tmp_path)})
    out = evaluate_checkpoint(path, ledger_path=tmp_path / "l.jsonl", dry_run=True)
    assert out["domains"] == ["pbc_val"]
    assert out["skipped_domains"] == {}


# --- the helper the check and the loader share -----------------------------------------

def test_the_reason_helper_agrees_with_the_loader(experiment_config, tiny_raw_loadable,
                                                   mapping, label_space, tmp_path):
    """Two sources of truth for 'can this domain be built' would drift, and the drift
    would be a domain that passes the check and then returns None."""
    from src.config import load_config
    from src.data.loaders import eval_loader_unavailable, make_eval_loader

    for overrides in ({}, {"paths.local.raabin_root": _missing(tmp_path)}):
        cfg = load_config(experiment_config(**{"data.image_size": 32,
                                               "train.batch_size": 2, **overrides}))
        for domain in DOMAINS + ["lisc"]:
            reason = eval_loader_unavailable(cfg, domain)
            loader = make_eval_loader(cfg, domain, mapping, label_space)
            assert (reason is None) == (loader is not None), (domain, reason)
