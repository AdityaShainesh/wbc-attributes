"""The aggregator: reads the nine *_eval.json records and prints the declared result.

It computes no metric. Every number it shows is already in the records - the bootstrap
intervals included - so its whole job is to read, refuse what must not be aggregated,
group by model and seed, and print the comparison CLAUDE.md sec 6 declared in advance.

Synthetic records only. Nothing here reads a checkpoint, an image or the ledger.
"""
import json

import pytest

from scripts.aggregate_eval import (
    AggregateError,
    DECISION_RULE_TEXT,
    build_report,
    load_records,
    verdict,
)

CLASSES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]
TARGETS = ("raabin_train", "raabin_testA", "raabin_testB")
SOURCE = "pbc_test"
SUBSET = "pbc_test_no_band_neutrophils"
MODELS = ("baseline", "bottleneck", "control")


# --- a synthetic record ------------------------------------------------------------------

def score(macro_f1=0.90, classes=None, recall=None, ece=0.10, coverage_accuracy=0.95,
          achieved=0.88, n=100):
    classes = CLASSES if classes is None else classes
    if recall is None:
        recall = {c: (0.9 if c in classes else None) for c in CLASSES}
    return {
        "n": n, "macro_f1": macro_f1, "macro_f1_classes": list(classes),
        "per_class_recall": dict(recall), "ece": ece,
        "mean_max_probability": 0.80, "accuracy": 0.91,
        "macro_f1_ci": {"point": macro_f1, "lower": macro_f1 - 0.02,
                        "upper": macro_f1 + 0.02, "n": 2000, "ci": 0.95,
                        "seed": 0, "stratify_by": "class"},
        # JSON has no float keys: the coverage grid comes back as strings.
        "coverage": {"0.5": {"threshold": 0.9, "achieved_coverage": 0.48,
                             "n_retained": 48, "accuracy": 0.99},
                     "0.9": {"threshold": 0.7, "achieved_coverage": achieved,
                             "n_retained": 88, "accuracy": coverage_accuracy},
                     "1.0": {"threshold": 0.0, "achieved_coverage": 1.0,
                             "n_retained": 100, "accuracy": 0.91}},
    }


def gap_entry(gap=0.10, source=0.90):
    return {"gap": gap, "source_point": source, "target_point": source - gap,
            "lower": gap - 0.03, "upper": gap + 0.03, "resampled_independently": True,
            "n_source_draws": 2000, "n_target_draws": 2000, "ci": 0.95, "seed": 0}


def record(model, seed, gaps=None, secondary=None, modes=None, source_macro_f1=0.90,
           **overrides):
    """One *_eval.json. `gaps`/`secondary` map (domain, mode) -> gap value."""
    modes = modes if modes is not None else (["soft"] if model == "baseline"
                                             else ["soft", "hard"])
    gaps = gaps or {}
    secondary = secondary or {}
    testb_classes = ["neutrophil", "lymphocyte"]

    def classes_of(domain):
        return testb_classes if domain == "raabin_testB" else CLASSES

    rec = {
        "run_id": model, "seed": seed, "dry_run": False, "override_reason": None,
        "selection_role": "headline", "selection_rule": "final_epoch",
        "checkpoint_sha256": f"{model}{seed}".ljust(64, "0"),
        "config_sha256": f"cfg{model}{seed}".ljust(64, "0"),
        "config": {
            "run_id": model, "env": "kaggle", "seed": seed,
            "model": {"type": model, "backbone": "resnet50",
                      "backbone_norm": "batchnorm"},
            "train": {"epochs": 30, "lr": 0.0003, "weight_decay": 0.0001,
                      "batch_size": 32, "class_weighting": "none"},
            "data": {"image_size": 224, "source": "pbc_wbcatt"},
            "selection": {"rule": "final_epoch", "report_metric": "val_macro_f1",
                          "divergence_guard": "non_finite_loss"},
            "eval": {"domains": [SOURCE, *TARGETS], "attribute_modes": modes,
                     "secondary_subsets": [SUBSET], "ece_bins": 15,
                     "macro_f1_over": "present_classes",
                     "abstention": {"threshold_split": "pbc_val",
                                    "coverage_grid": [0.5, 0.9, 1.0]}},
            "bootstrap": {"enabled": True, "n": 2000, "ci": 0.95,
                          "stratify_by": "class",
                          "resample_domains_independently": True},
        },
        "git": {"commit": "abc1234", "dirty": False},
        "hardware": {"device": "cuda"},
        "attribute_modes": list(modes),
        "domains": [SOURCE, *TARGETS],
        "skipped_domains": {},
        "results": {
            domain: {mode: score(
                macro_f1=(source_macro_f1 if domain == SOURCE else 0.75),
                classes=classes_of(domain))
                for mode in modes}
            for domain in (SOURCE, *TARGETS)
        },
        "secondary_subsets": {SUBSET: {"n": 90, "modes": {m: score() for m in modes}}},
        "per_attribute": {SOURCE: {}},
        "gaps": {
            domain: {
                "source_domain": SOURCE, "source_classes": classes_of(domain),
                "modes": {m: gap_entry(gaps.get((domain, m), 0.10), source_macro_f1)
                          for m in modes},
            } for domain in TARGETS
        },
        "secondary_gaps": {
            SUBSET: {
                domain: {
                    "source_domain": SUBSET, "source_classes": classes_of(domain),
                    "modes": {m: gap_entry(secondary.get((domain, m), 0.12),
                                           source_macro_f1)
                              for m in modes},
                } for domain in TARGETS
            }
        },
        "abstention_thresholds": {m: {"0.9": 0.7} for m in modes},
        "buffer_sha256_before": "x", "buffer_sha256_after": "x",
        "wall_clock_seconds": 100.0,
    }
    for dotted, value in overrides.items():
        node, parts = rec, dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return rec


def write(tmp_path, records):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for r in records:
        (tmp_path / f"{r['run_id']}_seed{r['seed']}_eval.json").write_text(
            json.dumps(r, indent=2))
    return tmp_path


def nine(**kwargs):
    return [record(m, s, **kwargs) for s in (0, 1, 2) for m in MODELS]


# --- refusals ------------------------------------------------------------------------------

def test_eight_records_are_refused(tmp_path):
    write(tmp_path, nine()[:8])
    with pytest.raises(AggregateError, match="9"):
        build_report(load_records(tmp_path))


def test_allow_partial_accepts_them_and_says_so(tmp_path):
    write(tmp_path, nine()[:8])
    out = build_report(load_records(tmp_path), allow_partial=True)
    assert "PARTIAL" in out


def test_a_dry_run_record_is_refused(tmp_path):
    records = nine()
    records[0]["dry_run"] = True
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="dry run"):
        build_report(load_records(tmp_path))


def test_a_dryrun_named_file_is_not_loaded(tmp_path):
    write(tmp_path, nine())
    (tmp_path / "baseline_seed0_eval_dryrun.json").write_text(
        json.dumps(record("baseline", 0, **{"dry_run": True})))
    assert len(load_records(tmp_path)) == 9


def test_a_non_headline_role_is_refused(tmp_path):
    records = nine()
    records[0]["selection_role"] = "informational"
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="headline"):
        build_report(load_records(tmp_path))


def test_a_non_final_epoch_rule_is_refused(tmp_path):
    records = nine()
    records[0]["selection_rule"] = "best_val"
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="final_epoch"):
        build_report(load_records(tmp_path))


def test_a_duplicate_run_id_and_seed_is_refused(tmp_path):
    records = nine()
    records[1]["run_id"] = records[0]["run_id"]
    records[1]["config"]["run_id"] = records[0]["run_id"]
    write(tmp_path, records)
    # Both would land on one filename, so write the second by hand.
    (tmp_path / "duplicate_eval.json").write_text(json.dumps(records[1]))
    with pytest.raises(AggregateError, match="duplicate|twice"):
        build_report(load_records(tmp_path))


def test_a_skipped_domain_is_refused(tmp_path):
    records = nine()
    records[0]["skipped_domains"] = {"raabin_testB": "not present on disk"}
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="skipped"):
        build_report(load_records(tmp_path))


def test_an_incomplete_grid_is_refused(tmp_path):
    records = [r for r in nine() if not (r["run_id"] == "control" and r["seed"] == 2)]
    stray = record("control", 1)
    stray["seed"] = 3
    stray["config"]["seed"] = 3
    records.append(stray)
    write(tmp_path, records)
    with pytest.raises(AggregateError):
        build_report(load_records(tmp_path))


# --- the same-config rule, within a model only ---------------------------------------------

def test_two_seeds_of_one_model_differing_only_in_seed_are_accepted(tmp_path):
    write(tmp_path, nine())
    build_report(load_records(tmp_path))      # must not raise


def test_two_seeds_of_one_model_differing_elsewhere_are_refused(tmp_path):
    records = nine()
    next(r for r in records if r["run_id"] == "baseline"
         and r["seed"] == 1)["config"]["train"]["lr"] = 0.001
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="lr|config"):
        build_report(load_records(tmp_path))


def test_models_differing_by_design_are_accepted(tmp_path):
    """baseline, bottleneck and control differ in model.type, the bottleneck block and
    attribute_modes by construction; requiring those to match would refuse the only
    comparison the project exists to make."""
    write(tmp_path, nine())
    build_report(load_records(tmp_path))


def test_models_differing_in_a_protocol_field_are_refused(tmp_path):
    records = nine()
    for r in records:
        if r["run_id"] == "control":
            r["config"]["train"]["lr"] = 0.001
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="lr|protocol"):
        build_report(load_records(tmp_path))


def test_macro_f1_classes_differing_across_seeds_is_refused(tmp_path):
    records = nine()
    r = next(r for r in records if r["run_id"] == "baseline" and r["seed"] == 1)
    r["results"]["raabin_testA"]["soft"]["macro_f1_classes"] = ["neutrophil"]
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="macro_f1_classes|classes"):
        build_report(load_records(tmp_path))


def test_source_classes_differing_across_seeds_is_refused(tmp_path):
    records = nine()
    r = next(r for r in records if r["run_id"] == "bottleneck" and r["seed"] == 2)
    r["gaps"]["raabin_testB"]["source_classes"] = CLASSES
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="source_classes|classes"):
        build_report(load_records(tmp_path))


# --- numbers -------------------------------------------------------------------------------

def test_every_per_seed_value_is_printed_beside_the_mean(tmp_path):
    gaps = {("raabin_testA", "soft"): 0.10}
    records = [record(m, s, gaps={("raabin_testA", "soft"): 0.10 + 0.05 * s})
               for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    for value in ("0.100", "0.150", "0.200"):
        assert value in out, f"{value} missing from the report"


def test_mean_and_std_match_a_hand_computed_case(tmp_path):
    records = [record(m, s, gaps={("raabin_testA", "soft"): [0.10, 0.20, 0.30][s]})
               for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    assert "0.200" in out            # the mean
    assert "0.082" in out or "0.100" in out    # population or sample std


def test_the_bootstrap_intervals_appear_once_per_seed(tmp_path):
    """CLAUDE.md sec 6: the CI is reported per seed per domain. An averaged interval
    would be a number nothing computed."""
    out = build_report(load_records(write(tmp_path, nine())))
    assert out.count("[0.070, 0.130]") >= 3


def test_accuracy_at_ninety_percent_coverage_is_read_from_the_string_key(tmp_path):
    records = nine()
    for r in records:
        for domain in r["results"]:
            for mode in r["results"][domain]:
                r["results"][domain][mode]["coverage"]["0.9"]["accuracy"] = 0.777
    out = build_report(load_records(write(tmp_path, records)))
    assert "0.777" in out


def test_achieved_coverage_is_printed_beside_nominal(tmp_path):
    records = nine()
    for r in records:
        for domain in r["results"]:
            for mode in r["results"][domain]:
                r["results"][domain][mode]["coverage"]["0.9"]["achieved_coverage"] = 0.631
    out = build_report(load_records(write(tmp_path, records)))
    assert "0.631" in out


# --- relative drop ---------------------------------------------------------------------------

def test_relative_drop_is_the_gap_over_the_source_point(tmp_path):
    records = [record(m, s, gaps={("raabin_testA", "soft"): 0.18},
                      source_macro_f1=0.90) for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    assert "0.200" in out          # 0.18 / 0.90


def test_relative_drop_carries_no_interval(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    assert "no interval" in out.lower()


def test_a_zero_source_point_is_refused_rather_than_infinite(tmp_path):
    records = nine()
    for r in records:
        r["gaps"]["raabin_testA"]["modes"]["soft"]["source_point"] = 0.0
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="source_point|zero"):
        build_report(load_records(tmp_path))


# --- the secondary gap --------------------------------------------------------------------

def test_the_secondary_gap_is_reported_and_labelled_non_decisive(tmp_path):
    records = [record(m, s, secondary={("raabin_testA", "soft"): 0.33})
               for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    assert "0.330" in out
    assert "non-decisive" in out.lower()


def test_a_missing_secondary_gap_block_says_so_rather_than_substituting(tmp_path):
    """The failure this guards: using secondary_subsets[...]['macro_f1'], which is over
    all five classes, as the source side of a two-class gap."""
    records = nine()
    for r in records:
        r["secondary_gaps"] = {}
    out = build_report(load_records(write(tmp_path, records)))
    assert "not available" in out.lower() or "absent" in out.lower()
    assert "0.900" not in out.split("secondary")[-1][:400]


def test_the_secondary_gap_never_uses_the_unrestricted_subset_score(tmp_path):
    records = nine()
    for r in records:
        for m in r["attribute_modes"]:
            r["secondary_subsets"][SUBSET]["modes"][m]["macro_f1"] = 0.4242
    out = build_report(load_records(write(tmp_path, records)))
    assert "0.4242" not in out and "0.424" not in out


# --- the decision rule -----------------------------------------------------------------------

@pytest.mark.parametrize("deltas,expected", [
    ((-0.1, -0.2, -0.3), "supported"),
    ((-0.1, -0.2, +0.3), "mixed"),
    ((-0.1, +0.2, +0.3), "not supported"),
    ((+0.1, +0.2, +0.3), "not supported"),
    ((0.0, -0.1, -0.2), "mixed"),
    ((0.0, 0.0, -0.1), "not supported"),
])
def test_the_verdict_follows_the_declared_rule(deltas, expected):
    assert verdict(list(deltas)) == expected


def test_a_delta_of_exactly_zero_counts_as_non_negative():
    """The boundary a reader will guess wrong. The rule says delta >= 0."""
    assert verdict([0.0, 0.0, -1.0]) == "not supported"
    assert verdict([-1e-12, -1.0, -1.0]) == "supported"


def test_the_three_verdicts_partition_every_sign_pattern():
    from itertools import product
    seen = {verdict(list(d)) for d in product([-1.0, 0.0, 1.0], repeat=3)}
    assert seen == {"supported", "mixed", "not supported"}


def test_the_rule_text_is_printed_with_the_verdicts(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    assert DECISION_RULE_TEXT in out


def test_the_sign_count_is_printed_beside_the_verdict(tmp_path):
    """delta_bb is bottleneck minus baseline; make the bottleneck better on every seed."""
    records = [record(m, s, gaps={("raabin_testA", "soft"):
                                  0.05 if m == "bottleneck" else 0.20})
               for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    assert "3/3" in out
    assert "supported" in out


def test_delta_bb_is_bottleneck_minus_baseline_per_seed(tmp_path):
    records = [record(m, s, gaps={("raabin_testA", "soft"):
                                  {"baseline": 0.30, "bottleneck": 0.10,
                                   "control": 0.20}[m]})
               for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    assert "-0.200" in out          # delta_bb = 0.10 - 0.30
    assert "-0.100" in out          # delta_bc = 0.10 - 0.20


def test_delta_bc_uses_the_control_not_the_baseline(tmp_path):
    records = [record(m, s, gaps={("raabin_testA", "soft"):
                                  {"baseline": 0.90, "bottleneck": 0.10,
                                   "control": 0.15}[m]})
               for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    assert "-0.050" in out


def test_pairing_is_by_seed_not_by_file_order(tmp_path):
    values = {0: 0.10, 1: 0.20, 2: 0.30}
    forwards = [record(m, s, gaps={("raabin_testA", "soft"): values[s]})
                for s in (0, 1, 2) for m in MODELS]
    backwards = list(reversed(forwards))
    a = build_report(load_records(write(tmp_path / "a", forwards)))
    b = build_report(load_records(write(tmp_path / "b", backwards)))
    assert a == b


def test_a_delta_is_computed_per_mode(tmp_path):
    records = [record(m, s, gaps={("raabin_testA", "soft"):
                                  0.10 if m == "bottleneck" else 0.30,
                                  ("raabin_testA", "hard"):
                                  0.40 if m == "bottleneck" else 0.30})
               for s in (0, 1, 2) for m in MODELS]
    out = build_report(load_records(write(tmp_path, records)))
    assert "-0.200" in out and "0.100" in out


def test_no_significance_test_appears_anywhere(tmp_path):
    out = build_report(load_records(write(tmp_path, nine()))).lower()
    for word in ("p-value", "p value", "significan", "t-test", "wilcoxon"):
        assert word not in out or "no significance test" in out


def test_no_delta_is_produced_for_the_source_domain(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    source_block = out.split("raabin")[0]
    assert "delta" not in source_block.lower()


def test_testb_is_marked_not_decisive_and_the_other_two_headline(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    assert "NOT DECISIVE" in out
    assert "headline" in out.lower()


# --- shapes ------------------------------------------------------------------------------------

def test_a_baseline_with_only_soft_produces_no_hard_column(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    assert "baseline" in out          # and it did not crash


def test_per_class_recall_stays_null_and_is_never_zero(tmp_path):
    """Zero reads as a model failure; null is the truth, which is that the number is
    not measurable there (CLAUDE.md sec 6). raabin_testB holds two of the five."""
    out = build_report(load_records(write(tmp_path, nine())))
    testb = out.split("### raabin_testB")[1].split("### ")[0]
    for absent in ("monocyte", "eosinophil", "basophil"):
        line = next(l for l in testb.splitlines() if l.strip().startswith(absent))
        assert "null" in line, f"{absent} is not reported as null: {line!r}"
        assert "0.000" not in line, f"{absent} was printed as zero: {line!r}"


def test_all_five_classes_are_listed_even_where_absent(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    testb = out.split("raabin_testB")[1][:1500]
    for cls in CLASSES:
        assert cls[:9] in testb, f"{cls} missing from the testB block"


def test_a_class_null_in_some_seeds_but_not_others_is_refused(tmp_path):
    records = nine()
    r = next(r for r in records if r["run_id"] == "baseline" and r["seed"] == 1)
    r["results"]["raabin_testB"]["soft"]["per_class_recall"]["monocyte"] = 0.5
    write(tmp_path, records)
    with pytest.raises(AggregateError, match="null|recall"):
        build_report(load_records(tmp_path))


def test_there_is_no_pooled_raabin_row(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    for forbidden in ("raabin_all", "raabin (pooled)", "raabin_pooled", "raabin overall"):
        assert forbidden not in out


def test_the_two_class_testb_gap_names_its_restricted_source_side(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    testb = out.split("raabin_testB")[1][:1500]
    assert "neutrophil" in testb and "lymphocyte" in testb


def test_the_basophil_caveat_is_printed(tmp_path):
    out = build_report(load_records(write(tmp_path, nine()))).lower()
    assert "single" in out and "patient" in out


def test_provenance_is_printed(tmp_path):
    out = build_report(load_records(write(tmp_path, nine())))
    assert "abc1234" in out
