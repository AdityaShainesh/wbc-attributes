"""Read the nine *_eval.json records and print the comparison CLAUDE.md sec 6 declares.

This script computes no metric. Every number it shows was computed during the
evaluation and written into the record, the bootstrap intervals included, so its whole
job is to read, refuse what must not be aggregated, group by model and seed, and print.
It selects nothing, and it touches no checkpoint, no image and not the ledger.

    python -m scripts.aggregate_eval                  # reads eval_records/
    python -m scripts.aggregate_eval --records DIR --json out.json

The refusals matter more than the formatting. A mean over three records that are not
three seeds of one protocol is not a mean over seeds, and nothing downstream would show
that it was not.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODELS = ("baseline", "bottleneck", "control")
SEEDS = (0, 1, 2)
SOURCE = "pbc_test"
TARGETS = ("raabin_train", "raabin_testA", "raabin_testB")
DOMAIN_ORDER = (SOURCE, *TARGETS)
HEADLINE_DOMAINS = ("raabin_train", "raabin_testA")
SECONDARY_SUBSET = "pbc_test_no_band_neutrophils"
CLASS_ORDER = ("neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil")

# JSON has no float keys, so the coverage grid comes back as strings.
HEADLINE_COVERAGE = "0.9"

DECISION_RULE_TEXT = (
    "supported on a domain: delta < 0 in all three seeds; "
    "not supported: delta >= 0 in at least two seeds; "
    "otherwise mixed"
)

# Fields every model must share, because they are the protocol rather than the
# representation. model.type, the bottleneck block and eval.attribute_modes are left out:
# the three models differ in exactly those by construction, and requiring them to match
# would refuse the only comparison this project exists to make.
PROTOCOL_FIELDS = (
    ("train",), ("data",), ("selection",), ("bootstrap",),
    ("model", "backbone"), ("model", "backbone_norm"),
    ("eval", "domains"), ("eval", "abstention"), ("eval", "ece_bins"),
    ("eval", "macro_f1_over"), ("eval", "secondary_subsets"),
)

BASOPHIL_CAVEAT = (
    "Raabin's basophils come from a single leukemia patient. Any basophil recall on a\n"
    "Raabin domain is one patient's cells, not a sample of basophils (CLAUDE.md sec 6)."
)


class AggregateError(Exception):
    """Something that must not be aggregated. Always fatal: a report that quietly
    dropped or averaged the wrong thing is worse than no report."""


# --- the declared rule ---------------------------------------------------------------------

def verdict(deltas: list[float]) -> str:
    """The rule declared in CLAUDE.md sec 6, before any Raabin number was seen.

    Exhaustive and disjoint over three seeds: three negatives is supported, exactly two
    is mixed, one or none is not supported. A delta of exactly zero counts as
    non-negative, which is the boundary a reader is most likely to guess wrong.
    """
    if all(d < 0 for d in deltas):
        return "supported"
    if sum(d >= 0 for d in deltas) >= 2:
        return "not supported"
    return "mixed"


# --- loading -------------------------------------------------------------------------------

def load_records(source) -> list[dict]:
    """Every *_eval.json under a directory, or the given files.

    *_eval_dryrun.json is excluded by name as well as by content: a dry run reads
    pbc_val, selects nothing and reports nothing, so it is not a result.
    """
    source = Path(source)
    if source.is_dir():
        paths = sorted(p for p in source.glob("*_eval.json")
                       if not p.name.endswith("_eval_dryrun.json"))
    else:
        paths = [source]
    records = []
    for path in paths:
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise AggregateError(f"{path} is not readable JSON: {exc}") from exc
        record["_path"] = str(path)
        records.append(record)
    return records


# --- refusals --------------------------------------------------------------------------------

def _dig(node, path):
    for key in path:
        node = node[key]
    return node


def validate(records: list[dict], allow_partial: bool = False) -> None:
    if not records:
        raise AggregateError("no records found")

    seen: dict[tuple[str, int], str] = {}
    for record in records:
        tag = f"{record.get('run_id')}_seed{record.get('seed')}"
        where = record.get("_path", tag)
        if record.get("dry_run"):
            raise AggregateError(f"{where}: this is a dry run; it reads pbc_val only")
        if record.get("selection_role") != "headline":
            raise AggregateError(
                f"{where}: selection_role is {record.get('selection_role')!r}, not "
                f"'headline'. best.pt is informational and is never the evaluated "
                f"checkpoint (CLAUDE.md sec 5).")
        if record.get("selection_rule") != "final_epoch":
            raise AggregateError(
                f"{where}: selection_rule is {record.get('selection_rule')!r}, not "
                f"'final_epoch'")
        if record.get("skipped_domains"):
            raise AggregateError(
                f"{where}: skipped domains {record['skipped_domains']}. A mean taken "
                f"over records that cover different domains is not a mean.")
        key = (record["run_id"], record["seed"])
        if key in seen:
            raise AggregateError(
                f"{key} appears twice: {seen[key]} and {where}. One of them is a "
                f"duplicate of a run, not a second seed.")
        seen[key] = where

    expected = {(m, s) for m in MODELS for s in SEEDS}
    if set(seen) != expected and not allow_partial:
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        raise AggregateError(
            f"expected the 9 records of {list(MODELS)} x {list(SEEDS)}; "
            f"missing {missing}, unexpected {extra}. Pass --allow-partial to report "
            f"on an incomplete set, which is then stamped PARTIAL.")

    _validate_configs(records, allow_partial)
    _validate_class_lists(records)


def _validate_configs(records: list[dict], allow_partial: bool) -> None:
    """Within one model, the three seeds must differ in seed alone. Across models, only
    the protocol fields must match - the three models differ by design elsewhere."""
    by_model: dict[str, list[dict]] = {}
    for record in records:
        by_model.setdefault(record["run_id"], []).append(record)

    for model, group in sorted(by_model.items()):
        first = group[0]
        for other in group[1:]:
            a = {k: v for k, v in first["config"].items() if k != "seed"}
            b = {k: v for k, v in other["config"].items() if k != "seed"}
            if a != b:
                differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                raise AggregateError(
                    f"{model} seeds {first['seed']} and {other['seed']} differ in "
                    f"config beyond the seed: {differing}. Three seeds of different "
                    f"configs are not three seeds.")

    models = sorted(by_model)
    reference = by_model[models[0]][0]
    for model in models[1:]:
        other = by_model[model][0]
        for field in PROTOCOL_FIELDS:
            try:
                left, right = _dig(reference["config"], field), _dig(other["config"], field)
            except KeyError:
                continue
            if left != right:
                raise AggregateError(
                    f"{models[0]} and {model} differ in the protocol field "
                    f"{'.'.join(field)}: {left!r} vs {right!r}. Differences between the "
                    f"three models must come from the representation, not the protocol "
                    f"(CLAUDE.md sec 5).")


def _validate_class_lists(records: list[dict]) -> None:
    """The class list behind a macro-F1, and the restriction behind a gap, must be the
    same across the seeds being averaged - otherwise the mean is over two quantities."""
    by_model: dict[str, list[dict]] = {}
    for record in records:
        by_model.setdefault(record["run_id"], []).append(record)

    for model, group in sorted(by_model.items()):
        first = group[0]
        for other in group[1:]:
            for domain in first["results"]:
                for mode in first["results"][domain]:
                    a = first["results"][domain][mode]["macro_f1_classes"]
                    b = other["results"][domain][mode]["macro_f1_classes"]
                    if a != b:
                        raise AggregateError(
                            f"{model} {domain}/{mode}: macro_f1_classes differ across "
                            f"seeds, {a} vs {b}")
                    left = first["results"][domain][mode]["per_class_recall"]
                    right = other["results"][domain][mode]["per_class_recall"]
                    nulls_a = {c for c, v in left.items() if v is None}
                    nulls_b = {c for c, v in right.items() if v is None}
                    if nulls_a != nulls_b:
                        raise AggregateError(
                            f"{model} {domain}/{mode}: per_class_recall is null for "
                            f"{sorted(nulls_a ^ nulls_b)} in some seeds and not others. "
                            f"Null means not measurable here; averaging over the seeds "
                            f"where it happens to be present would invent a number.")
            for domain in first.get("gaps", {}):
                a = first["gaps"][domain]["source_classes"]
                b = other["gaps"][domain]["source_classes"]
                if a != b:
                    raise AggregateError(
                        f"{model} {domain}: gap source_classes differ across seeds, "
                        f"{a} vs {b}")


# --- small helpers ------------------------------------------------------------------------

def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def _fmt(value, width=7) -> str:
    return "   —   " if value is None else f"{value:{width}.3f}"


def _seeds_and_summary(values: list) -> str:
    """Every per-seed value, then the mean ± std. CLAUDE.md sec 6 asks for the mean and
    std over three seeds; three values are a small enough sample that hiding them would
    hide most of what there is to know."""
    if any(v is None for v in values):
        return "  ".join(_fmt(v) for v in values) + "   |      —"
    mean, std = _mean_std(values)
    return "  ".join(_fmt(v) for v in values) + f"   | {mean:7.3f} ± {std:.3f}"


def _by_seed(records: list[dict], model: str) -> list[dict]:
    group = sorted((r for r in records if r["run_id"] == model),
                   key=lambda r: r["seed"])
    return group


# --- the report ------------------------------------------------------------------------------

def build_report(records: list[dict], allow_partial: bool = False) -> str:
    validate(records, allow_partial)
    out: list[str] = []
    w = out.append

    present = sorted({r["run_id"] for r in records})
    seeds_present = sorted({r["seed"] for r in records})

    w("=" * 92)
    w("SOURCE -> TARGET GENERALIZATION GAP")
    if allow_partial and len(records) != 9:
        w(f"*** PARTIAL: {len(records)} of 9 records. Not the declared result. ***")
    w("=" * 92)
    w("")
    w(_provenance(records))
    w("")
    w(_per_domain(records, present, seeds_present))
    w("")
    w(_gap_section(records, present))
    w("")
    w(_comparison(records, allow_partial))
    w("")
    w("-" * 92)
    w(BASOPHIL_CAVEAT)
    w("")
    w("Per-class recall carries no interval: the bootstrap is computed for macro-F1 and")
    w("for the gap, and not per class. Relative drop carries no interval either - the")
    w("bootstrap gives replicates for a gap, not for a ratio of a gap to a point value.")
    w("")
    w("There is no pooled Raabin number. Three domains, three rows (CLAUDE.md sec 6).")
    w("No significance test: three paired seeds do not support one.")
    return "\n".join(out)


def _provenance(records: list[dict]) -> str:
    lines = ["PROVENANCE", "-" * 92]
    boot = records[0]["config"]["bootstrap"]
    for record in sorted(records, key=lambda r: (r["seed"], r["run_id"])):
        git = record.get("git") or {}
        dirty = " DIRTY" if git.get("dirty") else ""
        lines.append(f"  {record['run_id']:<11} seed {record['seed']}  "
                     f"ckpt {record['checkpoint_sha256'][:12]}  "
                     f"cfg {str(record.get('config_sha256'))[:12]}  "
                     f"{git.get('commit', '?')[:7]}{dirty}")
    lines.append(f"  bootstrap: n={boot['n']} ci={boot['ci']} "
                 f"stratified by {boot['stratify_by']}, source and target resampled "
                 f"independently")
    lines.append("  The ledger is not read by this script; read it separately.")
    return "\n".join(lines)


def _per_domain(records: list[dict], models: list[str], seeds: list[int]) -> str:
    lines = ["PER DOMAIN", "-" * 92,
             f"{'':34}{'seed ' + '  seed '.join(str(s) for s in seeds):<33} | "
             f"mean ± std"]
    for domain in DOMAIN_ORDER:
        if not any(domain in r["results"] for r in records):
            continue
        lines.append("")
        lines.append(f"### {domain}")
        for model in models:
            group = _by_seed(records, model)
            if not group or domain not in group[0]["results"]:
                continue
            for mode in group[0]["attribute_modes"]:
                scores = [r["results"][domain][mode] for r in group]
                classes = scores[0]["macro_f1_classes"]
                lines.append(f"  {model} / {mode}   macro-F1 over {classes}")
                lines.append(f"    {'macro-F1':<26} "
                             f"{_seeds_and_summary([s['macro_f1'] for s in scores])}")
                for score, record in zip(scores, group):
                    ci = score.get("macro_f1_ci")
                    if ci:
                        lines.append(f"      seed {record['seed']} 95% CI        "
                                     f"[{ci['lower']:.3f}, {ci['upper']:.3f}]")
                lines.append(f"    {'ECE':<26} "
                             f"{_seeds_and_summary([s['ece'] for s in scores])}")
                lines.append(f"    {'mean max probability':<26} "
                             f"{_seeds_and_summary([s['mean_max_probability'] for s in scores])}")
                cov = [s.get("coverage", {}).get(HEADLINE_COVERAGE, {}) for s in scores]
                lines.append(f"    {'accuracy @ 90% coverage':<26} "
                             f"{_seeds_and_summary([c.get('accuracy') for c in cov])}")
                lines.append(f"    {'  coverage achieved':<26} "
                             f"{_seeds_and_summary([c.get('achieved_coverage') for c in cov])}")
                lines.append(f"    per-class recall  (null where the class is absent)")
                for cls in CLASS_ORDER:
                    values = [s["per_class_recall"].get(cls) for s in scores]
                    if all(v is None for v in values):
                        lines.append(f"      {cls:<24} {'null':>7}   "
                                     f"(not measurable on this domain)")
                    else:
                        lines.append(f"      {cls:<24} {_seeds_and_summary(values)}")
    return "\n".join(lines)


def _gap_values(records: list[dict], model: str, domain: str, mode: str,
                block: str = "gaps", subset: str | None = None) -> list[dict] | None:
    entries = []
    for record in _by_seed(records, model):
        node = record.get(block) or {}
        if subset is not None:
            node = node.get(subset) or {}
        if domain not in node or mode not in node[domain]["modes"]:
            return None
        entries.append(node[domain]["modes"][mode])
    return entries


def _gap_section(records: list[dict], models: list[str]) -> str:
    lines = ["SOURCE -> TARGET GAP   (source macro-F1 minus target macro-F1; "
             "positive is degradation)", "-" * 92]
    have_secondary = any(r.get("secondary_gaps") for r in records)
    if not have_secondary:
        lines.append("  Secondary gap: NOT AVAILABLE in these records. It is not")
        lines.append("  substituted from secondary_subsets, which is scored over all five")
        lines.append("  classes and would be a different quantity on a two-class domain.")
    for domain in TARGETS:
        lines.append("")
        lines.append(f"### {domain}")
        for model in models:
            group = _by_seed(records, model)
            if not group:
                continue
            source_classes = (group[0].get("gaps", {}).get(domain) or {}).get(
                "source_classes")
            if source_classes:
                lines.append(f"  {model}: source side is {SOURCE} restricted to "
                             f"{source_classes}")
            for mode in group[0]["attribute_modes"]:
                entries = _gap_values(records, model, domain, mode)
                if entries is None:
                    continue
                lines.append(f"    {model} / {mode}")
                lines.append(f"      {'gap':<24} "
                             f"{_seeds_and_summary([e['gap'] for e in entries])}")
                for entry, record in zip(entries, group):
                    lines.append(f"        seed {record['seed']} 95% CI      "
                                 f"[{entry['lower']:.3f}, {entry['upper']:.3f}]")
                drops = []
                for entry in entries:
                    source_point = entry["source_point"]
                    if not source_point:
                        raise AggregateError(
                            f"{model} {domain}/{mode}: source_point is "
                            f"{source_point!r}, so the relative drop is undefined. "
                            f"Refusing rather than printing an infinity.")
                    drops.append(entry["gap"] / source_point)
                lines.append(f"      {'relative drop (no interval)':<24} "
                             f"{_seeds_and_summary(drops)}")
                secondary = _gap_values(records, model, domain, mode,
                                        block="secondary_gaps",
                                        subset=SECONDARY_SUBSET)
                if secondary is not None:
                    lines.append(f"      {'secondary gap, non-decisive':<24} "
                                 f"{_seeds_and_summary([e['gap'] for e in secondary])}")
                    lines.append(f"        source side: {SECONDARY_SUBSET}, "
                                 f"restricted the same way")
    return "\n".join(lines)


def _comparison(records: list[dict], allow_partial: bool = False) -> str:
    """The primary comparison, declared in CLAUDE.md sec 6 before any Raabin number was
    seen: paired differences of gaps, by seed."""
    lines = ["PRIMARY COMPARISON   (paired by seed; negative favours the bottleneck)",
             "-" * 92,
             f"  rule: {DECISION_RULE_TEXT}",
             "  applied separately to each comparison and each domain",
             f"  headline domains: {', '.join(HEADLINE_DOMAINS)}",
             "  raabin_testB: camera-shift stress test, NOT DECISIVE"]
    present = {r["run_id"] for r in records}
    pairs = [("delta_bb", "bottleneck", "baseline"),
             ("delta_bc", "bottleneck", "control")]
    for domain in TARGETS:
        lines.append("")
        marker = "headline" if domain in HEADLINE_DOMAINS else "NOT DECISIVE"
        lines.append(f"### {domain}   [{marker}]")
        for name, left, right in pairs:
            if left not in present or right not in present:
                continue
            modes = sorted(set(_by_seed(records, left)[0]["attribute_modes"])
                           & set(_by_seed(records, right)[0]["attribute_modes"]))
            for mode in modes:
                a = _gap_values(records, left, domain, mode)
                b = _gap_values(records, right, domain, mode)
                if a is None or b is None:
                    continue
                left_seeds = [r["seed"] for r in _by_seed(records, left)]
                right_seeds = [r["seed"] for r in _by_seed(records, right)]
                if left_seeds != right_seeds:
                    message = (f"{name} on {domain}: {left} has seeds {left_seeds} and "
                               f"{right} has {right_seeds}. The difference is paired by "
                               f"seed and cannot be taken across different ones.")
                    if not allow_partial:
                        raise AggregateError(message)
                    # Under PARTIAL the set is known to be incomplete; say which
                    # comparison could not be made rather than quietly omitting it.
                    lines.append(f"  {name} / {mode}: SKIPPED - {message}")
                    continue
                deltas = [x["gap"] - y["gap"] for x, y in zip(a, b)]
                negatives = sum(d < 0 for d in deltas)
                lines.append(f"  {name} ({left} - {right}) / {mode}")
                lines.append(f"    {'delta':<24} {_seeds_and_summary(deltas)}")
                lines.append(f"    {'verdict':<24} {verdict(deltas):<14} "
                             f"{negatives}/{len(deltas)} seeds negative")
    return "\n".join(lines)


# --- the script -------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the *_eval.json records into the declared comparison.")
    parser.add_argument("--records", default=str(REPO / "eval_records"),
                        help="directory of *_eval.json records")
    parser.add_argument("--allow-partial", action="store_true",
                        help="report on fewer than the nine records, stamped PARTIAL")
    parser.add_argument("--json", default=None,
                        help="also write the assembled numbers as JSON")
    args = parser.parse_args(argv)

    records = load_records(args.records)
    try:
        report = build_report(records, allow_partial=args.allow_partial)
    except AggregateError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {f"{r['run_id']}_seed{r['seed']}": r for r in records},
            indent=2, default=str) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
