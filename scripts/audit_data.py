"""Read-only audit of everything under data/raw/.

Usage: python scripts/audit_data.py

Writes data/processed/audit_report.md (human-readable), a tracked copy at
reports/audit_report.md, and data/processed/audit_summary.json (machine-readable).
Nothing under data/raw/ is created, modified, moved or deleted.

The report separates facts measured by this run from claims already written into
configs/class_mapping.yaml, and gives each claim an explicit verdict. The mapping is
validated against the data, never trusted as ground truth and never edited here.

Target domains (LISC, and Raabin once present) are audited for file structure and
metadata only - counts, names, dimensions, formats, content hashes. No pixel, colour
or stain statistics are computed for any dataset, so nothing here can feed a
normalization or modelling decision. See CLAUDE.md section 3.
"""
import collections
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mapping import RESERVED  # noqa: E402  single source of truth

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PBC_DIR = RAW / "PBC_dataset_normal_DIB"
LISC_DIR = RAW / "LISC"
WBCATT_CSV = {s: RAW / f"pbc_attr_v1_{s}.csv" for s in ("train", "val", "test")}
MAPPING_PATH = ROOT / "configs" / "class_mapping.yaml"

OUT_JSON = ROOT / "data" / "processed" / "audit_summary.json"
OUT_MD = ROOT / "data" / "processed" / "audit_report.md"
OUT_MD_TRACKED = ROOT / "reports" / "audit_report.md"

# Raabin is not downloaded yet. Any of these appearing is enough for the audit to pick
# it up on a later run without this script changing.
RAABIN_CANDIDATES = ("raabin", "Raabin", "Raabin-WBC", "raabin-wbc", "RaabinWBC")

# Confirmed with Adi before the first run. WBCAtt nucleus_shape values split into
# segmented and not-segmented for the band-neutrophil check. The full cross-tabulation
# is reported too, so any other definition can be recomputed without re-running this.
SEGMENTED = {"segmented-bilobed", "segmented-multilobed"}
NOT_SEGMENTED = {"unsegmented-band", "unsegmented-round", "unsegmented-indented", "irregular"}

ID_COLUMNS = ("img_name", "label", "path")


def log(msg):
    print(msg, flush=True)


def prefix_of(filename):
    """Class prefix as it appears in a filename: BNE_7323.jpg -> BNE, neut_3-1.bmp -> neut."""
    return filename.split("_")[0]


def scan_image(path):
    """Read a file once; return hash, size on disk, format, dimensions, decodability."""
    data = path.read_bytes()
    rec = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    try:
        with Image.open(io.BytesIO(data)) as im:
            rec["format"] = im.format
            rec["dims"] = f"{im.size[0]}x{im.size[1]}"
            im.load()  # full decode, so truncated pixel data is caught too
        rec["readable"] = True
    except Exception as exc:
        rec.update(format=None, dims=None, readable=False, error=f"{type(exc).__name__}: {exc}")
    return rec


def scan_folder(folder):
    """Scan every file directly inside folder, hidden files included."""
    return {p.name: scan_image(p) for p in sorted(x for x in folder.iterdir() if x.is_file())}


def dup_groups(records):
    """Group names by content hash and keep only the collisions."""
    by_hash = collections.defaultdict(list)
    for name, rec in records.items():
        by_hash[rec["sha256"]].append(name)
    return {h: sorted(names) for h, names in by_hash.items() if len(names) > 1}


def git_state():
    def run(*args):
        try:
            out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


# --------------------------------------------------------------------------- PBC

def audit_pbc():
    log("PBC: scanning...")
    if not PBC_DIR.exists():
        return {"present": False, "folders": {}}
    per_folder, records = {}, {}
    for folder in sorted(d for d in PBC_DIR.iterdir() if d.is_dir()):
        scanned = scan_folder(folder)
        per_folder[folder.name] = {
            "files": len(scanned),
            "prefixes": dict(sorted(collections.Counter(prefix_of(n) for n in scanned).items())),
            "formats": dict(sorted(collections.Counter(str(r["format"]) for r in scanned.values()).items())),
            "dims": dict(collections.Counter(str(r["dims"]) for r in scanned.values()).most_common()),
            "hidden_files": sorted(n for n in scanned if n.startswith(".")),
            "unreadable": {n: r["error"] for n, r in scanned.items() if not r["readable"]},
        }
        for name, rec in scanned.items():
            records[f"{PBC_DIR.name}/{folder.name}/{name}"] = rec
        log(f"  {folder.name}: {len(scanned)}")
    return {
        "present": True,
        "total_files": len(records),
        "folders": per_folder,
        "duplicate_hash_groups": dup_groups(records),
        "_records": records,
    }


# ------------------------------------------------------------------------ WBCAtt

def read_wbcatt():
    rows = {}
    for split, path in WBCATT_CSV.items():
        with open(path, newline="", encoding="utf-8") as fh:
            rows[split] = list(csv.DictReader(fh))
    return rows


def audit_wbcatt(rows, pbc, mapping):
    log("WBCAtt: checking...")
    records = pbc.get("_records", {})
    attr_cols = [c for c in rows["train"][0] if c not in ID_COLUMNS]
    all_rows = [r for split_rows in rows.values() for r in split_rows]

    per_split = {}
    for split, split_rows in rows.items():
        per_split[split] = {
            "rows": len(split_rows),
            "classes": dict(sorted(collections.Counter(r["label"] for r in split_rows).items())),
            "unresolved_paths": [r["path"] for r in split_rows if r["path"] not in records],
            "duplicate_img_names": sorted(
                n for n, c in collections.Counter(r["img_name"] for r in split_rows).items() if c > 1
            ),
            "empty_cells": sum(1 for r in split_rows for v in r.values() if v in ("", None)),
        }

    attributes = {}
    for col in attr_cols:
        attributes[col] = {
            label: dict(sorted(collections.Counter(r[col] for r in all_rows if r["label"] == label).items()))
            for label in sorted({r["label"] for r in all_rows})
        }

    names = {s: {r["img_name"] for r in rs} for s, rs in rows.items()}
    hashes = {
        s: {records[r["path"]]["sha256"]: r["img_name"] for r in rs if r["path"] in records}
        for s, rs in rows.items()
    }
    overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(hashes[a]) & set(hashes[b])
        overlap[f"{a} vs {b}"] = {
            "by_filename": sorted(names[a] & names[b]),
            "by_hash": sorted((hashes[a][h], hashes[b][h]) for h in shared),
        }

    # WBC images present in PBC but absent from WBCAtt. "WBC" means the PBC folders the
    # mapping assigns to one of the five differential classes.
    wbc_folders = {
        e.split(":", 1)[1]
        for cls in ("neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil")
        for e in mapping[cls]
        if e.startswith("pbc:")
    }
    covered = {r["path"] for r in all_rows}
    uncovered = sorted(p for p in records if p.split("/")[1] in wbc_folders and p not in covered)

    # granule_type, granule_colour and granularity should agree on whether a cell is granular.
    inconsistent = [
        {
            "img_name": r["img_name"], "label": r["label"], "granule_type": r["granule_type"],
            "granule_colour": r["granule_colour"], "granularity": r["granularity"],
        }
        for r in all_rows
        if len({r["granule_type"] == "nil", r["granule_colour"] == "nil", r["granularity"] == "no"}) > 1
    ]

    return {
        "attribute_columns": attr_cols,
        "total_rows": len(all_rows),
        "per_split": per_split,
        "attributes_by_class": attributes,
        "split_overlap": overlap,
        "pbc_wbc_images_not_in_wbcatt": uncovered,
        "inconsistent_granule_rows": inconsistent,
    }


# -------------------------------------------------------------------- exclusions

def parse_exclusions(mapping):
    """Split excluded_files into file entries and folder entries (keys ending in /)."""
    files, folders = {}, {}
    for key, reason in mapping.get("excluded_files", {}).items():
        dataset, rel = key.split(":", 1)
        (folders if rel.endswith("/") else files)[key] = {
            "dataset": dataset, "rel": rel.rstrip("/"), "reason": reason,
        }
    return files, folders


def audit_exclusions(mapping, pbc, rows):
    """Check every exclusion resolves, then confirm the splits are clean once applied."""
    log("exclusions: resolving and re-checking for leakage...")
    files, folders = parse_exclusions(mapping)
    records = pbc.get("_records", {})
    roots = {"pbc": PBC_DIR, "lisc": LISC_DIR}

    resolved, stale = {}, {}
    for key, entry in files.items():
        if entry["dataset"] == "pbc":
            path = f"{PBC_DIR.name}/{entry['rel']}"
            if path in records:
                resolved[key] = {"path": path, "sha256": records[path]["sha256"]}
            else:
                stale[key] = f"no such file under {PBC_DIR.name}/"
        else:
            root = roots.get(entry["dataset"])
            if root is None:
                stale[key] = f"unknown dataset prefix '{entry['dataset']}'"
            elif (root / entry["rel"]).exists():
                resolved[key] = {"path": f"{root.name}/{entry['rel']}", "sha256": None}
            else:
                stale[key] = f"no such file under {root.name}/"
    for key, entry in folders.items():
        root = roots.get(entry["dataset"])
        if root is None:
            stale[key] = f"unknown dataset prefix '{entry['dataset']}'"
        elif not (root / entry["rel"]).is_dir():
            stale[key] = f"no such folder under {root.name}/"

    # WBCAtt rows are dropped by filename, which is unique across the whole dataset.
    dropped_names = {Path(e["rel"]).name for e in files.values() if e["dataset"] == "pbc"}
    per_split = {}
    for split, split_rows in rows.items():
        dropped = sorted(r["img_name"] for r in split_rows if r["img_name"] in dropped_names)
        per_split[split] = {"rows_before": len(split_rows), "rows_after": len(split_rows) - len(dropped),
                            "dropped": dropped}

    # Recompute cross-split identical images with the exclusions applied. This is the
    # check that matters: it should come back empty.
    kept = {s: [r for r in rs if r["img_name"] not in dropped_names] for s, rs in rows.items()}
    hashes = {
        s: {records[r["path"]]["sha256"]: r["img_name"] for r in rs if r["path"] in records}
        for s, rs in kept.items()
    }
    residual = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(hashes[a]) & set(hashes[b])
        residual[f"{a} vs {b}"] = sorted((hashes[a][h], hashes[b][h]) for h in shared)

    # Conflicting labels among exact duplicates that survive the exclusions.
    label_of = {r["path"]: r["label"] for rs in kept.values() for r in rs}
    conflicts = []
    for hash_, paths in dup_groups({p: r for p, r in records.items() if p in label_of}).items():
        labels = {label_of[p] for p in paths}
        if len(labels) > 1:
            conflicts.append({"sha256": hash_, "paths": paths, "labels": sorted(labels)})

    return {
        "file_entries": len(files),
        "folder_entries": len(folders),
        "resolved": resolved,
        "stale": stale,
        "per_split": per_split,
        "residual_cross_split_identical": residual,
        "residual_conflicting_labels": conflicts,
        "_dropped_names": dropped_names,
    }


# --------------------------------------------------------- band-neutrophil claim

def audit_band_claim(rows, mapping, dropped_names=frozenset()):
    log("band-neutrophil claim: recomputing...")
    all_rows = [r for split_rows in rows.values() for r in split_rows]
    neut = [r for r in all_rows if r["path"].split("/")[1] == "neutrophil"]

    crosstab = collections.defaultdict(collections.Counter)
    for r in neut:
        crosstab[prefix_of(r["img_name"])][r["nucleus_shape"]] += 1
    shapes = sorted({r["nucleus_shape"] for r in neut})

    sne_not_segmented = [r["img_name"] for r in neut
                         if prefix_of(r["img_name"]) == "SNE" and r["nucleus_shape"] not in SEGMENTED]
    bne_segmented = [r["img_name"] for r in neut
                     if prefix_of(r["img_name"]) == "BNE" and r["nucleus_shape"] in SEGMENTED]
    sne_band = sum(1 for r in neut
                   if prefix_of(r["img_name"]) == "SNE" and r["nucleus_shape"] == "unsegmented-band")

    # Secondary subsets, each by name, counted both before and after the exclusions.
    subsets = {}
    for name, spec in mapping.get("secondary_subsets", {}).items():
        split = spec["split"].replace("pbc_", "")
        drops = {k.split(":", 1)[1]: tuple(v) for k, v in spec["drop_filename_prefixes"].items()}

        def apply(source):
            kept_rows = [r for r in source
                         if not r["img_name"].startswith(drops.get(r["path"].split("/")[1], ()))]
            return {
                "rows": len(kept_rows),
                "neutrophils": sum(1 for r in kept_rows if r["path"].split("/")[1] == "neutrophil"),
                "band_labelled_survivors": sum(
                    1 for r in kept_rows
                    if r["path"].split("/")[1] == "neutrophil" and r["nucleus_shape"] == "unsegmented-band"),
            }

        subsets[name] = {
            "split": spec["split"],
            "drop_filename_prefixes": {k: list(v) for k, v in drops.items()},
            "before_exclusions": apply(rows[split]),
            "after_exclusions": apply([r for r in rows[split] if r["img_name"] not in dropped_names]),
        }

    return {
        "definition": {"segmented": sorted(SEGMENTED), "not_segmented": sorted(NOT_SEGMENTED)},
        "nucleus_shape_values_seen": shapes,
        "nucleus_shape_values_outside_definition": sorted(set(shapes) - SEGMENTED - NOT_SEGMENTED),
        "neutrophil_rows": len(neut),
        "crosstab": {p: dict(sorted(c.items())) for p, c in sorted(crosstab.items())},
        "disagreements": {
            "sne_not_segmented": len(sne_not_segmented),
            "bne_segmented": len(bne_segmented),
            "total": len(sne_not_segmented) + len(bne_segmented),
            "sne_labelled_unsegmented_band": sne_band,
        },
        "secondary_subsets": subsets,
    }


# -------------------------------------------------------------------------- LISC

def audit_lisc():
    log("LISC: scanning...")
    if not LISC_DIR.exists():
        return {"present": False, "folders": {}}
    folders = sorted(d.name for d in LISC_DIR.iterdir() if d.is_dir())
    scans, per_folder = {}, {}
    for name in folders:
        scanned = scan_folder(LISC_DIR / name)
        scans[name] = scanned
        dims = [tuple(int(x) for x in r["dims"].split("x")) for r in scanned.values() if r["dims"]]
        per_folder[name] = {
            "files": len(scanned),
            "classes": dict(sorted(collections.Counter(prefix_of(n) for n in scanned).items())),
            "formats": dict(sorted(collections.Counter(str(r["format"]) for r in scanned.values()).items())),
            "distinct_dims": len({r["dims"] for r in scanned.values()}),
            "dims_range": {
                "min_w": min(d[0] for d in dims), "max_w": max(d[0] for d in dims),
                "min_h": min(d[1] for d in dims), "max_h": max(d[1] for d in dims),
            } if dims else None,
            "unreadable": {n: r["error"] for n, r in scanned.items() if not r["readable"]},
            "duplicate_hash_groups": dup_groups(scanned),
        }
        log(f"  {name}: {len(scanned)}")

    def cross(a, b):
        if a not in scans or b not in scans:
            return None
        ha = {r["sha256"]: n for n, r in scans[a].items()}
        hb = {r["sha256"]: n for n, r in scans[b].items()}
        shared = set(ha) & set(hb)
        return {
            "shared_filenames": len(set(scans[a]) & set(scans[b])),
            "identical_images": len(shared),
            "examples": sorted((ha[h], hb[h]) for h in shared)[:5],
        }

    # Does every Train_Aug file trace back to a Train filename?
    traced = None
    if "Train" in scans and "Train_Aug" in scans:
        train_stems = {Path(n).stem for n in scans["Train"]}
        aug = list(scans["Train_Aug"])
        traced = {
            "exact_filename_matches": sum(1 for n in aug if Path(n).stem in train_stems),
            "stem_prefix_matches": sum(1 for n in aug if any(Path(n).stem.startswith(s) for s in train_stems)),
            "untraceable_examples": sorted(
                n for n in aug if not any(Path(n).stem.startswith(s) for s in train_stems))[:10],
        }

    jsons = {}
    for name in folders:
        path = LISC_DIR / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_prefix = collections.defaultdict(collections.Counter)
        for key, value in data.items():
            by_prefix[prefix_of(key)][str(value)] += 1
        jsons[path.name] = {
            "entries": len(data),
            "value_counts": dict(sorted(collections.Counter(str(v) for v in data.values()).items())),
            "value_types": dict(sorted(collections.Counter(type(v).__name__ for v in data.values()).items())),
            "class_prefix_to_value": {p: dict(sorted(c.items())) for p, c in sorted(by_prefix.items())},
            "keys_not_on_disk": sorted(set(data) - set(scans.get(name, {})))[:10],
            "files_not_in_json": sorted(set(scans.get(name, {})) - set(data))[:10],
        }

    return {
        "present": True,
        "folders": per_folder,
        "train_vs_test": cross("Train", "Test"),
        "train_vs_train_aug": cross("Train", "Train_Aug"),
        "train_aug_traceability": traced,
        "json_files": jsons,
    }


# ------------------------------------------------------------------------ Raabin

def audit_raabin():
    found = [c for c in RAABIN_CANDIDATES if (RAW / c).exists()]
    if not found:
        return {
            "present": False,
            "found_at": [],
            "folders": {},
            "note": "Not downloaded as of this run. Re-run this script once it is in "
                    "data/raw/ under one of: " + ", ".join(RAABIN_CANDIDATES) + ".",
        }

    root = RAW / found[0]
    log(f"Raabin: scanning {root.name}/ ...")
    per_split = {}
    for split in sorted(d for d in root.iterdir() if d.is_dir()):
        classes, dims, formats, unreadable = {}, collections.Counter(), collections.Counter(), {}
        for folder in sorted(d for d in split.iterdir() if d.is_dir()):
            scanned = scan_folder(folder)
            classes[folder.name] = len(scanned)
            dims.update(str(r["dims"]) for r in scanned.values())
            formats.update(str(r["format"]) for r in scanned.values())
            unreadable.update({f"{folder.name}/{n}": r["error"]
                               for n, r in scanned.items() if not r["readable"]})
        per_split[split.name] = {
            "images": sum(classes.values()),
            "classes": classes,
            "class_count": len(classes),
            "formats": dict(formats),
            "distinct_dims": len(dims),
            "unreadable": unreadable,
        }
        log(f"  {split.name}: {per_split[split.name]['images']} in {len(classes)} classes")

    return {
        "present": True,
        "found_at": found,
        "folders": per_split,
        "note": "Target domain. Class coverage per split is declared in "
                "configs/class_mapping.yaml under target_splits; Test-B carries only "
                "2 of the 5 classes and is reported as a two-class evaluation.",
    }


# ----------------------------------------------------------------- mapping check

def audit_mapping(mapping, pbc, rows, lisc, raabin):
    log("mapping: validating against the data...")
    observed = {f"pbc:{folder}" for folder in pbc.get("folders", {})}
    for split_rows in rows.values():
        observed |= {f"wbcatt:{r['label']}" for r in split_rows}
    for folder in ("Train", "Test"):  # Train_Aug is excluded as a folder, not a class source
        observed |= {f"lisc:{c}" for c in lisc.get("folders", {}).get(folder, {}).get("classes", {})}
    # Raabin's Train split is DANN-adaptation-only, but its class names still exist in
    # the data and so must be mapped; validation is about names, not about use.
    for split in raabin.get("folders", {}).values():
        observed |= {f"raabin:{c}" for c in split.get("classes", {})}

    # Single source of truth, shared with the loader: a second hand-maintained copy of
    # this list silently treated `target_splits` as a sixth class and reported its two
    # keys as stale mapping entries.
    classes = [k for k in mapping if k not in RESERVED]
    mapped = {}
    for cls in classes:
        for entry in mapping[cls]:
            mapped.setdefault(entry, []).append(cls)
    excluded = set(mapping.get("excluded", {}))

    return {
        "target_classes": classes,
        "observed_class_names": sorted(observed),
        "unmapped": sorted(observed - set(mapped) - excluded),
        "stale_entries": sorted((set(mapped) | excluded) - observed),
        "listed_under_multiple_classes": {e: c for e, c in mapped.items() if len(c) > 1},
        "excluded": mapping.get("excluded", {}),
        "excluded_files": mapping.get("excluded_files", {}),
    }


# ------------------------------------------------------------------------ report

def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def verdict(ok):
    return "**REPRODUCES**" if ok else "**DOES NOT REPRODUCE**"


def build_report(s):
    pbc, att, band = s["pbc"], s["wbcatt"], s["band_neutrophil_claim"]
    lisc, raabin, mapping = s["lisc"], s["raabin"], s["mapping_validation"]
    exc, git = s["exclusions"], s["git"]

    L = [
        "# Data audit",
        "",
        f"Generated by `scripts/audit_data.py` on {s['generated']} "
        f"(commit `{git['commit']}`{', dirty tree' if git['dirty'] else ''}) in {s['wall_clock_s']}s.",
        "",
        "Read-only: nothing under `data/raw/` was modified. Target domains are audited for",
        "structure and metadata only - no pixel, colour or stain statistics appear anywhere in",
        "this report (CLAUDE.md section 3).",
        "",
        "Sections 1-6 are measured from the data by this run. Section 7 gives verdicts on claims",
        "already written into `configs/class_mapping.yaml`. Section 8 validates that mapping",
        "against the class names actually present.",
        "",
        "## 1. PBC (source domain)",
        "",
    ]
    if not pbc["present"]:
        L += ["`data/raw/PBC_dataset_normal_DIB/` not found.", ""]
    else:
        L += [f"{pbc['total_files']} files across {len(pbc['folders'])} folders.", "",
              md_table(["folder", "files", "filename prefixes", "formats", "distinct dims", "hidden", "unreadable"],
                       [[name, f["files"],
                         ", ".join(f"{k} {v}" for k, v in f["prefixes"].items()),
                         ", ".join(f"{k} {v}" for k, v in f["formats"].items()),
                         len(f["dims"]), len(f["hidden_files"]), len(f["unreadable"])]
                        for name, f in pbc["folders"].items()]), ""]
        dims = collections.Counter()
        for f in pbc["folders"].values():
            dims.update(f["dims"])
        L += ["Dimensions across all folders: " +
              ", ".join(f"`{d}` x{n}" for d, n in dims.most_common(6)) + ".", ""]
        unreadable = {f"{name}/{fn}": err
                      for name, f in pbc["folders"].items() for fn, err in f["unreadable"].items()}
        L += [f"**Unreadable files: {len(unreadable)}.**" if unreadable else "All files decode cleanly.", ""]
        L += [f"- `{p}` - {e}" for p, e in unreadable.items()]
        if unreadable:
            L += [""]
        hidden = [f"{name}/{fn}" for name, f in pbc["folders"].items() for fn in f["hidden_files"]]
        L += [f"Hidden files: {', '.join('`' + h + '`' for h in hidden) if hidden else 'none'}.", "",
              f"Exact-duplicate image groups (by sha256): **{len(pbc['duplicate_hash_groups'])}**.", ""]
        L += [f"- `{h[:12]}`: {', '.join(n)}" for h, n in list(pbc["duplicate_hash_groups"].items())[:15]]
        if pbc["duplicate_hash_groups"]:
            L += [""]

    L += ["## 2. WBCAtt (attribute labels)", "",
          f"{att['total_rows']} rows over {len(att['attribute_columns'])} attribute columns.", "",
          md_table(["split", "rows", "classes", "unresolved paths", "dup img_name", "empty cells"],
                   [[name, d["rows"], ", ".join(f"{k} {v}" for k, v in d["classes"].items()),
                     len(d["unresolved_paths"]), len(d["duplicate_img_names"]), d["empty_cells"]]
                    for name, d in att["per_split"].items()]), "",
          "### Split overlap", "",
          md_table(["pair", "shared filenames", "identical images (by hash)"],
                   [[p, len(d["by_filename"]), len(d["by_hash"])] for p, d in att["split_overlap"].items()]), "",
          f"PBC WBC images not covered by WBCAtt: **{len(att['pbc_wbc_images_not_in_wbcatt'])}**.", ""]
    L += [f"- `{p}`" for p in att["pbc_wbc_images_not_in_wbcatt"][:10]]
    L += ["", f"### Logically inconsistent granule rows: {len(att['inconsistent_granule_rows'])}", "",
          "`granule_type`, `granule_colour` and `granularity` should agree on whether a cell is granular.", ""]
    if att["inconsistent_granule_rows"]:
        L += [md_table(["img_name", "label", "granule_type", "granule_colour", "granularity"],
                       [[r["img_name"], r["label"], r["granule_type"], r["granule_colour"], r["granularity"]]
                        for r in att["inconsistent_granule_rows"]]), ""]
    L += ["### Attribute values by class", ""]
    for col, by_class in att["attributes_by_class"].items():
        L += [f"**{col}**", "",
              md_table(["class", "values"],
                       [[c, ", ".join(f"{k} {v}" for k, v in vals.items())] for c, vals in by_class.items()]), ""]

    residual_pairs = sum(len(v) for v in exc["residual_cross_split_identical"].values())
    L += ["## 3. Exclusions declared in configs/class_mapping.yaml", "",
          f"{exc['file_entries']} file entries and {exc['folder_entries']} folder entries.", "",
          md_table(["check", "result"],
                   [["Entries that resolve to something on disk", len(exc["resolved"])],
                    ["Stale entries (nothing to exclude)",
                     ", ".join(f"`{k}`: {v}" for k, v in exc["stale"].items()) or "none"]]), "",
          "### Splits once the exclusions are applied", "",
          md_table(["split", "rows before", "rows after", "dropped"],
                   [[name, d["rows_before"], d["rows_after"], ", ".join(f"`{n}`" for n in d["dropped"]) or "-"]
                    for name, d in exc["per_split"].items()]), "",
          "### Residual leakage", "",
          "Cross-split byte-identical images remaining after exclusions "
          f"(**{residual_pairs}**; this is the number that must be zero):", "",
          md_table(["pair", "identical images", "examples"],
                   [[p, len(v), ", ".join(f"`{a}` = `{b}`" for a, b in v[:4]) or "-"]
                    for p, v in exc["residual_cross_split_identical"].items()]), "",
          f"Exact-duplicate groups still carrying conflicting WBCAtt labels: "
          f"**{len(exc['residual_conflicting_labels'])}**.", ""]
    L += [f"- `{c['sha256'][:12]}`: {', '.join(c['paths'])} labelled {', '.join(c['labels'])}"
          for c in exc["residual_conflicting_labels"]]
    if exc["residual_conflicting_labels"]:
        L += [""]

    L += ["## 4. Band-neutrophil check", "",
          "Definition used (agreed before the run): segmented = " +
          ", ".join(f"`{v}`" for v in band["definition"]["segmented"]) + "; everything else (" +
          ", ".join(f"`{v}`" for v in band["definition"]["not_segmented"]) + ") is not segmented.", ""]
    if band["nucleus_shape_values_outside_definition"]:
        L += ["**Values seen that the definition does not cover: " +
              ", ".join(f"`{v}`" for v in band["nucleus_shape_values_outside_definition"]) + ".**", ""]
    shapes = band["nucleus_shape_values_seen"]
    L += [f"{band['neutrophil_rows']} neutrophil rows. Full cross-tabulation:", "",
          md_table(["filename prefix"] + shapes + ["total"],
                   [[p] + [band["crosstab"][p].get(sh, 0) for sh in shapes] + [sum(band["crosstab"][p].values())]
                    for p in band["crosstab"]]), "",
          "Disagreements under that definition:", "",
          md_table(["direction", "count"],
                   [["`SNE_` not segmented", band["disagreements"]["sne_not_segmented"]],
                    ["`BNE_` segmented", band["disagreements"]["bne_segmented"]],
                    ["**total**", band["disagreements"]["total"]],
                    ["(`SNE_` labelled `unsegmented-band` specifically)",
                     band["disagreements"]["sne_labelled_unsegmented_band"]]]), "",
          "### Secondary subsets", "",
          md_table(["subset", "split", "rows (before -> after exclusions)", "neutrophils",
                    "of which WBCAtt calls `unsegmented-band`"],
                   [[f"`{name}`", d["split"],
                     f"{d['before_exclusions']['rows']} -> {d['after_exclusions']['rows']}",
                     d["after_exclusions"]["neutrophils"],
                     d["after_exclusions"]["band_labelled_survivors"]]
                    for name, d in band["secondary_subsets"].items()]), "",
          "The last column is the point to keep in view: filtering on filename prefixes does not",
          "remove the cells WBCAtt considers bands.", ""]

    L += ["## 5. LISC (target domain)", ""]
    if not lisc["present"]:
        L += ["`data/raw/LISC/` not found.", ""]
    else:
        L += [md_table(["folder", "files", "classes", "formats", "distinct dims", "w range", "h range",
                        "unreadable", "dup groups"],
                       [[n, f["files"], ", ".join(f"{k} {v}" for k, v in f["classes"].items()),
                         ", ".join(f"{k} {v}" for k, v in f["formats"].items()), f["distinct_dims"],
                         f"{f['dims_range']['min_w']}-{f['dims_range']['max_w']}" if f["dims_range"] else "",
                         f"{f['dims_range']['min_h']}-{f['dims_range']['max_h']}" if f["dims_range"] else "",
                         len(f["unreadable"]), len(f["duplicate_hash_groups"])]
                        for n, f in lisc["folders"].items()]), "",
              "### Overlap between folders", "",
              md_table(["pair", "shared filenames", "identical images"],
                       [[p, d["shared_filenames"], d["identical_images"]]
                        for p, d in (("Train vs Test", lisc["train_vs_test"]),
                                     ("Train vs Train_Aug", lisc["train_vs_train_aug"])) if d]), ""]
        if lisc["train_aug_traceability"]:
            t = lisc["train_aug_traceability"]
            L += [f"`Train_Aug` filename traceability: {t['exact_filename_matches']} share a filename with "
                  f"`Train`; {t['stem_prefix_matches']} begin with a `Train` filename stem.", ""]
            if t["untraceable_examples"]:
                L += ["Examples tracing to no `Train` file: " +
                      ", ".join(f"`{n}`" for n in t["untraceable_examples"]) + ".", ""]
        L += ["### The .json files", ""]
        for name, j in lisc["json_files"].items():
            L += [f"**{name}** - {j['entries']} entries; value types " +
                  ", ".join(f"{k} x{v}" for k, v in j["value_types"].items()) + ".", "",
                  md_table(["class prefix", "values"],
                           [[p, ", ".join(f"{k} x{v}" for k, v in vals.items())]
                            for p, vals in j["class_prefix_to_value"].items()]), ""]
            if j["keys_not_on_disk"] or j["files_not_in_json"]:
                L += [f"Keys with no file on disk: {len(j['keys_not_on_disk'])}; "
                      f"files absent from the json: {len(j['files_not_in_json'])}.", ""]

    L += ["## 6. Raabin-WBC (target domain)", "", raabin["note"], ""]
    if raabin.get("folders"):
        class_names = sorted({c for s in raabin["folders"].values() for c in s["classes"]})
        L += [md_table(["split", "images", "classes"] + class_names + ["unreadable"],
                       [[name, d["images"], d["class_count"]]
                        + [d["classes"].get(c, "-") for c in class_names]
                        + [len(d["unreadable"])]
                        for name, d in raabin["folders"].items()]), "",
              "Test-B covers only part of the label space, which is why "
              "`configs/class_mapping.yaml` declares its class coverage and the eval "
              "harness reports it separately.", ""]

    dis = band["disagreements"]
    claims = [
        ["662 `SNE_` files are labelled `unsegmented-band`",
         dis["sne_labelled_unsegmented_band"], verdict(dis["sne_labelled_unsegmented_band"] == 662)],
        ["670 `SNE_` files are not segmented at all",
         dis["sne_not_segmented"], verdict(dis["sne_not_segmented"] == 670)],
        ["48 `BNE_` files are labelled segmented",
         dis["bne_segmented"], verdict(dis["bne_segmented"] == 48)],
        ["718 disagreements in total",
         dis["total"], verdict(dis["total"] == 718)],
        ["No cross-split identical images survive the exclusions",
         f"{sum(len(v) for v in exc['residual_cross_split_identical'].values())} remaining",
         verdict(not any(exc["residual_cross_split_identical"].values()))],
        ["No exact duplicate carries conflicting labels after exclusions",
         f"{len(exc['residual_conflicting_labels'])} remaining",
         verdict(not exc["residual_conflicting_labels"])],
        ["Every exclusion entry resolves",
         f"{len(exc['stale'])} stale", verdict(not exc["stale"])],
    ]
    for name, d in band["secondary_subsets"].items():
        claims.append([f"`{name}` counts recorded in the mapping",
                       f"{d['after_exclusions']['rows']} rows, {d['after_exclusions']['neutrophils']} neutrophils "
                       f"(after exclusions)", ""])
    if pbc["present"]:
        neut = pbc["folders"].get("neutrophil", {})
        claims.append([
            "`neutrophil/.DS_169665.jpg` is not a real image and is absent from WBCAtt",
            f"hidden: {bool(neut.get('hidden_files'))}; "
            f"fails to decode: {'.DS_169665.jpg' in neut.get('unreadable', {})}; "
            f"absent from WBCAtt: {any('.DS_169665.jpg' in p for p in att['pbc_wbc_images_not_in_wbcatt'])}",
            ""])
    if lisc["present"] and lisc["train_vs_train_aug"]:
        claims.append([
            "`LISC/Train_Aug/` holds augmented copies of `LISC/Train`",
            f"{lisc['train_vs_train_aug']['identical_images']} byte-identical to a `Train` image, "
            f"out of {lisc['folders']['Train_Aug']['files']} files",
            ""])
    L += ["## 7. Verdicts on claims in configs/class_mapping.yaml", "",
          "These numbers were written into the mapping before any audit ran. Recomputed from scratch:", "",
          md_table(["claim", "measured", "verdict"], claims), ""]

    L += ["## 8. Mapping validation", "",
          f"Target classes: {', '.join('`' + c + '`' for c in mapping['target_classes'])}.", "",
          "Class names observed in the data: " +
          ", ".join(f"`{c}`" for c in mapping["observed_class_names"]) + ".", "",
          md_table(["check", "result"],
                   [["Unmapped names (hard error per CLAUDE.md section 4)",
                     ", ".join(f"`{e}`" for e in mapping["unmapped"]) or "none"],
                    ["Stale entries (mapped or excluded but not in the data)",
                     ", ".join(f"`{e}`" for e in mapping["stale_entries"]) or "none"],
                    ["Listed under more than one class",
                     ", ".join(f"`{e}` -> {v}" for e, v in mapping["listed_under_multiple_classes"].items())
                     or "none"]]), "",
          "Excluded classes and their stated reasons:", "",
          md_table(["entry", "reason"], [[f"`{k}`", v] for k, v in mapping["excluded"].items()]), ""]
    return "\n".join(L) + "\n"


def main():
    started = time.time()
    for path in (PBC_DIR, *WBCATT_CSV.values(), MAPPING_PATH):
        if not path.exists():
            sys.exit(f"missing: {path}")

    mapping = yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8"))
    pbc = audit_pbc()
    rows = read_wbcatt()
    att = audit_wbcatt(rows, pbc, mapping)
    exclusions = audit_exclusions(mapping, pbc, rows)
    band = audit_band_claim(rows, mapping, exclusions.pop("_dropped_names"))
    lisc = audit_lisc()
    raabin = audit_raabin()
    validation = audit_mapping(mapping, pbc, rows, lisc, raabin)

    pbc.pop("_records", None)
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_state(),
        "pbc": pbc,
        "wbcatt": att,
        "exclusions": exclusions,
        "band_neutrophil_claim": band,
        "lisc": lisc,
        "raabin": raabin,
        "mapping_validation": validation,
        "wall_clock_s": round(time.time() - started, 1),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD_TRACKED.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    report = build_report(summary)
    OUT_MD.write_text(report, encoding="utf-8")
    OUT_MD_TRACKED.write_text(report, encoding="utf-8")
    log(f"\nwrote {OUT_MD.relative_to(ROOT)}, {OUT_MD_TRACKED.relative_to(ROOT)} and "
        f"{OUT_JSON.relative_to(ROOT)} in {summary['wall_clock_s']}s")


if __name__ == "__main__":
    main()
