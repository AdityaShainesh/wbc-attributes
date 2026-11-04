"""Near-duplicate search over the PBC source domain.

Usage: python scripts/near_duplicates.py

Writes reports/near_duplicates.md and data/processed/near_duplicates.json. Read-only
with respect to data/raw/.

Why this exists: scripts/audit_data.py finds only byte-identical duplicates. That found
one genuine label error (BNE_191112 / EO_225902 share bytes but carry different WBCAtt
labels), which was luck -- a re-encoded or slightly cropped copy of the same cell would
be invisible to a sha256. This script looks for visually near-identical pairs instead,
to see whether more of the same problem is hiding.

Method: a 64-bit dHash per image (greyscale, resized to 9x8, each bit the sign of the
horizontal gradient), then Hamming distance between every pair. Distance 0 means the
images are visually indistinguishable at that resolution, not that the files match.

PBC only, and only the five differential classes. LISC and Raabin are target domains and
are deliberately not touched: a perceptual hash is derived from pixels, and the project
constraint (CLAUDE.md section 3) keeps target-domain pixels out of every decision.

Interpretation warning: every PBC image is one stained cell centred on a similar
background, so small Hamming distances occur by chance far more often than they would in
natural photographs. Distance 0 pairs are worth looking at; the report gives counts per
threshold rather than a single verdict, and no pair here is excluded automatically.
"""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PBC_DIR = RAW / "PBC_dataset_normal_DIB"
WBCATT_CSV = {s: RAW / f"pbc_attr_v1_{s}.csv" for s in ("train", "val", "test")}
MAPPING_PATH = ROOT / "configs" / "class_mapping.yaml"

OUT_MD = ROOT / "reports" / "near_duplicates.md"
OUT_JSON = ROOT / "data" / "processed" / "near_duplicates.json"

DIFFERENTIAL = ("neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil")
HASH_SIDE = 8                 # 8x8 comparisons -> 64 bits
REPORT_THRESHOLDS = (0, 2, 4, 6, 8)
LIST_THRESHOLD = 4            # pairs at or below this distance are listed individually
CHUNK = 512                   # rows of the distance matrix computed at once

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)


def log(msg):
    print(msg, flush=True)


def dhash(path):
    """64-bit difference hash, returned as 8 bytes, plus the file's sha256."""
    data = path.read_bytes()
    with Image.open(path) as im:
        small = im.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return np.packbits(bits), hashlib.sha256(data).hexdigest()


def load_wbcatt():
    """path -> (split, label) for every attribute-labelled image."""
    import csv
    out = {}
    for split, path in WBCATT_CSV.items():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[row["path"]] = (split, row["label"])
    return out


def excluded_relpaths(mapping):
    """Files already excluded by the mapping, as PBC-relative paths."""
    out = set()
    for key in mapping.get("excluded_files", {}):
        dataset, rel = key.split(":", 1)
        if dataset == "pbc" and not rel.endswith("/"):
            out.add(f"{PBC_DIR.name}/{rel}")
    return out


def pair_distances(hashes):
    """Hamming distances between all pairs; yields (i, j, distance) with i < j."""
    n = len(hashes)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        xor = np.bitwise_xor(hashes[start:stop, None, :], hashes[None, :, :])
        dist = POPCOUNT[xor].sum(axis=2)
        rows, cols = np.nonzero(dist <= max(REPORT_THRESHOLDS))
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                yield i, j, int(dist[r, c])
        log(f"  {stop}/{n}")


def main():
    started = time.time()
    if not PBC_DIR.exists():
        sys.exit(f"missing: {PBC_DIR}")
    mapping = yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8"))
    wbcatt = load_wbcatt()
    already_excluded = excluded_relpaths(mapping)

    log("hashing PBC (five differential classes)...")
    paths, hashes, shas = [], [], []
    skipped = []
    for folder in DIFFERENTIAL:
        for path in sorted(p for p in (PBC_DIR / folder).iterdir() if p.is_file()):
            rel = f"{PBC_DIR.name}/{folder}/{path.name}"
            if rel in already_excluded:
                skipped.append(rel)
                continue
            try:
                packed, sha = dhash(path)
            except Exception as exc:
                skipped.append(f"{rel} ({type(exc).__name__})")
                continue
            paths.append(rel)
            hashes.append(packed)
            shas.append(sha)
        log(f"  {folder}: {len(paths)} so far")
    hashes = np.vstack(hashes)

    log("comparing all pairs...")
    histogram = collections.Counter()
    pairs = []
    for i, j, dist in pair_distances(hashes):
        histogram[dist] += 1
        if dist <= LIST_THRESHOLD:
            split_i, label_i = wbcatt.get(paths[i], ("-", "-"))
            split_j, label_j = wbcatt.get(paths[j], ("-", "-"))
            pairs.append({
                "a": paths[i], "b": paths[j], "distance": dist,
                "byte_identical": shas[i] == shas[j],
                "split_a": split_i, "split_b": split_j,
                "label_a": label_i, "label_b": label_j,
                "cross_class": label_i != label_j,
                "cross_split": split_i != split_j,
            })
    pairs.sort(key=lambda p: (p["distance"], p["a"]))

    at_threshold = {t: sum(n for d, n in histogram.items() if d <= t) for t in REPORT_THRESHOLDS}
    cross_class = [p for p in pairs if p["cross_class"]]
    cross_split = [p for p in pairs if p["cross_split"]]

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": f"dHash {HASH_SIDE}x{HASH_SIDE} (64 bit), Hamming distance, PBC five classes only",
        "images_hashed": len(paths),
        "skipped": skipped,
        "pairs_at_or_below": at_threshold,
        "distance_histogram": dict(sorted(histogram.items())),
        "listed_threshold": LIST_THRESHOLD,
        "pairs": pairs,
        "cross_class_pairs": cross_class,
        "cross_split_pairs": cross_split,
        "wall_clock_s": round(time.time() - started, 1),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def table(headers, rows):
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(out)

    L = [
        "# Near-duplicate search (PBC source domain)",
        "",
        f"Generated by `scripts/near_duplicates.py` on {summary['generated']} in "
        f"{summary['wall_clock_s']}s. {len(paths)} images hashed across "
        f"{', '.join(DIFFERENTIAL)}.",
        "",
        "Method: 64-bit dHash, then Hamming distance over every pair. Distance 0 means the two",
        "images are indistinguishable at 8x8 greyscale, which is weaker evidence than an",
        "identical sha256 but catches re-encoded and lightly cropped copies that hashing misses.",
        "",
        "PBC only. LISC and Raabin are target domains and are deliberately not hashed: a",
        "perceptual hash is pixel-derived, and CLAUDE.md section 3 keeps target pixels out of",
        "every decision.",
        "",
        "**Every PBC image is one stained cell on a similar background, so low distances arise by",
        "chance much more often here than in natural images.** Nothing below is excluded",
        "automatically; this is evidence for a human call.",
        "",
        f"Files already excluded by the mapping were skipped: {len(skipped)}.",
        "",
        "## Pairs by distance",
        "",
        table(["Hamming distance <=", "pairs"], [[t, at_threshold[t]] for t in REPORT_THRESHOLDS]),
        "",
        "## What matters: cross-class pairs",
        "",
        "Two visually near-identical images carrying different WBCAtt labels. These are candidate",
        f"label errors. At distance <= {LIST_THRESHOLD}: **{len(cross_class)}**.",
        "",
    ]
    if cross_class:
        L += [table(["distance", "image A", "label A", "split A", "image B", "label B", "split B", "same bytes"],
                    [[p["distance"], f"`{p['a'].split('/', 1)[1]}`", p["label_a"], p["split_a"],
                      f"`{p['b'].split('/', 1)[1]}`", p["label_b"], p["split_b"],
                      "yes" if p["byte_identical"] else "no"]
                     for p in cross_class]), ""]
    L += ["## Cross-split pairs",
          "",
          "Near-identical images sitting in different splits. Not leakage in the strict sense --",
          "they are different files -- but a model can be credited at test time for a cell it",
          f"effectively saw in training. At distance <= {LIST_THRESHOLD}: **{len(cross_split)}**.",
          ""]
    if cross_split:
        L += [table(["distance", "image A", "split A", "image B", "split B", "same class", "same bytes"],
                    [[p["distance"], f"`{p['a'].split('/', 1)[1]}`", p["split_a"],
                      f"`{p['b'].split('/', 1)[1]}`", p["split_b"],
                      "yes" if not p["cross_class"] else "**no**",
                      "yes" if p["byte_identical"] else "no"]
                     for p in cross_split]), ""]
    same = [p for p in pairs if not p["cross_class"] and not p["cross_split"]]
    L += ["## Same class, same split",
          "",
          f"{len(same)} pairs at distance <= {LIST_THRESHOLD}. These affect nothing but the",
          "effective class weighting; listed in `data/processed/near_duplicates.json`.",
          ""]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"\nwrote {OUT_MD.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)} "
        f"in {summary['wall_clock_s']}s")


if __name__ == "__main__":
    main()
