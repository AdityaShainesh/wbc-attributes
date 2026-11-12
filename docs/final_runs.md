# The nine final runs

Three models × three seeds, on Kaggle, followed by one evaluation pass that touches
Raabin exactly once. Setup (cloning, dependencies, `torch.cuda.is_available()`,
`CUBLAS_WORKSPACE_CONFIG`, the test suite) is in [kaggle_runbook.md](kaggle_runbook.md)
and is not repeated here.

Read section 0 before booking any GPU time. Two things in it are blockers.

---

## 0. Blockers and prerequisites

### 0.1 RESOLVED — the nine configs

`seed` lives in the YAML and CLAUDE.md §5 forbids a CLI flag that changes behaviour the
config does not record, so each of the nine runs is its own config. Seed 0 keeps the
original names; seeds 1 and 2 are new files differing in `seed` alone:

| model | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| baseline | `baseline.yaml` | `baseline_seed1.yaml` | `baseline_seed2.yaml` |
| bottleneck | `bottleneck.yaml` | `bottleneck_seed1.yaml` | `bottleneck_seed2.yaml` |
| control | `control.yaml` | `control_seed1.yaml` | `control_seed2.yaml` |

`run_id` is shared across the three seeds of a model, so `src.train` writes to
`results/<model>_seed<N>/` — which for seeds 1 and 2 is also the config's own basename.
Tests assert `seed` is the only field that differs, and that no two experiment configs
can resolve to the same results directory.

### 0.2 RESOLVED — evaluation writes its record

Every evaluation now writes `<results_dir>/<run_id>_seed<seed>_eval.json` containing
everything `evaluate_checkpoint` returns: all metrics per domain and per mode, the
checkpoint hash, the resolved config, the git state, the selection role and rule, the
abstention thresholds and the buffer hashes. A dry run writes
`..._eval_dryrun.json` instead, so a cheap repeatable run cannot overwrite the record
of the one look.

The record is written **before** any ledger line, and a failed write aborts before the
ledger is touched: an entry saying a domain was looked at, with nothing saying what was
found, would spend the look and keep nothing. An existing record is not overwritten
without `--override-reason`.

### 0.3 ImageNet weights

All three configs carry `imagenet_weights: null`, which downloads. A *Save & Run All*
commit has no internet unless the notebook's internet setting is on, and that needs a
verified phone number on the account. If internet is off the run fails loudly rather than
training from random init (§5), which is correct but wastes the booking.

Either turn internet on for the notebook, or attach the ResNet-50 weights as a Kaggle
dataset and set `model.imagenet_weights` to that path in all nine configs. Decide before
booking.

### 0.4 Check the ledger is still empty

```bash
wc -l target_eval_ledger.jsonl        # expect 0
```

It is 0 as of this writing. A non-zero count before the final runs means a target or
`pbc_test` domain has already been looked at, and you need to read the file and know why
before adding to it.

### 0.5 Pre-run checklist

- [ ] 0.1 decided and the six configs committed, with a test
- [ ] 0.2 is in place (nothing to do)
- [ ] 0.3 decided; `imagenet_weights` consistent across all nine configs
- [ ] `target_eval_ledger.jsonl` is 0 lines
- [ ] Working tree clean and pushed — every result records the commit and a dirty flag,
      and a dirty final run is not reproducible
- [ ] `pytest` green on Kaggle, after the runbook's dependency step
- [ ] GPU quota: check the remaining weekly allowance against section 2's estimate

---

## 1. Why three notebooks and not one

Group by **seed**, not by model: one notebook runs baseline, bottleneck and control at a
single seed.

```
notebook A   baseline seed 0   bottleneck seed 0   control seed 0
notebook B   baseline seed 1   bottleneck seed 1   control seed 1
notebook C   baseline seed 2   bottleneck seed 2   control seed 2
```

After notebook A you hold a complete three-model comparison — single-seed, so not
reportable, but complete. Grouping by model would leave you after notebook A with three
baselines and nothing to compare them to. If a session dies, by-seed loses less.

It also keeps each notebook to three runs, which matters because Kaggle caps a GPU
session at 12 hours and all nine runs together are close enough to that cap to be a
gamble.

**The three seeds are not a reason to look at a target domain three times.** The ledger
keys on the checkpoint hash, so three seeds give three legitimate entries per domain —
that is expected, not an override. Section 4 covers this.

---

## 2. Expected length

From the wiring runs: **142 s/epoch**, 30 epochs, so **~71 min per run** and **~3 h 35
min per notebook**, ~10.7 GPU-hours for all nine. Against a 30 h/week quota that is about
a third of a week's allowance, and each notebook sits well inside the 12 h session cap.

Two caveats on that number:

- It predates A1 (`2685c9c`) and A2 (`ea01e25`), which remove 579 device
  synchronisations per epoch and let the host-to-device copy overlap. Neither changes a
  number; both may lower the time.
- The Step 0 measurement (`M1`/`M2'`/`M3`/`M4`) has not been run, so it is not yet known
  whether the 142 s is the data pipeline or the GPU step under deterministic kernels. If
  it is the GPU step, 142 s is roughly the floor.

Time the first run before assuming the rest, and compare against `seconds_per_epoch` in
its results JSON.

---

## 3. The notebook, cell by cell

One notebook per seed. Set `SEED` in cell 2 and change nothing else between the three.

### Cell 1 — setup

Clone, install dependencies **without torch/torchvision**, assert CUDA. Verbatim from
[kaggle_runbook.md](kaggle_runbook.md) §§2–4. Ends with:

```python
import torch
assert torch.cuda.is_available(), "no GPU: set Accelerator before Save & Run All"
print(torch.cuda.get_device_name(0))
```

### Cell 2 — the one thing that differs between notebooks

```python
SEED = 0          # 0, 1 or 2 — the ONLY edit between the three notebooks
SUFFIX = "" if SEED == 0 else f"_seed{SEED}"
CONFIGS = [f"configs/{m}{SUFFIX}.yaml" for m in ("baseline", "bottleneck", "control")]
print(CONFIGS)
```

Seed 0 uses the original filenames, seeds 1 and 2 the `_seed<N>` ones, so this cell is
the same text in all three notebooks and only `SEED` changes.

### Cell 3 — the test suite

```bash
!cd /kaggle/working/<repo> && python -m pytest -q
```

Non-negotiable before a 3.5-hour booking. The numeric lock in `tests/test_bit_identity.py`
will skip on Kaggle — it is gated on the environment it was captured in, and that is
expected, not a failure.

### Cell 4 — record what is about to run

```bash
!cd /kaggle/working/<repo> && git rev-parse HEAD && git status --porcelain && wc -l target_eval_ledger.jsonl
```

Empty `git status` output and a 0-line ledger. If either is wrong, stop here.

### Cells 5, 6, 7 — one cell per run

One run per cell, never a loop over all three. A cell is the unit Kaggle shows you
timing and output for, and if the third run dies you want the first two's output
unambiguous.

```bash
!cd /kaggle/working/<repo> && python -m src.train --config configs/baseline.yaml
```

Each prints its final-epoch `val_macro_f1`, the selected epoch, per-epoch seconds and the
checkpoint path. Expect `selected_epoch` to equal `epochs_run - 1` — that is the
final-epoch rule. A run that fires the divergence guard raises `DivergenceError`,
writes **no** checkpoint, and leaves a results JSON with `failed: true`; the cell fails
and the next one still runs.

### Cell 8 — manifest

```python
import json, glob, hashlib
for f in sorted(glob.glob("/kaggle/working/results/*/*_seed*.json")):
    d = json.load(open(f))
    print(f"{d['run_id']:<12} seed {d['seed']}  "
          f"{d['val_macro_f1']:.4f} @ epoch {d['selected_epoch']}  "
          f"failed={d['failed']}  {sum(d['seconds_per_epoch'])/60:.1f} min")
for f in sorted(glob.glob("/kaggle/working/results/*/last.pt")):
    print(hashlib.sha256(open(f,'rb').read()).hexdigest()[:16], f)
```

The `last.pt` hashes are what the ledger will key on. Keep this cell's output: it is how
you check later that the checkpoint you evaluated is the one this notebook produced.

### Cell 9 — nothing

Do **not** run `src.eval` in this notebook, not even `--dry-run`. Training and the
one-time target look are separate bookings on purpose (section 4).

---

## 4. Bringing the results home

Kaggle does not persist `/kaggle/working` beyond the session, and the ledger lives at
`<repo>/target_eval_ledger.jsonl` — inside the clone, so inside the ephemeral working
directory. **Nothing is safe until it is on your machine and committed.**

### 4.1 Download

After each notebook commit, from the notebook's **Output** tab, download the zip. It
contains `results/<run_id>_seed<seed>/` with `last.pt`, `best.pt` and the results JSON
for each of the three runs. Roughly 200 MB per run of checkpoints, so ~600 MB per
notebook.

### 4.2 What goes into git and what does not

```
results/<run_id>_seed<seed>/<run_id>_seed<seed>.json    keep locally; results/ is gitignored
results/<run_id>_seed<seed>/last.pt                     keep locally; never committed
results/<run_id>_seed<seed>/best.pt                     keep locally; informational
target_eval_ledger.jsonl                                COMMITTED — see 4.3
```

`results/` is gitignored and checkpoints are never committed. Keep them: `last.pt` is
what section 5 evaluates, and without it you retrain.

Back up the nine `last.pt` files somewhere other than the laptop before section 5. They
are ~900 MB total and ~10.7 GPU-hours of work, and the protocol gives you one look at
Raabin with them.

### 4.3 The ledger

Training writes no ledger line — no training run touches a ledgered domain. So after the
three training notebooks the ledger is still 0 lines and there is nothing to bring back.

It starts mattering in section 5. The rule there is: **download it and append to your
local copy after every evaluation, then commit and push immediately.** It is append-only
and committed to git precisely so repeat looks are visible in history; a ledger that is
recreated each session constrains nothing.

```bash
# after an evaluation session, from the repo root on your machine
cat ~/Downloads/target_eval_ledger.jsonl >> target_eval_ledger.jsonl
sort -u target_eval_ledger.jsonl | wc -l     # sanity: no duplicate lines
git add target_eval_ledger.jsonl
git commit -m "ledger: <what was evaluated and why>"
git push
```

Append, never overwrite. The Kaggle copy starts from whatever was in the clone, so if you
pushed earlier entries first the downloaded file already contains them — check the line
count before and after rather than assuming.

---

## 5. The one-time Raabin evaluation

A separate booking, after all nine training runs are complete and their checkpoints are
off Kaggle. This is the whole target-domain budget for the project. Read §5.1 before
booking the session.

### 5.1 What has to be true before you start

The checkpoints carry their config inside them, including **absolute paths resolved on
the training machine**. `src/eval.py` reads that embedded config and never a YAML passed
alongside it (CLAUDE.md §5), so the evaluation session has to look like the training
session or nothing loads.

| What the checkpoints record | What the session must provide |
|---|---|
| `/kaggle/working/coe/configs/class_mapping.yaml` | clone to **`/kaggle/working/coe`**, not any other directory |
| `/kaggle/working/coe/configs/label_space.yaml` | same |
| `/kaggle/input/datasets/anikakrishnaa/pbc-images/PBC_dataset_normal_DIB` | attach **pbc-images** |
| `/kaggle/input/datasets/anikakrishnaa/wbcatt` | attach **wbcatt** |
| `/kaggle/input/datasets/anikakrishnaa/raabin-double-labeled` | attach **raabin-double-labeled** |
| `/kaggle/working/results` | written by the session itself |

Plus **wbc-final-checkpoints**, the private dataset holding the nine `last.pt`. Kaggle
nests it, so the real path has the folder name twice:

```
/kaggle/input/datasets/anikakrishnaa/wbc-final-checkpoints/wbc-final-checkpoints/<model>_seed<N>/last.pt
```

**Do not attach the old training notebook's output.** It contains a `results/` directory
and a stale clone, and nothing in this session should be able to read either.

Two failure modes this section is built around:

- **A wrong clone directory fails loudly.** `ConfigError: data.class_mapping: not found`.
  Annoying, not dangerous.
- **A wrong dataset path used to fail silently.** `make_eval_loader` returned `None` for a
  Raabin root that was not a directory, the run put all three Raabin domains in
  `skipped_domains`, evaluated and ledgered `pbc_test` alone, and printed success — and
  `pbc_test` is held out, so that look was spent for nothing. As of the commit that added
  `_assert_domains_buildable`, any configured domain that cannot be built is a hard error
  raised **before** the ledger is read or written and before any image is decoded.
  `skipped_domains` is now always `{}`. The pre-flight cell below checks the same paths
  anyway, because a clear message beats a stack trace.

### 5.2 What one invocation does

`eval.domains` is identical in all nine configs:

```yaml
domains: [pbc_test, raabin_train, raabin_testA, raabin_testB]
```

So **one** `src.eval` invocation on **one** checkpoint evaluates the held-out source test
split and all three Raabin domains, and writes **four** ledger lines. Nine checkpoints ×
four domains = **36 ledger lines**.

`pbc_test` is ledgered too. It is held out for the source-domain report, so a repeat look
at it is the same error as a repeat look at Raabin.

Each invocation also writes one record, to the `results_dir` in its **embedded** config —
there is no flag for this:

```
/kaggle/working/results/<run_id>_seed<N>_eval.json          a real evaluation
/kaggle/working/results/<run_id>_seed<N>_eval_dryrun.json   --dry-run
```

`run_id` is `baseline`, `bottleneck` or `control`, so the nine records are
`baseline_seed0_eval.json` … `control_seed2_eval.json`, all in one flat directory. A dry
run cannot clobber a real record: the names differ.

**These records are the only copy of the numbers.** The ledger records only that a look
happened, not what it said.

One consequence worth knowing before you try: these checkpoints can no longer be
evaluated on your laptop at all, not even with `--dry-run`. `results_dir` is embedded as
`/kaggle/working/results`, and creating that locally fails with
`OSError: [Errno 30] Read-only file system: '/kaggle'`. Evaluation of a Kaggle-trained
checkpoint happens on Kaggle.

### 5.3 The notebook

Accelerator on. Internet on (for the clone). Save & Run All is not used here — run the
cells in order and stop if one fails.

#### Cell 1 — clone, check, CUDA

```python
# Before torch is imported anywhere: cuBLAS reads this at initialisation, and
# use_deterministic_algorithms(True) raises without it on a CUDA device.
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import importlib.util
import subprocess, sys
from pathlib import Path

REPO = Path("/kaggle/working/coe")
!rm -rf /kaggle/working/coe
!git clone -q https://github.com/niks2604/coe.git /kaggle/working/coe

print("HEAD:  ", subprocess.run(["git","-C",str(REPO),"rev-parse","--short","HEAD"],
                                capture_output=True, text=True).stdout.strip())
dirty = subprocess.run(["git","-C",str(REPO),"status","--porcelain"],
                       capture_output=True, text=True).stdout
print("dirty: ", repr(dirty) if dirty else "clean")

ledger = REPO / "target_eval_ledger.jsonl"
n = len(ledger.read_text().splitlines()) if ledger.is_file() else 0
print(f"ledger lines in the clone: {n}   (expect 0 before the first session, or "
      f"exactly what you have pushed)")

# NO `pip install -r requirements.txt`. It pins torch==2.7.0 and would replace Kaggle's
# 2.11.0+cu130 - the build these checkpoints were trained under. Install by name only
# what is actually missing.
MODULES = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
           "yaml": "PyYAML", "PIL": "pillow", "pandas": "pandas"}
missing = [pkg for mod, pkg in MODULES.items()
           if not importlib.util.find_spec(mod)]
if missing:
    print("installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("all imports present; nothing installed")

import torch
assert torch.cuda.is_available(), "no GPU: set Accelerator before running"
print(torch.__version__, torch.cuda.get_device_name(0))
```

#### Cell 2 — pre-flight: every path the checkpoints need

Reads one checkpoint's embedded config and asserts that everything it names exists. The
three Raabin split folders are checked **by existence only** — no file is opened, nothing
is counted, nothing derived from any pixel (CLAUDE.md §3).

```python
import os, sys, torch
from pathlib import Path

REPO = Path("/kaggle/working/coe")
CKPT_ROOT = Path("/kaggle/input/datasets/anikakrishnaa/wbc-final-checkpoints"
                 "/wbc-final-checkpoints")          # the folder name appears twice
os.chdir(REPO); sys.path.insert(0, str(REPO))

MODELS = ("baseline", "bottleneck", "control")
RUNS = [(m, s) for s in (0, 1, 2) for m in MODELS]   # seed-major
CKPTS = {(m, s): CKPT_ROOT / f"{m}_seed{s}" / "last.pt" for m, s in RUNS}

absent = [str(p) for p in CKPTS.values() if not p.is_file()]
assert not absent, "checkpoints not found:\n" + "\n".join(absent)
print(f"{len(CKPTS)} checkpoints present under {CKPT_ROOT}")

payload = torch.load(CKPTS[("baseline", 0)], map_location="cpu", weights_only=False)
cfg = payload["config"]
problems = []

def check(label, path, is_dir):
    ok = Path(path).is_dir() if is_dir else Path(path).is_file()
    print(f"  {label:<28} {'OK     ' if ok else 'MISSING'} {path}")
    if not ok:
        problems.append(f"{label}={path}")

print("\nembedded repo paths:")
check("data.class_mapping_path", cfg["data"]["class_mapping_path"], False)
check("data.label_space_path",   cfg["data"]["label_space_path"],   False)

print("\nembedded dataset paths:")
for key in ("pbc_images", "wbcatt_dir", "raabin_root"):
    check(f"paths.{key}", cfg["paths"][key], True)
print(f"  {'paths.results_dir':<28} (written by this session) "
      f"{cfg['paths']['results_dir']}")
print(f"  {'paths.lisc_root':<28} not checked: LISC is out of scope (CLAUDE.md §2)")

print("\nRaabin split folders (existence only; nothing opened or counted):")
from src.data.domains import target_folder
raabin = Path(cfg["paths"]["raabin_root"])
for did in ("raabin_train", "raabin_testA", "raabin_testB"):
    check(did, raabin / target_folder(did), True)

print("\nconfigured eval.domains:", cfg["eval"]["domains"])
assert not problems, (
    "STOP. Fix these before evaluating - the embedded config is not editable here:\n  "
    + "\n  ".join(problems))
print("\nall required paths present")
del payload
```

#### Cell 3 — checksums against the table

`src.ledger.checkpoint_hash` is a **plain SHA-256 of the file's bytes**, read in 1 MiB
chunks with no salt, prefix or length framing. It equals `shasum -a 256 <file>`, and the
cell proves that on one file rather than asserting it.

```python
import hashlib
from src.ledger import checkpoint_hash

EXPECTED = {
    ("baseline", 0): "79de1c4cb406", ("bottleneck", 0): "6c539cd6f1b9",
    ("control", 0):  "b70993fb5398", ("baseline", 1):   "40e06e74d209",
    ("bottleneck", 1): "b1abb3d45446", ("control", 1):  "e945cccf74ae",
    ("baseline", 2): "d8341cba0adc", ("bottleneck", 2): "3275a3b3a5c8",
    ("control", 2):  "17a7f81eb176",
}

probe = CKPTS[("baseline", 0)]
plain = hashlib.sha256(probe.read_bytes()).hexdigest()
assert checkpoint_hash(probe) == plain, "checkpoint_hash is not a plain sha256"
print(f"checkpoint_hash == plain sha256 of the bytes: {plain[:12]}\n")

bad = []
for (m, s) in RUNS:
    h = checkpoint_hash(CKPTS[(m, s)])
    ok = h.startswith(EXPECTED[(m, s)])
    print(f"  {m+'_seed'+str(s):<20} {h[:12]}  {'OK' if ok else 'MISMATCH, expected '
          + EXPECTED[(m, s)]}")
    if not ok:
        bad.append(f"{m}_seed{s}")
assert not bad, f"STOP. These are not the checkpoints that were uploaded: {bad}"
print("\nall nine match the table")
```

#### Cell 4 — dry run of ONE checkpoint

`pbc_val` only. No ledger line, no target domain, nothing spent. It proves the whole
path — a read-only mount, the embedded config, the model, the record write — and then
prints the record's keys, so every key the verification cell uses is checked against
reality before the real run.

```python
!cd /kaggle/working/coe && python -u -m src.eval \
    --checkpoint "/kaggle/input/datasets/anikakrishnaa/wbc-final-checkpoints/wbc-final-checkpoints/baseline_seed0/last.pt" \
    --dry-run
```

```python
import json
rec = json.loads(Path("/kaggle/working/results/baseline_seed0_eval_dryrun.json").read_text())
print("record keys:", sorted(rec))
for k in ("run_id", "seed", "dry_run", "domains", "skipped_domains", "selection_role",
          "selection_rule", "override_reason", "checkpoint_sha256",
          "buffer_sha256_before", "buffer_sha256_after", "attribute_modes"):
    assert k in rec, f"the verification cell expects {k!r} and it is not there"
    print(f"  {k:<22} {rec[k] if k != 'checkpoint_sha256' else rec[k][:12]}")
print("\nresults keys per domain:", {d: sorted(v) for d, v in rec["results"].items()})
print("score keys:", sorted(next(iter(next(iter(rec['results'].values())).values()))))
```

If anything here fails, **stop**. Nothing below has been spent yet.

#### Cells 5–13 — the nine evaluations

One cell each, seed-major, so a session that dies after three cells leaves a complete
three-model comparison at seed 0 rather than three baselines. No flags. `last.pt`, never
`best.pt` — `src/eval.py` refuses a non-headline checkpoint outright. `python -u` on
every call, so a cell that dies mid-domain has already printed how far it got; without it
the buffered output is lost with the process and you cannot tell which domains were
reached.

```python
D = "/kaggle/input/datasets/anikakrishnaa/wbc-final-checkpoints/wbc-final-checkpoints"
```

```python
!cd /kaggle/working/coe && python -u -m src.eval --checkpoint "$D/baseline_seed0/last.pt"
```

Then one cell each, in this order:

```
baseline_seed0   bottleneck_seed0   control_seed0
baseline_seed1   bottleneck_seed1   control_seed1
baseline_seed2   bottleneck_seed2   control_seed2
```

Each prints the record path, macro-F1 and ECE per domain per attribute mode, the class
list each macro-F1 was computed over, and whether the model's buffers changed during
evaluation (they must not). The abstention threshold comes from `pbc_val` and is applied
unchanged to every domain.

If a cell fails *before* its ledger lines are written, re-run it. If it fails *after*,
re-running needs `--override-reason` (§5.6).

#### Cell 14 — verification

```python
import json, collections
from pathlib import Path

RESULTS = Path("/kaggle/working/results")
records = sorted(RESULTS.glob("*_seed*_eval.json"))     # excludes *_eval_dryrun.json
assert len(records) == 9, f"expected 9 records, found {len(records)}: {records}"

bad, seen = [], set()
for f in records:
    r = json.loads(f.read_text())
    tag = f"{r['run_id']}_seed{r['seed']}"
    if (r["run_id"], r["seed"]) in seen: bad.append(f"{tag}: duplicate")
    seen.add((r["run_id"], r["seed"]))
    if r["dry_run"]:                     bad.append(f"{tag}: dry run")
    if r["skipped_domains"]:             bad.append(f"{tag}: SKIPPED {r['skipped_domains']}")
    if sorted(r["domains"]) != ["pbc_test","raabin_testA","raabin_testB","raabin_train"]:
        bad.append(f"{tag}: domains={r['domains']}")
    if r["selection_role"] != "headline": bad.append(f"{tag}: role={r['selection_role']}")
    if r["selection_rule"] != "final_epoch": bad.append(f"{tag}: rule={r['selection_rule']}")
    if r["override_reason"]:             bad.append(f"{tag}: override {r['override_reason']!r}")
    if r["buffer_sha256_before"] != r["buffer_sha256_after"]:
        bad.append(f"{tag}: BUFFERS CHANGED during evaluation")
    print(f"{tag:<20} {len(r['domains'])} domains  modes={r['attribute_modes']}  "
          f"pbc_test soft macro-F1 {r['results']['pbc_test']['soft']['macro_f1']:.4f}")

rows = [json.loads(l) for l in (Path("/kaggle/working/coe/target_eval_ledger.jsonl")
                                ).read_text().splitlines() if l.strip()]
per_domain = collections.Counter(r["domain"] for r in rows)
print("\nledger:", dict(per_domain), " total", len(rows))
print("roles:", collections.Counter(r.get("selection_role") for r in rows))
print("overrides:", [r.get("override_reason") for r in rows if r.get("override_reason")])

if len(rows) != 36:                      bad.append(f"ledger has {len(rows)} lines, expected 36")
if set(per_domain.values()) != {9}:      bad.append(f"not 9 per domain: {dict(per_domain)}")
assert not bad, "\n".join(bad)
print("\nall nine records and 36 ledger lines verified")
```

#### Cell 15 — collect

```python
!mkdir -p /kaggle/working/outputs
!cp /kaggle/working/coe/target_eval_ledger.jsonl /kaggle/working/outputs/
!cp /kaggle/working/results/*_eval.json /kaggle/working/outputs/
!cp /kaggle/working/results/*_eval_dryrun.json /kaggle/working/outputs/ 2>/dev/null || true
!ls -la /kaggle/working/outputs
print("\nDownload /kaggle/working/outputs, then STOP THE GPU SESSION.")
```

**Stop the session.** Kaggle bills the accelerator until you do, and a session left
running is a session someone can re-run a cell in.

### 5.4 The ledger does not persist, so you have to carry it

The clone's `target_eval_ledger.jsonl` starts at whatever is in git and dies with the
session. Nothing on Kaggle remembers it. So, **after every session, before booking
another**, bring it home — but verify before appending rather than appending blind.

The downloaded ledger is the local one plus whatever this session added, so its first *n*
lines must be byte-identical to the local file. If they are not, the clone started from a
different ledger than the one you hold, and concatenating the two would produce a file
that is a record of neither.

```bash
# on your machine, in the repo
DL=~/Downloads/outputs/target_eval_ledger.jsonl
n=$(wc -l < target_eval_ledger.jsonl)

# 1. the downloaded ledger must BEGIN with the local one, byte for byte.
#    Guarded on n > 0: BSD head rejects -n 0, and the local ledger is empty
#    before the first session.
if [ "$n" -gt 0 ]; then
  head -n "$n" "$DL" | cmp - target_eval_ledger.jsonl || {
    echo "STOP: the session's ledger does not extend the local one."
    echo "Do not append. Work out which clone it came from first."; exit 1; }
fi

# 2. append only what is new
tail -n +$((n + 1)) "$DL" >> target_eval_ledger.jsonl

# 3. check what you now hold
wc -l < target_eval_ledger.jsonl                 # expect 36 after a complete pass
python -c "
import json, collections
rows=[json.loads(l) for l in open('target_eval_ledger.jsonl') if l.strip()]
per=collections.Counter(r['domain'] for r in rows)
print(dict(per), 'total', len(rows))
print('roles:', collections.Counter(r.get('selection_role') for r in rows))
print('overrides:', [r.get('override_reason') for r in rows if r.get('override_reason')])
assert len(rows) == 36, f'expected 36 lines, got {len(rows)}'
assert set(per.values()) == {9}, f'expected 9 per domain, got {dict(per)}'
assert set(per) == {'pbc_test','raabin_train','raabin_testA','raabin_testB'}, dict(per)
print('ledger verified')
"
git add target_eval_ledger.jsonl && git commit -m "ledger: Raabin evaluation, 9 checkpoints x 4 domains" && git push
```

`n` is counted **before** the append, and `tail -n +$((n+1))` takes only the lines past
it, so running the block twice on the same download adds nothing the second time. Never
overwrite the local file with the downloaded one: the local file is authoritative, and an
overwrite silently drops any look recorded in a session whose output you did not keep.

**Push before the next session**, because the next session's clone starts from the pushed
ledger, and a clone that has forgotten an earlier look will cheerfully repeat it.

### 5.5 Where the records go

They are written to `/kaggle/working/results/`, and `results/` is gitignored, so **the
nine records are not tracked by git unless you move them**. Losing them loses the numbers;
the ledger does not hold them.

`eval_records/` at the repo root is their tracked home. It is committed, and
`eval_records/README.md` records why.

```bash
mkdir -p eval_records
cp ~/Downloads/outputs/*_eval.json eval_records/
git add eval_records/ && git commit -m "results: the nine Raabin evaluation records"
```

The justification is the ledger's own, from CLAUDE.md §5: a record that resets when
`results/` is cleared does not constrain anything. Nine JSON files, each an embedded
resolved config plus metrics, well under 2 MB in total, and `scripts/aggregate_eval.py`
reads exactly this directory. CLAUDE.md §5 now names **three** things committed despite
`results/` being gitignored, and a test asserts that paragraph names all three.

Real evaluations only. `*_eval_dryrun.json` does not belong here: a dry run reads
`pbc_val`, selects nothing and reports nothing, and keeping those alongside the records
would stop the directory answering "which looks were spent".

### 5.6 Ledger rules for this section

- **Three seeds are three legitimate entries per domain.** The ledger keys on the
  checkpoint hash and three seeds are three different checkpoints. No override needed.
- **A second evaluation of the same (checkpoint, domain) is refused.** That is the point.
- **Retraining produces a new hash**, so a re-run of a training job is a new checkpoint and
  a legitimate new entry — which is also why you do not retrain casually after looking at
  Raabin.
- **A legitimate override reason is a mechanical failure**, not a number you did not like.
  "Session died after `raabin_testA`, re-running for `testB`" is a reason. "Checking
  whether the gap looks better" is the thing the ledger exists to make visible.
- **An override also replaces the record file**, not just the ledger line. If both looks
  matter, move the old `*_eval.json` aside first.
- `--override-reason` with nothing to override is itself an error, so it cannot be left on
  out of habit.

### 5.7 Reminders

- Stop the GPU session when the last cell finishes.
- Raabin's basophils come from a single leukemia patient. Any basophil recall on a Raabin
  domain carries that caveat and is one patient's cells, not a sample of basophils
  (CLAUDE.md §6).
- `raabin_testB` holds two of the five classes, so it reports a 2-class macro-F1 and the
  source side of its gap is `pbc_test` restricted to those two classes. Predictions are
  still an argmax over all five. Per-class recall is `null`, not 0, for a class absent from
  an eval set.
- There is no pooled Raabin number. Three domains, three rows.
