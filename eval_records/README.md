# eval_records/

The `*_eval.json` record written by each `python -m src.eval` run, one per
(checkpoint, evaluation). Committed.

## Why this directory exists

`src/eval.py` writes its record to the `results_dir` in the checkpoint's **embedded**
config — there is no flag for it — and `results/` is gitignored. So the records have no
tracked home unless they are moved to one, and they are the **only** copy of the
numbers: `target_eval_ledger.jsonl` records that a look happened, not what it said.

For the nine final runs the records are written on Kaggle, to
`/kaggle/working/results/<run_id>_seed<N>_eval.json`, and die with the session. The
justification for committing them here is the ledger's own (CLAUDE.md §5): a record that
resets when `results/` is cleared does not constrain anything.

## What goes here

Real evaluations only — `<run_id>_seed<N>_eval.json`. Not `*_eval_dryrun.json`: a dry run
reads `pbc_val`, selects nothing and reports nothing, and keeping those here would mean
the directory no longer answers "which looks were spent".

Expected after the one-time Raabin evaluation: nine files.

```
eval_records/
├── baseline_seed0_eval.json     ├── bottleneck_seed0_eval.json   ├── control_seed0_eval.json
├── baseline_seed1_eval.json     ├── bottleneck_seed1_eval.json   ├── control_seed1_eval.json
└── baseline_seed2_eval.json     └── bottleneck_seed2_eval.json   └── control_seed2_eval.json
```

Each holds the full resolved config, the checkpoint sha256, the git commit and dirty
flag, the selection role and rule, every metric per domain and per class, the bootstrap
intervals, and the wall clock and hardware.

## How they get here

See `docs/final_runs.md` §5.5. In short, after downloading the session's outputs:

```bash
cp ~/Downloads/outputs/*_eval.json eval_records/
git add eval_records/ && git commit -m "results: the nine Raabin evaluation records"
```

`scripts/aggregate_eval.py` reads this directory and nothing else.
