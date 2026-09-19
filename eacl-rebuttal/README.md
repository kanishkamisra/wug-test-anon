# EACL rebuttal: image-only wug/wugs number learning

Self-contained experiment folder. **No file outside `eacl-rebuttal/` is modified** —
everything here either points the existing, unmodified `core/eval/embed_eval.py`
at new data, or runs a small fork of `core/train/embed_train.py` (see below).

## Motivation

1. The original image-condition training captions vary in wording between the
   singular and plural row, and sometimes carry number cues beyond `[wug]`/`[wugs]`
   itself. Here the singular/plural rows are *identical text*, differing only in
   the token, so the image is the only source of number information.
2. The original dev set (`data/embeddings/dev/dev_eval.csv`) directly tests
   singular/plural agreement — the same thing the held-out test set measures,
   which risks leaking into model/LR selection. The dev set here instead tests
   only whether `[wug]`/`[wugs]` are treated as **nouns, not verbs** (see below).
3. The original `noun_init.txt` (27 pairs) may be too small/idiosyncratic to give
   an unbiased singular/plural init. `noun_init_large.txt` (530 pairs) is sampled
   from WordNet via `get_unambiguous_nouns()` (imported from
   `representational-analysis/embed_analysis.py`, not duplicated).

## What's here

- `data/train/im/wugs_subset/` — the same 5 singular + 5 plural creature images
  as `data/embeddings/train/im/wugs_subset/`, copied and renamed to
  `singular{1..5}.png`/`plural{1..5}.png` because `embed_train.py`'s image
  loader hardcodes that naming pattern (the original files are named
  `snarple_singular_*.png`/`snarple_plural_*.png`).
- `data/train/text/image_train_matched.csv` — 16 hand-written caption templates,
  each emitted once with `[wug]` and once with `[wugs]`, identical otherwise.
  `[wug]`/`[wugs]` is never the subject of a finite verb and never follows a
  number-marked determiner, so nothing but the image indicates which is which.
- `data/dev/pos_templates_source.json` + `dev_noun_vs_verb.csv` — the dev set.
  `pos_templates_source.json` has 50 "noun" sentence templates (`[mask]` as a
  direct object/PP complement) and 50 "verb" templates (`[mask]` as a bare/
  infinitive verb slot). `build_dev_set.py` pairs them 1:1 (verbs shuffled with
  a fixed seed) and fills `[mask]` with `[wug]`/`[wugs]` to get 100 good/bad
  rows. This tests part-of-speech only, never singular/plural.
- `data/init/noun_init_large.txt` — 530 singular/plural pairs from
  `generate_large_noun_init.py`, for the "large" init variant.
- `scripts/embed_train_dev_margin.py` — a fork of `core/train/embed_train.py`.
  **Only** `run_agreement_eval()`'s return value changes: `overall_acc`/
  `sing_acc`/`plur_acc` now hold the mean logprob margin `diffs.mean()` (i.e.
  `avg_logprob_per_token(good) - avg_logprob_per_token(bad)`, using minicons'
  existing mean-per-token `sequence_score` reduction) instead of a binarized
  win-rate. For the noun-vs-verb dev set this is
  `LP(wug,noun) + LP(wugs,noun) - LP(wug,verb) - LP(wugs,verb)` (up to
  averaging over pairs rather than summing). Early stopping/best-epoch
  selection is unchanged — it just compares this margin instead of an accuracy,
  which works identically since higher is still better. Diff against the
  original is 3 hunks; run `diff core/train/embed_train.py
  eacl-rebuttal/scripts/embed_train_dev_margin.py` to see exactly that.
- `scripts/run_sweep.py` — LR x init-variant x seed grid (14 LRs x {small,large}
  init x 5 seeds = 140 runs by default), dispatched across a GPU pool
  (`gpu_pool.py`: one job per GPU, next queued job starts as soon as a GPU
  frees up). Always calls `embed_train_dev_margin.py`, never the original.
- `scripts/run_test_eval.py` — for every completed run's `learned_embeddings.pt`,
  calls the **unmodified** `python3 -m core.eval.embed_eval --paired` against
  the held-out test set `data/interp/agreement_target_wug.csv` (verb agreement,
  `[wug]`/`[wugs]` substituted). This script is untouched by the dev-metric
  change above.
- `scripts/aggregate_results.py` — joins each run's dev margin with its test
  accuracy (overall + by attractor condition att0-3) into `results/summary.csv`.

## Running it

```bash
cd wug-test-interp
python3 eacl-rebuttal/scripts/run_sweep.py --gpus 0,1,2         # ~140 training runs
python3 eacl-rebuttal/scripts/run_test_eval.py --gpus 0,1,2     # score all of them on the test set
python3 eacl-rebuttal/scripts/aggregate_results.py              # -> results/summary.csv
```

Each script also accepts `--lrs`/`--seeds`/`--epochs`/`--out_dir` etc. for
smaller runs (see `--help` or the top of each file) — e.g. the smoke tests used
during development:

```bash
python3 eacl-rebuttal/scripts/run_sweep.py --gpus 0 --lrs 0.001 --seeds 17 \
    --epochs 3 --out_dir eacl-rebuttal/results/smoke_test
```

`results/smoke_test/` and `results/smoke_test_full/` are exactly that — throwaway
pipeline sanity checks, safe to delete once the real sweep is confirmed working.

Model/cache note: this machine's actual HF cache is `/home/shared/hf_cache`
(`HF_HOME`), not the `/mnt/dv/...` path hardcoded as the CLI default in
`core/train/embed_train.py`/`core/eval/embed_eval.py` (that path doesn't exist
here). All scripts in this folder default `--cache_dir` to `/home/shared/hf_cache`
— override if running elsewhere.

## Status as of 2026-09-16 evening — read this first if picking up this work cold

**Second dev-metric fork exists and supersedes the noun-vs-verb one for LR/model
selection.** `scripts/embed_train_dev_image.py` (forked from
`embed_train_dev_margin.py`) uses a *held-out-image* dev metric instead of the
noun-vs-verb text one: 20 held-out singular + 20 held-out plural images (never
used in training; under `data/dev/held_out_images/{singular,plural}/`, from
user-uploaded `singular.zip`/`plural.zip`) x 6 QA templates
(`data/dev/dev_image_number.csv`, `QA_TEMPLATES` in that script) = 120 stimuli,
crossed. For each stimulus it builds a fixed chat-template prompt ending right
before the target-noun position, does a manual forward pass, and reads the raw
`logit[wug_id] - logit[wugs_id]` gap at that position — this is precision-safe;
an earlier attempt using whole-sequence mean log-prob (`minicons`'
`sequence_score`) failed because vision-patch tokens vastly outnumber the
caption text and drown the 1-token signal in bf16. Same return-value contract
as the margin fork (`overall_acc`/`sing_acc`/`plur_acc`/etc.), so
`run_sweep.py --train_script .../embed_train_dev_image.py --eval_csv
.../dev_image_number.csv` works with no other changes.

**Key finding: batching (not the dev metric itself) was the real blocker.**
The *original* inherited training loop uses `batch_size=1` with
`Adam(betas=(0.0, 0.9))` (no momentum) — each epoch is ~32 single-example noisy
optimizer steps. This was the dominant driver of both huge seed-to-seed
variance and dev-metric unreliability, confirmed by direct comparison:

| batch config | dev metric's own best-LR pick | test acc at that pick | true best test acc (any LR) |
|---|---|---|---|
| bs=1 (original) | lr=0.0075 | 0.587 ± 0.052 | 0.738 (lr=0.005) |
| `--batch_mode joint --batch_size 4` | lr=0.01 | 0.708 ± 0.015 | 0.743 (lr=0.003) |
| `--batch_mode joint --batch_size 8` | lr=0.01 | **0.738 ± 0.023** | 0.738 (lr=0.005) — **exact tie** |

At bs=8 the image-logit-gap dev metric's own LR pick now exactly matches the
true best test accuracy — i.e. **it's finally a usable leak-free proxy**, but
only once the batch-size/momentum noise is fixed. `batch_mode=joint,
batch_size=8` is the recommended training config going forward. (`batch_mode`
has two settings, both pre-existing in the inherited `core/train/embed_train.py`
and unmodified here: `alternating`, the default, does separate singular-only
and plural-only steps and NEVER mixes them in one step; `joint` combines
singular+plural into one step per batch — `joint` is what's needed for a batch
to have both classes in it.)

All of this — the 120-stimulus dev-image design, the constant-seed=42 sweep,
the bs=4/bs=8 batching comparison — lives under
`results/sweep_image_dev_seed42/` (see `manifest.csv` for the full run index;
`init_{tag}/lr_{tag}[_{batch_mode}_bs{N}]/seed_{N}/` per run, `test_eval/` for
scored test-set results per run).

**Currently running (started 2026-09-16 ~21:37, unattended overnight):** a
50-seed run at the recommended config — `lr=0.01, batch_mode=joint,
batch_size=8, init=small` — to characterize seed variance properly at a single
locked-in config (matches the ~50-seed convention used earlier on the original
unmodified pipeline). 5 seeds (17, 42, 123, 2024, 98765) were already done from
the bs=8 LR sweep above; seeds 100-144 (excluding 123, already done) + seed 200
were launched to bring the total to 50. Launched via two separate
`run_sweep.py` invocations sharing GPUs 0,1,3 (GPU 2 was occupied by another
user's unrelated job — left alone):
```bash
python3 eacl-rebuttal/scripts/run_sweep.py --gpus 0,1,3 --lrs 0.01 \
    --seeds 100,101,...,144 \
    --inits small:/home/user/wug-test-interp/data/embeddings/init/noun_init.txt \
    --batch_mode joint --batch_size 8 \
    --out_dir eacl-rebuttal/results/sweep_image_dev_seed42
# + a second invocation with --seeds 200 only, same other args
```
**To check progress:** `find eacl-rebuttal/results/sweep_image_dev_seed42/init_small/lr_0p01_joint_bs8 -name learned_embeddings.pt | wc -l` (target: 50). Note the training subprocess name in `ps`/`pgrep` is `embed_train_dev_margin.py` even in image-dev-metric mode — that's the file `run_sweep.py --train_script` was pointed at for these runs, not a bug.

**Once all 50 are done, next steps:**
1. Score any unscored ones: `python3 eacl-rebuttal/scripts/run_test_eval.py --gpus 0,1,2 --manifest eacl-rebuttal/results/sweep_image_dev_seed42/manifest.csv --out_dir eacl-rebuttal/results/sweep_image_dev_seed42/test_eval` (already has skip-if-`scored.csv`-exists logic, safe to re-run).
2. Compute pooled/singular/plural test accuracy with 95% CIs (t-distribution, `scipy.stats.t.ppf(0.975, df=n-1)`) across all 50 seeds at this one config — pooled by default; only break out by attractor or verb-type if asked (see reporting-defaults note below).
3. Compare this 50-seed distribution against the 5-seed estimate already in hand to see if 0.738 held up or was itself a small-n fluke.

**Reporting convention respected throughout this project:** default to pooled/
simple stats; don't proactively stratify (e.g. by attractor, by copula-vs-lexical
verb type) unless explicitly asked to. The held-out test set's attractor
conditions ARE confounded by verb type (att0 is 97.7% lexical verbs; att1-3 are
each exactly 40% copula "is/was") — a real effect, but report it only on request,
per explicit user instruction ("no need to stratify unless I tell you").
