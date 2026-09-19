# Claude Code handoff: 4B image-only number-learning experiments

Last updated: 2026-09-17 (America/Chicago)

## Read this first

The latest controlled initialization-scale experiment is **complete**. All 25
training runs succeeded, dev selection completed, and all five test evaluations
for the dev-selected configuration succeeded. No related GPU jobs or tmux
sessions remain active.

The dev-selected configuration is:

- model: `Qwen/Qwen3-VL-4B-Instruct`
- initialization: small noun list, controlled spherical symmetry
- initial `[wug]`/`[wugs]` separation: **0.25x** the mean real-noun
  singular/plural pair distance
- learning rate: `0.01`
- batching: `joint`, batch size `8` per class
- checkpoint selection: balanced logistic score on held-out images
- early stopping: patience `5`, maximum `50` epochs
- training objective: unchanged full-vocabulary cross-entropy

Five-seed held-out test result at this selected configuration:

| metric | mean | 95% CI |
|---|---:|---:|
| pooled | **0.7103** | [0.6870, 0.7335] |
| singular | **0.7732** | [0.7432, 0.8033] |
| plural | **0.6473** | [0.5759, 0.7186] |

The next planned experiment is to run this locked configuration on the other
45 seeds, yielding 50 seeds total. See **Next action** below.

## User decisions and constraints

- Do **not** change the training loss. The user explicitly wants ordinary
  full-vocabulary cross-entropy retained.
- Symmetric initialization is desirable, but its initial separation must be
  controlled rather than accidentally changed by normalization.
- Use only the **small** noun initialization list; do not repeat large-init
  experiments.
- Use the correct held-out-image dev metric for checkpoint/config selection.
- The test set must not select the initialization scale or checkpoint. It is
  evaluated only after dev selection.
- Once the best five-seed configuration is selected, characterize it using 45
  additional seeds (50 total).

## Latest experiment: controlled spherical separation

Output directory:

```text
eacl-rebuttal/results/spherical_symmetric_4B_scale_search/
```

Runner:

```text
eacl-rebuttal/scripts/run_spherical_scale_4b_sweep.sh
```

Grid:

- exact post-normalization separation scales: `0, 0.25, 0.5, 1.0, 1.5`
- LR: `0.01`
- seeds: `17, 42, 123, 2024, 98765`
- 25 cells total; 25/25 succeeded

Dev results (higher/less negative balanced-logistic score is better):

| separation scale | mean dev score | SD | mean best epoch | n |
|---:|---:|---:|---:|---:|
| 0.00x | -0.055427 | 0.092623 | 12.0 | 5 |
| **0.25x** | **-0.012061** | **0.008267** | 13.6 | 5 |
| 0.50x | -0.017678 | 0.006590 | 12.0 | 5 |
| 1.00x | -0.025189 | 0.013391 | 19.2 | 5 |
| 1.50x | -0.552053 | 1.046029 | 17.4 | 5 |

The decisive result is that a modest `0.25x` head start is best on dev. Exact
co-location (`0x`) works and learns, but is less stable. Large initial
separation (`1.5x`) is clearly unstable.

### Selected configuration: test accuracy by attractor count

These are means over the five seeds selected above. CIs are t intervals over
seeds and are available in `selected_test_summary.csv`.

| attractors | pooled | singular | plural |
|---:|---:|---:|---:|
| 0 | 0.8107 | 0.6663 | 0.9551 |
| 1 | 0.7844 | 0.8446 | 0.7243 |
| 2 | 0.6507 | 0.8103 | 0.4911 |
| 3 | 0.5951 | 0.7717 | 0.4186 |

The plural collapse at high attractor counts is reduced relative to the prior
4B symmetric run, but it is not eliminated.

### Per-seed selected test results

| seed | pooled | singular | plural |
|---:|---:|---:|---:|
| 17 | 0.7123 | 0.7843 | 0.6404 |
| 42 | 0.6934 | 0.7800 | 0.6068 |
| 123 | 0.7252 | 0.7346 | 0.7157 |
| 2024 | 0.7313 | 0.7682 | 0.6943 |
| 98765 | 0.6891 | 0.7989 | 0.5793 |

Authoritative files:

- `dev_summary.csv`: all five scale cells aggregated over seeds
- `selected_configs.csv`: dev-selected `0.25x` configuration
- `selected_test_per_run.csv`: five test-scored seeds
- `selected_test_summary.csv`: pooled and attractor-level means/CIs
- `selected_test_eval/*/scored.csv`: item-level test results
- `manifest.csv`: all 25 training runs
- `orchestrator.log`: successful end-to-end execution log

All paths above are under
`eacl-rebuttal/results/spherical_symmetric_4B_scale_search/`.

## What `0x` means and its sanity check

At `0x`, both special tokens begin at the exact same normalized noun-centroid
embedding:

```text
e_[wug] = e_[wugs] = normalized noun centroid
```

They are not constrained to remain identical. Singular and plural image
examples give their output rows different gradients, so CE separates them
immediately.

Smoke-test output:

```text
eacl-rebuttal/results/spherical_4B_zero_smoke/
```

- initial separation: `0.0000`
- separation after epoch 1: `1.242188`
- separation after epoch 2: `1.601562`

This confirms that `0x` is a valid learnable condition rather than a frozen
symmetry.

## Controlled spherical initialization formula

Implemented in `scripts/embed_train_dev_image.py` as
`init_mode=spherical_symmetric`.

Let:

- `c` be the unit-normalized noun centroid,
- `u` be a sampled unit vector orthogonal to `c`,
- `r` be the model's target embedding norm,
- `D` be the mean singular/plural distance among the real initialization nouns,
- `s = scale * D` be the requested final chord distance.

Then:

```text
sin(theta) = s / (2r)
e_[wug]  = r * (cos(theta) c + sin(theta) u)
e_[wugs] = r * (cos(theta) c - sin(theta) u)
```

Both rows have norm `r`, their midpoint direction is `c`, neither token gets an
arbitrary directional advantage, and their final separation is exactly `s`
(apart from bf16 rounding).

This was needed because the earlier pre-normalization `+epsilon/-epsilon`
implementation did **not** match the independent baseline's final distance:

- old symmetric final initialization distance: approximately `1.9297`
- independent initialization distance: approximately `1.43-1.47`

Thus the earlier apparent symmetry benefit was confounded with a much larger
initial separation.

## Dev metric and early stopping

The correct dev set is:

```text
eacl-rebuttal/data/dev/dev_image_number.csv
```

It contains 20 held-out singular and 20 held-out plural images crossed with six
QA templates (120 stimuli). The training script reads the next-token logit gap:

```text
g_i = logit([wug]) - logit([wugs])
```

The signed margin is `z_i = g_i` for a singular image and `z_i = -g_i` for a
plural image.

Checkpoint selection uses a class-balanced logistic score:

```text
score = -0.5 * (mean_singular softplus(-z_i)
                + mean_plural softplus(-z_i))
```

Higher is better; the optimum is `0`. This prevents a very confident result on
one number from hiding poor behavior on the other number. It changes only
checkpoint/config selection, **not** the training loss.

Early stopping is active with patience 5. Each run restores and saves the
embedding rows from the best dev epoch, so it does not blindly use the final
epoch. Every epoch is also saved because the experiments use `--write_all`.

## Prior 4B results and why this experiment was run

### Independent small initialization, 50 seeds

Configuration: LR `0.01`, joint batching, batch size 8, raw held-out-image
checkpoint selection.

| metric | accuracy |
|---|---:|
| pooled | 0.6414 |
| singular | 0.7340 |
| plural | 0.5487 |

By attractor:

| attractors | pooled | singular | plural |
|---:|---:|---:|---:|
| 0 | 0.7636 | 0.6164 | 0.9107 |
| 1 | 0.6971 | 0.8065 | 0.5877 |
| 2 | 0.5755 | 0.7744 | 0.3766 |
| 3 | 0.5293 | 0.7387 | 0.3199 |

Training CE at selected checkpoints was strongly imbalanced: approximately
`2.55` singular versus `0.015` plural. The base 4B model itself is capable of
the grammar: known nouns score about `0.832` pooled, and directly initializing
separate singular/plural noun centroids without training scores about `0.843`.
The issue therefore appears to be learning/selection dynamics, not lack of
grammatical knowledge.

### Earlier uncontrolled symmetric initialization, five seeds

Eight-LR screen (`0.0003, 0.001, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02`)
selected LR `0.01` using balanced-logistic dev selection.

| metric | mean | 95% CI |
|---|---:|---:|
| pooled | 0.6720 | [0.6639, 0.6802] |
| singular | 0.7793 | [0.7426, 0.8160] |
| plural | 0.5648 | [0.5283, 0.6013] |

By attractor:

| attractors | pooled | singular | plural |
|---:|---:|---:|---:|
| 0 | 0.8117 | 0.7026 | 0.9209 |
| 1 | 0.7256 | 0.8417 | 0.6094 |
| 2 | 0.5991 | 0.8031 | 0.3951 |
| 3 | 0.5517 | 0.7697 | 0.3337 |

This was only about `+0.0088` pooled over the exact same five independent seeds,
with only 2/5 seeds improving. More importantly, its initialization distance
was accidentally larger, motivating the controlled scale screen.

### 2B comparison

The 2B model did not show the same plural collapse. At LR `0.01`, batch size 8,
50 seeds:

| metric | accuracy |
|---|---:|
| pooled | 0.742 |
| singular | 0.705 |
| plural | 0.779 |

| attractors | pooled | singular | plural |
|---:|---:|---:|---:|
| 0 | 0.755 | 0.628 | 0.883 |
| 1 | 0.783 | 0.769 | 0.798 |
| 2 | 0.736 | 0.738 | 0.733 |
| 3 | 0.695 | 0.687 | 0.704 |

Its selected-checkpoint CE was also much more balanced (approximately `0.236`
singular, `0.300` plural).

## Code changes made during this work

### `eacl-rebuttal/scripts/embed_train_dev_image.py`

- added `selection_metric=balanced_logistic`
- best-checkpoint/early-stop score can now use the balanced logistic formula
- fixed best-checkpoint restoration when max epochs are reached without an
  early-stop break
- added `init_mode=symmetric` (legacy pre-normalization implementation)
- added `init_mode=spherical_symmetric`
- added `--symmetric_separation_scale`
- records initialization mode and separation scale in configs/summaries
- training remains ordinary full-vocabulary CE

### `eacl-rebuttal/scripts/run_sweep.py`

- accepts `spherical_symmetric`
- accepts comma-separated `--symmetric_separation_scales`
- includes scale in paths, job names, and manifest rows
- supports resumable multi-scale grids without name collisions
- manifest writer now unions fields when merging rows

### `eacl-rebuttal/scripts/select_lr_and_score.py`

- treats separation scale as a selectable dev hyperparameter
- writes the winning scale into selected manifests and summaries
- test-scores only the dev-selected configuration

### `eacl-rebuttal/scripts/run_spherical_scale_4b_sweep.sh`

- reproducible end-to-end launcher for the completed 25-cell screen
- chains training, dev selection, test scoring, and summary generation

There are unrelated or older working-tree changes, including
`core/eval/embed_eval.py` and `utils/vlm_loading.py`. Do not discard or reset
them. The repository/worktree is not clean, and much of `eacl-rebuttal/` may be
untracked in Git.

## Next action: 45 more seeds at the locked winner

The canonical 50-seed set used by the prior 2B experiment is:

```text
17,42,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,
115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,
132,133,134,135,136,137,138,139,140,141,142,143,144,200,2024,98765
```

The five already run here are `17,42,123,2024,98765`. Therefore the additional
45 are `100-144` excluding `123` (44 seeds), plus `200`.

Run the 45 additional seeds in a **new output directory**, for example:

```text
eacl-rebuttal/results/spherical_symmetric_4B_sep0p25_more45/
```

Do not append them to the five-scale search manifest: that would leave the
other four scale cells at `n=5` and the selected-config script correctly rejects
the resulting incomplete grid.

Suggested training command (the `100-144` list must omit `123`):

```bash
python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.01 \
  --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,200 \
  --inits small:/home/user/wug-test-interp/data/embeddings/init/noun_init.txt \
  --epochs 50 \
  --terminate_cond_epochs 5 \
  --batch_mode joint \
  --batch_size 8 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --out_dir eacl-rebuttal/results/spherical_symmetric_4B_sep0p25_more45 \
  --init_mode spherical_symmetric \
  --symmetric_separation_scales 0.25 \
  --selection_metric balanced_logistic \
  --write_all
```

Then score the 45-run manifest directly with `scripts/run_test_eval.py`, combine
those per-run test results with the existing five rows from
`spherical_symmetric_4B_scale_search/selected_test_per_run.csv`, and report
pooled/singular/plural means plus 95% t intervals across all 50 seeds. Also
report the attractor breakdown if requested.

Launch the training and scoring as one persistent chained job (tmux or an
equivalent persistent mechanism) so scoring starts automatically when training
finishes. Confirm actual GPU availability before launching.

## Useful status and recovery commands

Completed-run count for the scale screen:

```bash
find eacl-rebuttal/results/spherical_symmetric_4B_scale_search \
  -name learned_embeddings.pt | wc -l
# expected: 25
```

Check active jobs:

```bash
pgrep -af 'embed_train_dev_image|run_test_eval|spherical_symmetric_4B'
```

Inspect the completed orchestration:

```bash
tail -100 eacl-rebuttal/results/spherical_symmetric_4B_scale_search/orchestrator.log
```

The sweep runner skips completed runs with `learned_embeddings.pt`, so rerunning
the exact command is safe unless `--force` is supplied.

