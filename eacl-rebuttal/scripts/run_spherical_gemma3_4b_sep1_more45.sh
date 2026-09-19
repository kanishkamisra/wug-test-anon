#!/usr/bin/env bash
set -euo pipefail

cd /home/user/wug-test-interp

export HF_HOME=/home/user/.cache/hf_gemma3
export TRANSFORMERS_CACHE=/home/user/.cache/hf_gemma3
export HUGGINGFACE_HUB_CACHE=/home/user/.cache/hf_gemma3

out_dir="eacl-rebuttal/results/spherical_symmetric_gemma3_4b_sep1_more45"
model="google/gemma-3-4b-it"
inits="small:/home/user/wug-test-interp/data/embeddings/init/noun_init.txt"
existing_per_run="eacl-rebuttal/results/spherical_symmetric_gemma3_4b_bs4/selected_test_per_run.csv"

python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.003 \
  --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,200 \
  --inits "$inits" \
  --epochs 50 \
  --terminate_cond_epochs 5 \
  --batch_mode joint \
  --batch_size 4 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --out_dir "$out_dir" \
  --init_mode spherical_symmetric \
  --symmetric_separation_scales 1.0 \
  --selection_metric balanced_logistic \
  --write_all

python3 eacl-rebuttal/scripts/run_test_eval.py \
  --manifest "$out_dir/manifest.csv" \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --gpus 0,1,2,3 \
  --out_dir "$out_dir/test_eval" \
  --batch_size 16

python3 eacl-rebuttal/scripts/combine_50_seed_test_results.py \
  --new_manifest "$out_dir/manifest.csv" \
  --new_test_dir "$out_dir/test_eval" \
  --existing_per_run "$existing_per_run" \
  --out_dir "$out_dir"

echo "SPHERICAL_GEMMA3_4B_SEP1_MORE45_COMPLETE"
