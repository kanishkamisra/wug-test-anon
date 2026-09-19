#!/usr/bin/env bash
set -euo pipefail

cd /home/user/wug-test-interp

export HF_HOME=/home/user/.cache/hf_gemma3
export TRANSFORMERS_CACHE=/home/user/.cache/hf_gemma3
export HUGGINGFACE_HUB_CACHE=/home/user/.cache/hf_gemma3

out_dir="eacl-rebuttal/results/spherical_symmetric_gemma3_4b_bs4"
model="google/gemma-3-4b-it"
inits="small:/home/user/wug-test-interp/data/embeddings/init/noun_init.txt"

python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.003 \
  --seeds 17,42,123,2024,98765 \
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
  --symmetric_separation_scales 0,0.25,0.5,1,1.5 \
  --selection_metric balanced_logistic \
  --write_all

python3 eacl-rebuttal/scripts/select_lr_and_score.py \
  --sweep_dir "$out_dir" \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --gpus 0,1,2,3 \
  --batch_size 16

echo "SPHERICAL_SCALE_GEMMA3_4B_BS4_COMPLETE"
