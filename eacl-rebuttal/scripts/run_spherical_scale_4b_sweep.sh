#!/usr/bin/env bash
set -euo pipefail

cd /home/user/wug-test-interp

out_dir="eacl-rebuttal/results/spherical_symmetric_4B_scale_search"
model="Qwen/Qwen3-VL-4B-Instruct"
inits="small:/home/user/wug-test-interp/data/embeddings/init/noun_init.txt"

python3 eacl-rebuttal/scripts/run_sweep.py \
    --gpus 0,1,2,3 \
    --lrs 0.01 \
    --seeds 17,42,123,2024,98765 \
    --inits "$inits" \
    --epochs 50 \
    --terminate_cond_epochs 5 \
    --batch_mode joint \
    --batch_size 8 \
    --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
    --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
    --model "$model" \
    --out_dir "$out_dir" \
    --init_mode spherical_symmetric \
    --symmetric_separation_scales 0,0.25,0.5,1,1.5 \
    --selection_metric balanced_logistic \
    --write_all

python3 eacl-rebuttal/scripts/select_lr_and_score.py \
    --sweep_dir "$out_dir" \
    --model "$model" \
    --gpus 0,1,2,3 \
    --batch_size 32

echo "SPHERICAL_SCALE_4B_SWEEP_COMPLETE"
