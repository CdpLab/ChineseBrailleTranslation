#!/bin/bash
set -e

output_dir="./save_model"

accelerate launch train.py \
  --model_name_or_path "./down_model" \
  --output_dir "$output_dir" \
  --train_file "./Chinese_braille_data/Parallel Corpus/train.json" \
  --validation_file "./Chinese_braille_data/Parallel Corpus/val.json" \
  --test_file "./Chinese_braille_data/Parallel Corpus/test.json" \
  --source_prefix "translate Chinese to Braille: " \
  --do_train \
  --num_train_epochs 15 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --overwrite_output_dir True \
  --seed 42 \
  --logging_steps 100 \
  --max_source_length 128 \
  --max_target_length 300 \
  --val_max_target_length 300 \
  --use_fast_tokenizer False \
  --preprocessing_num_workers 4 \
  --overwrite_cache True \
  --warmup_ratio 0.01 \
  --fp16 False \
  --save_steps 10000 \
  --save_strategy "steps" \
  --learning_rate 5e-5 \
  --lr_scheduler_type "cosine"
