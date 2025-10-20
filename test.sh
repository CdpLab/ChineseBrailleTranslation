#!/bin/bash
set -e

output_dir="./save_model/evaluation-final"

accelerate launch run_translation.py \
  --model_name_or_path "./save_model" \
  --output_dir "$output_dir" \
  --validation_file "./Chinese_braille_data/Parallel Corpus/val.json" \
  --test_file "./Chinese_braille_data/Parallel Corpus/test.json" \
  --source_prefix "translate Chinese to Braille: " \
  --do_eval \
  --do_predict \
  --per_device_eval_batch_size 16 \
  --overwrite_output_dir True \
  --seed 42 \
  --logging_steps 10 \
  --max_source_length 128 \
  --max_target_length 300 \
  --val_max_target_length 300 \
  --generation_max_length 300 \
  --generation_num_beams 1 \
  --use_fast_tokenizer False \
  --preprocessing_num_workers 4
