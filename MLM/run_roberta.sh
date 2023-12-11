#!/usr/bin/sh
python run_nli4ct.py \
  --model_name_or_path roberta-base \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 128 \
  --output_dir ../debug_nli4ct/ \
  --save_steps -1