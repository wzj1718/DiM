#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


MODEL="$1"
TASK="$2"

python3 -m lmms_eval \
    --model llava \
    --model_args pretrained=${MODEL}\
    --tasks ${TASK} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${TASK} \
    --output_path eval_logs\-${MODEL}-${TASK}

