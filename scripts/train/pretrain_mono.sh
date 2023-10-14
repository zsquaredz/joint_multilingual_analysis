#!/bin/bash

LOG_DIR="./log/"
OUTPUT_DIR="./out"
LANG='en'

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
    src/train/train_distributed_monolingual.py \
    --source_train_directory ./data/processed/pretraining/downsampled_train/ \
    --source_dev_directory ./data/processed/pretraining/downsampled_dev/ \
    --source_lang $LANG \
    --langs_to_use $LANG \
    --num_langs 1 \
    --save_at_end \
    --tokenizer_path xlm-roberta-base \
    --use_multi_tokenizer \
    --output_directory $OUTPUT_DIR/${LANG}_pretraining_output/ \
    --experiment_name ${LANG} \
    --train_sampler baseline \
    --max_seq_len 512 \
    --local_rank 0 \
    --max_steps 150001 \
    --batch_size 1 \
    --no_early_stop \
    --save_total_limit 3 \
    --num_gpus 4 \
    --fp16 \
    --learning_rate 1e-4 \
    --log_dir $LOG_DIR \
    --gradient_accumulation_steps 2
