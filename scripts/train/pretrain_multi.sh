#!/bin/bash

LOG_DIR="./log/"
OUTPUT_DIR="./out"
export PYTHONPATH="./"

# All-33
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/train/train_distributed_multilingual.py \
    --source_train_directory ./data/processed/pretraining/downsampled_train/ \
    --source_dev_directory ./data/processed/pretraining/downsampled_dev/ \
    --source_lang en \
    --langs_to_use ar bg ca zh hr cs da nl en et fi fr gl de el he hi hu id it ja ko pl pt ro ru sk sl es sv tr uk vi \
    --num_langs 33 \
    --save_at_end \
    --output_directory $OUTPUT_DIR/All33_pretraining_output/ \
    --experiment_name all33 \
    --train_sampler baseline \
    --max_seq_len 512 \
    --local_rank 0 \
    --max_steps 300001 \
    --batch_size 1 \
    --no_early_stop \
    --num_gpus 4 \
    --fp16 \
    --learning_rate 1e-4 \
    --log_dir $LOG_DIR \
    --gradient_accumulation_steps 2

# Div-10
# torchrun --standalone --nnodes=1 --nproc_per_node=4 \
#     src/train/train_distributed_multilingual.py \
#     --source_train_directory ./data/processed/pretraining/downsampled_train/ \
#     --source_dev_directory ./data/processed/pretraining/downsampled_dev/ \
#     --source_lang en \
#     --langs_to_use en ru zh ar hi es tr el fi id \
#     --num_langs 10 \
#     --save_at_end \
#     --output_directory $OUTPUT_DIR/Div10_pretraining_output/ \
#     --experiment_name div10 \
#     --train_sampler baseline \
#     --max_seq_len 512 \
#     --local_rank 0 \
#     --max_steps 150001 \
#     --batch_size 1 \
#     --no_early_stop \
#     --num_gpus 4 \
#     --fp16 \
#     --learning_rate 1e-4 \
#     --log_dir $LOG_DIR \
#     --gradient_accumulation_steps 2

# # Rel-5
# torchrun --standalone --nnodes=1 --nproc_per_node=4 \
#     src/train/train_distributed_multilingual.py \
#     --source_train_directory ./data/processed/pretraining/downsampled_train/ \
#     --source_dev_directory ./data/processed/pretraining/downsampled_dev/ \
#     --source_lang en \
#     --langs_to_use en de sv nl da \
#     --num_langs 5 \
#     --save_at_end \
#     --output_directory $OUTPUT_DIR/Rel5_pretraining_output/ \
#     --experiment_name rel5 \
#     --train_sampler baseline \
#     --max_seq_len 512 \
#     --local_rank 0 \
#     --max_steps 150001 \
#     --batch_size 1 \
#     --no_early_stop \
#     --num_gpus 4 \
#     --fp16 \
#     --learning_rate 1e-4 \
#     --log_dir $LOG_DIR \
#     --gradient_accumulation_steps 2



