#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=telechat
MODEL_VERSION="telechat2-7B"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./pretrain_model/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./data/sft/llava_instruct_361k_zh_merged.json \
    --image_folder ./data/sft/llava_instruct_sft \
    --vision_tower ./pretrain_model/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 16 \
    --attn_implementation eager
