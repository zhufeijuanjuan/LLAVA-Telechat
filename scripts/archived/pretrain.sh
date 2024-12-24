#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
MODEL_VERSION="telechat2-7B"
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=telechat
########### DO NOT CHANGE ###########

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./pretrain_model/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path data/pretrain/blip_558k/blip_laion_cc_sbu_558k_QwenVL_AF1112_20wPicked.json \
    --image_folder data/pretrain/blip_558k/images \
    --vision_tower pretrain_model/siglip-so400m-patch14-384 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain-v1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --attn_implementation eager \
    --lazy_preprocess True
