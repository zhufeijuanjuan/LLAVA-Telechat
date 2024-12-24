export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

LLM_VERSION="pretrain_model/telechat2-7B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="pretrain_model/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

#PROMPT_VERSION=plain
PROMPT_VERSION=telechat
BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-telechat2"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

NUM_GPUS=${NUM_GPUS:-7}
NNODES=${NNODES:-1}
RANK=${RANK:-0}
ADDR=${ADDR:-'192.168.100.10'} #localhost
PORT=${PORT:-'19995'}

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NNODES: ${NNODES}"
echo "RANK: ${RANK}"
echo "ADDR: ${ADDR}"
echo "PORT: ${PORT}"

#ACCELERATE_CPU_AFFINITY=1no
torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2_offload.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path data/pretrain/blip_558k/blip_laion_cc_sbu_558k_QwenVL_AF1112_5wPicked.json \
    --image_folder data/pretrain/blip_558k/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --is_multimodal True \
    --run_name $BASE_RUN_NAME \
    --attn_implementation eager \
    --verbose_logging True

# You can delete the sdpa attn_implementation if you want to use flash attn
