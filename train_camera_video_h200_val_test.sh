#!/bin/bash

# ==============================================================================
# 1. 脚本安全与错误处理设置
# ==============================================================================
# -e: 当任何命令执行失败时，立即退出脚本。
# -u: 当使用未定义的变量时，立即退出脚本。
# -o pipefail: 在管道命令中，只要有任何一个命令失败，整个管道就视为失败。
set -euo pipefail

# ==============================================================================
# 2. 参数配置区 (在此处修改所有参数)
# ==============================================================================
# -- GPU 设置 --
GPUS="0"
# GPUS="3,4,5,6"

# -- 核心路径 --
DATASET_PATH="/mnt/data/camera_datasets_ssd/MuteApo/RealCam-Vid"
DATASET_LIST="/mnt/data/camera_datasets_ssd/MuteApo/RealCam-Vid/RealCam-Vid_train_camera.npz"
OUTPUT_PATH="./models_debug"
DIT_PATH="/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"
IMAGE_ENCODER_PATH="/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
TEXT_ENCODER_PATH="/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth"
VAE_PATH="/mnt/data/video_public_ckpt/Wan-AI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth"

# -- 训练超参数 --
PER_DEVICE_BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=1
LEARNING_RATE=1e-5
MAX_EPOCHS=15
NUM_FRAMES=81
DATALOADER_NUM_WORKERS=4

# -- LoRA 配置 --
LORA_RANK=16
LORA_ALPHA=16
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"

# -- 其他设置 --
# 根据你的GPU架构设置，H100/H200/A100后 Ampere 架构等使用 9.0, 8.0
export TORCH_CUDA_ARCH_LIST="9.0"
export TOKENIZERS_PARALLELISM=false

# ==============================================================================
# 3. 自动清理函数 (核心功能)
# ==============================================================================
# 定义一个清理函数，用于终止此脚本启动的所有子进程
cleanup() {
    echo "接收到退出信号，正在清理所有子进程..."
    # $$ 是当前脚本的 PID。pkill -P $$ 会杀死所有父进程PID为当前脚本的进程。
    # 这是一种非常可靠的清理方法。
    pkill -P $$ || true # '|| true' 确保即使没有子进程可杀，脚本也不会因为 pkill 的失败而报错退出
    echo "清理完毕。"
}

# 使用 trap 命令捕获退出信号。
# SIGINT: Ctrl+C
# SIGTERM: kill 命令（默认）
# EXIT: 脚本退出时（无论是正常结束、出错还是被中断）
# 当捕获到这些信号时，会自动调用 cleanup 函数。
trap cleanup SIGINT SIGTERM EXIT

# ==============================================================================
# 4. 日志和启动准备
# ==============================================================================
# 创建输出目录和日志目录
LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "${LOG_DIR}"

# 创建带时间戳的日志文件名
LOG_FILE="${LOG_DIR}/training_$(date +'%Y%m%d_%H%M%S').log"

# 自动计算要使用的 GPU 数量
NPROC=$(echo $GPUS | tr ',' ' ' | wc -w)

echo "=================================================="
echo "训练即将开始..."
echo "使用的 GPUs: ${GPUS} (共 ${NPROC} 张卡)"
echo "批次大小 (每卡): ${PER_DEVICE_BATCH_SIZE}"
echo "梯度累加步数: ${ACCUMULATE_GRAD_BATCHES}"
echo "日志将保存到: ${LOG_FILE}"
echo "=================================================="

# ==============================================================================
# 5. 执行训练命令
# ==============================================================================
# 使用 torchrun 启动分布式训练
# 将所有输出 (stdout 和 stderr) 通过 tee 同时打印到控制台并写入日志文件
# CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} \
CUDA_VISIBLE_DEVICES=${GPUS} python -m wanvideo.train_wan_camera_t2v_plucker \
    --task train \
    --train_architecture adapter \
    --dataset_path "${DATASET_PATH}" \
    --dataset_list "${DATASET_LIST}" \
    --output_path "${OUTPUT_PATH}" \
    --dit_path "${DIT_PATH}" \
    --image_encoder_path "${IMAGE_ENCODER_PATH}" \
    --text_encoder_path "${TEXT_ENCODER_PATH}" \
    --vae_path "${VAE_PATH}" \
    --steps_per_epoch 0 \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --num_frames ${NUM_FRAMES} \
    --max_epochs ${MAX_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES} \
    --use_gradient_checkpointing \
    --use_tensorboard \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成。"