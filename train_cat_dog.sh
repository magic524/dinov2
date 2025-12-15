#!/bin/bash
# 单GPU训练脚本 - 适用于自定义猫狗数据集

# 激活conda环境
source /home/magic524/miniconda3/bin/activate dinov2

# 设置Python路径
export PYTHONPATH=/home/magic524/projects/no4/dinov2:$PYTHONPATH

# 配置参数
CONFIG_FILE="/home/magic524/projects/no4/dinov2/dinov2/configs/train/cat_dog_vitb16.yaml"
OUTPUT_DIR="/home/magic524/projects/no4/dinov2/outputs/cat_dog_vitb16_run2"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 打印信息
echo "======================================================================"
echo "Starting DINOv2 Training on Custom Cat-Dog Dataset"
echo "======================================================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "======================================================================"

# 启动训练 - 使用torchrun进行单GPU训练
cd /home/magic524/projects/no4/dinov2

torchrun \
    --nproc_per_node=1 \
    --standalone \
    dinov2/train/train.py \
    --config-file="$CONFIG_FILE" \
    --output-dir="$OUTPUT_DIR"

echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
