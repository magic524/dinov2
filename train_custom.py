#!/usr/bin/env python3
"""
简单的单GPU训练脚本，适用于自定义小数据集
"""

import sys
import os

# 将项目路径加入Python路径
sys.path.insert(0, '/home/magic524/projects/no4/dinov2')
os.environ['PYTHONPATH'] = '/home/magic524/projects/no4/dinov2'

import subprocess

# 训练参数
config_file = "/home/magic524/projects/no4/dinov2/dinov2/configs/train/cat_dog_custom.yaml"
output_dir = "/home/magic524/projects/no4/dinov2/outputs/cat_dog_run1"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 构建训练命令
# 使用torchrun进行单GPU训练
cmd = [
    "torchrun",
    "--nproc_per_node=1",  # 单GPU
    "--standalone",
    "/home/magic524/projects/no4/dinov2/dinov2/train/train.py",
    f"--config-file={config_file}",
    f"--output-dir={output_dir}",
]

print("="*70)
print("Starting DINOv2 training on custom cat_dog dataset")
print("="*70)
print(f"Config file: {config_file}")
print(f"Output dir: {output_dir}")
print("="*70)
print("Command:", " ".join(cmd))
print("="*70)

# 运行训练
subprocess.run(cmd)
