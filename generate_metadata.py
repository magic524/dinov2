#!/usr/bin/env python3
"""生成自定义数据集的元数据文件"""

import sys
sys.path.insert(0, '/home/magic524/projects/no4/dinov2')

from dinov2.data.datasets import CustomImageFolder

# 数据集路径
dataset_root = "/mnt/e/Datasets/DINO_datasets/cat_dog"
extra_dir = "/mnt/e/Datasets/DINO_datasets/cat_dog"

# 为每个split生成元数据
for split in CustomImageFolder.Split:
    print(f"\n{'='*50}")
    print(f"Generating metadata for {split.value} split")
    print(f"{'='*50}")
    
    try:
        dataset = CustomImageFolder(
            split=split,
            root=dataset_root,
            extra=extra_dir
        )
        dataset.dump_extra()
        print(f"✓ Successfully generated metadata for {split.value}")
    except Exception as e:
        print(f"✗ Error generating metadata for {split.value}: {e}")

print("\n" + "="*50)
print("Metadata generation completed!")
print("="*50)
