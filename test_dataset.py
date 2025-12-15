#!/usr/bin/env python3
"""测试自定义数据集是否能正常加载"""

import sys
sys.path.insert(0, '/home/magic524/projects/no4/dinov2')

from dinov2.data.datasets import CustomImageFolder

# 测试数据集加载
dataset_root = "/mnt/e/Datasets/DINO_datasets/cat_dog"
extra_dir = "/mnt/e/Datasets/DINO_datasets/cat_dog"

print("="*70)
print("Testing CustomImageFolder dataset")
print("="*70)

for split in [CustomImageFolder.Split.TRAIN, CustomImageFolder.Split.TEST]:
    print(f"\n{split.value.upper()} Split:")
    print("-"*70)
    
    dataset = CustomImageFolder(
        split=split,
        root=dataset_root,
        extra=extra_dir
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # 测试加载第一张图片
    try:
        img, label = dataset[0]
        print(f"First image loaded successfully")
        print(f"  Image type: {type(img)}")
        print(f"  Image size: {img.size}")
        print(f"  Label: {label}")
        print(f"  Class name: {dataset.get_class_name(0)}")
    except Exception as e:
        print(f"Error loading first image: {e}")
    
    # 显示类别信息
    try:
        class_names = dataset._get_class_names()
        print(f"Classes: {[cn.decode('utf-8') for cn in class_names]}")
    except Exception as e:
        print(f"Error getting class names: {e}")

print("\n" + "="*70)
print("Dataset test completed!")
print("="*70)
