#!/usr/bin/env python3
"""评估训练好的DINOv2模型"""

import sys
sys.path.insert(0, '/home/magic524/projects/no4/dinov2')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
from tqdm import tqdm

from dinov2.models.vision_transformer import vit_base

def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    # 创建模型
    model = vit_base(
        patch_size=16,
        img_size=518,
        init_values=1e-5,
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取student模型的权重
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    else:
        state_dict = checkpoint
    
    # 移除不需要的前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', '')
            new_state_dict[new_key] = v
        elif not any(x in k for x in ['dino_head', 'ibot_head']):
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model

def extract_features(model, image_paths, device='cuda'):
    """提取图像特征"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    labels = []
    
    model = model.to(device)
    
    print(f"提取 {len(image_paths)} 张图片的特征...")
    
    with torch.no_grad():
        for img_path, label in tqdm(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # 提取特征 (使用CLS token)
                output = model(img_tensor)
                feat = output.cpu().numpy().squeeze()
                
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"处理图片出错 {img_path}: {e}")
                continue
    
    return np.array(features), np.array(labels)

def load_dataset(root_dir, split='val'):
    """加载数据集"""
    image_paths = []
    
    split_dir = os.path.join(root_dir, split)
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    
    print(f"\n加载 {split} 数据集，类别: {class_names}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append((img_path, class_idx))
    
    print(f"总共加载 {len(image_paths)} 张图片")
    return image_paths, class_names

def evaluate_knn(checkpoint_path, dataset_root, k=5):
    """使用KNN评估模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 加载模型
    model = load_model(checkpoint_path)
    
    # 加载训练集和验证集
    train_paths, class_names = load_dataset(dataset_root, 'train')
    val_paths, _ = load_dataset(dataset_root, 'val')
    
    # 提取特征
    print("\n" + "="*60)
    print("提取训练集特征...")
    train_features, train_labels = extract_features(model, train_paths, device)
    
    print("\n" + "="*60)
    print("提取验证集特征...")
    val_features, val_labels = extract_features(model, val_paths, device)
    
    # 训练KNN分类器
    print("\n" + "="*60)
    print(f"训练KNN分类器 (k={k})...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)
    
    # 预测
    predictions = knn.predict(val_features)
    
    # 计算准确率
    accuracy = accuracy_score(val_labels, predictions)
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"验证集准确率: {accuracy*100:.2f}%\n")
    
    # 详细分类报告
    print("分类报告:")
    print(classification_report(val_labels, predictions, target_names=class_names))
    
    return accuracy

if __name__ == "__main__":
    checkpoint_path = "/home/magic524/projects/no4/dinov2/outputs/cat_dog_vitb16_run1/final_model.rank_0.pth"
    dataset_root = "/mnt/e/Datasets/DINO_datasets/cat_dog"
    
    print("="*60)
    print("DINOv2 模型评估")
    print("="*60)
    
    accuracy = evaluate_knn(checkpoint_path, dataset_root, k=5)
    
    print("\n" + "="*60)
    print(f"最终准确率: {accuracy*100:.2f}%")
    print("="*60)
