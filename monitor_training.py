#!/usr/bin/env python3
"""监控DINOv2训练进度"""

import json
import os
import time

output_dir = "/home/magic524/projects/no4/dinov2/outputs/cat_dog_vitb16_run1"
metrics_file = os.path.join(output_dir, "training_metrics.json")

def read_latest_metrics(n=5):
    """读取最新的n条训练指标"""
    if not os.path.exists(metrics_file):
        return []
    
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    metrics = []
    for line in lines[-n:]:
        try:
            metrics.append(json.loads(line.strip()))
        except:
            pass
    return metrics

def format_metric(value):
    """格式化指标值"""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def main():
    print("="*80)
    print("DINOv2 训练进度监控 - 猫狗数据集")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print(f"指标文件: {metrics_file}")
    print("="*80)
    
    while True:
        try:
            metrics = read_latest_metrics(n=1)
            
            if not metrics:
                print("等待训练开始...")
                time.sleep(5)
                continue
            
            latest = metrics[0]
            
            # 清屏并显示最新状态
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("="*80)
            print(f"{'DINOv2 训练进度监控':^80}")
            print("="*80)
            print(f"\n迭代次数: {latest['iteration']}")
            print(f"当前Epoch: {latest['iteration'] / 500:.2f} / 20")  # 500 iterations per epoch
            print(f"完成度: {latest['iteration'] / (20 * 500) * 100:.1f}%")
            print("\n" + "-"*80)
            print("Loss指标:")
            print(f"  总Loss:           {format_metric(latest['total_loss'])}")
            print(f"  DINO Global:      {format_metric(latest['dino_global_crops_loss'])}")
            print(f"  DINO Local:       {format_metric(latest['dino_local_crops_loss'])}")
            print(f"  IBOT Loss:        {format_metric(latest['ibot_loss'])}")
            print(f"  Koleo Loss:       {format_metric(latest['koleo_loss'])}")
            print("\n" + "-"*80)
            print("训练参数:")
            print(f"  学习率:           {format_metric(latest['lr'])}")
            print(f"  权重衰减:         {format_metric(latest['wd'])}")
            print(f"  动量:             {format_metric(latest['mom'])}")
            print(f"  Batch Size:       {int(latest['current_batch_size'])}")
            print("\n" + "-"*80)
            print("性能指标:")
            print(f"  迭代耗时:         {format_metric(latest['iter_time'])}s")
            print(f"  数据加载耗时:     {format_metric(latest['data_time'])}s")
            
            # 估算剩余时间
            total_iters = 20 * 500
            remaining_iters = total_iters - latest['iteration']
            estimated_time = remaining_iters * latest['iter_time'] / 3600  # 小时
            
            print(f"\n估算剩余时间:     {estimated_time:.2f} 小时")
            print("="*80)
            print("\n按Ctrl+C退出监控")
            
            time.sleep(10)  # 每10秒更新一次
            
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
