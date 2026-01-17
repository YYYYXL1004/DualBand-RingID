#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
密钥容量极限测试
测试不同密钥数量下的识别准确率，找到容量-准确率权衡曲线

用法: 
  # 运行 5,10,20 的实验（使用 GPU 3）
  python scripts/key_capacity_test.py --keys 5,10,20,30 --gpu 3
  
  # 运行 30,50 的实验（使用 GPU 5）
  python scripts/key_capacity_test.py --keys 50 --gpu 5
  
  # 只汇总已有结果（不运行新实验）
  python scripts/key_capacity_test.py --collect-only
"""

import subprocess
import os
import re
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 测试配置
DEFAULT_KEY_COUNTS = [5, 10, 20, 30, 50]  # 要测试的密钥数量
TRIALS_PER_KEY = 10  # 每个密钥生成的图片数
OUTPUT_DIR = './runs/key_capacity_test'

def parse_script_args():
    parser = argparse.ArgumentParser(description='密钥容量极限测试')
    parser.add_argument('--keys', type=str, default='5,10,20,30,50', help='要测试的密钥数量，逗号分隔')
    parser.add_argument('--gpu', type=int, default=3, help='使用的 GPU ID')
    parser.add_argument('--trials-per-key', type=int, default=10, help='每个密钥的测试图片数')
    parser.add_argument('--collect-only', action='store_true', help='只汇总已有结果，不运行新实验')
    return parser.parse_args()

def run_experiment(num_keys, trials, gpu_id):
    """运行单次实验"""
    run_name = f'capacity_test_keys_{num_keys}'
    cmd = [
        'python', 'identify.py',
        '--gpu_id', str(gpu_id),
        '--run_name', run_name,
        '--trials', str(trials),
        '--assigned_keys', str(num_keys),
        '--save_generated_imgs', '1'  # 不保存图片，加快速度
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing with {num_keys} keys, {trials} trials on GPU {gpu_id}...")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 解析输出
    output = result.stdout + result.stderr
    print(output)
    
    # 提取准确率
    accuracies = parse_accuracies(output)
    
    return accuracies

def parse_accuracies(output):
    """从输出中解析准确率"""
    accuracies = {}
    
    # 查找表格行
    lines = output.split('\n')
    for line in lines:
        if '|a-b|' in line and 'L1' in line:
            # 解析表格行: | L1 |a-b|         | 1.000 | 1.000  |  1.000  | 0.800  | 1.000  |   1.000   |       1.000       | 0.971 |
            # parts[0]='', parts[1]='L1', parts[2]='|a-b|...', parts[3]='Clean', ...
            parts = line.split('|')
            if len(parts) >= 11:
                try:
                    accuracies['Clean'] = float(parts[3].strip())
                    accuracies['Rot_75'] = float(parts[4].strip())
                    accuracies['JPEG_25'] = float(parts[5].strip())
                    accuracies['CS_75'] = float(parts[6].strip())
                    accuracies['Blur_8'] = float(parts[7].strip())
                    accuracies['Noise_01'] = float(parts[8].strip())
                    accuracies['Brightness'] = float(parts[9].strip())
                    accuracies['Avg'] = float(parts[10].strip())
                except (ValueError, IndexError) as e:
                    print(f"解析错误: {e}, line: {line}")
    
    return accuracies

def plot_results(results, output_path):
    """绘制容量-准确率曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 提取数据
    key_counts = sorted(results.keys())
    avg_accs = [results[k].get('Avg', 0) for k in key_counts]
    clean_accs = [results[k].get('Clean', 0) for k in key_counts]
    cs_accs = [results[k].get('CS_75', 0) for k in key_counts]
    
    # 图1: 平均准确率 vs 密钥数量
    ax1 = axes[0]
    ax1.plot(key_counts, avg_accs, 'b-o', linewidth=2, markersize=8, label='Average')
    ax1.plot(key_counts, clean_accs, 'g--s', linewidth=1.5, markersize=6, label='Clean')
    ax1.plot(key_counts, cs_accs, 'r--^', linewidth=1.5, markersize=6, label='Crop&Scale')
    ax1.set_xlabel('Number of Keys', fontsize=12)
    ax1.set_ylabel('Identification Accuracy', fontsize=12)
    ax1.set_title('Key Capacity vs Accuracy', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 图2: 所有攻击类型的准确率
    ax2 = axes[1]
    attack_types = ['Clean', 'Rot_75', 'JPEG_25', 'CS_75', 'Blur_8', 'Noise_01', 'Brightness']
    x = np.arange(len(key_counts))
    width = 0.12
    
    for i, attack in enumerate(attack_types):
        accs = [results[k].get(attack, 0) for k in key_counts]
        ax2.bar(x + i*width, accs, width, label=attack)
    
    ax2.set_xlabel('Number of Keys', fontsize=12)
    ax2.set_ylabel('Identification Accuracy', fontsize=12)
    ax2.set_title('Accuracy by Attack Type', fontsize=14)
    ax2.set_xticks(x + width * 3)
    ax2.set_xticklabels(key_counts)
    ax2.legend(loc='lower left', fontsize=8)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")

def collect_existing_results():
    """从已有的 log.csv 文件中收集结果"""
    results = {}
    runs_dir = './runs'
    for dirname in os.listdir(runs_dir):
        if 'capacity_test_keys_' in dirname:
            log_path = os.path.join(runs_dir, dirname, 'log.csv')
            if os.path.exists(log_path):
                with open(log_path) as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        header = lines[0].strip().split(',')
                        data = lines[1].strip().split(',')
                        num_keys = int(dirname.split('keys_')[1])
                        try:
                            clean_idx = header.index('Clean')
                            results[num_keys] = {
                                'Clean': float(data[clean_idx]),
                                'Rot_75': float(data[clean_idx+1]),
                                'JPEG_25': float(data[clean_idx+2]),
                                'CS_75': float(data[clean_idx+3]),
                                'Blur_8': float(data[clean_idx+4]),
                                'Noise_01': float(data[clean_idx+5]),
                                'Brightness': float(data[clean_idx+6]),
                                'Avg': float(data[clean_idx+7]),
                            }
                        except (ValueError, IndexError) as e:
                            print(f"解析 {dirname} 失败: {e}")
    return results

def main():
    args = parse_script_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 先收集已有结果
    results = collect_existing_results()
    print(f"已收集到 {len(results)} 个已有结果: {sorted(results.keys())}")
    
    if args.collect_only:
        print("仅汇总模式，不运行新实验")
    else:
        # 解析要测试的密钥列表
        key_counts = [int(k) for k in args.keys.split(',')]
        
        for num_keys in key_counts:
            # 跳过已有结果的实验
            if num_keys in results and results[num_keys].get('Avg', 0) > 0:
                print(f"\n跳过 {num_keys} keys（已有结果）")
                continue
            
            trials = num_keys * args.trials_per_key
            accuracies = run_experiment(num_keys, trials, args.gpu)
            if accuracies:
                results[num_keys] = accuracies
            
            # 保存中间结果
            with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("KEY CAPACITY TEST RESULTS")
    print("="*80)
    print(f"{'Keys':<8} {'Clean':<8} {'Rot75':<8} {'JPEG25':<8} {'C&S75':<8} {'Blur8':<8} {'Noise':<8} {'Bright':<8} {'Avg':<8}")
    print("-"*80)
    for num_keys in sorted(results.keys()):
        r = results[num_keys]
        print(f"{num_keys:<8} {r.get('Clean', 0):<8.3f} {r.get('Rot_75', 0):<8.3f} {r.get('JPEG_25', 0):<8.3f} {r.get('CS_75', 0):<8.3f} {r.get('Blur_8', 0):<8.3f} {r.get('Noise_01', 0):<8.3f} {r.get('Brightness', 0):<8.3f} {r.get('Avg', 0):<8.3f}")
    print("="*80)
    
    # 绘制图表
    plot_results(results, os.path.join(OUTPUT_DIR, 'key_capacity_curve.png'))
    
    print("\nDone!")

if __name__ == '__main__':
    main()
