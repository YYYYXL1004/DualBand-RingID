#!/usr/bin/env python
"""
频段对比实验脚本
测试 3 种频段编码策略对多密钥可识别性的影响：
1. 低频编码 (R=3-7): 5 个环
2. 高频编码 (R=8-14): 7 个环  
3. 双频带编码: 低频+高频交替

用法:
    python scripts/frequency_band_test.py --gpu 3
    python scripts/frequency_band_test.py --strategy low --gpu 3
    python scripts/frequency_band_test.py --collect-only
"""

import argparse
import subprocess
import os
import sys
import json
import shutil
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 实验配置
STRATEGIES = {
    'low': {
        'name': '低频编码',
        'RADIUS': 7,
        'RADIUS_CUTOFF': 3,
        'description': 'R=3-7, 4个环'
    },
    'high': {
        'name': '高频编码',
        'RADIUS': 14,
        'RADIUS_CUTOFF': 8,
        'description': 'R=8-14, 6个环'
    },
    'dual': {
        'name': '双频带编码',
        'RADIUS': 14,
        'RADIUS_CUTOFF': 3,
        'description': 'R=3-14, 11个环 (原始设置)'
    }
}

# 实验参数
NUM_KEYS = 5
NUM_TRIALS = 50  # 每组 50 张图，约 6 分钟


def modify_utils_py(radius, radius_cutoff):
    """临时修改 utils.py 中的 RADIUS 和 RADIUS_CUTOFF"""
    utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils.py')
    
    with open(utils_path, 'r') as f:
        content = f.read()
    
    # 备份原始值
    import re
    original_radius = re.search(r'^RADIUS = (\d+)', content, re.MULTILINE).group(1)
    original_cutoff = re.search(r'^RADIUS_CUTOFF = (\d+)', content, re.MULTILINE).group(1)
    
    # 修改值
    content = re.sub(r'^RADIUS = \d+', f'RADIUS = {radius}', content, flags=re.MULTILINE)
    content = re.sub(r'^RADIUS_CUTOFF = \d+', f'RADIUS_CUTOFF = {radius_cutoff}', content, flags=re.MULTILINE)
    
    with open(utils_path, 'w') as f:
        f.write(content)
    
    return int(original_radius), int(original_cutoff)


def restore_utils_py(original_radius, original_cutoff):
    """恢复 utils.py 的原始值"""
    utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils.py')
    
    with open(utils_path, 'r') as f:
        content = f.read()
    
    import re
    content = re.sub(r'^RADIUS = \d+', f'RADIUS = {original_radius}', content, flags=re.MULTILINE)
    content = re.sub(r'^RADIUS_CUTOFF = \d+', f'RADIUS_CUTOFF = {original_cutoff}', content, flags=re.MULTILINE)
    
    with open(utils_path, 'w') as f:
        f.write(content)


def run_experiment(strategy_name, gpu_id):
    """运行单个策略的实验"""
    strategy = STRATEGIES[strategy_name]
    print(f"\n{'='*60}")
    print(f"运行实验: {strategy['name']} ({strategy['description']})")
    print(f"{'='*60}")
    
    # 修改 utils.py
    original_radius, original_cutoff = modify_utils_py(
        strategy['RADIUS'], 
        strategy['RADIUS_CUTOFF']
    )
    
    try:
        # 构建运行命令
        run_name = f"freq_band_{strategy_name}"
        cmd = [
            'python', 'identify.py',
            '--run_name', run_name,
            '--trials', str(NUM_TRIALS),
            '--assigned_keys', str(NUM_KEYS),
            '--num_images', '1',
            '--reference_model', 'None',
            '--gpu_id', str(gpu_id),
            '--general_seed', '42'
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 运行实验
        project_root = os.path.dirname(os.path.dirname(__file__))
        result = subprocess.run(
            cmd, 
            cwd=project_root,
            capture_output=False
        )
        
        if result.returncode != 0:
            print(f"警告: 实验 {strategy_name} 返回非零退出码: {result.returncode}")
        
    finally:
        # 恢复 utils.py
        restore_utils_py(original_radius, original_cutoff)
        print(f"已恢复 utils.py 原始设置 (RADIUS={original_radius}, RADIUS_CUTOFF={original_cutoff})")


def collect_results():
    """收集所有实验结果"""
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs')
    results = {}
    
    for strategy_name in STRATEGIES:
        pattern = f"freq_band_{strategy_name}"
        
        # 查找最新的匹配目录
        matching_dirs = []
        for d in os.listdir(runs_dir):
            if pattern in d:
                matching_dirs.append(os.path.join(runs_dir, d))
        
        if not matching_dirs:
            print(f"未找到策略 {strategy_name} 的结果目录")
            continue
        
        # 选择最新的
        latest_dir = max(matching_dirs, key=os.path.getmtime)
        log_file = os.path.join(latest_dir, 'log.csv')
        
        if os.path.exists(log_file):
            results[strategy_name] = {
                'dir': latest_dir,
                'log': log_file,
                'config': STRATEGIES[strategy_name]
            }
            print(f"找到 {strategy_name}: {latest_dir}")
        else:
            print(f"警告: {latest_dir} 中未找到 log.csv")
    
    return results


def generate_comparison_report(results):
    """生成对比报告"""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs', 'frequency_band_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = [
        "# 频段编码策略对比实验报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验配置",
        "",
        "| 策略 | RADIUS | RADIUS_CUTOFF | 使用的环 |",
        "|------|--------|---------------|----------|",
    ]
    
    for name, config in STRATEGIES.items():
        num_rings = config['RADIUS'] - config['RADIUS_CUTOFF']
        report_lines.append(f"| {config['name']} | {config['RADIUS']} | {config['RADIUS_CUTOFF']} | {num_rings} 个 |")
    
    report_lines.extend([
        "",
        "## 实验结果",
        "",
        "请查看各策略的 log.csv 文件获取详细结果。",
        "",
    ])
    
    for name, data in results.items():
        report_lines.append(f"### {STRATEGIES[name]['name']}")
        report_lines.append(f"- 结果目录: `{data['dir']}`")
        report_lines.append("")
    
    report_path = os.path.join(output_dir, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='频段对比实验')
    parser.add_argument('--strategy', type=str, choices=['low', 'high', 'dual', 'all'], 
                        default='all', help='要运行的策略')
    parser.add_argument('--gpu', type=int, default=3, help='GPU ID')
    parser.add_argument('--collect-only', action='store_true', help='只收集结果，不运行实验')
    parser.add_argument('--trials', type=int, default=50, help='每组实验的图片数')
    parser.add_argument('--keys', type=int, default=5, help='密钥数量')
    args = parser.parse_args()
    
    global NUM_TRIALS, NUM_KEYS
    NUM_TRIALS = args.trials
    NUM_KEYS = args.keys
    
    if args.collect_only:
        results = collect_results()
        if results:
            generate_comparison_report(results)
        return
    
    strategies_to_run = list(STRATEGIES.keys()) if args.strategy == 'all' else [args.strategy]
    
    print(f"将运行以下策略: {strategies_to_run}")
    print(f"GPU: {args.gpu}, 每组 {NUM_TRIALS} 张图, {NUM_KEYS} 个密钥")
    
    for strategy in strategies_to_run:
        run_experiment(strategy, args.gpu)
    
    # 收集结果
    print("\n收集实验结果...")
    results = collect_results()
    if results:
        generate_comparison_report(results)


if __name__ == '__main__':
    main()
