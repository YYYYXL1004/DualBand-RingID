#!/bin/bash

# 路径偏转超参数调优实验脚本
# 使用方法: screen -S deflection_exp && bash scripts/run_deflection_experiments.sh

echo "=========================================="
echo "开始路径偏转超参数调优实验"
echo "时间: $(date)"
echo "=========================================="

# 通用参数
GPU_ID=2
TRIALS=100
ASSIGNED_KEYS=5
SAVE_IMGS=1

echo "Baseline"
python identify.py --use_deflection 0 --trials 100 --gpu_id 2 --save_generated_imgs 1 --run_name baseline_no_deflection

# ========== 阶段一：验证偏转不破坏识别 ==========
echo ""
echo "[阶段一] 验证偏转不破坏识别"
echo "=========================================="

# 偏转 + 补偿（默认参数）
echo "[实验 1.1] 偏转 steps=5, strength=0.3"
python identify.py \
    --use_deflection 1 \
    --assigned_keys $ASSIGNED_KEYS \
    --deflection_steps 5 \
    --deflection_strength 0.3 \
    --trials $TRIALS \
    --gpu_id $GPU_ID \
    --save_generated_imgs $SAVE_IMGS \
    --run_name deflection_keys5_s5_m0.3

# ========== 阶段二：强度调优 ==========
echo ""
echo "[阶段二] 强度调优"
echo "=========================================="

echo "[实验 2.1] 强度 0.1"
python identify.py \
    --use_deflection 1 \
    --assigned_keys $ASSIGNED_KEYS \
    --deflection_steps 5 \
    --deflection_strength 0.1 \
    --trials $TRIALS \
    --gpu_id $GPU_ID \
    --save_generated_imgs $SAVE_IMGS \
    --run_name deflection_keys5_s5_m0.1

echo "[实验 2.2] 强度 0.2"
python identify.py \
    --use_deflection 1 \
    --assigned_keys $ASSIGNED_KEYS \
    --deflection_steps 5 \
    --deflection_strength 0.2 \
    --trials $TRIALS \
    --gpu_id $GPU_ID \
    --save_generated_imgs $SAVE_IMGS \
    --run_name deflection_keys5_s5_m0.2

echo "[实验 2.3] 强度 0.5"
python identify.py \
    --use_deflection 1 \
    --assigned_keys $ASSIGNED_KEYS \
    --deflection_steps 5 \
    --deflection_strength 0.5 \
    --trials $TRIALS \
    --gpu_id $GPU_ID \
    --save_generated_imgs $SAVE_IMGS \
    --run_name deflection_keys5_s5_m0.5

# ========== 阶段三：步数调优 ==========
echo ""
echo "[阶段三] 步数调优"
echo "=========================================="

echo "[实验 3.1] 步数 3"
python identify.py \
    --use_deflection 1 \
    --assigned_keys $ASSIGNED_KEYS \
    --deflection_steps 3 \
    --deflection_strength 0.3 \
    --trials $TRIALS \
    --gpu_id $GPU_ID \
    --save_generated_imgs $SAVE_IMGS \
    --run_name deflection_keys5_s3_m0.3

echo "[实验 3.2] 步数 10"
python identify.py \
    --use_deflection 1 \
    --assigned_keys $ASSIGNED_KEYS \
    --deflection_steps 10 \
    --deflection_strength 0.3 \
    --trials $TRIALS \
    --gpu_id $GPU_ID \
    --save_generated_imgs $SAVE_IMGS \
    --run_name deflection_keys5_s10_m0.3

echo "[实验 3.3] 步数 15"
python identify.py \
    --use_deflection 1 \
    --assigned_keys $ASSIGNED_KEYS \
    --deflection_steps 15 \
    --deflection_strength 0.3 \
    --trials $TRIALS \
    --gpu_id $GPU_ID \
    --save_generated_imgs $SAVE_IMGS \
    --run_name deflection_keys5_s15_m0.3

echo ""
echo "=========================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "结果保存在 ./runs/ 目录下"
echo "=========================================="
