import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_folder_metrics(folder_path):
    """计算单个文件夹内所有配对图片的平均 PSNR 和 SSIM"""
    all_files = os.listdir(folder_path)
    ref_files = sorted([f for f in all_files if f.endswith('_no_watermark.png')])
    
    psnrs, ssims = [], []

    for ref_name in ref_files:
        prefix = ref_name.replace('_no_watermark.png', '')
        dist_name = f"{prefix}_watermarked.png"
        
        ref_img = cv2.imread(os.path.join(folder_path, ref_name))
        dist_img = cv2.imread(os.path.join(folder_path, dist_name))

        if ref_img is None or dist_img is None:
            continue

        # 统一尺寸
        if ref_img.shape != dist_img.shape:
            dist_img = cv2.resize(dist_img, (ref_img.shape[1], ref_img.shape[0]))

        # 计算指标
        p = psnr(ref_img, dist_img, data_range=255)
        s = ssim(ref_img, dist_img, channel_axis=-1, data_range=255)
        
        psnrs.append(p)
        ssims.append(s)

    if not psnrs:
        return None, None
    return np.mean(psnrs), np.mean(ssims)

def main():
    parent_dir = 'C:\\Files\\tree_ring_results\\tree_ring_results' 
    sub_folders = ['Dual-Ring', 'High-Freq', 'Low-Freq']
    
    summary = []

    print(f"{'Folder Name':<15} | {'Avg PSNR (dB)':<15} | {'Avg SSIM':<10}")
    print("-" * 45)

    for folder in sub_folders:
        folder_path = os.path.join(parent_dir, folder)
        
            
        avg_p, avg_s = calculate_folder_metrics(folder_path)
        
        if avg_p is not None:
            summary.append([folder, avg_p, avg_s])
            print(f"{folder:<15} | {avg_p:>15.4f} | {avg_s:>10.4f}")
        else:
            print(f"{folder:<15} | 无匹配图片")

    # 最后的总结对比
    if summary:
        # 按 PSNR 从高到低排序，方便看哪个效果最好
        summary.sort(key=lambda x: x[1], reverse=True) 
        for row in summary:
            print(f" {row[0]}: PSNR={row[1]:.2f}, SSIM={row[2]:.4f}")

if __name__ == "__main__":
    main()