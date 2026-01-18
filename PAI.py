import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageFilter
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import json

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "D:/models/stable-diffusion-2-1-base" 
PROMPTS_ID = "D:/prompts/prompts_for_alignment.txt"
NUM_PROMPTS = 50
INFERENCE_STEPS = 50
SEED = 42
IMAGE_SIZE = 512
OUTPUT_DIR = "pai_results"

# 水印配置：优化参数以提高检测率
WATERMARK_CONFIGS = {
    "PAI-Framework-Enhanced": {
        "low_r": 4,           # 低频环半径
        "high_r": 18,         # 高频环半径
        "alpha": 0.8,         # 增加注入强度 (论文中 alpha 较高以对抗去噪)
        "lambda_p": 0.3,      # 提高最小感知阈值
        "method": "pai",
        "dual": True
    }
}

def set_random_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_prompts(num_prompts, prompt_file_path):
    prompts = []
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line: line = line.split(":", 1)[1].strip()
                if line: prompts.append(line)
    
    if len(prompts) < num_prompts:
        prompts += ["A high-quality photo of a futuristic city"] * (num_prompts - len(prompts))
    return prompts[:num_prompts]

# --- PAI 核心算法实现 ---

def get_perceptual_weight(latents_fft, lambda_p=0.2):
    """
    感知权重计算优化：
    由于初始 latent 是纯噪声，其振幅分布平坦。
    这里对振幅进行非线性增强，以突出频率特征。
    """
    magnitude = torch.abs(latents_fft)
    # 对数域增强处理
    mag_log = torch.log1p(magnitude)
    min_val = mag_log.min()
    max_val = mag_log.max()
    norm_magnitude = (mag_log - min_val) / (max_val - min_val + 1e-8)
    
    return torch.clamp(norm_magnitude, min=lambda_p, max=1.0)

def inject_pai_watermark(init_latents, config):
    """
    优化后的 PAI 注入方法。
    核心：确保注入的 Target Patch 与原始信号有相似的功率谱密度（PSD）。
    """
    device = init_latents.device
    dtype = init_latents.dtype
    batch, channels, height, width = init_latents.shape
    
    # 1. 转换到频域
    latents_f32 = init_latents.to(torch.float32)
    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents_f32, dim=(-2, -1)), dim=(-2, -1))
    
    # 2. 生成目标水印 Patch (固定种子)
    generator = torch.Generator(device=device).manual_seed(SEED)
    real_part = torch.randn(latents_fft.shape, device=device, generator=generator)
    imag_part = torch.randn(latents_fft.shape, device=device, generator=generator)
    target_patch = torch.complex(real_part, imag_part)
    
    # 能量对齐：让 target_patch 的标准差匹配当前频率带的标准差
    target_patch = target_patch * (latents_fft.std(dim=(-2, -1), keepdim=True) * 1.5)

    # 构造空间掩码 Mask
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, width, device=device), indexing='ij')
    dist = torch.sqrt(x**2 + y**2)
    
    mask_low = (dist <= (config["low_r"] / (height/2))) & (dist > (1 / (height/2)))
    mask_high = (dist <= (config["high_r"] / (height/2))) & (dist > ((config["high_r"]-2) / (height/2)))
    mask_s = (mask_low | mask_high).float()

    # 计算感知权重
    mask_p = get_perceptual_weight(latents_fft, lambda_p=config["lambda_p"])
    
    # 语义偏转合成
    # 增加 alpha 权重，确保信号能穿透扩散过程
    weight = config["alpha"] * mask_s.unsqueeze(0).unsqueeze(0) * mask_p
    
    # 注入：在频域进行插值
    latents_fft_w = (1 - weight) * latents_fft + weight * target_patch
    
    # 逆变换回空域
    res_latents = torch.fft.ifft2(torch.fft.ifftshift(latents_fft_w, dim=(-2, -1)), dim=(-2, -1)).real
    return res_latents.to(dtype)

def detect_pai_watermark(image, pipe, config):
    device = DEVICE
    img_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float16) / 127.5 - 1.0
    with torch.no_grad():
        # 获取重构潜变量
        latents = pipe.vae.encode(img_tensor).latent_dist.mode() * 0.18215
    
    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.to(torch.float32), dim=(-2, -1)), dim=(-2, -1))
    
    # 获取目标 Patch
    generator = torch.Generator(device=device).manual_seed(SEED)
    real_part = torch.randn(latents_fft.shape, device=device, generator=generator)
    imag_part = torch.randn(latents_fft.shape, device=device, generator=generator)
    target_patch = torch.complex(real_part, imag_part)

    height = latents.shape[-2]
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, height, device=device), indexing='ij')
    dist = torch.sqrt(x**2 + y**2)
    mask_s = ((dist <= (config["low_r"] / (height/2))) & (dist > (1 / (height/2)))) | \
             ((dist <= (config["high_r"] / (height/2))) & (dist > ((config["high_r"]-2) / (height/2))))
    
    mask_bool = mask_s > 0
    if not mask_bool.any(): return 0.0
    
    # 提取对应频段的特征
    feat_lat = latents_fft[:, :, mask_bool].flatten()
    feat_tar = target_patch[:, :, mask_bool].flatten()
    
    # 标准化特征：
    feat_lat = (feat_lat - feat_lat.mean()) / (feat_lat.std() + 1e-8)
    feat_tar = (feat_tar - feat_tar.mean()) / (feat_tar.std() + 1e-8)
    
    # 计算复数余弦相似度（转化为实数向量）
    vec_lat = torch.view_as_real(feat_lat).flatten()
    vec_tar = torch.view_as_real(feat_tar).flatten()
    
    sim = F.cosine_similarity(vec_lat.view(1, -1), vec_tar.view(1, -1)).item()
    return abs(sim)

# 攻击模拟
def apply_attacks(image):
    attacks = {}
    img_array = np.array(image).astype(np.float32)
    # 高斯噪声
    noise = np.random.normal(0, 15, img_array.shape)
    attacks['noise'] = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
    # 模糊
    attacks['blur'] = image.filter(ImageFilter.GaussianBlur(radius=2))
    # JPEG
    temp_p = "temp_atk.jpg"
    image.save(temp_p, "JPEG", quality=60)
    attacks['jpeg'] = Image.open(temp_p)
    return attacks


def main():
    set_random_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    prompts = load_prompts(NUM_PROMPTS, PROMPTS_ID)
    print(f"Loading Model: {MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        scheduler=DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler"),
        torch_dtype=torch.float16, 
        safety_checker=None
    ).to(DEVICE)

    results_summary = {}

    for group_name, config in WATERMARK_CONFIGS.items():
        print(f"\n>>> Running Evaluation: {group_name}")
        group_path = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_path, exist_ok=True)
        
        scores = {"clean": [], "noise": [], "blur": [], "jpeg": [], "no_watermark": []}

        for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            # 1. 初始潜变量
            gen = torch.Generator(device=DEVICE).manual_seed(SEED + i)
            init_latents = torch.randn((1, 4, 64, 64), generator=gen, device=DEVICE, dtype=torch.float16)
            
            # 2. PAI 注入
            w_latents = inject_pai_watermark(init_latents, config)
            
            # 3. 推理生成
            with torch.autocast(DEVICE):
                img_w = pipe(prompt=prompt, latents=w_latents, num_inference_steps=INFERENCE_STEPS).images[0]
                img_no = pipe(prompt=prompt, latents=init_latents, num_inference_steps=INFERENCE_STEPS).images[0]
            
            # 4. 保存
            img_w.save(os.path.join(group_path, f"{i:03d}_w.png"))
            
            # 5. 攻击与检测
            atk_imgs = apply_attacks(img_w)
            scores["clean"].append(detect_pai_watermark(img_w, pipe, config))
            scores["no_watermark"].append(detect_pai_watermark(img_no, pipe, config))
            for k, img in atk_imgs.items():
                scores[k].append(detect_pai_watermark(img, pipe, config))

        # 统计
        group_stats = {k: float(np.mean(v)) for k, v in scores.items()}
        results_summary[group_name] = group_stats
        
        print(f"\n--- {group_name} Results ---")
        for k, v in group_stats.items():
            print(f"{k:<15}: {v:.4f}")

    with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
        json.dump(results_summary, f, indent=4)

if __name__ == "__main__":
    main()