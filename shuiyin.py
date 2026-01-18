import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageFilter
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "D:/models/stable-diffusion-2-1-base" 
PROMPTS_ID = "D:/prompts/prompts_for_alignment.txt"
NUM_PROMPTS = 50  
INFERENCE_STEPS = 50
SEED = 42
IMAGE_SIZE = 512
OUTPUT_DIR = "tree_ring_results"

# 水印配置：包含单频段（低/高）与双频段（标准/高斯）共四种方法
WATERMARK_CONFIGS = {
    "Low-Freq": {
        "radius": 4, 
        "alpha": 0.3, 
        "method": "standard",
        "dual": False
    },
    "High-Freq": {
        "radius": 18, 
        "alpha": 0.5, 
        "method": "standard",
        "dual": False
    },
    "Dual-Ring-Standard": {
        "low_r": 4,
        "high_r": 18,
        "low_alpha": 0.25,
        "high_alpha": 0.5,
        "dual": True,
        "method": "standard"
    },
    "Dual-Ring-Gaussian": {
        "low_r": 4,
        "high_r": 18,
        "low_alpha": 0.25,
        "high_alpha": 0.3,
        "low_sigma": 1.5,
        "high_sigma": 1.0,
        "dual": True,
        "method": "gaussian"
    },
    "Dual-Ring-Gaussian_1": {
        "low_r": 4,
        "high_r": 18,
        "low_alpha": 0.25,   
        "high_alpha": 0.3,   
        "low_sigma": 1.5,    
        "high_sigma": 1.0,   
        "dual": True,
        "method": "gaussian"
    }
    
}

def set_random_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_prompts(num_prompts, prompt_file_path):
    prompts = []
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line: line = line.split(":", 1)[1].strip()
                if line: prompts.append(line)
    
    if len(prompts) < num_prompts:
        default = "A cinematic shot of a mystical forest, highly detailed, 8k"
        prompts += [default] * (num_prompts - len(prompts))
    return prompts[:num_prompts]

# 遮罩生成
def get_ring_mask(size, r_outer, r_inner=0, device="cpu"):
    x, y = torch.meshgrid(torch.linspace(-1, 1, size, device=device), torch.linspace(-1, 1, size, device=device), indexing='ij')
    dist = torch.sqrt(x**2 + y**2)
    mask = (dist <= (r_outer / (size / 2))) & (dist > (r_inner / (size / 2)))
    return mask

def get_gaussian_ring_mask(size, r, sigma, r_inner=0, device="cpu"):
    x, y = torch.meshgrid(torch.linspace(-1, 1, size, device=device), torch.linspace(-1, 1, size, device=device), indexing='ij')
    dist = torch.sqrt(x**2 + y**2)
    center_dist = r / (size / 2)
    norm_sigma = sigma / (size / 2)
    weight = torch.exp(-((dist - center_dist)**2) / (2 * norm_sigma**2))
    if r_inner > 0:
        weight = weight * (dist > (r_inner / (size / 2)))
    return weight

# 注入与检测
def get_target_patch(shape, device):
    """根据固定种子生成目标水印 Patch"""
    generator = torch.Generator(device=device).manual_seed(SEED)
    patch = torch.randn(shape, device=device, generator=generator)
    return patch.to(torch.complex64)

def inject_watermark(init_latents, config):
    device = init_latents.device
    dtype = init_latents.dtype
    _, _, height, _ = init_latents.shape
    
    latents_f32 = init_latents.to(torch.float32)
    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents_f32, dim=(-2, -1)), dim=(-2, -1))
    
    latent_std = latents_fft.std(dim=(-2, -1), keepdim=True)
    target_patch = get_target_patch(latents_fft.shape, device) * latent_std

    method = config.get("method", "standard")
    
    if config.get("dual", False):
        if method == "gaussian":
            low_sigma = config.get("low_sigma", 2.0)
            high_sigma = config.get("high_sigma", 2.0)
            w_low = get_gaussian_ring_mask(height, config["low_r"], low_sigma, r_inner=1, device=device)
            w_high = get_gaussian_ring_mask(height, config["high_r"], high_sigma, device=device)
            
            latents_fft = (1 - config["low_alpha"] * w_low) * latents_fft + (config["low_alpha"] * w_low) * target_patch
            latents_fft = (1 - config["high_alpha"] * w_high) * latents_fft + (config["high_alpha"] * w_high) * target_patch
        else:
            mask_low = get_ring_mask(height, config["low_r"], r_inner=1, device=device)
            mask_high = get_ring_mask(height, config["high_r"], r_inner=config["high_r"]-2, device=device)
            
            latents_fft[:, :, mask_low] = (1 - config["low_alpha"]) * latents_fft[:, :, mask_low] + config["low_alpha"] * target_patch[:, :, mask_low]
            latents_fft[:, :, mask_high] = (1 - config["high_alpha"]) * latents_fft[:, :, mask_high] + config["high_alpha"] * target_patch[:, :, mask_high]
    else:
        # 单频段注入逻辑
        r = config["radius"]
        alpha = config["alpha"]
        mask = get_ring_mask(height, r, r_inner=1 if r > 1 else 0, device=device)
        latents_fft[:, :, mask] = (1 - alpha) * latents_fft[:, :, mask] + alpha * target_patch[:, :, mask]
            
    res_latents = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-2, -1)), dim=(-2, -1)).real
    return res_latents.to(dtype)

def detect_watermark(image, pipe, config):
    """从生成的图像中反推并检测水印"""
    device = DEVICE
    img_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float16) / 127.5 - 1.0
    with torch.no_grad():
        latents = pipe.vae.encode(img_tensor).latent_dist.sample() * 0.18215
    
    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents.to(torch.float32), dim=(-2, -1)), dim=(-2, -1))
    target_patch = get_target_patch(latents_fft.shape, device)
    height = latents.shape[-2]
    
    # 准备权重掩码 (Weight Mask)
    if config.get("dual", False):
        if config.get("method") == "gaussian":
            mask_weight = torch.max(
                get_gaussian_ring_mask(height, config["low_r"], config.get("low_sigma", 1.5), r_inner=1, device=device),
                get_gaussian_ring_mask(height, config["high_r"], config.get("high_sigma", 1.0), device=device)
            )
        else:
            mask_weight = torch.max(
                get_ring_mask(height, config["low_r"], r_inner=1, device=device),
                get_ring_mask(height, config["high_r"], r_inner=config["high_r"]-2, device=device)
            ).float()
    else:
        r = config["radius"]
        mask_weight = get_ring_mask(height, r, r_inner=1 if r > 1 else 0, device=device).float()
        
    
    # 只要权重大于一个极小值，就认为该点参与计算
    bool_mask = mask_weight > 0.001
    
    if not bool_mask.any(): return 0.0
    
    # 提取特征并应用权重提升检测精度
    vec_target = target_patch[:, :, bool_mask].flatten()
    vec_latents = latents_fft[:, :, bool_mask].flatten()
    vec_weight = mask_weight[bool_mask].repeat(latents_fft.shape[1]) 

    # 将复数拆分为实部和虚部
    target_flat = torch.cat([vec_target.real, vec_target.imag])
    latents_flat = torch.cat([vec_latents.real, vec_latents.imag])
    weight_flat = torch.cat([vec_weight, vec_weight])

    # 加权余弦相似度计算
    # 先乘权重再计算，使重要区域贡献更大
    weighted_target = target_flat * weight_flat
    weighted_latents = latents_flat * weight_flat
    
    cos_sim = F.cosine_similarity(weighted_target.view(1, -1), weighted_latents.view(1, -1), dim=1).item()
    return abs(cos_sim)

# 攻击模拟
def apply_attacks(image):
    """模拟常见的图像攻击"""
    attacks = {}
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 15, img_array.shape)
    attacks['noise'] = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
    attacks['blur'] = image.filter(ImageFilter.GaussianBlur(radius=2))
    temp_path = "temp_attack.jpg"
    image.save(temp_path, "JPEG", quality=50)
    attacks['jpeg'] = Image.open(temp_path)
    return attacks

def load_pipeline():
    print(f"正在从 {MODEL_ID} 加载模型...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, scheduler=scheduler, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    ).to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe

def main():
    set_random_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    prompts = load_prompts(NUM_PROMPTS, PROMPTS_ID)
    pipe = load_pipeline()

    for group_name, config in WATERMARK_CONFIGS.items():
        print(f"\n>>> 正在处理分组: {group_name}")
        group_path = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_path, exist_ok=True)
        
        detection_scores = {"clean": [], "noise": [], "blur": [], "jpeg": [], "none": []}

        for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            torch.cuda.empty_cache() 
            
            generator = torch.Generator(device=DEVICE).manual_seed(SEED + i)
            shape = (1, pipe.unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
            init_latents = torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.float16)
            
            watermarked_latents = inject_watermark(init_latents, config)
            with torch.autocast(DEVICE):
                out_w = pipe(prompt=prompt, latents=watermarked_latents, num_inference_steps=INFERENCE_STEPS).images[0]
                out_no = pipe(prompt=prompt, latents=init_latents, num_inference_steps=INFERENCE_STEPS).images[0]
            
            attacked_images = apply_attacks(out_w)
            
            score_clean = detect_watermark(out_w, pipe, config)
            score_none = detect_watermark(out_no, pipe, config) 
            
            detection_scores["clean"].append(score_clean)
            detection_scores["none"].append(score_none)
            
            for atk_name, atk_img in attacked_images.items():
                score_atk = detect_watermark(atk_img, pipe, config)
                detection_scores[atk_name].append(score_atk)
            
            out_w.save(os.path.join(group_path, f"{i:03d}_w.png"))

        print(f"\n--- {group_name} 检测统计 ---")
        print(f"{'攻击类型':<15} | {'平均得分':<10}")
        print("-" * 30)
        for k, v in detection_scores.items():
            avg_score = np.mean(v)
            print(f"{k:<15} | {avg_score:.4f}")
            
    print(f"\n评估已完成。")

if __name__ == "__main__":
    main()