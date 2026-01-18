import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageFilter
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import json
from typing import Union

# 基础配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "D:/models/stable-diffusion-2-1-base"
PROMPTS_ID = "D:/prompts/prompts_for_alignment.txt"
NUM_PROMPTS = 50
INFERENCE_STEPS = 50
SEED = 42
OUTPUT_DIR = "pai_results"

# 水印配置
WATERMARK_CONFIGS = {
    "PAI": {
        "wm_steps": 10,
        "start_wm_step": 20,
        
        # 水印强度
        "wm_strength": 0.15,
        
        # 频段选择
        "ring_r_min": 6,
        "ring_r_max": 14,
        
        # 密钥种子
        "key_seed": 12345,
        
        # CFG
        "guidance_scale": 7.5,
    }
}


def set_random_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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

class PAIFrequencyDomainV4:
    """
    频域水印 - 基于频域能量注入
    
    核心思想：
    1. 生成固定的频域pattern（在特定频率位置有确定的复数值）
    2. 在去噪过程中，将pattern加到pred_x0上
    3. 检测时，计算图像频域与pattern的相关性
    """
    
    def __init__(self, model_name, config, device=DEVICE):
        self.device = device
        self.config = config
        self.num_ddim_steps = INFERENCE_STEPS
        self.GUIDANCE_SCALE = config["guidance_scale"]
        self.wm_strength = config["wm_strength"]
        self.wm_steps = config["wm_steps"]
        self.start_wm_step = config["start_wm_step"]
        
        # 水印组件
        self.wm_pattern_latent = None
        self.freq_mask = None 
        self.wm_timesteps = []
        self.context = None
        self.prompt = None
        
        self._load_model(model_name)
    
    def _load_model(self, model_name):
        print(f"Loading model from {model_name}")
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet"
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        self.scheduler.set_timesteps(self.num_ddim_steps)
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
        print("Model loaded successfully")
    
    def _create_frequency_mask(self, height, width):
        """
        创建2D频域掩码 [H, W]
        """
        y, x = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32) - height // 2,
            torch.arange(width, device=self.device, dtype=torch.float32) - width // 2,
            indexing='ij'
        )
        dist = torch.sqrt(x**2 + y**2)
        r_min = self.config["ring_r_min"]
        r_max = self.config["ring_r_max"]
        mask = ((dist >= r_min) & (dist <= r_max)).float()
        
        # 排除DC及其附近
        dc_exclude = dist < 3
        mask = mask * (~dc_exclude).float()
        return mask  # [H, W]
    
    def _generate_watermark_pattern(self, latent_shape):
        """
        生成水印pattern
        在频域的特定位置放置固定的复数值，然后转换到空域
        """
        batch, channels, height, width = latent_shape
        
        # 创建2D频域掩码
        self.freq_mask = self._create_frequency_mask(height, width) 
        # 使用固定种子生成pattern
        torch.manual_seed(self.config["key_seed"])
        # 在频域创建pattern [1, channels, H, W]
        freq_pattern = torch.zeros((1, channels, height, width), 
                                    device=self.device, dtype=torch.complex64)
        mask_2d = self.freq_mask > 0.5  # [H, W] bool
        num_points = mask_2d.sum().item()
        
        if num_points == 0:
            print("Warning: No frequency points selected")
            self.wm_pattern_latent = torch.zeros((batch, channels, height, width), 
                                                   device=self.device, dtype=torch.float32)
            return self.wm_pattern_latent
        
        # 生成固定的随机相位pattern（每个通道相同以增强鲁棒性）
        torch.manual_seed(self.config["key_seed"])
        phases = torch.rand(num_points, device=self.device) * 2 * np.pi
        # 固定幅度
        amplitude = 1.0
        complex_values = amplitude * torch.exp(1j * phases)
        
        # 对所有通道使用相同的pattern
        for c in range(channels):
            freq_pattern[0, c, mask_2d] = complex_values
        # 强制共轭对称（确保IFFT后为实数）
        freq_pattern = self._make_hermitian_symmetric(freq_pattern)
        # 转换到空域
        spatial_pattern = torch.fft.ifft2(
            torch.fft.ifftshift(freq_pattern, dim=(-2, -1)),
            dim=(-2, -1)
        ).real
        # 归一化
        max_val = spatial_pattern.abs().max()
        if max_val > 1e-8:
            spatial_pattern = spatial_pattern / max_val
        self.wm_pattern_latent = spatial_pattern.expand(batch, -1, -1, -1).clone()
        return self.wm_pattern_latent
    
    def _make_hermitian_symmetric(self, freq_tensor):
        """
        强制频域张量具有Hermitian对称性
        确保IFFT后的结果为实数
        """
        h, w = freq_tensor.shape[-2:]
        result = freq_tensor.clone()
        
        for i in range(h):
            for j in range(w):
                i_sym = (h - i) % h
                j_sym = (w - j) % w
                if i == i_sym and j == j_sym:
                    # DC或Nyquist点：虚部为0
                    result[..., i, j] = result[..., i, j].real.to(result.dtype)
                elif i < i_sym or (i == i_sym and j < j_sym):
                    # 只处理一半，另一半设为共轭
                    result[..., i_sym, j_sym] = result[..., i, j].conj()
        return result
    
    def _update_wm_timesteps(self):
        """更新水印时间步"""
        self.wm_timesteps = []
        for i in range(self.wm_steps):
            step_idx = self.start_wm_step + i
            if step_idx < len(self.scheduler.timesteps):
                t = self.scheduler.timesteps[step_idx]
                self.wm_timesteps.append(t.item() if torch.is_tensor(t) else t)
        
    
    @torch.no_grad()
    def init_prompt(self, prompt):
        self.prompt = prompt
        text_input = self.tokenizer(
            [prompt], padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]
        uncond_input = self.tokenizer(
            [""], padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
    
    def prev_step_with_watermark(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        inject_watermark: bool = False
    ) -> torch.FloatTensor:
        """DDIM步骤 + 水印注入"""
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(self.device)
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep].to(self.device)
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod.to(self.device)
        )
        beta_prod_t = 1 - alpha_prod_t
        
        # 预测原始样本
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        # 水印注入
        if inject_watermark and self.wm_pattern_latent is not None:
            pred_original_sample = self._inject_watermark(pred_original_sample)
            
        # DDIM步骤
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample
    
    def _inject_watermark(self, pred_x0: torch.FloatTensor) -> torch.FloatTensor:
        """水印注入"""
        wm_pattern = self.wm_pattern_latent.to(pred_x0.dtype)
        # 自适应强度：根据pred_x0的标准差调整
        adaptive_strength = self.wm_strength * pred_x0.std()
        watermarked = pred_x0 + adaptive_strength * wm_pattern
        
        return watermarked
    
    @torch.no_grad()
    def get_noise_pred(self, latents, t):
        uncond_emb, cond_emb = self.context.chunk(2)
        noise_pred_uncond = self.unet(latents, t, uncond_emb)["sample"]
        noise_pred_cond = self.unet(latents, t, cond_emb)["sample"]
        noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
        
        return noise_pred
    
    @torch.no_grad()
    def generate(self, prompt: str, init_latents: torch.FloatTensor, inject_watermark: bool = True):
        self.init_prompt(prompt)
        # 生成水印pattern
        self._generate_watermark_pattern(init_latents.shape)
        self._update_wm_timesteps()
        latents = init_latents.clone()
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Generating", leave=False)):
            latents_input = latents.to(self.unet.dtype)
            noise_pred = self.get_noise_pred(latents_input, t)
            t_value = t.item() if torch.is_tensor(t) else t
            should_inject = inject_watermark and (t_value in self.wm_timesteps)
            
            latents = self.prev_step_with_watermark(
                noise_pred.to(torch.float32),
                t_value,
                latents.to(torch.float32),
                inject_watermark=should_inject
            )
        return self.latent2image(latents)
    
    @torch.no_grad()
    def latent2image(self, latents, return_type='pil'):
        latents = latents / self.vae.config.scaling_factor
        latents = latents.to(self.vae.dtype)
        
        image = self.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).astype(np.uint8)
        
        if return_type == 'pil':
            return Image.fromarray(image)
        return image
    
    @torch.no_grad()
    def image2latent(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=self.device, dtype=self.vae.dtype)
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * self.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def detect(self, image) -> float:
        """
        频域相关性检测
        方法：
        1. 将图像编码到latent空间
        2. 计算latent的频域表示
        3. 在目标频段计算与水印pattern的归一化互相关
        """
        latents = self.image2latent(image)
        
        # 确保pattern已初始化
        if self.wm_pattern_latent is None or self.wm_pattern_latent.shape != latents.shape:
            self._generate_watermark_pattern(latents.shape)
        latents_f32 = latents.to(torch.float32)
        # 计算latent的频域表示 [B, C, H, W]
        latent_fft = torch.fft.fftshift(
            torch.fft.fft2(latents_f32, dim=(-2, -1)),
            dim=(-2, -1)
        )
        # 计算pattern的频域表示
        pattern_fft = torch.fft.fftshift(
            torch.fft.fft2(self.wm_pattern_latent, dim=(-2, -1)),
            dim=(-2, -1)
        )
        
        # 获取2D掩码 [H, W]
        mask_2d = self.freq_mask > 0.5  # [H, W] bool
        if not mask_2d.any():
            return 0.0
        # 逐通道计算相关性，然后平均
        correlations = []
        for c in range(latent_fft.shape[1]):
            # 提取当前通道在掩码位置的值
            lat_c = latent_fft[0, c, mask_2d] 
            pat_c = pattern_fft[0, c, mask_2d] 
            if lat_c.numel() == 0:
                continue
            # 复数内积相关性
            inner_prod = (lat_c * pat_c.conj()).sum()
            norm_lat = torch.sqrt((lat_c.abs() ** 2).sum() + 1e-8)
            norm_pat = torch.sqrt((pat_c.abs() ** 2).sum() + 1e-8)
            corr = (inner_prod.abs() / (norm_lat * norm_pat)).item()
            correlations.append(corr)
        if len(correlations) == 0:
            return 0.0
        freq_corr = np.mean(correlations)
    
        # 空域相关
        lat_flat = latents_f32.flatten()
        pat_flat = self.wm_pattern_latent.flatten()
        lat_norm = (lat_flat - lat_flat.mean()) / (lat_flat.std() + 1e-8)
        pat_norm = (pat_flat - pat_flat.mean()) / (pat_flat.std() + 1e-8)
        spatial_corr = (lat_norm * pat_norm).mean().abs().item()
        
        # 计算目标频段的能量占比
        total_energy = (latent_fft.abs() ** 2).sum().item()
        target_energy = 0
        for c in range(latent_fft.shape[1]):
            target_energy += (latent_fft[0, c, mask_2d].abs() ** 2).sum().item()
        energy_ratio = target_energy / (total_energy + 1e-8)
        # 综合得分
        score = 0.5 * freq_corr + 0.3 * spatial_corr + 0.2 * min(energy_ratio * 20, 1.0)
        
        return float(score)


# 无水印
class BaselineGenerator:
    def __init__(self, pai_model):
        self.pai = pai_model
    @torch.no_grad()
    def generate(self, prompt: str, init_latents: torch.FloatTensor):
        return self.pai.generate(prompt, init_latents, inject_watermark=False)

# 攻击模拟
def apply_attacks(image):
    attacks = {}
    img_array = np.array(image).astype(np.float32)
    # 高斯噪声
    noise = np.random.normal(0, 15, img_array.shape)
    attacks['noise'] = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
    # 高斯模糊
    attacks['blur'] = image.filter(ImageFilter.GaussianBlur(radius=2))
    # JPEG压缩
    temp_path = "temp_attack.jpg"
    image.save(temp_path, "JPEG", quality=60)
    attacks['jpeg'] = Image.open(temp_path).copy()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    # 亮度调整
    from PIL import ImageEnhance
    attacks['brightness'] = ImageEnhance.Brightness(image).enhance(1.3)
    # 缩放攻击
    w, h = image.size
    attacks['resize'] = image.resize((w//2, h//2)).resize((w, h), Image.LANCZOS)
    # 裁剪
    crop_size = int(min(w, h) * 0.1)
    attacks['crop'] = image.crop(
        (crop_size, crop_size, w-crop_size, h-crop_size)
    ).resize((w, h), Image.LANCZOS)
    
    return attacks

# 评估指标
def calculate_psnr_ssim(img1, img2):
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    
    mse = np.mean((arr1 - arr2) ** 2)
    psnr = 50.0 if mse < 1e-10 else 20 * np.log10(255.0 / np.sqrt(mse))
    
    c1, c2 = 6.5025, 58.5225
    mu1, mu2 = arr1.mean(), arr2.mean()
    s1, s2 = arr1.std(), arr2.std()
    cov = np.mean((arr1 - mu1) * (arr2 - mu2))
    ssim = ((2*mu1*mu2 + c1) * (2*cov + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2))
    return psnr, ssim


def calculate_auc(wm_scores, no_wm_scores):
    all_scores = np.concatenate([wm_scores, no_wm_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)
    
    tprs, fprs = [], []
    for th in thresholds:
        tp = np.sum(wm_scores >= th)
        fn = np.sum(wm_scores < th)
        fp = np.sum(no_wm_scores >= th)
        tn = np.sum(no_wm_scores < th)
        tprs.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))
    
    auc = 0
    for i in range(1, len(fprs)):
        auc += (fprs[i-1] - fprs[i]) * (tprs[i-1] + tprs[i]) / 2
    return abs(auc)


def main():
    set_random_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    prompts = load_prompts(NUM_PROMPTS, PROMPTS_ID)
    
    all_results = {}
    
    for cfg_name, cfg in WATERMARK_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Configuration: {cfg_name}")
        print(f"{'='*70}")
        print("Parameters:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print()
        
        out_dir = os.path.join(OUTPUT_DIR, cfg_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # 使用V4版本
        pai = PAIFrequencyDomainV4(MODEL_ID, cfg, device=DEVICE)
        baseline = BaselineGenerator(pai)
        
        scores = {
            "watermarked": [],
            "clean": [],
            "noise": [], "blur": [], "jpeg": [],
            "brightness": [], "resize": [], "crop": []
        }
        quality = {"psnr": [], "ssim": []}
        
        for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Evaluating"):
            gen = torch.Generator(device=DEVICE).manual_seed(SEED + i)
            init_latents = torch.randn(
                (1, 4, 64, 64),
                generator=gen,
                device=DEVICE,
                dtype=torch.float32
            )
            
            img_wm = pai.generate(prompt, init_latents, inject_watermark=True)
            img_clean = baseline.generate(prompt, init_latents)
            
            img_wm.save(os.path.join(out_dir, f"{i:03d}_watermarked.png"))
            img_clean.save(os.path.join(out_dir, f"{i:03d}_clean.png"))
            
            psnr, ssim = calculate_psnr_ssim(img_wm, img_clean)
            quality["psnr"].append(psnr)
            quality["ssim"].append(ssim)
            
            scores["watermarked"].append(pai.detect(img_wm))
            scores["clean"].append(pai.detect(img_clean))
            
            for atk_name, atk_img in apply_attacks(img_wm).items():
                scores[atk_name].append(pai.detect(atk_img))
        wm_scores = np.array(scores["watermarked"])
        clean_scores = np.array(scores["clean"])
        
        stats = {}
        for k, v in scores.items():
            v = np.array(v)
            stats[k] = {"mean": float(v.mean()), "std": float(v.std())}
        
        stats["psnr"] = float(np.mean(quality["psnr"]))
        stats["ssim"] = float(np.mean(quality["ssim"]))
        stats["gap"] = stats["watermarked"]["mean"] - stats["clean"]["mean"]
        stats["auc"] = calculate_auc(wm_scores, clean_scores)
        
        best_threshold = (stats["watermarked"]["mean"] + stats["clean"]["mean"]) / 2
        tpr = np.sum(wm_scores >= best_threshold) / len(wm_scores)
        tnr = np.sum(clean_scores < best_threshold) / len(clean_scores)
        stats["tpr"] = float(tpr)
        stats["tnr"] = float(tnr)
        stats["threshold"] = float(best_threshold)
        
        all_results[cfg_name] = stats
        
        # 打印结果
        print(f"Results for {cfg_name}")
        
        print(f"\nImage Quality")
        print(f"  PSNR: {stats['psnr']:.2f} dB")
        print(f"  SSIM: {stats['ssim']:.4f}")
        
        print(f"\nDetection Performance")
        print(f"  Watermarked: {stats['watermarked']['mean']:.4f} ± {stats['watermarked']['std']:.4f}")
        print(f"  Clean:       {stats['clean']['mean']:.4f} ± {stats['clean']['std']:.4f}")
        print(f"  Gap:         {stats['gap']:.4f}")
        print(f"  AUC:         {stats['auc']:.4f}")
        print(f"  Threshold:   {stats['threshold']:.4f}")
        print(f"  TPR:         {stats['tpr']:.2%}")
        print(f"  TNR:         {stats['tnr']:.2%}")
        
        print(f"\nRobustness")
        for atk_name in ["noise", "blur", "jpeg", "brightness", "resize", "crop"]:
            s = stats[atk_name]
            print(f"  {atk_name:<12}: {s['mean']:.4f} ± {s['std']:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()