"""SER Custom Diffusion evaluation.

- Load base SD1.5
- Load UNet K/V deltas (delta.bin)
- Load learned modifier_token embedding (learned_token.bin)
- Replace [V] in prompts with modifier_token

Metrics: LPIPS, SSIM, CPS, CPIS history avg

Usage:
  python evaluate_ser_custom_diffusion.py --test_json ... --sd15_path ... --delta_path ... --token_path ... --modifier_token <new1>
"""

import os
import json
import argparse
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import clip
from torchvision import transforms

from delta_utils import load_delta


@dataclass
class EvalConfig:
    test_json: str = "{SER_DATASET_BASE_PATH}/processed_masked/test.json"
    sd15_path: str = "{SD15_MODEL_PATH}"

    delta_path: str = "{SER_DATASET_BASE_PATH}/custom_diffusion_sd15/delta.bin"
    token_path: str = "{SER_DATASET_BASE_PATH}/custom_diffusion_sd15/learned_token.bin"

    target_token: str = "[V]"
    modifier_token: str = "<new1>"

    output_dir: str = "{SER_DATASET_BASE_PATH}/evaluation_results_custom_diffusion"
    generated_images_dir: str = "{SER_DATASET_BASE_PATH}/evaluation_generated_images_custom_diffusion"

    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512
    num_samples: int = None
    seed: int = 42


class SEREvaluatorCustomDiffusion:
    def __init__(self, config: EvalConfig):
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Warning: CUDA not available; evaluation will be very slow.")

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.generated_images_dir, exist_ok=True)

        self._init_metrics()
        self._load_model()
        self._load_test_data()

    def _init_metrics(self):
        self.lpips_metric = lpips.LPIPS(net="vgg").to(self.device).eval()
        self.ssim_metric = SSIM(data_range=2.0).to(self.device)

        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        self.clip_transform = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _load_model(self):
        from diffusers import StableDiffusionPipeline

        print(f"Loading SD1.5 from {self.config.sd15_path}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config.sd15_path,
            safety_checker=None,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)

        if self.config.delta_path and os.path.exists(self.config.delta_path):
            print(f"Loading delta from {self.config.delta_path}...")
            missing, unexpected = load_delta(self.pipe.unet, self.config.delta_path, map_location="cpu")
            if unexpected:
                print(f"Warning: unexpected keys in delta: {len(unexpected)}")
            if missing:
                # missing is expected because delta is partial
                pass

        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        token_id = tokenizer.convert_tokens_to_ids(self.config.modifier_token)
        if token_id == tokenizer.unk_token_id:
            tokenizer.add_tokens([self.config.modifier_token])
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(self.config.modifier_token)

        if self.config.token_path and os.path.exists(self.config.token_path):
            emb = torch.load(self.config.token_path, map_location="cpu", weights_only=False)
            if self.config.modifier_token in emb:
                vec = emb[self.config.modifier_token]
            else:
                vec = list(emb.values())[0]
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[token_id] = vec.to(text_encoder.get_input_embeddings().weight.dtype)

        print("Model loaded (base + delta + token).")

    def _load_test_data(self):
        with open(self.config.test_json, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)
        if self.config.num_samples is not None:
            self.test_data = self.test_data[: self.config.num_samples]
        print(f"Loaded {len(self.test_data)} samples")

    def _replace_token_in_prompt(self, prompt: str) -> str:
        return str(prompt).replace(self.config.target_token, self.config.modifier_token)

    def _image_to_tensor(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.image_size, self.config.image_size), Image.BICUBIC)
        tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        return tensor

    def _calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1_n = img1 * 2.0 - 1.0
        img2_n = img2 * 2.0 - 1.0
        with torch.no_grad():
            v = self.lpips_metric(img1_n, img2_n)
        return float(v.item())

    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1_n = img1 * 2.0 - 1.0
        img2_n = img2 * 2.0 - 1.0
        with torch.no_grad():
            v = self.ssim_metric(img1_n, img2_n)
        return float(v.item())

    def _clip_img_feat(self, img: torch.Tensor) -> torch.Tensor:
        img = self.clip_transform(img)
        with torch.no_grad():
            feat = self.clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def _calculate_clip_image_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        f1 = self._clip_img_feat(img1)
        f2 = self._clip_img_feat(img2)
        return float((f1 @ f2.T).mean().item())

    def _calculate_cps(self, generated_tensor: torch.Tensor, topic: str) -> float:
        image_features = self._clip_img_feat(generated_tensor)
        text_tokens = clip.tokenize([topic]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return float((image_features @ text_features.T).mean().item())

    def evaluate(self):
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        results = []
        all_metrics = {
            "lpips_target": [],
            "lpips_history_avg": [],
            "ssim_target": [],
            "ssim_history_avg": [],
            "cps": [],
            "cpis_history_avg": [],
        }

        for idx, sample in enumerate(tqdm(self.test_data, desc="Evaluating")):
            try:
                target_image_path = sample.get("target_image_path")
                target_caption = sample.get("target_caption", "")
                target_masked_caption = sample.get("target_masked_caption", "")

                if not target_image_path or not os.path.exists(target_image_path):
                    continue

                prompt = target_caption if target_caption else target_masked_caption
                if not prompt:
                    continue

                generation_prompt = self._replace_token_in_prompt(prompt)

                with torch.no_grad():
                    generated_image = self.pipe(
                        generation_prompt,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator=generator,
                    ).images[0]

                topic = sample.get("history_topic", "unknown")
                target_item_id = sample.get("target_item_id", f"sample_{idx}")
                save_path = os.path.join(
                    self.config.generated_images_dir,
                    f"{topic}_{str(target_item_id).replace('/', '_')}.png",
                )
                generated_image.save(save_path)

                target_tensor = self._image_to_tensor(target_image_path)
                generated_tensor = self._image_to_tensor(save_path)

                lpips_target = self._calculate_lpips(generated_tensor, target_tensor)
                ssim_target = self._calculate_ssim(generated_tensor, target_tensor)
                cps = self._calculate_cps(generated_tensor, topic)

                history_items = sample.get("history_items_info", [])[:10]
                lp_hist, ssim_hist, cpis_hist = [], [], []
                for hist_item in history_items:
                    hist_path = hist_item.get("image_path")
                    if hist_path and os.path.exists(hist_path):
                        try:
                            hist_tensor = self._image_to_tensor(hist_path)
                            lp_hist.append(self._calculate_lpips(generated_tensor, hist_tensor))
                            ssim_hist.append(self._calculate_ssim(generated_tensor, hist_tensor))
                            cpis_hist.append(self._calculate_clip_image_similarity(generated_tensor, hist_tensor))
                        except Exception:
                            pass

                lpips_history_avg = float(np.mean(lp_hist)) if lp_hist else None
                ssim_history_avg = float(np.mean(ssim_hist)) if ssim_hist else None
                cpis_history_avg = float(np.mean(cpis_hist)) if cpis_hist else None

                results.append(
                    {
                        "sample_idx": idx,
                        "topic": topic,
                        "target_item_id": target_item_id,
                        "prompt": prompt,
                        "generation_prompt": generation_prompt,
                        "generated_image_path": save_path,
                        "target_image_path": target_image_path,
                        "lpips_target": lpips_target,
                        "lpips_history_avg": lpips_history_avg,
                        "ssim_target": ssim_target,
                        "ssim_history_avg": ssim_history_avg,
                        "cps": cps,
                        "cpis_history_avg": cpis_history_avg,
                    }
                )

                all_metrics["lpips_target"].append(lpips_target)
                all_metrics["ssim_target"].append(ssim_target)
                all_metrics["cps"].append(cps)
                if lpips_history_avg is not None:
                    all_metrics["lpips_history_avg"].append(lpips_history_avg)
                if ssim_history_avg is not None:
                    all_metrics["ssim_history_avg"].append(ssim_history_avg)
                if cpis_history_avg is not None:
                    all_metrics["cpis_history_avg"].append(cpis_history_avg)

            except Exception as e:
                print(f"Error at sample {idx}: {e}")
                continue

        summary = {}
        for k, vs in all_metrics.items():
            summary[f"avg_{k}"] = float(np.mean(vs)) if vs else None
            summary[f"std_{k}"] = float(np.std(vs)) if vs else None

        out_path = os.path.join(self.config.output_dir, "ser_custom_diffusion_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results, "config": vars(self.config)}, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)
        print("Summary:", summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_json", type=str, default="{SER_DATASET_BASE_PATH}/processed_masked/test.json")
    p.add_argument("--sd15_path", type=str, default="{SD15_MODEL_PATH}")
    p.add_argument("--delta_path", type=str, default="{SER_DATASET_BASE_PATH}/custom_diffusion_sd15/delta.bin")
    p.add_argument("--token_path", type=str, default="{SER_DATASET_BASE_PATH}/custom_diffusion_sd15/learned_token.bin")
    p.add_argument("--target_token", type=str, default="[V]")
    p.add_argument("--modifier_token", type=str, default="<new1>")
    p.add_argument("--output_dir", type=str, default="{SER_DATASET_BASE_PATH}/evaluation_results_custom_diffusion")
    p.add_argument("--generated_images_dir", type=str, default="{SER_DATASET_BASE_PATH}/evaluation_generated_images_custom_diffusion")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = EvalConfig(
        test_json=args.test_json,
        sd15_path=args.sd15_path,
        delta_path=args.delta_path,
        token_path=args.token_path,
        target_token=args.target_token,
        modifier_token=args.modifier_token,
        output_dir=args.output_dir,
        generated_images_dir=args.generated_images_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    SEREvaluatorCustomDiffusion(cfg).evaluate()
