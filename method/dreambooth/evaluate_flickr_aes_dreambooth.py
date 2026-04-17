"""FLICKR-AES DreamBooth (LoRA) evaluation.

This is a light adaptation of method/textual_inversion/evaluate_flickr_aes.py:
- Load base SD1.5
- Load UNet LoRA
- Load learned instance_token embedding
- Replace [V] with instance_token in prompts

Metrics implemented here (subset to keep it robust):
- LPIPS / SSIM (vs a chosen target image if present)
- CLIP similarities (CPIS history avg)
- Optional Verifier score if verifier checkpoints are provided

Note: FLICKR-AES processed_dataset samples typically contain `interaction_sequence` (history).
This evaluator generates one image per sample using a prompt sampled from masked captions.
"""

import os
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import clip
from torchvision import transforms

from lora_utils import load_lora_into_pipeline


@dataclass
class EvalConfig:
    test_json: str = "{FLICKR_AES_BASE_PATH}/processed_dataset/test.json"
    sd15_path: str = "{SD15_MODEL_PATH}"

    lora_dir: str = "{FLICKR_AES_BASE_PATH}/dreambooth_lora_sd15"
    token_path: str = "{FLICKR_AES_BASE_PATH}/dreambooth_lora_sd15/learned_token.bin"

    target_token: str = "[V]"
    instance_token: str = "sks"

    output_dir: str = "{FLICKR_AES_BASE_PATH}/evaluation_results_dreambooth"
    generated_images_dir: str = "{FLICKR_AES_BASE_PATH}/evaluation_generated_images_dreambooth"

    images_dir: str = "{FLICKR_AES_BASE_PATH}/40K"
    captions_json: str = "{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json"

    verifier_model_path: str = "{FLICKR_AES_BASE_PATH}/verifier_checkpoints/best_model.pth"
    verifier_user_map_path: str = "{FLICKR_AES_BASE_PATH}/verifier_checkpoints/user_map.json"

    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512
    num_samples: int = None
    seed: int = 42


class VerifierScorer:
    def __init__(self, model_path: str, user_map_path: str, device: str = "cuda"):
        self.device = torch.device(device)

        if not (os.path.exists(model_path) and os.path.exists(user_map_path)):
            raise FileNotFoundError("Verifier files missing")

        with open(user_map_path, "r", encoding="utf-8") as f:
            self.user_map = json.load(f)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        num_users = len(self.user_map)
        input_dim = checkpoint["config"]["input_dim"]
        user_emb_dim = checkpoint["config"]["user_emb_dim"]
        hidden_dim = checkpoint["config"]["hidden_dim"]
        dropout = checkpoint["config"]["dropout"]

        # Import verifier net from method/data_process
        import sys
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from data_process.train_verifier import VerifierNet

        self.model = VerifierNet(
            num_users=num_users,
            input_img_dim=input_dim,
            user_emb_dim=user_emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

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
        self.input_dim = input_dim

    def score_image(self, user_id: str, image_path: str) -> Optional[float]:
        if user_id not in self.user_map:
            return None
        if not os.path.exists(image_path):
            return None

        user_idx = self.user_map[user_id]
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            image_tensor = self.clip_transform(image_tensor)
            feat = self.clip_model.encode_image(image_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

            # adjust dim
            if feat.shape[-1] != self.input_dim:
                if feat.shape[-1] > self.input_dim:
                    feat = feat[:, : self.input_dim]
                else:
                    pad = torch.zeros((feat.shape[0], self.input_dim - feat.shape[-1]), device=feat.device)
                    feat = torch.cat([feat, pad], dim=-1)

            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            score = self.model(user_tensor, feat)
            return float(score.item())


class FlickrAESEvaluatorDreamBooth:
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
        self._load_captions()
        self._maybe_init_verifier()

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

        # Load LoRA
        print(f"Loading LoRA from {self.config.lora_dir}...")
        ok = load_lora_into_pipeline(self.pipe, self.config.lora_dir)
        if not ok:
            raise FileNotFoundError(f"Failed to load LoRA from: {self.config.lora_dir}")

        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        token_id = tokenizer.convert_tokens_to_ids(self.config.instance_token)
        if token_id == tokenizer.unk_token_id:
            tokenizer.add_tokens([self.config.instance_token])
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(self.config.instance_token)

        if self.config.token_path and os.path.exists(self.config.token_path):
            emb = torch.load(self.config.token_path, map_location="cpu")
            vec = emb.get(self.config.instance_token, list(emb.values())[0])
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[token_id] = vec.to(
                    text_encoder.get_input_embeddings().weight.dtype
                )

        print("Model loaded (base + LoRA + token).")

    def _load_test_data(self):
        with open(self.config.test_json, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)
        if self.config.num_samples is not None:
            self.test_data = self.test_data[: self.config.num_samples]
        print(f"Loaded {len(self.test_data)} samples")

    def _load_captions(self):
        self.masked_captions: Dict[str, str] = {}
        if os.path.exists(self.config.captions_json):
            with open(self.config.captions_json, "r", encoding="utf-8") as f:
                self.masked_captions = json.load(f)

    def _maybe_init_verifier(self):
        self.verifier = None
        try:
            self.verifier = VerifierScorer(
                self.config.verifier_model_path,
                self.config.verifier_user_map_path,
                device=str(self.device),
            )
            print("Verifier enabled")
        except Exception as e:
            print(f"Verifier disabled: {e}")
            self.verifier = None

    def _replace_token_in_prompt(self, prompt: str) -> str:
        return str(prompt).replace(self.config.target_token, self.config.instance_token)

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

    def evaluate(self):
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        results = []
        all_metrics = {
            "cpis_history_avg": [],
            "verifier_score": [],
        }

        for idx, sample in enumerate(tqdm(self.test_data, desc="Evaluating")):
            try:
                user_id = sample.get("user_id", "unknown")
                seq = sample.get("interaction_sequence", [])
                if not seq:
                    continue

                # pick an item that has a masked caption
                prompt = ""
                picked_item_id = None
                for _ in range(20):
                    it = random.choice(seq)
                    item_id = it.get("item_id", "")
                    key = item_id[:-4] if item_id.endswith(".jpg") else item_id
                    cap = self.masked_captions.get(key, "")
                    if cap:
                        prompt = cap
                        picked_item_id = item_id
                        break

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

                save_path = os.path.join(
                    self.config.generated_images_dir,
                    f"{user_id}_{str(picked_item_id or idx).replace('/', '_')}.png",
                )
                generated_image.save(save_path)

                generated_tensor = self._image_to_tensor(save_path)

                # CPIS vs history
                cpis_hist = []
                for hist in seq[:10]:
                    hist_path = hist.get("image_path")
                    if hist_path and os.path.exists(hist_path):
                        try:
                            hist_tensor = self._image_to_tensor(hist_path)
                            cpis_hist.append(self._calculate_clip_image_similarity(generated_tensor, hist_tensor))
                        except Exception:
                            pass
                cpis_history_avg = float(np.mean(cpis_hist)) if cpis_hist else None

                verifier_score = None
                if self.verifier is not None:
                    verifier_score = self.verifier.score_image(user_id, save_path)

                results.append(
                    {
                        "sample_idx": idx,
                        "user_id": user_id,
                        "picked_item_id": picked_item_id,
                        "prompt": prompt,
                        "generation_prompt": generation_prompt,
                        "generated_image_path": save_path,
                        "cpis_history_avg": cpis_history_avg,
                        "verifier_score": verifier_score,
                    }
                )

                if cpis_history_avg is not None:
                    all_metrics["cpis_history_avg"].append(cpis_history_avg)
                if verifier_score is not None:
                    all_metrics["verifier_score"].append(verifier_score)

            except Exception as e:
                print(f"Error at sample {idx}: {e}")
                continue

        summary = {}
        for k, vs in all_metrics.items():
            summary[f"avg_{k}"] = float(np.mean(vs)) if vs else None
            summary[f"std_{k}"] = float(np.std(vs)) if vs else None

        out_path = os.path.join(self.config.output_dir, "flickr_aes_dreambooth_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results, "config": vars(self.config)}, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)
        print("Summary:", summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_json", type=str, default="{FLICKR_AES_BASE_PATH}/processed_dataset/test.json")
    p.add_argument("--sd15_path", type=str, default="{SD15_MODEL_PATH}")
    p.add_argument("--lora_dir", type=str, default="{FLICKR_AES_BASE_PATH}/dreambooth_lora_sd15")
    p.add_argument("--token_path", type=str, default="{FLICKR_AES_BASE_PATH}/dreambooth_lora_sd15/learned_token.bin")
    p.add_argument("--target_token", type=str, default="[V]")
    p.add_argument("--instance_token", type=str, default="sks")
    p.add_argument("--output_dir", type=str, default="{FLICKR_AES_BASE_PATH}/evaluation_results_dreambooth")
    p.add_argument("--generated_images_dir", type=str, default="{FLICKR_AES_BASE_PATH}/evaluation_generated_images_dreambooth")
    p.add_argument("--images_dir", type=str, default="{FLICKR_AES_BASE_PATH}/40K")
    p.add_argument("--captions_json", type=str, default="{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json")
    p.add_argument("--verifier_model_path", type=str, default="{FLICKR_AES_BASE_PATH}/verifier_checkpoints/best_model.pth")
    p.add_argument("--verifier_user_map_path", type=str, default="{FLICKR_AES_BASE_PATH}/verifier_checkpoints/user_map.json")
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
        lora_dir=args.lora_dir,
        token_path=args.token_path,
        target_token=args.target_token,
        instance_token=args.instance_token,
        output_dir=args.output_dir,
        generated_images_dir=args.generated_images_dir,
        images_dir=args.images_dir,
        captions_json=args.captions_json,
        verifier_model_path=args.verifier_model_path,
        verifier_user_map_path=args.verifier_user_map_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    FlickrAESEvaluatorDreamBooth(cfg).evaluate()
