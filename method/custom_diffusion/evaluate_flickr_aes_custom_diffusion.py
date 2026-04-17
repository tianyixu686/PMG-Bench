"""FLICKR-AES Custom Diffusion evaluation.

- Load base SD1.5
- Load delta.bin (UNet cross-attn K/V)
- Load learned modifier_token embedding
- Replace [V] with modifier_token

Metrics (subset): LPIPS/SSIM (if target available), CLIP history similarity, optional verifier.
"""

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Optional

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
    test_json: str = "{FLICKR_AES_BASE_PATH}/processed_dataset/test.json"
    sd15_path: str = "{SD15_MODEL_PATH}"

    delta_path: str = "{FLICKR_AES_BASE_PATH}/custom_diffusion_sd15/delta.bin"
    token_path: str = "{FLICKR_AES_BASE_PATH}/custom_diffusion_sd15/learned_token.bin"

    target_token: str = "[V]"
    modifier_token: str = "<new1>"

    output_dir: str = "{FLICKR_AES_BASE_PATH}/evaluation_results_custom_diffusion"
    generated_images_dir: str = "{FLICKR_AES_BASE_PATH}/evaluation_generated_images_custom_diffusion"

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
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            score = self.model(feat, user_tensor)
        return float(score.item())


class FlickrAESEvaluatorCustomDiffusion:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.generated_images_dir, exist_ok=True)

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

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.sd15_path,
            safety_checker=None,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)

        if self.cfg.delta_path and os.path.exists(self.cfg.delta_path):
            missing, unexpected = load_delta(self.pipe.unet, self.cfg.delta_path, map_location="cpu")
            if unexpected:
                print(f"Warning: unexpected keys in delta: {len(unexpected)}")

        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        token_id = tokenizer.convert_tokens_to_ids(self.cfg.modifier_token)
        if token_id == tokenizer.unk_token_id:
            tokenizer.add_tokens([self.cfg.modifier_token])
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(self.cfg.modifier_token)

        if self.cfg.token_path and os.path.exists(self.cfg.token_path):
            emb = torch.load(self.cfg.token_path, map_location="cpu", weights_only=False)
            vec = emb.get(self.cfg.modifier_token, list(emb.values())[0])
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[token_id] = vec.to(text_encoder.get_input_embeddings().weight.dtype)

        print("Model loaded (base + delta + token).")

    def _load_test_data(self):
        with open(self.cfg.test_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if self.cfg.num_samples is not None:
            self.data = self.data[: self.cfg.num_samples]
        print(f"Loaded {len(self.data)} samples")

    def _load_captions(self):
        if self.cfg.captions_json and os.path.exists(self.cfg.captions_json):
            with open(self.cfg.captions_json, "r", encoding="utf-8") as f:
                self.captions = json.load(f)
        else:
            self.captions = {}

    def _maybe_init_verifier(self):
        self.verifier = None
        try:
            self.verifier = VerifierScorer(
                self.cfg.verifier_model_path,
                self.cfg.verifier_user_map_path,
                device=str(self.device),
            )
            print("Verifier enabled")
        except Exception as e:
            print(f"Verifier disabled: {e}")

    def _replace(self, prompt: str) -> str:
        return str(prompt).replace(self.cfg.target_token, self.cfg.modifier_token)

    def _image_to_tensor(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.cfg.image_size, self.cfg.image_size), Image.BICUBIC)
        return transforms.ToTensor()(image).unsqueeze(0).to(self.device)

    def _lpips(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a * 2.0 - 1.0
        b = b * 2.0 - 1.0
        with torch.no_grad():
            v = self.lpips_metric(a, b)
        return float(v.item())

    def _ssim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a * 2.0 - 1.0
        b = b * 2.0 - 1.0
        with torch.no_grad():
            v = self.ssim_metric(a, b)
        return float(v.item())

    def _clip_img_feat(self, img: torch.Tensor) -> torch.Tensor:
        img = self.clip_transform(img)
        with torch.no_grad():
            feat = self.clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def _clip_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        fa = self._clip_img_feat(a)
        fb = self._clip_img_feat(b)
        return float((fa @ fb.T).mean().item())

    def evaluate(self):
        g = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        results = []
        metrics = {
            "lpips_target": [],
            "ssim_target": [],
            "cpis_history_avg": [],
            "verifier_score": [],
        }

        for idx, sample in enumerate(tqdm(self.data, desc="Evaluating")):
            try:
                seq = sample.get("interaction_sequence", [])
                if not seq:
                    continue

                item = random.choice(seq)
                image_path = item.get("image_path")
                item_id = item.get("item_id", "")
                user_id = sample.get("user_id", "")

                key = item_id[:-4] if item_id.endswith(".jpg") else item_id
                prompt = self.captions.get(key, "")
                if not prompt:
                    continue

                gen_prompt = self._replace(prompt)
                with torch.no_grad():
                    img = self.pipe(
                        gen_prompt,
                        num_inference_steps=self.cfg.num_inference_steps,
                        guidance_scale=self.cfg.guidance_scale,
                        generator=g,
                    ).images[0]

                save_path = os.path.join(self.cfg.generated_images_dir, f"{str(key).replace('/', '_')}.png")
                img.save(save_path)

                gen_t = self._image_to_tensor(save_path)

                lp, ss = None, None
                if image_path and os.path.exists(image_path):
                    tgt_t = self._image_to_tensor(image_path)
                    lp = self._lpips(gen_t, tgt_t)
                    ss = self._ssim(gen_t, tgt_t)
                    metrics["lpips_target"].append(lp)
                    metrics["ssim_target"].append(ss)

                sims = []
                for h in seq[:10]:
                    hp = h.get("image_path")
                    if hp and os.path.exists(hp):
                        try:
                            sims.append(self._clip_sim(gen_t, self._image_to_tensor(hp)))
                        except Exception:
                            pass
                cpis = float(np.mean(sims)) if sims else None
                if cpis is not None:
                    metrics["cpis_history_avg"].append(cpis)

                verifier_score = None
                if self.verifier is not None and user_id:
                    verifier_score = self.verifier.score_image(str(user_id), save_path)
                    if verifier_score is not None:
                        metrics["verifier_score"].append(verifier_score)

                results.append(
                    {
                        "sample_idx": idx,
                        "user_id": user_id,
                        "item_id": item_id,
                        "prompt": prompt,
                        "generation_prompt": gen_prompt,
                        "generated_image_path": save_path,
                        "target_image_path": image_path,
                        "lpips_target": lp,
                        "ssim_target": ss,
                        "cpis_history_avg": cpis,
                        "verifier_score": verifier_score,
                    }
                )

            except Exception as e:
                print(f"Error at sample {idx}: {e}")

        summary = {}
        for k, vs in metrics.items():
            summary[f"avg_{k}"] = float(np.mean(vs)) if vs else None
            summary[f"std_{k}"] = float(np.std(vs)) if vs else None

        out_path = os.path.join(self.cfg.output_dir, "flickr_aes_custom_diffusion_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results, "config": vars(self.cfg)}, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)
        print("Summary:", summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_json", type=str, default="{FLICKR_AES_BASE_PATH}/processed_dataset/test.json")
    p.add_argument("--sd15_path", type=str, default="{SD15_MODEL_PATH}")
    p.add_argument("--delta_path", type=str, default="{FLICKR_AES_BASE_PATH}/custom_diffusion_sd15/delta.bin")
    p.add_argument("--token_path", type=str, default="{FLICKR_AES_BASE_PATH}/custom_diffusion_sd15/learned_token.bin")
    p.add_argument("--target_token", type=str, default="[V]")
    p.add_argument("--modifier_token", type=str, default="<new1>")
    p.add_argument("--captions_json", type=str, default="{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json")
    p.add_argument("--output_dir", type=str, default="{FLICKR_AES_BASE_PATH}/evaluation_results_custom_diffusion")
    p.add_argument("--generated_images_dir", type=str, default="{FLICKR_AES_BASE_PATH}/evaluation_generated_images_custom_diffusion")
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
        delta_path=args.delta_path,
        token_path=args.token_path,
        target_token=args.target_token,
        modifier_token=args.modifier_token,
        captions_json=args.captions_json,
        output_dir=args.output_dir,
        generated_images_dir=args.generated_images_dir,
        verifier_model_path=args.verifier_model_path,
        verifier_user_map_path=args.verifier_user_map_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    FlickrAESEvaluatorCustomDiffusion(cfg).evaluate()
