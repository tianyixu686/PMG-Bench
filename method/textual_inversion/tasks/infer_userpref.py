import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import time
from PIL import Image
from tqdm import tqdm

from method.dreambooth.utils.style_mask import mask_prompt_to_instance_token


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(img_path: str, *, repo_root: Path, data_root: Path) -> str:
    if not img_path:
        return ""
    p = Path(str(img_path))
    if p.is_absolute():
        return str(p)
    cand1 = (data_root / p).resolve()
    if cand1.exists() or str(img_path).replace("\\", "/").startswith("images/"):
        return str(cand1)
    return str((repo_root / p).resolve())


def resize_rgb(img_path: str, size: int, *, repo_root: Path, data_root: Path) -> np.ndarray:
    img_path = resolve_image_path(img_path, repo_root=repo_root, data_root=data_root)
    if not img_path or not os.path.exists(img_path):
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    try:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((size, size), Image.BICUBIC)
        return np.ascontiguousarray(np.array(image, dtype=np.uint8))
    except Exception:
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def save_image_grid(images: List[np.ndarray], save_path: str):
    if not images:
        return
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    h, w = images[0].shape[:2]
    grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    for idx, im in enumerate(images):
        r, c = idx // cols, idx % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    Image.fromarray(grid).save(save_path)


def load_textual_inversion_embeddings(text_encoder, tokenizer, embeds_path: Path, *, target_token: str, num_vectors: int):
    obj = torch.load(str(embeds_path), map_location="cpu")
    embeds = obj.get(target_token)
    if embeds is None and isinstance(obj, dict) and len(obj) == 1:
        embeds = next(iter(obj.values()))
    if embeds is None:
        raise ValueError(f"Invalid embeds file: {embeds_path}")

    placeholder_tokens = [f"<v_{i}>" for i in range(int(num_vectors))]
    num_added = tokenizer.add_tokens(placeholder_tokens)
    if num_added != int(num_vectors):
        raise ValueError(f"Failed to add placeholder tokens: expected {num_vectors}, got {num_added}")
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    w = text_encoder.get_input_embeddings().weight.data
    embeds = embeds.to(dtype=w.dtype)
    if embeds.shape[0] != len(placeholder_token_ids):
        raise ValueError(f"Embeds shape mismatch: {embeds.shape} vs expected {len(placeholder_token_ids)} vectors")
    with torch.no_grad():
        for j, token_id in enumerate(placeholder_token_ids):
            w[token_id] = embeds[j].clone()
    return " ".join(placeholder_tokens)


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    data_root_default = repo_root / "data" / "userpref_v1"

    p = argparse.ArgumentParser(description="Per-user Textual Inversion inference for userpref_v1 (SD1.5)")
    p.add_argument("--test_json", type=str, default=str(data_root_default / "processed_dataset" / "test.json"))
    p.add_argument("--sd15_path", type=str, required=True)
    p.add_argument(
        "--ti_root",
        type=str,
        default=str(repo_root / "outputs" / "userpref_v1_textual_inversion"),
        help="Root directory containing per-user TI outputs (each has learned_embeds.bin)",
    )
    p.add_argument("--output_dir", type=str, default=str(repo_root / "outputs" / "userpref_v1_textual_inversion_infer"))

    p.add_argument("--num_images_per_sample", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_prompt", type=str, default="lowres, text, error, cropped, worst quality, low quality")

    p.add_argument("--target_token", type=str, default="[V]")
    p.add_argument("--initializer_token", type=str, default="style")
    p.add_argument("--num_vectors", type=int, default=8)

    p.add_argument("--no_style_mask", action="store_true")
    p.add_argument(
        "--style_mask_terms",
        type=str,
        default=str(repo_root / "method" / "dreambooth" / "style_mask_terms.json"),
    )

    p.add_argument("--user_ids", type=str, default="", help="Comma-separated subset users")
    return p.parse_args()


def main():
    args = parse_args()

    from diffusers import StableDiffusionPipeline

    repo_root = Path(__file__).resolve().parents[3]
    test_json_path = Path(args.test_json).resolve()
    if test_json_path.name.lower() == "test.json" and test_json_path.parent.name == "processed_dataset":
        data_root = test_json_path.parents[1]
    else:
        data_root = repo_root / "data" / "userpref_v1"

    test_data = load_json(Path(args.test_json))

    allow_users = None
    if str(args.user_ids).strip():
        allow_users = {x.strip() for x in str(args.user_ids).split(",") if x.strip()}

    user2samples: Dict[str, List[Tuple[int, dict]]] = {}
    for sample_idx, sample in enumerate(test_data):
        uid = str(sample.get("worker_id") or "").strip()
        if not uid:
            continue
        if allow_users is not None and uid not in allow_users:
            continue
        user2samples.setdefault(uid, []).append((int(sample_idx), sample))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rng = torch.Generator(device=device).manual_seed(int(args.seed))

    for uid, samples in tqdm(list(user2samples.items()), desc="Users"):
        ti_dir = Path(args.ti_root) / uid
        embeds_path = ti_dir / "learned_embeds.bin"
        if not embeds_path.exists():
            print(f"[SKIP] user={uid}: missing {embeds_path}")
            continue

        t_user_load_start = time.perf_counter()
        pipe = StableDiffusionPipeline.from_pretrained(str(args.sd15_path), safety_checker=None).to(device)
        placeholder_tokens_str = load_textual_inversion_embeddings(
            pipe.text_encoder,
            pipe.tokenizer,
            embeds_path=embeds_path,
            target_token=str(args.target_token),
            num_vectors=int(args.num_vectors),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_user_load_end = time.perf_counter()

        for sample_idx, sample in samples:
            target_item = sample.get("target_item_info", {})
            query_variant = str(target_item.get("query_variant") or "")
            prompt_raw = str(target_item.get("caption") or "").strip()
            if not prompt_raw:
                continue

            # Ensure prompt contains the TI "concept token" exactly once.
            style_mask_enabled = not bool(args.no_style_mask)
            if style_mask_enabled:
                prompt_masked = mask_prompt_to_instance_token(
                    prompt_raw,
                    instance_token=str(args.target_token),
                    terms_path=str(args.style_mask_terms),
                    enabled=True,
                ).strip()
                prompt = prompt_masked
            else:
                prompt = prompt_raw

            prompt = prompt.replace(str(args.target_token), placeholder_tokens_str)
            if placeholder_tokens_str not in prompt:
                prompt = f"{prompt}, {placeholder_tokens_str}".strip(" ,")

            sample_dir = output_dir / f"sample_{int(sample_idx):04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            images = []
            per_image_times_s: List[float] = []
            t_sample_start = time.perf_counter()
            for img_idx in range(int(args.num_images_per_sample)):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = pipe(
                        prompt,
                        negative_prompt=str(args.negative_prompt),
                        num_inference_steps=int(args.num_inference_steps),
                        guidance_scale=float(args.guidance_scale),
                        generator=rng,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                per_image_times_s.append(float(t1 - t0))
                image = out.images[0]
                image_np = np.array(image.convert("RGB"))
                images.append(image_np)
                image.save(sample_dir / f"gen_{img_idx}.jpg")
            t_sample_end = time.perf_counter()

            save_image_grid(images, str(sample_dir / "grid.jpg"))

            history_items = sample.get("history_items_info", [])
            history_images = [
                resize_rgb(str(it.get("image_path") or ""), int(args.image_size), repo_root=repo_root, data_root=data_root)
                for it in history_items
            ]
            target_ref = resize_rgb(
                str(target_item.get("image_path") or ""),
                int(args.image_size),
                repo_root=repo_root,
                data_root=data_root,
            )
            save_image_grid(history_images + [target_ref], str(sample_dir / "references.jpg"))

            meta = {
                "sample_idx": int(sample_idx),
                "user_id": uid,
                "query_variant": query_variant,
                "prompt_raw": prompt_raw,
                "prompt": prompt,
                "style_mask_enabled": bool(style_mask_enabled),
                "target_token": str(args.target_token),
                "placeholder_tokens": placeholder_tokens_str,
                "embeds_path": str(embeds_path),
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "seed": int(args.seed),
                "negative_prompt": str(args.negative_prompt),
                "timing": {
                    "pipeline_load_s": float(t_user_load_end - t_user_load_start),
                    "sample_total_s": float(t_sample_end - t_sample_start),
                    "per_image_s": per_image_times_s,
                    "num_images": int(args.num_images_per_sample),
                },
            }
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()

