import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from method.dreambooth.core.pipeline import build_infer_pipeline
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
    candidate1 = (data_root / p).resolve()
    if candidate1.exists() or str(img_path).replace("\\", "/").startswith("images/"):
        return str(candidate1)
    candidate2 = (repo_root / p).resolve()
    return str(candidate2)


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
    for idx, image in enumerate(images):
        r, c = idx // cols, idx % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = image
    Image.fromarray(grid).save(save_path)


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    data_root_default = repo_root / "data" / "userpref_v1"

    p = argparse.ArgumentParser(description="Per-user DreamBooth LoRA inference for userpref_v1")
    p.add_argument("--test_json", type=str, default=str(data_root_default / "processed_dataset" / "test.json"))
    p.add_argument("--sd15_path", type=str, required=True)
    p.add_argument(
        "--lora_root",
        type=str,
        default=str(repo_root / "outputs" / "userpref_v1_dreambooth_lora"),
        help="Root directory containing per-user LoRA outputs",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_root / "outputs" / "userpref_v1_dreambooth_infer"),
    )

    p.add_argument("--num_images_per_sample", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_prompt", type=str, default="lowres, text, error, cropped, worst quality, low quality")

    p.add_argument("--instance_token", type=str, default="sks")
    p.add_argument("--initializer_token", type=str, default="photo")

    p.add_argument("--no_mask", action="store_true", help="Deprecated: same as --no_style_mask")
    p.add_argument("--no_style_mask", action="store_true")
    p.add_argument(
        "--style_mask_terms",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "style_mask_terms.json"),
    )

    p.add_argument("--user_ids", type=str, default="", help="Comma-separated subset users")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    data_root_default = repo_root / "data" / "userpref_v1"

    test_json_path = Path(args.test_json).resolve()
    if test_json_path.name.lower() == "test.json" and test_json_path.parent.name == "processed_dataset":
        data_root = test_json_path.parents[1]
    else:
        data_root = data_root_default

    test_data = load_json(Path(args.test_json))

    allow_users = None
    if args.user_ids.strip():
        allow_users = {x.strip() for x in args.user_ids.split(",") if x.strip()}

    user2samples: Dict[str, List[tuple]] = {}
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
        lora_dir = Path(args.lora_root) / uid
        pipe, token_loaded, token_initialized = build_infer_pipeline(
            sd15_path=str(args.sd15_path),
            lora_dir=str(lora_dir),
            token=str(args.instance_token),
            initializer_token=str(args.initializer_token),
            device=device,
        )
        if pipe is None:
            print(f"[SKIP] user={uid}: failed to load LoRA from {lora_dir}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        token = str(args.instance_token).strip() or "sks"
        token_path = lora_dir / "learned_token.bin"

        for sample_idx, sample in samples:
            target_item = sample.get("target_item_info", {})
            query_variant = target_item.get("query_variant", "")

            style_mask_enabled = (not args.no_mask) and (not args.no_style_mask)
            prompt_raw = str(target_item.get("caption") or "").strip()
            prompt_masked_in_data = str(target_item.get("masked_caption") or "").strip()

            if style_mask_enabled:
                prompt_masked = mask_prompt_to_instance_token(
                    prompt_raw,
                    instance_token=token,
                    terms_path=args.style_mask_terms,
                    enabled=True,
                ).strip()
                prompt = prompt_masked
            else:
                prompt_masked = prompt_masked_in_data
                prompt = prompt_raw

            if not prompt:
                continue

            sample_dir = output_dir / f"sample_{int(sample_idx):04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            images = []
            for img_idx in range(int(args.num_images_per_sample)):
                with torch.no_grad():
                    out = pipe(
                        prompt,
                        negative_prompt=args.negative_prompt,
                        num_inference_steps=int(args.num_inference_steps),
                        guidance_scale=float(args.guidance_scale),
                        generator=rng,
                    )
                image = out.images[0]
                image_np = np.array(image.convert("RGB"))
                images.append(image_np)
                image.save(sample_dir / f"gen_{img_idx}.jpg")

            save_image_grid(images, str(sample_dir / "grid.jpg"))

            history_items = sample.get("history_items_info", [])
            history_images = [
                resize_rgb(str(item.get("image_path") or ""), int(args.image_size), repo_root=repo_root, data_root=data_root)
                for item in history_items
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
                "prompt": prompt,
                "prompt_raw": prompt_raw,
                "prompt_masked": prompt_masked,
                "style_mask_enabled": bool(style_mask_enabled),
                "style_mask_terms": str(args.style_mask_terms),
                "instance_token": token,
                "instance_token_loaded": bool(token_loaded),
                "instance_token_initialized": bool(token_initialized),
                "instance_token_path": str(token_path),
                "lora_dir": str(lora_dir),
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "seed": int(args.seed),
                "negative_prompt": args.negative_prompt,
                "target_item_info": target_item,
                "stage2_candidates": sample.get("stage2_candidates", []),
            }
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
