import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lora_utils import load_lora_into_pipeline
from style_mask import mask_prompt_to_instance_token


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_path(img_path: str, *, repo_root: Path, data_root: Path) -> str:
    if not img_path:
        return ""
    p = Path(str(img_path))
    if p.is_absolute():
        return str(p)
    # Prefer data_root-relative layout (images/...) for server portability
    cand1 = (data_root / p).resolve()
    if cand1.exists() or str(img_path).replace("\\", "/").startswith("images/"):
        return str(cand1)
    # Backward compat: repo-root relative (data/userpref_v1/images/...)
    cand2 = (repo_root / p).resolve()
    return str(cand2)


def _resize_rgb(img_path: str, size: int, *, repo_root: Path, data_root: Path) -> np.ndarray:
    img_path = _resolve_image_path(img_path, repo_root=repo_root, data_root=data_root)
    if not img_path or not os.path.exists(img_path):
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    try:
        im = Image.open(img_path).convert("RGB")
        im = im.resize((size, size), Image.BICUBIC)
        return np.ascontiguousarray(np.array(im, dtype=np.uint8))
    except Exception:
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def save_image_grid(images: List[np.ndarray], save_path: str):
    if not images:
        return
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    img_h, img_w = images[0].shape[:2]
    grid = np.ones((rows * img_h, cols * img_w, 3), dtype=np.uint8) * 255
    for idx, img in enumerate(images):
        r, c = idx // cols, idx % cols
        grid[r * img_h : (r + 1) * img_h, c * img_w : (c + 1) * img_w] = img
    Image.fromarray(grid).save(save_path)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_root_default = repo_root / "data" / "userpref_v1"

    p = argparse.ArgumentParser(description="Per-user DreamBooth(LoRA) inference for userpref_v1")
    p.add_argument("--test_json", type=str, default=str(data_root / "processed_dataset" / "test.json"))

    p.add_argument("--sd15_path", type=str, required=True)
    p.add_argument(
        "--lora_root",
        type=str,
        default=str(repo_root / "outputs" / "userpref_v1_dreambooth_lora"),
        help="Directory containing per-user lora dirs (each has pytorch_lora_weights.*)",
    )

    p.add_argument("--output_dir", type=str, default=str(repo_root / "outputs" / "userpref_v1_dreambooth_infer"))

    p.add_argument("--num_images_per_sample", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_prompt", type=str, default="lowres, text, error, cropped, worst quality, low quality")

    p.add_argument(
        "--instance_token",
        type=str,
        default="sks",
        help="DreamBooth instance token to inject into masked prompts and load from learned_token.bin when available.",
    )

    p.add_argument(
        "--initializer_token",
        type=str,
        default="photo",
        help="Initializer token used to initialize instance_token embedding when learned_token.bin is unavailable.",
    )

    p.add_argument(
        "--no_mask",
        action="store_true",
        help="[DEPRECATED] Same as --no_style_mask.",
    )

    p.add_argument(
        "--no_style_mask",
        action="store_true",
        help="Disable style masking for DreamBooth inference prompts (default: enabled).",
    )
    p.add_argument(
        "--style_mask_terms",
        type=str,
        default=str(Path(__file__).resolve().parent / "style_mask_terms.json"),
        help="JSON file with style masking terms; used when style mask is enabled.",
    )

    p.add_argument("--user_ids", type=str, default="", help="Comma-separated subset of users to run")

    args = p.parse_args()

    # Infer data_root from test_json when possible
    test_json_path = Path(args.test_json).resolve()
    if test_json_path.name.lower() == "test.json" and test_json_path.parent.name == "processed_dataset":
        data_root = test_json_path.parents[1]
    else:
        data_root = data_root_default

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    test_data = _load_json(Path(args.test_json))

    allow_users = None
    if args.user_ids.strip():
        allow_users = {x.strip() for x in args.user_ids.split(",") if x.strip()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group samples by user for efficient pipeline loading (keep global sample_idx for eval compatibility)
    user2samples: Dict[str, List[tuple]] = {}
    for sample_idx, s in enumerate(test_data):
        uid = str(s.get("worker_id") or "").strip()
        if not uid:
            continue
        if allow_users is not None and uid not in allow_users:
            continue
        user2samples.setdefault(uid, []).append((int(sample_idx), s))

    from diffusers import StableDiffusionPipeline

    base_pipe = StableDiffusionPipeline.from_pretrained(
        args.sd15_path,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)

    rng = torch.Generator(device=device).manual_seed(int(args.seed))

    for uid, samples in tqdm(list(user2samples.items()), desc="Users"):
        # Create a fresh pipe per user (simple + safe). If you want speed, cache and unload LoRA.
        pipe = base_pipe
        # Clone base pipe by reloading to avoid LoRA accumulation
        pipe = StableDiffusionPipeline.from_pretrained(
            args.sd15_path,
            safety_checker=None,
            torch_dtype=dtype,
        ).to(device)

        lora_dir = Path(args.lora_root) / uid
        ok = load_lora_into_pipeline(pipe, str(lora_dir))
        if not ok:
            print(f"[SKIP] user={uid}: LoRA not found or failed to load: {lora_dir}")
            del pipe
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Ensure instance token exists as a single token and load its embedding if available.
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        token = str(args.instance_token).strip() or "sks"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            tokenizer.add_tokens([token])
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(token)

        token_path = lora_dir / "learned_token.bin"
        token_loaded = False
        token_initialized = False
        if token_path.exists():
            try:
                emb = torch.load(str(token_path), map_location="cpu")
                if isinstance(emb, dict) and emb:
                    vec = emb.get(token)
                    if vec is None:
                        vec = list(emb.values())[0]
                    with torch.no_grad():
                        text_encoder.get_input_embeddings().weight[token_id] = vec.to(
                            text_encoder.get_input_embeddings().weight.dtype
                        )
                    token_loaded = True
            except Exception:
                token_loaded = False

        # Fallback: if token file missing/failed, initialize the newly-added token row
        # to the initializer_token embedding (similar to trainer's _maybe_add_token).
        if (not token_loaded) and (token_id != tokenizer.unk_token_id):
            init_id = tokenizer.convert_tokens_to_ids(str(args.initializer_token))
            if init_id == tokenizer.unk_token_id:
                init_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            if init_id != tokenizer.unk_token_id:
                with torch.no_grad():
                    embeds = text_encoder.get_input_embeddings().weight
                    embeds[token_id] = embeds[init_id].clone()
                token_initialized = True

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

            # Match `method/PMG/evaluation.py` convention: eval_output_dir/sample_0000/gen_0.jpg
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
                img = out.images[0]
                img_np = np.array(img.convert("RGB"))
                images.append(img_np)
                img.save(sample_dir / f"gen_{img_idx}.jpg")

            save_image_grid(images, str(sample_dir / "grid.jpg"))

            history_items = sample.get("history_items_info", [])
            history_imgs = [
                _resize_rgb(it.get("image_path", ""), int(args.image_size), repo_root=repo_root, data_root=data_root)
                for it in history_items
            ]
            target_ref = _resize_rgb(
                target_item.get("image_path", ""),
                int(args.image_size),
                repo_root=repo_root,
                data_root=data_root,
            )
            save_image_grid(history_imgs + [target_ref], str(sample_dir / "references.jpg"))

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
