import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Reuse the existing IP-Adapter + metrics implementation (SER version).
# Note: this directory name contains a hyphen ("ip-adater"), so it can't be imported
# as a normal Python package. We load the module by file path.
#
# IMPORTANT: we do this lazily so `--help` works even if diffusers/transformers
# deps aren't available in the current environment.
import importlib.util  # noqa: E402


def _load_ser_impl():
    ser_impl_path = Path(__file__).with_name("personalized_generation.py")
    spec = importlib.util.spec_from_file_location("ip_adapter_ser_impl", str(ser_impl_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec from {ser_impl_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


def _resolve_under_root(data_root: Path, maybe_rel_path: str) -> str:
    p = Path(str(maybe_rel_path))
    if p.is_absolute():
        return str(p)
    return str((data_root / p).resolve())


def _load_images_abs(image_paths_abs: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for img_path in image_paths_abs:
        try:
            if os.path.exists(img_path):
                images.append(Image.open(img_path).convert("RGB"))
            else:
                print(f"Warning: Image not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images


def _select_history_paths_userpref(
    sample: dict,
    data_root: Path,
    min_preference_score: float,
    max_images: int,
) -> Tuple[List[str], List[dict]]:
    raw_items = sample.get("history_items_info") or []
    kept: List[dict] = []

    for it in raw_items:
        try:
            score = float(it.get("preference_score", float("-inf")))
        except Exception:
            score = float("-inf")
        if score >= min_preference_score:
            kept.append(it)

    # Keep last N (closest to "recent history" if the list is chronological).
    if max_images is not None and len(kept) > max_images:
        kept = kept[-max_images:]

    paths_abs = [
        _resolve_under_root(data_root, it["image_path"])
        for it in kept
        if isinstance(it, dict) and it.get("image_path")
    ]
    return paths_abs, kept


def process_userpref_dataset(
    test_json_path: str,
    data_root: str,
    output_dir: str,
    sd_model_path: str,
    ip_adapter_path: str,
    ip_adapter_weight: str,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    start_idx: int = 0,
    num_inference_steps: int = 50,
    guidance_scale: float = 10.0,
    ip_adapter_scale: float = 0.3,
    save_history_images: bool = False,
    seed: int = 42,
    min_preference_score: float = 4.0,
    max_history_images: int = 10,
):
    ser_impl = _load_ser_impl()
    MetricsEvaluator = ser_impl.MetricsEvaluator
    setup_pipeline = ser_impl.setup_pipeline
    load_multiple_ip_adapters = ser_impl.load_multiple_ip_adapters
    generate_personalized_image = ser_impl.generate_personalized_image

    data_root_p = Path(data_root).resolve()
    os.makedirs(output_dir, exist_ok=True)
    generated_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(generated_dir, exist_ok=True)

    if save_history_images:
        history_dir = os.path.join(output_dir, "history_images")
        os.makedirs(history_dir, exist_ok=True)

    print(f"Loading userpref test dataset from {test_json_path}...")
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total_samples = len(test_data)
    if max_samples is not None:
        test_data = test_data[start_idx : start_idx + max_samples]
    else:
        test_data = test_data[start_idx:]

    print(f"Processing {len(test_data)} samples (starting from index {start_idx})...")

    pipe = setup_pipeline(
        sd_model_path=sd_model_path,
        ip_adapter_path=ip_adapter_path,
        ip_adapter_weight=ip_adapter_weight,
        device=device,
    )

    print("Initializing metrics evaluator...")
    evaluator = MetricsEvaluator(device=device, image_size=512)

    results = []
    failed_samples = []
    all_metrics = {
        "lpips_target": [],
        "lpips_history_avg": [],
        "ssim_target": [],
        "ssim_history_avg": [],
        "cps": [],
        "cpis_history_avg": [],
        "hpsv2": [],
        "laion_aesthetic": [],
    }

    pbar = tqdm(test_data, desc="Generating images (IP-Adapter userpref)")
    for idx, sample in enumerate(pbar):
        try:
            # userpref schema
            worker_id = sample.get("worker_id", "unknown_user")
            target_info = sample.get("target_item_info") or {}
            query_variant = target_info.get("query_variant", "NA")
            target_item_name = target_info.get("item_name", f"sample_{idx + start_idx}")

            # History images: preference_score >= threshold
            history_paths_abs, kept_items = _select_history_paths_userpref(
                sample=sample,
                data_root=data_root_p,
                min_preference_score=min_preference_score,
                max_images=max_history_images,
            )
            history_images = _load_images_abs(history_paths_abs)

            if len(history_images) == 0:
                failed_samples.append(
                    {
                        "index": idx + start_idx,
                        "worker_id": worker_id,
                        "target_item_name": target_item_name,
                        "query_variant": query_variant,
                        "reason": f"No valid history images with preference_score>={min_preference_score}",
                    }
                )
                continue

            # Ensure enough IP-Adapter instances if we use multi-image conditioning
            if len(history_images) > 1:
                num_loaded = 1
                if hasattr(pipe.unet, "encoder_hid_proj") and hasattr(
                    pipe.unet.encoder_hid_proj, "image_projection_layers"
                ):
                    num_loaded = len(pipe.unet.encoder_hid_proj.image_projection_layers)
                if num_loaded < len(history_images):
                    load_multiple_ip_adapters(
                        pipe,
                        ip_adapter_path,
                        ip_adapter_weight,
                        len(history_images),
                    )

            prompt = target_info.get("caption", "") or ""
            target_image_rel = target_info.get("image_path", None)
            target_image_path = (
                _resolve_under_root(data_root_p, target_image_rel) if target_image_rel else None
            )

            generated_image = generate_personalized_image(
                pipe=pipe,
                history_images=history_images,
                target_caption=prompt,
                target_image_path=target_image_path,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ip_adapter_scale=ip_adapter_scale,
                seed=seed,
            )

            safe_id = str(target_item_name).replace("/", "_").replace("\\", "_")
            safe_user = str(worker_id).replace("/", "_").replace("\\", "_")
            fname = f"user{safe_user}_{query_variant}_{safe_id}"
            output_path = os.path.join(generated_dir, f"{fname}_generated.png")
            generated_image.save(output_path)

            if save_history_images:
                for i, (hist_img, meta) in enumerate(zip(history_images, kept_items)):
                    hist_name = meta.get("item_name", f"hist_{i}")
                    hist_safe = str(hist_name).replace("/", "_").replace("\\", "_")
                    hist_output_path = os.path.join(
                        history_dir, f"{fname}_history_{i}_{hist_safe}.png"
                    )
                    hist_img.save(hist_output_path)

            # For userpref baseline we don't have SER-style "history_topic".
            metrics = evaluator.evaluate_generated_image(
                generated_image=generated_image,
                target_image_path=target_image_path,
                history_images=history_images,
                history_topic="",  # disables CPS (no preference keywords)
                target_caption=prompt,
                generated_image_path=output_path,
            )

            result = {
                "index": idx + start_idx,
                "worker_id": worker_id,
                "query_variant": query_variant,
                "target_item_name": target_item_name,
                "prompt": prompt,
                "target_image_path": target_image_path,
                "num_history_images": len(history_images),
                "history_filter": {
                    "min_preference_score": min_preference_score,
                    "max_history_images": max_history_images,
                },
                "output_path": output_path,
                "success": True,
                "metrics": metrics,
            }
            results.append(result)

            for key in all_metrics.keys():
                if metrics.get(key) is not None:
                    all_metrics[key].append(metrics[key])

        except SystemExit:
            raise
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback

            traceback.print_exc()
            failed_samples.append(
                {
                    "index": idx + start_idx,
                    "worker_id": sample.get("worker_id", None),
                    "reason": str(e),
                }
            )
            continue

    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            # values may contain tensors/np scalars
            avg_metrics[f"avg_{key}"] = float(torch.tensor(values).mean().item())
            avg_metrics[f"std_{key}"] = float(torch.tensor(values).std(unbiased=False).item())

    results_path = os.path.join(output_dir, "generation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "userpref_v1",
                "split": "test",
                "data_root": str(data_root_p),
                "total_samples": len(test_data),
                "successful": len(results),
                "failed": len(failed_samples),
                "results": results,
                "failed_samples": failed_samples,
                "average_metrics": avg_metrics,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'='*60}")
    print("Generation and Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(test_data)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_samples)}")
    print("\nAverage Metrics:")
    for key in sorted(all_metrics.keys()):
        if f"avg_{key}" in avg_metrics:
            print(
                f"  {key.upper():25s}: {avg_metrics[f'avg_{key}']:.4f} ± {avg_metrics.get(f'std_{key}', 0):.4f}"
            )
        else:
            print(f"  {key.upper():25s}: N/A (all values were None)")
    print(f"\nResults saved to {results_path}")
    print(f"{'='*60}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data" / "userpref_v1"
    default_test_json = default_data_root / "processed_dataset" / "test.json"

    parser = argparse.ArgumentParser(
        description="UserPref v1 test split: personalized generation with IP-Adapter (inference-only baseline)"
    )
    parser.add_argument("--test_json", type=str, default=str(default_test_json), help="Test JSON path")
    parser.add_argument("--data_root", type=str, default=str(default_data_root), help="Dataset root (data/userpref_v1)")
    parser.add_argument("--output_dir", type=str, default=str(default_data_root / "ip_adapter_generated"), help="Output directory")
    parser.add_argument("--sd_model_path", type=str, default="{SD15_MODEL_PATH}", help="Stable Diffusion model path")
    parser.add_argument("--ip_adapter_path", type=str, default="{IP_ADAPTER_PATH}", help="IP-Adapter model path")
    parser.add_argument("--ip_adapter_weight", type=str, default="ip-adapter_sd15.bin", help="IP-Adapter weight filename")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples (None for all)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=10.0, help="Guidance scale")
    parser.add_argument("--ip_adapter_scale", type=float, default=0.3, help="IP-Adapter scale")
    parser.add_argument("--save_history_images", action="store_true", help="Save selected history images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_preference_score", type=float, default=4.0, help="Keep history images with preference_score >= this")
    parser.add_argument("--max_history_images", type=int, default=10, help="Max number of history images to use")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    process_userpref_dataset(
        test_json_path=args.test_json,
        data_root=args.data_root,
        output_dir=args.output_dir,
        sd_model_path=args.sd_model_path,
        ip_adapter_path=args.ip_adapter_path,
        ip_adapter_weight=args.ip_adapter_weight,
        device=args.device,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        save_history_images=args.save_history_images,
        seed=args.seed,
        min_preference_score=args.min_preference_score,
        max_history_images=args.max_history_images,
    )


if __name__ == "__main__":
    main()

