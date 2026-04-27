import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(raw_path: str, *, repo_root: Path, data_root: Path) -> str:
    if not raw_path:
        return ""
    p = Path(str(raw_path))
    if p.is_absolute():
        return str(p)

    candidate1 = (data_root / p).resolve()
    if candidate1.exists() or str(raw_path).replace("\\", "/").startswith("images/"):
        return str(candidate1)

    candidate2 = (repo_root / p).resolve()
    return str(candidate2)


def parse_score(value, default=-1e9) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    if val != val:
        return default
    return val


def select_target_from_candidates(sample: dict, *, repo_root: Path, data_root: Path) -> Tuple[str, Optional[float]]:
    candidates = sample.get("stage2_candidates") or []
    best_path = ""
    best_score = -1e9

    for item in candidates:
        path = resolve_image_path(str(item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)
        score = parse_score(item.get("preference_score"))
        if (score > best_score) and path and os.path.exists(path):
            best_score = score
            best_path = path

    if best_path:
        return best_path, (None if best_score <= -1e8 else float(best_score))

    target_item = sample.get("target_item_info") or {}
    fallback = resolve_image_path(str(target_item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)
    return fallback, None


def image_to_tensor(image_path: str, *, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


def clip_embedder(device: torch.device):
    try:
        import clip
    except ImportError:
        return None, None

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    clip_transform = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return model, clip_transform


def clip_image_feature(model, image_tensor: torch.Tensor, clip_transform) -> torch.Tensor:
    image_tensor = clip_transform(image_tensor)
    with torch.no_grad():
        feat = model.encode_image(image_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def main():
    repo_root = Path(__file__).resolve().parents[3]
    data_root_default = repo_root / "data" / "userpref_v1"

    parser = argparse.ArgumentParser(description="Evaluate userpref DreamBooth generation outputs")
    parser.add_argument("--test_json", type=str, default=str(data_root_default / "processed_dataset" / "test.json"))
    parser.add_argument(
        "--infer_output_dir",
        type=str,
        default=str(repo_root / "outputs" / "userpref_v1_dreambooth_infer"),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(repo_root / "outputs" / "userpref_v1_dreambooth_eval" / "metrics.json"),
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_history", type=int, default=10)
    parser.add_argument("--user_ids", type=str, default="")
    args = parser.parse_args()

    test_json_path = Path(args.test_json).resolve()
    if test_json_path.name.lower() == "test.json" and test_json_path.parent.name == "processed_dataset":
        data_root = test_json_path.parents[1]
    else:
        data_root = data_root_default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = load_json(Path(args.test_json))

    allow_users = None
    if args.user_ids.strip():
        allow_users = {x.strip() for x in args.user_ids.split(",") if x.strip()}

    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
    except Exception:
        lpips_metric = None

    try:
        from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

        ssim_metric = SSIM(data_range=1.0).to(device)
    except Exception:
        ssim_metric = None

    clip_model, clip_transform = clip_embedder(device)

    results = []
    aggregate = {
        "lpips_target": [],
        "ssim_target": [],
        "lpips_history_avg": [],
        "ssim_history_avg": [],
        "cpis_history_avg": [],
        "pref_align_spearman": [],
        "pref_align_hit_at_1": [],
    }

    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        uid = str(sample.get("worker_id") or "").strip()
        if allow_users is not None and uid not in allow_users:
            continue

        gen_path = Path(args.infer_output_dir) / f"sample_{idx:04d}" / "gen_0.jpg"
        if not gen_path.exists():
            continue

        target_path, target_pref = select_target_from_candidates(sample, repo_root=repo_root, data_root=data_root)
        if not target_path or not os.path.exists(target_path):
            continue

        generated_tensor = image_to_tensor(str(gen_path), image_size=int(args.image_size), device=device)
        target_tensor = image_to_tensor(str(target_path), image_size=int(args.image_size), device=device)

        sample_metrics = {
            "sample_idx": idx,
            "user_id": uid,
            "target_preference": target_pref,
            "lpips_target": None,
            "ssim_target": None,
            "lpips_history_avg": None,
            "ssim_history_avg": None,
            "cpis_history_avg": None,
            "pref_align_spearman": None,
            "pref_align_hit_at_1": None,
        }

        if lpips_metric is not None:
            with torch.no_grad():
                lpips_target = lpips_metric(generated_tensor * 2 - 1, target_tensor * 2 - 1).item()
            sample_metrics["lpips_target"] = float(lpips_target)
            aggregate["lpips_target"].append(float(lpips_target))

        if ssim_metric is not None:
            with torch.no_grad():
                ssim_target = ssim_metric(generated_tensor, target_tensor).item()
            sample_metrics["ssim_target"] = float(ssim_target)
            aggregate["ssim_target"].append(float(ssim_target))

        history_items = sample.get("history_items_info", [])[: max(1, int(args.num_history))]
        lp_hist = []
        ssim_hist = []
        cpis_hist = []

        gen_feat = None
        if clip_model is not None and clip_transform is not None:
            gen_feat = clip_image_feature(clip_model, generated_tensor, clip_transform)

        for item in history_items:
            hist_path = resolve_image_path(str(item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)
            if not hist_path or not os.path.exists(hist_path):
                continue
            hist_tensor = image_to_tensor(hist_path, image_size=int(args.image_size), device=device)

            if lpips_metric is not None:
                with torch.no_grad():
                    lp_hist.append(float(lpips_metric(generated_tensor * 2 - 1, hist_tensor * 2 - 1).item()))
            if ssim_metric is not None:
                with torch.no_grad():
                    ssim_hist.append(float(ssim_metric(generated_tensor, hist_tensor).item()))
            if gen_feat is not None:
                hist_feat = clip_image_feature(clip_model, hist_tensor, clip_transform)
                cpis_hist.append(float((gen_feat @ hist_feat.T).mean().item()))

        if lp_hist:
            sample_metrics["lpips_history_avg"] = float(np.mean(lp_hist))
            aggregate["lpips_history_avg"].append(sample_metrics["lpips_history_avg"])
        if ssim_hist:
            sample_metrics["ssim_history_avg"] = float(np.mean(ssim_hist))
            aggregate["ssim_history_avg"].append(sample_metrics["ssim_history_avg"])
        if cpis_hist:
            sample_metrics["cpis_history_avg"] = float(np.mean(cpis_hist))
            aggregate["cpis_history_avg"].append(sample_metrics["cpis_history_avg"])

        if gen_feat is not None:
            candidates = sample.get("stage2_candidates") or []
            sims = []
            prefs = []
            for item in candidates:
                cand_path = resolve_image_path(str(item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)
                if not cand_path or not os.path.exists(cand_path):
                    continue
                pref = parse_score(item.get("preference_score"), default=float("nan"))
                if pref != pref:
                    continue
                cand_tensor = image_to_tensor(cand_path, image_size=int(args.image_size), device=device)
                cand_feat = clip_image_feature(clip_model, cand_tensor, clip_transform)
                sim = float((gen_feat @ cand_feat.T).mean().item())
                sims.append(sim)
                prefs.append(float(pref))

            if len(sims) >= 2:
                rank_sim = np.argsort(np.argsort(np.array(sims)))
                rank_pref = np.argsort(np.argsort(np.array(prefs)))
                corr = np.corrcoef(rank_sim, rank_pref)[0, 1]
                if corr == corr:
                    sample_metrics["pref_align_spearman"] = float(corr)
                    aggregate["pref_align_spearman"].append(float(corr))

                hit = int(int(np.argmax(sims)) == int(np.argmax(prefs)))
                sample_metrics["pref_align_hit_at_1"] = hit
                aggregate["pref_align_hit_at_1"].append(float(hit))

        results.append(sample_metrics)

    avg_metrics = {
        key: (float(np.mean(values)) if values else None)
        for key, values in aggregate.items()
    }

    output = {
        "total_samples": len(test_data),
        "evaluated_samples": len(results),
        "average_metrics": avg_metrics,
        "per_sample_metrics": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics: {out_path}")
    print(json.dumps(avg_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
