#!/usr/bin/env python3
"""
实验三主实验：在统一协议下对一次生成结果（generation_results.json 或 sample 目录）计算多维自动指标。

输出 JSON：每样本 CLIP-I / CLIP-T / DINO-I / LPIPS / SSIM / Gram distance /（可选）HPSv2 /（可选）FID。

用法:
  python -m experiments.exp3_bench_metrics.run_eval_benchmark \\
    --test_json data/userpref_v1/processed_dataset/test.json \\
    --input outputs/.../generation_results.json \\
    --output_json outputs/experiments/exp3_bench_metrics/ip_adapter_eval.json

可选: --compute_hps --compute_fid（较慢且依赖环境）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.common.gram_style import GramStyleDistance
from experiments.common.repo import repo_root
from experiments.common.userpref_io import data_root_from_split_json, load_split, select_target_from_candidates
from tools.eval_userpref_outputs import (
    build_clip,
    build_dino,
    build_hpsv2,
    build_sample_map_from_dir,
    build_sample_map_from_ip_adapter_results,
    clip_image_feature,
    clip_text_feature,
    dino_feature,
    image_path_to_pil,
)


def image_to_tensor(image_path: str, *, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


def _mean_std(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    a = np.array(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(len(a))}


def main():
    repo = repo_root()
    default_data = repo / "data" / "userpref_v1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", type=str, default=str(default_data / "processed_dataset" / "test.json"))
    ap.add_argument("--input", type=str, required=True, help="generation_results.json 或包含 sample_XXXX/gen_0.jpg 的目录")
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--compute_hps", action="store_true")
    ap.add_argument("--compute_fid", action="store_true")
    ap.add_argument("--prefer_meta_prompt", action="store_true")
    args = ap.parse_args()

    test_json_path = Path(args.test_json).resolve()
    data_root = data_root_from_split_json(test_json_path)
    test_data = load_split(test_json_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
    except Exception:
        lpips_metric = None

    clip_model, clip_transform = build_clip(device)
    dino_model, dino_transform = build_dino(device)
    if dino_model is None:
        print(
            "Warning: DINO backbone unavailable (install `timm`, e.g. "
            "`pip install -r experiments/exp3_bench_metrics/requirements_eval_extras.txt`); "
            "dino_i will be null."
        )
    gram_metric = GramStyleDistance(device, image_size=256)
    hpsv2_model = build_hpsv2() if args.compute_hps else None
    if args.compute_hps and hpsv2_model is None:
        print(
            "Warning: hpsv2 not importable; install `hpsv2` and ensure HuggingFace cache "
            "has CLIP ViT-H-14 + xswu/HPSv2 weights (offline: set HF_HOME / TRANSFORMERS_CACHE)."
        )

    fid_metric = None
    if args.compute_fid:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
        except ModuleNotFoundError as e:
            msg = str(e).lower()
            if "fidelity" in msg or "torch-fidelity" in msg:
                print(
                    "Warning: FID needs `torch-fidelity` "
                    "(see experiments/exp3_bench_metrics/requirements_eval_extras.txt); skipping FID."
                )
            else:
                print(f"Warning: FID unavailable ({e!r}); skipping FID.")
        except Exception as e:
            print(f"Warning: FID init failed ({type(e).__name__}: {e}); skipping FID.")
            fid_metric = None

    input_path = Path(args.input).expanduser().resolve()
    idx2gen: Dict[int, str] = {}
    idx2meta: Dict[int, str] = {}
    input_kind = ""
    if input_path.is_file():
        input_kind = "ip_adapter_generation_results"
        idx2gen = build_sample_map_from_ip_adapter_results(input_path)
    else:
        input_kind = "directory_samples"
        idx2gen, idx2meta = build_sample_map_from_dir(input_path)

    from tools.eval_userpref_outputs import load_meta_prompt

    agg_keys = ["clip_i", "clip_t", "dino_i", "lpips_target", "gram_target"]
    if args.compute_hps:
        agg_keys.append("hpsv2")
    agg: Dict[str, List[float]] = {k: [] for k in agg_keys}
    per_sample: List[dict] = []

    for idx, sample in enumerate(tqdm(test_data, desc="benchmark metrics")):
        gen_path = idx2gen.get(int(idx), "")
        gen_ok = bool(gen_path and Path(gen_path).is_file())
        target_path = select_target_from_candidates(sample, repo_root=repo, data_root=data_root)
        tgt_ok = bool(target_path and Path(target_path).is_file())

        prompt_raw = str((sample.get("target_item_info") or {}).get("caption") or "").strip()
        meta_path_s = idx2meta.get(int(idx), "")
        if args.prefer_meta_prompt and meta_path_s:
            pr, _pu = load_meta_prompt(Path(meta_path_s))
            prompt_raw = pr.strip() or prompt_raw

        uid = str(sample.get("worker_id") or "").strip()
        qv = str((sample.get("target_item_info") or {}).get("query_variant") or "").strip()

        m: Dict[str, Optional[float]] = {
            "clip_i": None,
            "clip_t": None,
            "dino_i": None,
            "lpips_target": None,
            "gram_target": None,
            "hpsv2": None,
        }

        if gen_ok and tgt_ok:
            gen_t = image_to_tensor(gen_path, image_size=int(args.image_size), device=device)
            tgt_t = image_to_tensor(str(target_path), image_size=int(args.image_size), device=device)
            gen_pil = image_path_to_pil(gen_path)
            tgt_pil = image_path_to_pil(str(target_path))

            if lpips_metric is not None:
                with torch.no_grad():
                    v = float(lpips_metric(gen_t * 2 - 1, tgt_t * 2 - 1).item())
                    m["lpips_target"] = v
                    agg["lpips_target"].append(v)

            if clip_model is not None and clip_transform is not None:
                with torch.no_grad():
                    gf = clip_image_feature(clip_model, gen_t, clip_transform)
                    tf_ = clip_image_feature(clip_model, tgt_t, clip_transform)
                    m["clip_i"] = float((gf @ tf_.T).mean().item())
                    agg["clip_i"].append(float(m["clip_i"]))
                    if prompt_raw:
                        txt = clip_text_feature(clip_model, prompt_raw.replace("*", ""), device=device)
                        m["clip_t"] = float((gf @ txt.T).mean().item())
                        agg["clip_t"].append(float(m["clip_t"]))

            if dino_model is not None and dino_transform is not None:
                with torch.no_grad():
                    df = dino_feature(dino_model, dino_transform, gen_pil, device=device)
                    dt = dino_feature(dino_model, dino_transform, tgt_pil, device=device)
                    m["dino_i"] = float((df @ dt.T).mean().item())
                    agg["dino_i"].append(float(m["dino_i"]))

            try:
                g = float(gram_metric(gen_pil, tgt_pil))
                m["gram_target"] = g
                agg["gram_target"].append(g)
            except Exception:
                pass

            if hpsv2_model is not None and prompt_raw:
                from tools.eval_userpref_outputs import _hpsv2_score

                hv = _hpsv2_score(hpsv2_model, gen_path, prompt_raw)
                if hv is not None:
                    m["hpsv2"] = hv
                    agg["hpsv2"].append(hv)

            if fid_metric is not None:
                fid_metric.update(gen_t.clamp(0, 1), real=False)
                fid_metric.update(tgt_t.clamp(0, 1), real=True)

        per_sample.append(
            {
                "sample_idx": idx,
                "user_id": uid,
                "query_variant": qv,
                "gen_path": gen_path or None,
                "target_path": str(target_path) if tgt_ok else None,
                "prompt": prompt_raw,
                "metrics": m,
            }
        )

    summary = {k: _mean_std(v) for k, v in agg.items()}
    fid_score = None
    if fid_metric is not None:
        try:
            fid_metric.eval()
            fid_score = float(fid_metric.compute().item())
        except Exception as e:
            fid_score = None
            summary["fid_error"] = str(e)

    out = {
        "dataset": "userpref_v1",
        "test_json": str(test_json_path),
        "input": str(input_path),
        "input_kind": input_kind,
        "summary": summary,
        "fid_vs_target_distribution": fid_score,
        "per_sample": per_sample,
        "notes": {
            "gram_target": "sum of Frobenius norms between Gram matrices (relu1_1..relu4_1 VGG19); lower is closer style",
            "fid": "torchmetrics FrechetInceptionDistance(feature=64) between generated and target images (paired pool); interpret with care for small N",
            "deps": "FID needs pip install torch-fidelity. DINO-I uses torch.hub facebookresearch/dino by default (or timm+USERPREF_DINO_CKPT). HPSv2 needs pip install hpsv2 + HF weights (see tools/eval_userpref_outputs.build_hpsv2).",
        },
    }

    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", outp, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        json.dumps(
            {"fid_vs_target_distribution": fid_score},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
