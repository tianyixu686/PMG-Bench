import argparse
import json
import os
import re
import shutil
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
    cand1 = (data_root / p).resolve()
    if cand1.exists() or str(raw_path).replace("\\", "/").startswith("images/"):
        return str(cand1)
    return str((repo_root / p).resolve())


def select_target_from_candidates(sample: dict, *, repo_root: Path, data_root: Path) -> str:
    # Prefer the best (highest preference_score) stage2 candidate if available; otherwise fallback to target_item_info.
    candidates = sample.get("stage2_candidates") or []
    best_path = ""
    best_score = -1e18
    for item in candidates:
        path = resolve_image_path(str(item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)
        if not path or not os.path.exists(path):
            continue
        try:
            score = float(item.get("preference_score"))
        except Exception:
            score = float("nan")
        if score == score and score > best_score:
            best_score = score
            best_path = path
    if best_path:
        return best_path
    target_item = sample.get("target_item_info") or {}
    return resolve_image_path(str(target_item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)


def image_to_tensor(image_path: str, *, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


def image_path_to_pil(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def _safe_std(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(np.array(values, dtype=np.float64)))


def build_clip(device: torch.device):
    try:
        import clip  # openai/clip
    except Exception:
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


@torch.no_grad()
def clip_image_feature(clip_model, image_tensor: torch.Tensor, clip_transform) -> torch.Tensor:
    x = clip_transform(image_tensor)
    feat = clip_model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


@torch.no_grad()
def clip_text_feature(clip_model, text: str, device: torch.device) -> torch.Tensor:
    import clip

    tokens = clip.tokenize([text]).to(device)
    feat = clip_model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def build_dino(device: torch.device):
    """
    DINO ViT-B/16 image embedding for cosine similarity (DINO-I).

    Order:
      1) timm `vit_base_patch16_224.dino` — respects HF_ENDPOINT (e.g. https://hf-mirror.com) for weights.
      2) torch.hub facebookresearch/dino — needs GitHub zip + dl.fbaipublicfiles.com (can be slow/blocked).
      3) timm + USERPREF_DINO_CKPT — fully offline if you have a local checkpoint file.
    """
    timm_err: Optional[str] = None

    # --- 1) timm + Hugging Face (mirror-friendly) ---
    try:
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
    except Exception as e:
        timm_err = f"timm import: {type(e).__name__}: {e}"
    else:
        ckpt = os.environ.get("USERPREF_DINO_CKPT", "").strip()
        try:
            if ckpt and Path(ckpt).is_file():
                model = timm.create_model(
                    "vit_base_patch16_224.dino",
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                    checkpoint_path=ckpt,
                )
            else:
                os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
                os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
                model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0, global_pool="avg")
            model.eval().to(device)
            cfg = resolve_data_config({}, model=model)
            tfm = create_transform(**cfg, is_training=False)
            return model, tfm
        except Exception as e:
            timm_err = f"timm load: {type(e).__name__}: {e}"

    # --- 2) torch.hub Facebook DINO ---
    try:
        import torchvision.transforms as T

        model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=True,
            trust_repo=True,
        )
        model.eval().to(device)
        tfm = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return model, tfm
    except Exception as hub_e:
        hub_msg = f"{type(hub_e).__name__}: {hub_e}"

    print(
        "Warning: DINO unavailable. "
        f"timm branch: {timm_err}; torch.hub: {hub_msg}. "
        "Use HF_ENDPOINT mirror for timm, or set USERPREF_DINO_CKPT, or pre-populate torch hub cache."
    )
    return None, None


@torch.no_grad()
def dino_feature(dino_model, dino_transform, image_pil: Image.Image, device: torch.device) -> torch.Tensor:
    x = dino_transform(image_pil).unsqueeze(0).to(device)
    feat = dino_model(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def build_ssim(device: torch.device):
    try:
        from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

        # Images are in [0,1] from ToTensor, so data_range=1.0 is the consistent choice.
        metric = SSIM(data_range=1.0).to(device)
        return metric
    except Exception:
        return None


def _ensure_hpsv2_open_clip_bpe() -> None:
    """PyPI wheel for hpsv2 often omits open_clip BPE vocab; reuse OpenAI clip package file."""
    try:
        import clip  # type: ignore
        import hpsv2  # type: ignore

        dst_dir = Path(hpsv2.__file__).resolve().parent / "src" / "open_clip"
        dst = dst_dir / "bpe_simple_vocab_16e6.txt.gz"
        if dst.is_file():
            return
        src = Path(clip.__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
        if not src.is_file():
            return
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception:
        pass


def build_hpsv2():
    try:
        import hpsv2  # type: ignore

        _ensure_hpsv2_open_clip_bpe()
        return hpsv2
    except Exception:
        try:
            from hpsv2 import HPSv2  # type: ignore

            _ensure_hpsv2_open_clip_bpe()

            # normalize to an object exposing .score(img_path, prompt, hps_version="v2.1")
            class _Wrapper:
                def __init__(self):
                    self._m = HPSv2()

                def score(self, imgs_path, prompt, hps_version="v2.1"):
                    return self._m.score(imgs_path, prompt, hps_version=hps_version)

            return _Wrapper()
        except Exception:
            return None


def _hpsv2_score(hpsv2_model, image_path: str, prompt: str) -> Optional[float]:
    if hpsv2_model is None:
        return None
    try:
        out = hpsv2_model.score(image_path, prompt, hps_version="v2.1")
        if isinstance(out, (list, tuple)) and out:
            out = out[0]
        if hasattr(out, "item"):
            return float(out.item())
        return float(out)
    except Exception as e:
        if not getattr(_hpsv2_score, "_warned", False):
            setattr(_hpsv2_score, "_warned", True)
            print(f"Warning: HPSv2 scoring failed ({type(e).__name__}: {e}); further HPSv2 errors suppressed.")
        return None


def build_aesthetic():
    """
    Optional LAION aesthetic predictor.
    We keep this best-effort: if deps/weights aren't available, returns None.
    """
    try:
        from aesthetic_predictor_v2_5 import AestheticPredictorV2_5  # type: ignore

        return AestheticPredictorV2_5
    except Exception:
        return None


def _aesthetic_score(predictor_ctor, device: torch.device, image_pil: Image.Image) -> Optional[float]:
    if predictor_ctor is None:
        return None
    try:
        # Lazy init per-process; predictor is small but loads weights.
        if not hasattr(_aesthetic_score, "_predictor"):
            _aesthetic_score._predictor = predictor_ctor().to(device)  # type: ignore[attr-defined]
            _aesthetic_score._predictor.eval()  # type: ignore[attr-defined]
        pred = _aesthetic_score._predictor  # type: ignore[attr-defined]
        with torch.no_grad():
            score = pred(image_pil)
        if hasattr(score, "item"):
            return float(score.item())
        return float(score)
    except Exception:
        return None


def load_meta_prompt(meta_path: Path) -> Tuple[str, str]:
    # Returns (prompt_raw, prompt_used) if meta exists, else ("", "").
    if not meta_path.exists():
        return "", ""
    try:
        meta = load_json(meta_path)
        return str(meta.get("prompt_raw") or ""), str(meta.get("prompt") or "")
    except Exception:
        return "", ""


def load_meta_timing(meta_path: Path) -> Optional[dict]:
    if not meta_path.exists():
        return None
    try:
        meta = load_json(meta_path)
        timing = meta.get("timing")
        return timing if isinstance(timing, dict) else None
    except Exception:
        return None


def build_sample_map_from_dir(root: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Recursively find images matching **/sample_????/gen_0.jpg under root.
    Returns:
      - sample_idx -> gen_image_path
      - sample_idx -> meta_json_path (if exists)

    This adapts to outputs organized as:
      outputs/<method>/<exp>/sample_0000/gen_0.jpg
      outputs/<method>/<user_id>/sample_0000/gen_0.jpg
      outputs/<method>/<exp>/<user_id>/sample_0000/gen_0.jpg
    """
    idx2gen: Dict[int, str] = {}
    idx2meta: Dict[int, str] = {}
    pat = re.compile(r"sample_(\d{4})$")

    for cur_root, _dirs, files in os.walk(str(root)):
        base = os.path.basename(cur_root)
        m = pat.match(base)
        if not m:
            continue
        idx = int(m.group(1))
        if "gen_0.jpg" in files and idx not in idx2gen:
            gen_path = os.path.join(cur_root, "gen_0.jpg")
            if os.path.exists(gen_path):
                idx2gen[idx] = gen_path
        if "meta.json" in files and idx not in idx2meta:
            meta_path = os.path.join(cur_root, "meta.json")
            if os.path.exists(meta_path):
                idx2meta[idx] = meta_path

    return idx2gen, idx2meta


def build_sample_map_from_ip_adapter_results(path: Path) -> Dict[int, str]:
    """
    IP-Adapter userpref output:
      outputs/.../generation_results.json
      { results: [ { index: int, output_path: str, ... } ] }
    """
    obj = load_json(path)
    rows = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Not an ip-adapter generation_results.json: {path}")
    out: Dict[int, str] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        idx = r.get("index")
        img = r.get("output_path")
        try:
            idx_i = int(idx)
        except Exception:
            continue
        if not img:
            continue
        img_s = str(img)
        if os.path.exists(img_s):
            out[idx_i] = img_s
    return out


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_root_default = repo_root / "data" / "userpref_v1"

    p = argparse.ArgumentParser(
        description="Evaluate userpref_v1 runs with unified metrics (directory outputs or ip-adapter generation_results.json)"
    )
    p.add_argument("--test_json", type=str, default=str(data_root_default / "processed_dataset" / "test.json"))
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Either a directory (searched for **/sample_XXXX/gen_0.jpg) or an ip-adapter generation_results.json",
    )
    p.add_argument("--output_json", type=str, required=True, help="Where to write metrics JSON")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--user_ids", type=str, default="", help="Comma-separated subset users")
    p.add_argument("--prefer_meta_prompt", action="store_true", help="Use meta.json prompt_raw if present")

    args = p.parse_args()

    test_json_path = Path(args.test_json).resolve()
    if test_json_path.name.lower() == "test.json" and test_json_path.parent.name == "processed_dataset":
        data_root = test_json_path.parents[1]
    else:
        data_root = data_root_default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import lpips

        lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
    except Exception:
        lpips_metric = None

    clip_model, clip_transform = build_clip(device)
    dino_model, dino_transform = build_dino(device)
    ssim_metric = build_ssim(device)
    # NOTE: user request: only compute CLIP-I/DINO-I/CLIP-T + LPIPS/SSIM(target).
    # We intentionally do NOT compute history-avg, HPSv2, or aesthetic metrics here.

    test_data = load_json(Path(args.test_json))

    allow_users = None
    if str(args.user_ids).strip():
        allow_users = {x.strip() for x in str(args.user_ids).split(",") if x.strip()}

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

    results: List[dict] = []

    agg: Dict[str, List[float]] = {
        "clip_i": [],
        "dino_i": [],
        "clip_t": [],
        "lpips_target": [],
        "ssim_target": [],
    }

    for idx, sample in enumerate(tqdm(test_data, desc="Evaluating outputs")):
        uid = str(sample.get("worker_id") or "").strip()
        if allow_users is not None and uid not in allow_users:
            continue

        gen_path_s = idx2gen.get(int(idx), "")
        gen_found = bool(gen_path_s and os.path.exists(gen_path_s))

        target_path = select_target_from_candidates(sample, repo_root=repo_root, data_root=data_root)
        target_found = bool(target_path and os.path.exists(target_path))

        prompt_raw = str((sample.get("target_item_info") or {}).get("caption") or "").strip()
        prompt_used = ""
        meta_path_s = idx2meta.get(int(idx), "")
        if bool(args.prefer_meta_prompt) and meta_path_s:
            pr, pu = load_meta_prompt(Path(meta_path_s))
            prompt_raw = pr.strip() or prompt_raw
            prompt_used = pu.strip()

        m: Dict[str, Optional[float]] = {
            "clip_i": None,
            "dino_i": None,
            "clip_t": None,
            "lpips_target": None,
            "ssim_target": None,
        }

        if gen_found and target_found:
            generated_tensor = image_to_tensor(gen_path_s, image_size=int(args.image_size), device=device)
            target_tensor = image_to_tensor(str(target_path), image_size=int(args.image_size), device=device)

            if lpips_metric is not None:
                with torch.no_grad():
                    m["lpips_target"] = float(lpips_metric(generated_tensor * 2 - 1, target_tensor * 2 - 1).item())
                    agg["lpips_target"].append(float(m["lpips_target"]))

            if ssim_metric is not None:
                with torch.no_grad():
                    m["ssim_target"] = float(ssim_metric(generated_tensor, target_tensor).item())
                    agg["ssim_target"].append(float(m["ssim_target"]))

            gen_feat = None
            if clip_model is not None and clip_transform is not None:
                gen_feat = clip_image_feature(clip_model, generated_tensor, clip_transform)
                tgt_feat = clip_image_feature(clip_model, target_tensor, clip_transform)
                m["clip_i"] = float((gen_feat @ tgt_feat.T).mean().item())
                agg["clip_i"].append(float(m["clip_i"]))

                if prompt_raw:
                    txt_feat = clip_text_feature(clip_model, prompt_raw.replace("*", ""), device=device)
                    m["clip_t"] = float((gen_feat @ txt_feat.T).mean().item())
                    agg["clip_t"].append(float(m["clip_t"]))

            if dino_model is not None and dino_transform is not None:
                gen_pil = image_path_to_pil(gen_path_s)
                tgt_pil = image_path_to_pil(str(target_path))
                gen_df = dino_feature(dino_model, dino_transform, gen_pil, device=device)
                tgt_df = dino_feature(dino_model, dino_transform, tgt_pil, device=device)
                m["dino_i"] = float((gen_df @ tgt_df.T).mean().item())
                agg["dino_i"].append(float(m["dino_i"]))

        results.append(
            {
                "sample_idx": int(idx),
                "user_id": uid,
                "gen_path": (gen_path_s or None),
                "gen_found": bool(gen_found),
                "target_path": (str(target_path) if target_found else None),
                "target_found": bool(target_found),
                "prompt_raw": prompt_raw,
                "prompt_used": prompt_used,
                "metrics": m,
                "input_kind": input_kind,
                "input_root": str(input_path),
            }
        )

    summary = {
        k: {"mean": _safe_mean(v), "std": _safe_std(v), "n": int(len(v))}
        for k, v in agg.items()
    }

    out = {
        "dataset": "userpref_v1",
        "test_json": str(test_json_path),
        "input": str(input_path),
        "input_kind": str(input_kind),
        "total_samples_in_test_json": int(len(test_data)),
        "evaluated_samples": int(len(results)),
        "summary": summary,
        "per_sample": results,
        "notes": {
            "clip": "OpenAI CLIP ViT-B/32; cosine similarity on L2-normalized embeddings",
            "dino": "timm vit_base_patch16_224.dino; cosine similarity on L2-normalized pooled features",
            "clip_t_text": "uses target_item_info.caption (weak request text); --prefer_meta_prompt can override when meta.json is available",
            "lpips_target": "lpips vgg between gen and target; lower is better",
            "ssim_target": "SSIM (data_range=1.0) between gen and target; higher is better",
            "missing_samples": "If a sample's gen image is missing under the input, metrics are null; each metric's n is reported in summary",
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

