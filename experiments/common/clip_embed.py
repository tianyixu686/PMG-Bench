"""CLIP image embeddings (ViT-B/32), same normalization as tools/eval_userpref_outputs.py."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms


def build_clip(device: torch.device):
    try:
        import clip  # type: ignore
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


def tensor_from_rgb_path(path: str, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    t = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return t


@torch.no_grad()
def embed_image_paths(
    paths: List[str],
    *,
    device: torch.device,
    image_size: int = 512,
    batch_size: int = 32,
) -> Tuple[Optional[torch.Tensor], List[int]]:
    """
    Returns (embeddings [N,D], valid_indices) for paths that exist.
    If CLIP unavailable, returns (None, []).
    """
    valid_idx: List[int] = []
    valid_paths: List[str] = []
    for i, p in enumerate(paths):
        if p and Path(p).is_file():
            valid_idx.append(i)
            valid_paths.append(p)
    if not valid_paths:
        return None, []

    model, tfm = build_clip(device)
    if model is None or tfm is None:
        return None, valid_idx

    out_chunks: List[torch.Tensor] = []
    for start in range(0, len(valid_paths), batch_size):
        batch_paths = valid_paths[start : start + batch_size]
        tensors = []
        for vp in batch_paths:
            tensors.append(tensor_from_rgb_path(vp, image_size, device))
        batch = torch.cat(tensors, dim=0)
        feat = clip_image_feature(model, batch, tfm)
        out_chunks.append(feat.cpu())
    emb = torch.cat(out_chunks, dim=0)
    return emb, valid_idx
