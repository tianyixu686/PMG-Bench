from __future__ import annotations

import os
from typing import List, Literal, Tuple

import torch


FreezeModel = Literal["crossattn_kv", "crossattn"]


def select_trainable_parameters(
    unet,
    *,
    freeze_model: FreezeModel = "crossattn_kv",
    train_self_attention: bool = False,
) -> Tuple[List[torch.nn.Parameter], List[str]]:
    """Return (params, param_names) for Custom Diffusion training.

    - freeze_model="crossattn_kv": only trains cross-attention K/V projections ("attn2.to_k"/"attn2.to_v").
    - freeze_model="crossattn": trains all cross-attention parameters (any parameter name containing "attn2").

    If train_self_attention=True, also includes self-attention blocks ("attn1").

    Returned names match `unet.state_dict()` keys so they can be saved/loaded as a delta.
    """

    if freeze_model not in ("crossattn_kv", "crossattn"):
        raise ValueError("freeze_model must be 'crossattn_kv' or 'crossattn'")

    params: List[torch.nn.Parameter] = []
    names: List[str] = []

    for name, param in unet.named_parameters():
        is_cross = "attn2" in name
        is_self = "attn1" in name
        if not (is_cross or (train_self_attention and is_self)):
            continue

        if freeze_model == "crossattn_kv":
            if ("attn2.to_k" not in name) and ("attn2.to_v" not in name):
                continue

        params.append(param)
        names.append(name)

    if not params:
        raise RuntimeError("No trainable attention parameters found in UNet; diffusers version may be incompatible.")

    return params, names


def save_delta(
    unet,
    delta_path: str,
    train_param_names: List[str],
    *,
    metadata: dict | None = None,
):
    os.makedirs(os.path.dirname(delta_path) or ".", exist_ok=True)
    state = unet.state_dict()
    delta = {k: state[k].detach().cpu() for k in train_param_names if k in state}
    payload = {
        "state_dict": delta,
        "train_param_names": list(train_param_names),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    torch.save(payload, delta_path)


def load_delta(unet, delta_path: str, *, map_location: str | torch.device = "cpu"):
    payload = torch.load(delta_path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    else:
        # legacy format: directly a state_dict subset
        state_dict = payload

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    return missing, unexpected
