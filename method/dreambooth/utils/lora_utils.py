from __future__ import annotations

import os
from typing import Optional


def load_lora_into_pipeline(
    pipe,
    lora_dir: str,
    *,
    weight_name: Optional[str] = None,
) -> bool:
    candidates = []
    if weight_name:
        candidates.append(weight_name)
    candidates.extend(["pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"])

    for cand in candidates:
        if not os.path.exists(os.path.join(lora_dir, cand)):
            continue
        try:
            pipe.load_lora_weights(lora_dir, weight_name=cand)
            return True
        except Exception:
            return False

    try:
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(lora_dir)
            return True
    except Exception:
        return False

    return False
