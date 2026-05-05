from pathlib import Path

import torch

from method.dreambooth.utils.lora_utils import load_lora_into_pipeline


def maybe_add_instance_token(tokenizer, text_encoder, token: str, initializer_token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id != tokenizer.unk_token_id:
        return token_id

    num_added = tokenizer.add_tokens([token])
    if num_added != 1:
        raise ValueError(f"Failed to add instance token '{token}'.")

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"Token '{token}' still maps to unk after adding.")

    init_id = tokenizer.convert_tokens_to_ids(initializer_token)
    if init_id == tokenizer.unk_token_id:
        init_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    with torch.no_grad():
        embeds = text_encoder.get_input_embeddings().weight
        embeds[token_id] = embeds[init_id].clone()

    return token_id


def build_infer_pipeline(*, sd15_path: str, lora_dir: str, token: str, initializer_token: str, device: torch.device):
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(sd15_path, safety_checker=None, torch_dtype=dtype).to(device)

    ok = load_lora_into_pipeline(pipe, lora_dir)
    if not ok:
        return None, False, False

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    token_id_before = tokenizer.convert_tokens_to_ids(token)
    _ = maybe_add_instance_token(tokenizer, text_encoder, token, initializer_token)
    token_initialized = bool(token_id_before == tokenizer.unk_token_id)
    token_loaded = False
    return pipe, token_loaded, token_initialized
