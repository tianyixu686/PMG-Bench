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


def load_token_embedding_if_exists(pipe, *, token: str, initializer_token: str, lora_dir: str) -> tuple[bool, bool]:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        tokenizer.add_tokens([token])
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(token)

    token_path = Path(lora_dir) / "learned_token.bin"
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

    if not token_loaded:
        init_id = tokenizer.convert_tokens_to_ids(initializer_token)
        if init_id == tokenizer.unk_token_id:
            init_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        if init_id != tokenizer.unk_token_id:
            with torch.no_grad():
                embeds = text_encoder.get_input_embeddings().weight
                embeds[token_id] = embeds[init_id].clone()
            token_initialized = True

    return token_loaded, token_initialized


def build_infer_pipeline(*, sd15_path: str, lora_dir: str, token: str, initializer_token: str, device: torch.device):
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(sd15_path, safety_checker=None, torch_dtype=dtype).to(device)

    ok = load_lora_into_pipeline(pipe, lora_dir)
    if not ok:
        return None, False, False

    token_loaded, token_initialized = load_token_embedding_if_exists(
        pipe,
        token=token,
        initializer_token=initializer_token,
        lora_dir=lora_dir,
    )
    return pipe, token_loaded, token_initialized
