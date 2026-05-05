import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

def _to_transformers_cache(past_key_values_legacy):
    """
    Convert legacy tuple(layer)->(k,v) cache to a transformers Cache object when required.
    Newer transformers versions expect `past_key_values.get_seq_length()` to exist.
    """
    try:
        from transformers.cache_utils import DynamicCache  # type: ignore

        return DynamicCache.from_legacy_cache(past_key_values_legacy)
    except Exception:
        return past_key_values_legacy

def _layer_device(llama_model, layer_idx: int) -> torch.device:
    layer = llama_model.model.layers[int(layer_idx)]
    try:
        return next(layer.parameters()).device
    except StopIteration:
        return getattr(llama_model, "device", torch.device("cpu"))


def _torch_dtype(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def _resolve_image_path(img_path: str, *, repo_root: Path, data_root: Path) -> str:
    if not img_path:
        return ""
    p = Path(str(img_path))
    if p.is_absolute():
        return str(p)
    cand1 = (data_root / p).resolve()
    if cand1.exists() or str(img_path).replace("\\", "/").startswith("images/"):
        return str(cand1)
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


def save_image_grid(images, save_path: str):
    if len(images) == 0:
        return
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    img_h, img_w = images[0].shape[:2]
    grid = np.ones((rows * img_h, cols * img_w, 3), dtype=np.uint8) * 255
    for idx, img in enumerate(images):
        row, col = idx // cols, idx % cols
        grid[row * img_h : (row + 1) * img_h, col * img_w : (col + 1) * img_w] = img
    Image.fromarray(grid).save(save_path)


def prompt_preprocess(history_captions: str) -> str:
    prompt = (
        '### Human: A person rated the following images highly: "<Images/>". '
        'Describe their visual taste. ###Assistant: '
    )
    return prompt.replace("<Images/>", history_captions)


def load_user_styles(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either list[{user_id, style_text}] or dict[user_id]=style_text
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    out: Dict[str, str] = {}
    if isinstance(data, list):
        for row in data:
            uid = row.get("user_id") if isinstance(row, dict) else None
            st = row.get("style") if isinstance(row, dict) else None
            if uid is None or st is None:
                continue
            out[str(uid)] = str(st)
    return out


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_processed_root = repo_root / "data" / "userpref_v1" / "processed_dataset"

    p = argparse.ArgumentParser(description="PMG inference on userpref_v1")
    p.add_argument("--test_json", type=str, default=str(default_processed_root / "test.json"))

    p.add_argument("--llama_path", type=str, required=True)
    p.add_argument("--sd_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)

    p.add_argument("--output_dir", type=str, default=str(repo_root / "outputs" / "userpref_v1_pmg_infer"))

    p.add_argument("--user_style_json", type=str, default="", help="Optional mapping for 3rd conditioning route")

    p.add_argument("--num_image_prompt", type=int, default=2)
    p.add_argument("--num_prefix_prompt", type=int, default=2)
    p.add_argument("--max_txt_len", type=int, default=600)
    p.add_argument("--image_size", type=int, default=512)

    p.add_argument("--weight_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--num_images_per_sample", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg_prompt", type=str, default="lowres, text, error, cropped, worst quality, low quality")

    p.add_argument("--weight_target", type=float, default=1.0)
    p.add_argument("--weight_image", type=float, default=1.0)
    p.add_argument("--weight_preference", type=float, default=0.0, help="Set >0 to enable 3rd route")

    # PMG baseline uses raw captions by default (no DreamBooth-style masking layer here).

    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = _torch_dtype(a.weight_dtype)

    test_json = Path(a.test_json)
    test_json_abs = test_json.resolve()
    if test_json_abs.name.lower() == "test.json" and test_json_abs.parent.name == "processed_dataset":
        data_root = test_json_abs.parents[1]
    else:
        data_root = repo_root / "data" / "userpref_v1"
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with test_json.open("r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples: {test_json}")

    user_styles = load_user_styles(Path(a.user_style_json)) if a.user_style_json else {}

    def get_user_style_text(user_id: str) -> str:
        if a.weight_preference <= 0:
            return ""
        st = user_styles.get(str(user_id), "")
        return st

    # LLaMA
    from transformers import LlamaForCausalLM, LlamaTokenizer

    llama_tokenizer = LlamaTokenizer.from_pretrained(a.llama_path)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model = LlamaForCausalLM.from_pretrained(
        a.llama_path,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    llama_model.eval()

    # SD
    from myCustomPipeline import SDPipeline

    sd_pipeline = SDPipeline(weight_dtype, model_id=a.sd_path).to(device)

    # Inference model (same structure as training)
    class PrefixEncoder(torch.nn.Module):
        def __init__(self, num_hidden_layers, hidden_size, pre_seq_len, prefix_projection=False, prefix_hidden_size=4096):
            super().__init__()
            self.prefix_projection = prefix_projection
            if self.prefix_projection:
                self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
                self.trans = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, prefix_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size),
                )
            else:
                self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

        def forward(self, prefix):
            if self.prefix_projection:
                past_key_values = self.trans(self.embedding(prefix))
            else:
                past_key_values = self.embedding(prefix)
            return past_key_values.to(weight_dtype)

    class InferenceModel(torch.nn.Module):
        def __init__(self, layer_num, num_image_prompt, num_prefix_prompt, emb_dim, sd_hidden_state_dim):
            super().__init__()
            self.layer_num = layer_num
            self.num_image_prompt = num_image_prompt
            self.num_prefix_prompt = num_prefix_prompt
            self.emb_dim = emb_dim
            self.mapping_layer = torch.nn.Linear(emb_dim, sd_hidden_state_dim)
            self.trainable_prompt = torch.nn.Parameter(torch.randn((1, num_image_prompt, emb_dim), requires_grad=True))
            self.prefix_tokens = torch.arange(num_prefix_prompt).long()
            self.prefix_encoder = PrefixEncoder(layer_num, 4096, num_prefix_prompt)

        def forward(self, llama_model, token, token_len):
            bsz = token.shape[0]
            attention_mask = token != llama_tokenizer.pad_token_id
            emb = llama_model.model.embed_tokens(token)
            for i in range(bsz):
                l = token_len[i].item()
                emb[i, l : l + self.num_image_prompt] = self.trainable_prompt
                attention_mask[i, l : l + self.num_image_prompt] = 1
            attention_mask = torch.concat(
                [torch.ones((bsz, self.num_prefix_prompt), device=attention_mask.device), attention_mask], dim=1
            )

            # Transformers compatibility: some versions expose `num_heads`, others only have it in config.
            num_head = getattr(llama_model.config, "num_attention_heads", None)
            if num_head is None:
                num_head = getattr(llama_model.config, "num_heads", None)
            if num_head is None:
                num_head = getattr(getattr(llama_model.model.layers[0], "self_attn", object()), "num_heads", None)
            if num_head is None:
                raise AttributeError("Cannot resolve number of attention heads from llama_model")
            prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(token.device)
            past = self.prefix_encoder(prefix_tokens)
            # Convert to legacy KV cache format: tuple[layer] of (key, value)
            past = past.view(bsz, self.num_prefix_prompt, self.layer_num, 2, num_head, -1).permute(2, 3, 0, 4, 1, 5)
            legacy_list = []
            for i in range(self.layer_num):
                dev = _layer_device(llama_model, i)
                legacy_list.append((past[i, 0].to(dev), past[i, 1].to(dev)))
            past_key_values_legacy = tuple(legacy_list)
            past_key_values = _to_transformers_cache(past_key_values_legacy)

            outputs = llama_model.model.forward(
                inputs_embeds=emb,
                output_hidden_states=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            encoder_hidden_states = [
                outputs.last_hidden_state[i, token_len[i] : token_len[i] + self.num_image_prompt] for i in range(bsz)
            ]
            encoder_hidden_states = torch.stack(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.to(self.mapping_layer.weight.device)
            return self.mapping_layer(encoder_hidden_states)

    infer_model = InferenceModel(
        layer_num=len(llama_model.model.layers),
        num_image_prompt=a.num_image_prompt,
        num_prefix_prompt=a.num_prefix_prompt,
        emb_dim=4096,
        sd_hidden_state_dim=768,
    ).to(device)

    state = torch.load(a.checkpoint_path, map_location="cpu")
    infer_model.load_state_dict(state, strict=True)
    infer_model = infer_model.to(weight_dtype)
    infer_model.eval()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    for idx, sample in enumerate(tqdm(test_data, desc="Generating")):
        user_id = str(sample.get("worker_id"))
        history_items = sample.get("history_items_info", [])
        target_item = sample.get("target_item_info", {})

        query_variant = target_item.get("query_variant", "")

        history_captions = []
        for k, item in enumerate(history_items):
            cap_raw = str(item.get("caption") or "").strip()
            cap = cap_raw
            if cap:
                history_captions.append(f"{k+1}. {cap}")
        history_text = " ".join(history_captions)
        prompt_text = prompt_preprocess(history_text)

        max_prompt_len = max(1, int(a.max_txt_len) - int(a.num_image_prompt))
        product_token = (
            llama_tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
            )
            .input_ids[0]
            .tolist()
        )
        token_len = len(product_token)
        product_token += [llama_tokenizer.pad_token_id] * (a.max_txt_len - len(product_token))
        input_ids = torch.tensor(product_token).unsqueeze(0).to(device)
        token_len_tensor = torch.tensor([token_len]).to(device)

        target_caption = str(target_item.get("caption") or "").strip()
        if target_caption:
            target_emb = sd_pipeline.textEncode(target_caption, num_tokens=40)
        else:
            target_emb = torch.zeros(1, 40, 768, device=device, dtype=weight_dtype)

        with torch.no_grad():
            image_emb = infer_model.forward(llama_model, input_ids, token_len_tensor)

        style_text = get_user_style_text(user_id)
        if a.weight_preference > 0 and style_text:
            style_emb = sd_pipeline.textEncode(style_text, num_tokens=35)
        else:
            style_emb = torch.zeros(1, 0, 768, device=device, dtype=weight_dtype)

        combined_emb = torch.cat(
            [
                target_emb * float(a.weight_target),
                image_emb * float(a.weight_image),
                style_emb * float(a.weight_preference) if style_emb.shape[1] > 0 else style_emb,
            ],
            dim=1,
        )

        generated_images = []
        for img_idx in range(a.num_images_per_sample):
            gen = sd_pipeline.generate(
                combined_emb,
                negative_prompt=a.neg_prompt,
                generator=[torch.manual_seed(a.seed + idx * 100 + img_idx)],
                show_processbar=False,
            )
            generated_images.append(gen[0])

        # Match `method/PMG/evaluation.py` convention: eval_output_dir/sample_0000/gen_0.jpg
        sample_dir = output_dir / f"sample_{idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, img in enumerate(generated_images):
            Image.fromarray(img).save(sample_dir / f"gen_{img_idx}.jpg")
        save_image_grid(generated_images, str(sample_dir / "grid.jpg"))

        # references (history + target reference)
        history_imgs = [
            _resize_rgb(item.get("image_path", ""), a.image_size, repo_root=repo_root, data_root=data_root)
            for item in history_items
        ]
        target_img = _resize_rgb(target_item.get("image_path", ""), a.image_size, repo_root=repo_root, data_root=data_root)
        save_image_grid(history_imgs + [target_img], str(sample_dir / "references.jpg"))

        meta = {
            "sample_idx": int(idx),
            "user_id": user_id,
            "query_variant": query_variant,
            "history_captions_raw": [str(item.get("caption") or "") for item in history_items],
            "target_caption": target_caption,
            "target_item": target_item,
            "stage2_candidates": sample.get("stage2_candidates", []),
            "conditioning": {
                "target_weight": float(a.weight_target),
                "image_weight": float(a.weight_image),
                "preference_weight": float(a.weight_preference),
                "style_text": style_text,
            },
        }
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
