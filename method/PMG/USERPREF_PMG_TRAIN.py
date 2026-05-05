import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

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


@dataclass
class Args:
    # Train loop
    validation_steps: int
    save_steps: int
    train_batch_size: int
    vaild_batch_size: int
    gradient_accumulation_steps: int
    vaild_image_num: int
    num_image_prompt: int
    num_prefix_prompt: int
    num_train_epochs: int
    learning_rate: float
    lr_scheduler: str
    lr_warmup_steps: int
    lr_num_cycles: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    image_size: int
    max_txt_len: int

    # Paths
    train_json: Path
    val_json: Path
    llama_path: Path
    sd_path: Path
    output_dir: Path
    model_name: str

    # Precision / accelerate
    weight_dtype: torch.dtype
    mixed_precision: str
    allow_tf32: bool
    report_to: str
    logging_dir: str
    dataloader_num_workers: int
    seed: int
    resume_from: str


def parse_args() -> Args:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data" / "userpref_v1" / "processed_dataset"

    p = argparse.ArgumentParser(description="PMG training on userpref_v1 (history -> taste embedding + query caption -> SD)")

    p.add_argument("--train_json", type=str, default=str(default_data_root / "train.json"))
    p.add_argument("--val_json", type=str, default=str(default_data_root / "val.json"))

    p.add_argument("--llama_path", type=str, required=True, help="Path to LLaMA weights/tokenizer")
    p.add_argument("--sd_path", type=str, required=True, help="Path or model id for Stable Diffusion (e.g. sd1.5)")

    p.add_argument("--output_dir", type=str, default=str(repo_root / "outputs" / "userpref_v1_pmg"))
    p.add_argument("--model_name", type=str, default="userpref_v1")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from", type=str, default="")

    p.add_argument("--validation_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--train_batch_size", type=int, default=6)
    p.add_argument("--vaild_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)

    p.add_argument("--vaild_image_num", type=int, default=1)
    p.add_argument("--num_image_prompt", type=int, default=2)
    p.add_argument("--num_prefix_prompt", type=int, default=2)

    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--lr_scheduler", type=str, default="linear")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_num_cycles", type=int, default=1)

    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-6)

    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--max_txt_len", type=int, default=600)

    p.add_argument("--weight_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    a = p.parse_args()

    return Args(
        validation_steps=a.validation_steps,
        save_steps=a.save_steps,
        train_batch_size=a.train_batch_size,
        vaild_batch_size=a.vaild_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        vaild_image_num=a.vaild_image_num,
        num_image_prompt=a.num_image_prompt,
        num_prefix_prompt=a.num_prefix_prompt,
        num_train_epochs=a.num_train_epochs,
        learning_rate=a.learning_rate,
        lr_scheduler=a.lr_scheduler,
        lr_warmup_steps=a.lr_warmup_steps,
        lr_num_cycles=a.lr_num_cycles,
        adam_beta1=a.adam_beta1,
        adam_beta2=a.adam_beta2,
        adam_weight_decay=a.adam_weight_decay,
        adam_epsilon=a.adam_epsilon,
        image_size=a.image_size,
        max_txt_len=a.max_txt_len,
        train_json=Path(a.train_json),
        val_json=Path(a.val_json),
        llama_path=Path(a.llama_path),
        sd_path=Path(a.sd_path),
        output_dir=Path(a.output_dir),
        model_name=a.model_name,
        weight_dtype=_torch_dtype(a.weight_dtype),
        mixed_precision=a.mixed_precision,
        allow_tf32=bool(a.allow_tf32),
        report_to=a.report_to,
        logging_dir=a.logging_dir,
        dataloader_num_workers=a.dataloader_num_workers,
        seed=a.seed,
        resume_from=a.resume_from,
    )


def prompt_preprocess(history_captions: str) -> str:
    prompt = (
        '### Human: A person rated the following images highly: "<Images/>". '
        'Describe their visual taste. ###Assistant: '
    )
    return prompt.replace("<Images/>", history_captions)


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
        arr = np.array(im, dtype=np.uint8)
        return np.ascontiguousarray(arr)
    except Exception:
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def _item_id(item_info: Dict[str, Any]) -> str:
    # Backward-compat with existing PMG scripts: prefer item_id, else item_name
    if "item_id" in item_info and item_info.get("item_id") is not None:
        return str(item_info.get("item_id"))
    return str(item_info.get("item_name"))


class UserPrefDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer,
        sd_pipeline,
        num_image_prompt: int,
        repo_root: Path,
        data_root: Path,
        repeats: int = 1,
        max_len: int = 600,
        mode: str = "train",
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.data = data
        self.max_len = max_len
        self._length = len(data) * repeats
        self.sd_pipeline = sd_pipeline
        self.num_image_prompt = int(num_image_prompt)
        self.repo_root = Path(repo_root)
        self.data_root = Path(data_root)

        self.item_emb_dict: Dict[str, torch.Tensor] = {}
        self.item_ids_dict: Dict[str, torch.Tensor] = {}

        unique_items: Dict[str, str] = {}
        for sample in data:
            for item_info in sample.get("history_items_info", []):
                item_id = _item_id(item_info)
                cap = item_info.get("caption")
                if item_id not in unique_items and cap:
                    unique_items[item_id] = cap
            target_info = sample.get("target_item_info", {})
            tid = _item_id(target_info)
            tcap = target_info.get("caption")
            if tid not in unique_items and tcap:
                unique_items[tid] = tcap

        print(f"[{mode}] Computing embeddings for {len(unique_items)} unique items...")
        items_list = list(unique_items.items())
        bs = 256
        for t in tqdm(range(0, len(items_list), bs), desc=f"[{mode}] Encoding"):
            batch_items = items_list[t : t + bs]
            batch_ids = [item_id for item_id, _ in batch_items]
            batch_captions = [caption for _, caption in batch_items]

            tokens_list = []
            for cap in batch_captions:
                # SD1.5 / CLIP typical max length = 77
                tokens = self.sd_pipeline.textEncode(cap, num_tokens=77, return_tokens=True).detach()[0]
                tokens_list.append(tokens)

            embs = self.sd_pipeline.textEncode(tokens=torch.stack(tokens_list, dim=0))
            for i, item_id in enumerate(batch_ids):
                self.item_ids_dict[item_id] = tokens_list[i].cpu()
                self.item_emb_dict[item_id] = embs[i].detach().cpu()

    def __len__(self):
        return self._length

    def __getitem__(self, i: int) -> Dict[str, Any]:
        i = i % len(self.data)
        sample = self.data[i]
        history_items = sample.get("history_items_info", [])
        target_item = sample.get("target_item_info", {})

        history_captions = [
            f"{k+1}. {item['caption']}"
            for k, item in enumerate(history_items)
            if item.get("caption")
        ]
        history_text = " ".join(history_captions)
        prompt_text = prompt_preprocess(history_text)
        # Ensure prompt always fits `max_len` while reserving slots for `num_image_prompt`.
        # We truncate tokenization instead of asserting/crashing on long histories.
        max_prompt_len = max(1, int(self.max_len) - int(self.num_image_prompt))
        product_token = (
            self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
            )
            .input_ids[0]
            .tolist()
        )

        example: Dict[str, Any] = {}
        example["token_len"] = len(product_token)
        # After truncation this should always hold, but keep a safe guard.
        assert self.max_len >= len(product_token) + self.num_image_prompt, f"len:{len(product_token)} max_len:{self.max_len}"
        product_token += [self.tokenizer.pad_token_id] * (self.max_len - len(product_token))
        example["input_ids"] = torch.tensor(product_token)

        target_id = _item_id(target_item)
        example["keywords_ids"] = self.item_ids_dict.get(target_id, torch.zeros(77, dtype=torch.long))
        example["keywords_emb"] = self.item_emb_dict.get(target_id, torch.zeros(77, 768))
        example["pixel_values"] = _resize_rgb(
            target_item.get("image_path", ""),
            size=args.image_size,
            repo_root=self.repo_root,
            data_root=self.data_root,
        )

        # negative sample
        nega_id = np.random.choice(list(self.item_emb_dict.keys()))
        example["nega_keywords_ids"] = self.item_ids_dict[nega_id]
        example["nega_keywords_emb"] = self.item_emb_dict[nega_id]

        nega_img_path = None
        for s in self.data:
            tinfo = s.get("target_item_info", {})
            if _item_id(tinfo) == nega_id:
                nega_img_path = tinfo.get("image_path")
                break
        if not nega_img_path:
            for s in self.data:
                for h in s.get("history_items_info", []):
                    if _item_id(h) == nega_id:
                        nega_img_path = h.get("image_path")
                        break
                if nega_img_path:
                    break

        example["nega_pixel_values"] = _resize_rgb(
            nega_img_path or "",
            size=args.image_size,
            repo_root=self.repo_root,
            data_root=self.data_root,
        )
        return example


def main():
    global args
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    train_json_abs = args.train_json.resolve()
    if train_json_abs.parent.name == "processed_dataset" and train_json_abs.parent.parent.name == "userpref_v1":
        data_root = train_json_abs.parents[1]
    else:
        data_root = repo_root / "data" / "userpref_v1"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger = get_logger(__name__)
    logging_dir = str(args.output_dir / args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=str(args.output_dir), logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Load data
    accelerator.print(f"Loading train: {args.train_json}")
    with args.train_json.open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    accelerator.print(f"Loading val: {args.val_json}")
    with args.val_json.open("r", encoding="utf-8") as f:
        valid_data = json.load(f)

    # LLaMA
    from transformers import LlamaForCausalLM, LlamaTokenizer

    llama_tokenizer = LlamaTokenizer.from_pretrained(str(args.llama_path))
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_model = LlamaForCausalLM.from_pretrained(
        str(args.llama_path),
        torch_dtype=args.weight_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    llama_model.requires_grad_(False)
    llama_model = accelerator.prepare(llama_model)

    # SD
    from myCustomPipeline import SDPipeline

    sd_pipeline = SDPipeline(args.weight_dtype, model_id=str(args.sd_path))
    torch.cuda.empty_cache()
    sd_pipeline = accelerator.prepare(sd_pipeline)

    # Dataset / loaders
    train_dataset = UserPrefDataset(
        train_data,
        tokenizer=llama_tokenizer,
        sd_pipeline=sd_pipeline,
        num_image_prompt=args.num_image_prompt,
        repo_root=repo_root,
        data_root=data_root,
        mode="train",
        repeats=1,
        max_len=args.max_txt_len,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    vaild_dataset = UserPrefDataset(
        valid_data,
        tokenizer=llama_tokenizer,
        sd_pipeline=sd_pipeline,
        num_image_prompt=args.num_image_prompt,
        repo_root=repo_root,
        data_root=data_root,
        mode="val",
        repeats=1,
        max_len=args.max_txt_len,
    )
    vaild_dataloader = torch.utils.data.DataLoader(
        vaild_dataset,
        batch_size=args.vaild_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    vaild_dataloader = accelerator.prepare(vaild_dataloader)

    # Model
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

        def forward(self, prefix: torch.Tensor):
            if self.prefix_projection:
                prefix_tokens = self.embedding(prefix)
                past_key_values = self.trans(prefix_tokens)
            else:
                past_key_values = self.embedding(prefix)
            return past_key_values

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
            # `past` is (B, P, L*2*H). Convert to legacy KV cache format:
            # tuple[layer] of (key, value), each (B, num_heads, P, head_dim).
            past = past.view(bsz, self.num_prefix_prompt, self.layer_num, 2, num_head, -1).permute(2, 3, 0, 4, 1, 5)
            # IMPORTANT: when LLaMA is sharded by `device_map="auto"`, each layer can live on a different GPU.
            # Cache tensors must be placed on the same device as the corresponding layer to avoid device mismatch
            # during `past_key_values.update(...)`.
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
            encoder_hidden_states = []
            for i in range(bsz):
                l = token_len[i].item()
                encoder_hidden_states.append(outputs.last_hidden_state[i, l : l + self.num_image_prompt])
            encoder_hidden_states = torch.stack(encoder_hidden_states)
            # If LLaMA is sharded across GPUs, last_hidden_state can live on a different device than mapping_layer.
            encoder_hidden_states = encoder_hidden_states.to(self.mapping_layer.weight.device)
            encoder_hidden_states = self.mapping_layer(encoder_hidden_states)
            return encoder_hidden_states

    model = InferenceModel(
        layer_num=len(llama_model.model.layers),
        num_image_prompt=args.num_image_prompt,
        num_prefix_prompt=args.num_prefix_prompt,
        emb_dim=4096,
        sd_hidden_state_dim=768,
    )
    model = accelerator.prepare(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    accelerator.print(f"Num train samples: {len(train_dataset)}")
    accelerator.print(f"Num epochs: {args.num_train_epochs}")
    accelerator.print(f"Steps/epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"Total steps: {max_train_steps}")

    if accelerator.is_main_process:
        from datetime import datetime

        accelerator.init_trackers(
            f"{args.model_name}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_p{args.num_prefix_prompt}_i{args.num_image_prompt}",
            config={
                "train_json": str(args.train_json),
                "val_json": str(args.val_json),
                "llama_path": str(args.llama_path),
                "sd_path": str(args.sd_path),
            },
        )

    logger.info("***** Running training *****")
    global_step = 0
    first_epoch = 0

    if args.resume_from:
        resume_from = str(args.resume_from).strip()
        accelerator.print(f"[RESUME] requested: {resume_from}")
        resume_path = Path(resume_from)
        trainer_state_path = resume_path / "trainer_state.json"

        if resume_path.is_file() and resume_path.suffix == ".pth":
            accelerator.print("[RESUME] Detected .pth weights; loading model weights only (optimizer/scheduler not restored).")
            state = torch.load(str(resume_path), map_location="cpu")
            model.load_state_dict(state, strict=True)
        else:
            accelerator.print("[RESUME] Detected accelerate checkpoint directory; restoring full training state.")
            accelerator.load_state(resume_from)
            if trainer_state_path.exists():
                try:
                    with trainer_state_path.open("r", encoding="utf-8") as f:
                        ts = json.load(f) or {}
                    global_step = int(ts.get("global_step", 0))
                    first_epoch = int(ts.get("epoch", 0))
                    accelerator.print(f"[RESUME] restored epoch={first_epoch}, global_step={global_step}")
                except Exception as e:
                    accelerator.print(f"[RESUME] failed to read trainer_state.json: {e}")
            else:
                # best-effort fallback: try to infer from directory name
                m = re.search(r"epoch(\d+)", resume_from)
                if m:
                    first_epoch = int(m.group(1))
                    accelerator.print(f"[RESUME] inferred epoch={first_epoch} from path name")

    def log_validation(model, llama_model, sd_pipeline, global_step, batch, with_his_emb=True, with_keyword=True, name=""):
        torch.cuda.empty_cache()
        with torch.no_grad():
            keywords_emb = batch["keywords_emb"]
            if with_his_emb:
                image_emb = model.forward(llama_model, batch["input_ids"], batch["token_len"])
                if with_keyword:
                    image_emb = torch.concat([keywords_emb, image_emb], dim=1)
            else:
                image_emb = keywords_emb

            img_bsz = 1
            gen_images = []
            for t in range(args.vaild_image_num):
                gen_images.append(
                    sd_pipeline.generate(
                        image_emb.repeat(img_bsz, 1, 1),
                        negative_prompt="lowres, text, error, cropped, worst quality, low quality",
                        generator=[torch.manual_seed(i + t * img_bsz) for i in range(img_bsz)],
                        show_processbar=False,
                    )
                )

            gen_images = np.concatenate(gen_images, axis=0)
            for tracker in accelerator.trackers:
                tracker.writer.add_images(f"validation_{name}", gen_images, global_step, dataformats="NHWC")
            return gen_images

    def calc_loss(image_emb, keywords_emb, pixel_values):
        keywords_emb = keywords_emb.to(image_emb.device)
        image_emb = torch.concat([keywords_emb, image_emb], dim=1)
        pixel_values = (pixel_values.to(args.weight_dtype) / 127.5 - 1).permute(0, 3, 1, 2)
        pixel_values = torch.nn.functional.interpolate(pixel_values, (args.image_size, args.image_size), mode="bilinear")
        image_loss = sd_pipeline.forward(image_emb, pixel_values)
        return image_loss, image_loss, 0

    for param in model.parameters():
        param.requires_grad = True

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            with accelerator.accumulate(model):
                image_emb = model.forward(llama_model, batch["input_ids"], batch["token_len"])
                loss, image_loss, _ = calc_loss(image_emb, batch["keywords_emb"], batch["pixel_values"])
                nega_loss, _, _ = calc_loss(image_emb, batch["nega_keywords_emb"], batch["nega_pixel_values"])
                final_loss = loss - nega_loss * 0.5
                accelerator.backward(final_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    # 1) Lightweight model-only weights (compatible with USERPREF_PMG_INFER.py)
                    save_path = args.output_dir / f"save-steps-{global_step}.pth"
                    if accelerator.is_main_process:
                        torch.save(accelerator.unwrap_model(model).state_dict(), str(save_path))

                    # 2) Full training state (optimizer/scheduler/grad scaler/etc) for true resume
                    ckpt_dir = args.output_dir / f"ckpt-step-{global_step}"
                    if accelerator.is_main_process:
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        with (ckpt_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
                            json.dump(
                                {"epoch": int(epoch), "global_step": int(global_step)},
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    accelerator.wait_for_everyone()
                    accelerator.save_state(str(ckpt_dir))

                if global_step % args.validation_steps == 1:
                    for vstep, vbatch in enumerate(vaild_dataloader):
                        if global_step == 1:
                            his_text = llama_tokenizer.decode(vbatch["input_ids"][0])
                            for tracker in accelerator.trackers:
                                tracker.writer.add_text(f"validation_{vstep}", f"### History:\n{his_text}")
                                log_validation(model, llama_model, sd_pipeline, global_step, vbatch, False, name=f"{vstep}_only_kw")
                        _ = log_validation(model, llama_model, sd_pipeline, global_step, vbatch, name=str(vstep))

            logs = {
                "epoch": epoch,
                "loss": float(loss.detach().item()),
                "image_loss": float(image_loss.detach().item()),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "sync_gradients": bool(accelerator.sync_gradients),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        save_path = args.output_dir / f"model-epoch{epoch}.pth"
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(), str(save_path))
        # Epoch-level accelerate checkpoint for robust resume
        ckpt_dir = args.output_dir / f"ckpt-epoch{epoch}"
        if accelerator.is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with (ckpt_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"epoch": int(epoch + 1), "global_step": int(global_step)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        accelerator.wait_for_everyone()
        accelerator.save_state(str(ckpt_dir))
        accelerator.print(f"Completed epoch {epoch+1}/{args.num_train_epochs}; saved: {save_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
