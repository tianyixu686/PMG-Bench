import argparse
import json
import logging
import math
import os
import random
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from datasets import (
    FlickrAESCustomDiffusionDataConfig,
    FlickrAESCustomDiffusionDataset,
    POGCustomDiffusionDataConfig,
    POGCustomDiffusionDataset,
    SERCustomDiffusionDataConfig,
    SERCustomDiffusionDataset,
)
from delta_utils import save_delta, select_trainable_parameters

check_min_version("0.21.4")
logger = get_logger(__name__)


@dataclass
class TrainConfig:
    dataset: str = "ser"  # ser|pog|flickr_aes

    # Data
    train_json: str = "{SER_DATASET_BASE_PATH}/processed_masked/train.json"
    masked_captions_file: str = ""  # used for pog/flickr_aes
    original_captions_file: str = ""  # used for pog/flickr_aes

    # Model
    sd15_path: str = "{SD15_MODEL_PATH}"
    revision: Optional[str] = None

    # Prompt/token
    target_token: str = "[V]"
    modifier_token: str = "<new1>"  # supports + separated tokens
    initializer_token: str = "photo"  # supports + separated tokens

    # Custom Diffusion params
    freeze_model: str = "crossattn_kv"  # crossattn_kv|crossattn
    train_self_attention: bool = False
    train_text_encoder: bool = False
    train_modifier_token_embedding: bool = True

    # Prior preservation
    with_prior_preservation: bool = False
    class_data_dir: str = ""
    class_prompt: str = ""
    num_class_images: int = 100
    prior_loss_weight: float = 1.0
    prior_generation_precision: str = "fp16"  # no|fp32|fp16|bf16

    # Training
    resolution: int = 512
    center_crop: bool = False
    train_batch_size: int = 1
    sample_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = 800
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    token_learning_rate: float = 5e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    dataloader_num_workers: int = 0

    # Optimizer
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Perf
    mixed_precision: Optional[str] = None  # no|fp16|bf16
    allow_tf32: bool = False
    gradient_checkpointing: bool = False
    enable_xformers_memory_efficient_attention: bool = False

    # Logging/output
    output_dir: str = "{SER_DATASET_BASE_PATH}/custom_diffusion_sd15"
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    save_steps: int = 500
    seed: int = 42


def _read_image_rgb(path: str, size: int, *, center_crop: bool) -> torch.Tensor:
    image = Image.open(path).convert("RGB")

    if center_crop:
        w, h = image.size
        crop = min(w, h)
        left = (w - crop) // 2
        top = (h - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))

    image = image.resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class ClassImagesDataset(Dataset):
    def __init__(self, class_data_dir: str, class_prompt: str, *, resolution: int, center_crop: bool):
        self.class_prompt = class_prompt
        self.resolution = resolution
        self.center_crop = center_crop

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths: List[str] = []
        for name in os.listdir(class_data_dir):
            p = os.path.join(class_data_dir, name)
            if os.path.isfile(p) and Path(p).suffix.lower() in exts:
                paths.append(p)
        if not paths:
            raise ValueError(f"No class images found in: {class_data_dir}")

        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = _read_image_rgb(self.paths[idx], self.resolution, center_crop=self.center_crop)
        return {"pixel_values": image, "prompt": self.class_prompt}


class PriorPreservationDataset(Dataset):
    def __init__(self, instance_dataset: Dataset, class_dataset: Dataset):
        self.instance_dataset = instance_dataset
        self.class_dataset = class_dataset

    def __len__(self):
        return len(self.instance_dataset)

    def __getitem__(self, idx: int):
        inst = self.instance_dataset[idx]
        cls = self.class_dataset[random.randint(0, len(self.class_dataset) - 1)]
        return {
            "instance_pixel_values": inst["pixel_values"],
            "instance_prompt": inst["prompt"],
            "class_pixel_values": cls["pixel_values"],
            "class_prompt": cls["prompt"],
        }


def _collate_fn(examples, *, with_prior_preservation: bool, resolution: int):
    latent_h = resolution // 8
    latent_w = resolution // 8

    if with_prior_preservation:
        inst_pixels = torch.stack([e["instance_pixel_values"] for e in examples])
        cls_pixels = torch.stack([e["class_pixel_values"] for e in examples])
        pixel_values = torch.cat([inst_pixels, cls_pixels], dim=0)

        prompts = [e["instance_prompt"] for e in examples] + [e["class_prompt"] for e in examples]

        mask = torch.ones((pixel_values.shape[0], 1, latent_h, latent_w), dtype=torch.float32)
        return {"pixel_values": pixel_values, "prompt": prompts, "mask": mask}

    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    prompts = [e["prompt"] for e in examples]
    mask = torch.ones((pixel_values.shape[0], 1, latent_h, latent_w), dtype=torch.float32)
    return {"pixel_values": pixel_values, "prompt": prompts, "mask": mask}


def _replace_token(text: str, target_token: str, modifier_token: str) -> str:
    if not text:
        return ""
    return str(text).replace(target_token, modifier_token)


def _build_instance_dataset(cfg: TrainConfig) -> Dataset:
    ds = cfg.dataset.lower()
    if ds == "ser":
        data_cfg = SERCustomDiffusionDataConfig(
            train_json=cfg.train_json,
            image_size=cfg.resolution,
            target_token=cfg.target_token,
            modifier_token=cfg.modifier_token.split("+")[0],
        )
        return SERCustomDiffusionDataset(data_cfg)

    if ds == "pog":
        masked = cfg.masked_captions_file or "{POG_BASE_PATH}/POG_captions_sampled_masked.json"
        original = cfg.original_captions_file or "{POG_BASE_PATH}/POG_captions_sampled.json"
        data_cfg = POGCustomDiffusionDataConfig(
            train_json=cfg.train_json,
            masked_captions_file=masked,
            original_captions_file=original,
            image_size=cfg.resolution,
            target_token=cfg.target_token,
            modifier_token=cfg.modifier_token.split("+")[0],
        )
        return POGCustomDiffusionDataset(data_cfg)

    if ds == "flickr_aes":
        masked = cfg.masked_captions_file or "{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json"
        original = cfg.original_captions_file or "{FLICKR_AES_BASE_PATH}/FLICKR_captions.json"
        data_cfg = FlickrAESCustomDiffusionDataConfig(
            train_json=cfg.train_json,
            masked_captions_file=masked,
            original_captions_file=original,
            image_size=cfg.resolution,
            target_token=cfg.target_token,
            modifier_token=cfg.modifier_token.split("+")[0],
        )
        return FlickrAESCustomDiffusionDataset(data_cfg)

    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _add_modifier_tokens(tokenizer, text_encoder, modifier_token: str, initializer_token: str) -> List[int]:
    modifier_tokens = [t for t in str(modifier_token).split("+") if t]
    initializer_tokens = [t for t in str(initializer_token).split("+") if t]
    if len(modifier_tokens) > len(initializer_tokens):
        raise ValueError("You must specify + separated initializer token for each modifier token")

    modifier_token_ids: List[int] = []
    initializer_token_ids: List[int] = []

    for mt, it in zip(modifier_tokens, initializer_tokens[: len(modifier_tokens)]):
        added = tokenizer.add_tokens([mt])
        if added == 0:
            raise ValueError(f"Tokenizer already contains token: {mt}")

        token_ids = tokenizer.encode(it, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError("initializer_token must be a single token")

        initializer_token_ids.append(token_ids[0])
        modifier_token_ids.append(tokenizer.convert_tokens_to_ids(mt))

    text_encoder.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for mt_id, it_id in zip(modifier_token_ids, initializer_token_ids):
            token_embeds[mt_id] = token_embeds[it_id]

    # Freeze everything except token embeddings
    if hasattr(text_encoder, "text_model"):
        params_to_freeze = [
            *text_encoder.text_model.encoder.parameters(),
            *text_encoder.text_model.final_layer_norm.parameters(),
            *text_encoder.text_model.embeddings.position_embedding.parameters(),
        ]
        for p in params_to_freeze:
            p.requires_grad = False

    return modifier_token_ids


def _parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="ser", choices=["ser", "pog", "flickr_aes"])
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--masked_captions_file", type=str, default="")
    p.add_argument("--original_captions_file", type=str, default="")

    p.add_argument("--sd15_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)

    p.add_argument("--target_token", type=str, default="[V]")
    p.add_argument("--modifier_token", type=str, default="<new1>")
    p.add_argument("--initializer_token", type=str, default="photo")

    p.add_argument("--freeze_model", type=str, default="crossattn_kv", choices=["crossattn_kv", "crossattn"])
    p.add_argument("--train_self_attention", action="store_true")
    p.add_argument("--train_text_encoder", action="store_true")

    p.add_argument("--train_modifier_token_embedding", action="store_true")
    p.add_argument("--no_train_modifier_token_embedding", action="store_true")

    p.add_argument("--with_prior_preservation", action="store_true")
    p.add_argument("--class_data_dir", type=str, default="")
    p.add_argument("--class_prompt", type=str, default="")
    p.add_argument("--num_class_images", type=int, default=100)
    p.add_argument("--prior_loss_weight", type=float, default=1.0)
    p.add_argument("--prior_generation_precision", type=str, default=None, choices=["no", "fp32", "fp16", "bf16"])

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--sample_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=800)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--token_learning_rate", type=float, default=5e-4)
    p.add_argument("--scale_lr", action="store_true")
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dry_run", action="store_true", help="Run ~2 optimizer steps to sanity-check wiring")

    return p.parse_args()


def _ensure_min_class_images(
    *,
    accelerator: Accelerator,
    pretrained_model_name_or_path: str,
    class_data_dir: str,
    class_prompt: str,
    num_class_images: int,
    sample_batch_size: int,
    revision: Optional[str],
    prior_generation_precision: Optional[str],
):
    class_images_dir = Path(class_data_dir)
    class_images_dir.mkdir(parents=True, exist_ok=True)
    cur_class_images = len([p for p in class_images_dir.iterdir() if p.is_file()])
    if cur_class_images >= num_class_images:
        return

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    if prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16

    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        revision=revision,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)

    num_new_images = num_class_images - cur_class_images
    logger.info(f"Generating {num_new_images} class images...")

    class_idx = 0
    while class_idx < num_new_images:
        cur_bs = min(sample_batch_size, num_new_images - class_idx)
        prompts = [class_prompt] * cur_bs
        images = pipeline(prompts, num_inference_steps=50, guidance_scale=6.0).images
        for img in images:
            out = class_images_dir / f"{cur_class_images + class_idx}.jpg"
            img.save(out)
            class_idx += 1

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()


def main():
    args = _parse_args()

    if args.train_modifier_token_embedding and args.no_train_modifier_token_embedding:
        raise ValueError("Pass at most one of --train_modifier_token_embedding / --no_train_modifier_token_embedding")
    train_modifier_token_embedding = True
    if args.no_train_modifier_token_embedding:
        train_modifier_token_embedding = False
    if args.train_modifier_token_embedding:
        train_modifier_token_embedding = True

    cfg = TrainConfig(
        dataset=args.dataset,
        train_json=args.train_json,
        masked_captions_file=args.masked_captions_file,
        original_captions_file=args.original_captions_file,
        sd15_path=args.sd15_path,
        revision=args.revision,
        target_token=args.target_token,
        modifier_token=args.modifier_token,
        initializer_token=args.initializer_token,
        freeze_model=args.freeze_model,
        train_self_attention=bool(args.train_self_attention),
        train_text_encoder=bool(args.train_text_encoder),
        train_modifier_token_embedding=train_modifier_token_embedding,
        with_prior_preservation=bool(args.with_prior_preservation),
        class_data_dir=args.class_data_dir,
        class_prompt=args.class_prompt,
        num_class_images=args.num_class_images,
        prior_loss_weight=args.prior_loss_weight,
        prior_generation_precision=args.prior_generation_precision or "fp16",
        resolution=args.resolution,
        center_crop=bool(args.center_crop),
        train_batch_size=args.train_batch_size,
        sample_batch_size=args.sample_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=2 if args.dry_run else args.max_train_steps,
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        learning_rate=args.learning_rate,
        token_learning_rate=args.token_learning_rate,
        scale_lr=bool(args.scale_lr),
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        use_8bit_adam=bool(args.use_8bit_adam),
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        allow_tf32=bool(args.allow_tf32),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        enable_xformers_memory_efficient_attention=bool(args.enable_xformers_memory_efficient_attention),
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        save_steps=args.save_steps,
        seed=args.seed,
    )

    logging_dir = Path(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if cfg.with_prior_preservation:
        if not cfg.class_data_dir or not cfg.class_prompt:
            raise ValueError("--with_prior_preservation requires --class_data_dir and --class_prompt")

        _ensure_min_class_images(
            accelerator=accelerator,
            pretrained_model_name_or_path=cfg.sd15_path,
            class_data_dir=cfg.class_data_dir,
            class_prompt=cfg.class_prompt,
            num_class_images=cfg.num_class_images,
            sample_batch_size=cfg.sample_batch_size,
            revision=cfg.revision,
            prior_generation_precision=cfg.prior_generation_precision,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.sd15_path,
        subfolder="tokenizer",
        revision=cfg.revision,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(cfg.sd15_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(cfg.sd15_path, subfolder="text_encoder", revision=cfg.revision)
    vae = AutoencoderKL.from_pretrained(cfg.sd15_path, subfolder="vae", revision=cfg.revision)
    unet = UNet2DConditionModel.from_pretrained(cfg.sd15_path, subfolder="unet", revision=cfg.revision)

    vae.requires_grad_(False)
    if not cfg.train_text_encoder and not cfg.train_modifier_token_embedding:
        text_encoder.requires_grad_(False)
    else:
        text_encoder.requires_grad_(True)

    # Select UNet trainables
    unet.requires_grad_(False)
    train_params, train_param_names = select_trainable_parameters(
        unet,
        freeze_model=cfg.freeze_model,
        train_self_attention=cfg.train_self_attention,
    )
    for p in train_params:
        p.requires_grad_(True)

    # Add/initialize modifier token(s)
    modifier_token_ids: List[int] = []
    if cfg.train_modifier_token_embedding:
        if cfg.train_text_encoder:
            raise ValueError("Use either --train_text_encoder or --train_modifier_token_embedding, not both")
        modifier_token_ids = _add_modifier_tokens(tokenizer, text_encoder, cfg.modifier_token, cfg.initializer_token)

    # Performance options
    if cfg.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available. Install it or disable --enable_xformers_memory_efficient_attention")
        unet.enable_xformers_memory_efficient_attention()

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if cfg.train_text_encoder or cfg.train_modifier_token_embedding:
            text_encoder.gradient_checkpointing_enable()

    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Mixed precision casting for inference-only modules
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if accelerator.mixed_precision != "fp16":
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if cfg.scale_lr:
        cfg.learning_rate = cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        if cfg.with_prior_preservation:
            cfg.learning_rate *= 2.0

    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_class = bnb.optim.AdamW8bit
        except ImportError as e:
            raise ImportError("To use 8-bit Adam, install bitsandbytes") from e
    else:
        optimizer_class = torch.optim.AdamW

    params = [
        {
            "params": train_params,
            "lr": cfg.learning_rate,
            "weight_decay": cfg.adam_weight_decay,
        }
    ]

    if cfg.train_text_encoder:
        params.append(
            {
                "params": [p for p in text_encoder.parameters() if p.requires_grad],
                "lr": cfg.learning_rate,
                "weight_decay": cfg.adam_weight_decay,
            }
        )
    elif cfg.train_modifier_token_embedding:
        params.append(
            {
                "params": [text_encoder.get_input_embeddings().weight],
                "lr": cfg.token_learning_rate,
                "weight_decay": 0.0,
            }
        )

    optimizer = optimizer_class(
        params,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
    )

    instance_dataset = _build_instance_dataset(cfg)

    if cfg.with_prior_preservation:
        class_dataset = ClassImagesDataset(
            cfg.class_data_dir,
            cfg.class_prompt,
            resolution=cfg.resolution,
            center_crop=cfg.center_crop,
        )
        train_dataset: Dataset = PriorPreservationDataset(instance_dataset, class_dataset)
    else:
        train_dataset = instance_dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=lambda ex: _collate_fn(ex, with_prior_preservation=cfg.with_prior_preservation, resolution=cfg.resolution),
    )

    # Scheduler and math around training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps,
    )

    if cfg.train_text_encoder or cfg.train_modifier_token_embedding:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    if accelerator.is_main_process:
        accelerator.init_trackers("custom-diffusion")

    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")

    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    global_step = 0

    for _epoch in range(cfg.num_train_epochs):
        unet.train()
        if cfg.train_text_encoder or cfg.train_modifier_token_embedding:
            text_encoder.train()

        for _step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                tokenized = tokenizer(
                    list(batch["prompt"]),
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(accelerator.device)

                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if cfg.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    mask = torch.chunk(batch["mask"], 2, dim=0)[0].to(model_pred.device)

                    loss_map = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    inst_loss = ((loss_map * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = inst_loss + cfg.prior_loss_weight * prior_loss
                else:
                    mask = batch["mask"].to(model_pred.device)
                    loss_map = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss_map * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                accelerator.backward(loss)

                if cfg.train_modifier_token_embedding and modifier_token_ids:
                    if accelerator.num_processes > 1 and hasattr(text_encoder, "module"):
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad

                    if grads is not None:
                        index = torch.ones((grads.shape[0],), device=grads.device, dtype=torch.bool)
                        for tid in modifier_token_ids:
                            index = index & (torch.arange(grads.shape[0], device=grads.device) != tid)
                        grads.data[index, :] = 0

                if accelerator.sync_gradients:
                    to_clip = [p for p in unet.parameters() if p.requires_grad]
                    if cfg.train_text_encoder or cfg.train_modifier_token_embedding:
                        to_clip.extend([p for p in text_encoder.parameters() if p.requires_grad])
                    accelerator.clip_grad_norm_(to_clip, cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and (global_step % cfg.save_steps == 0 or global_step == cfg.max_train_steps):
                    unet_to_save = accelerator.unwrap_model(unet)
                    delta_out = os.path.join(cfg.output_dir, "delta.bin")
                    save_delta(
                        unet_to_save,
                        delta_out,
                        train_param_names,
                        metadata={
                            "freeze_model": cfg.freeze_model,
                            "train_self_attention": cfg.train_self_attention,
                        },
                    )

                    if cfg.train_modifier_token_embedding and modifier_token_ids:
                        emb_weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.detach().cpu()
                        token_out = os.path.join(cfg.output_dir, "learned_token.bin")
                        token_map = {}
                        for tok, tid in zip(str(cfg.modifier_token).split("+"), modifier_token_ids):
                            token_map[tok] = emb_weight[tid]
                        torch.save(token_map, token_out)

            if global_step >= cfg.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if global_step >= cfg.max_train_steps:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
