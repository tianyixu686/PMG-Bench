import argparse
import json
import os
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    SERDreamBoothDataConfig,
    SERDreamBoothDataset,
    POGDreamBoothDataConfig,
    POGDreamBoothDataset,
    FlickrAESDreamBoothDataConfig,
    FlickrAESDreamBoothDataset,
)


@dataclass
class TrainConfig:
    dataset: str = "ser"  # ser|pog|flickr_aes

    # Data
    train_json: str = "{SER_DATASET_BASE_PATH}/processed_masked/train.json"
    masked_captions_file: str = ""  # used for pog/flickr_aes
    original_captions_file: str = ""  # used for pog/flickr_aes

    # Model
    pretrained_model_name_or_path: str = "{SD15_MODEL_PATH}"

    # Prompt/token
    target_token: str = "[V]"
    instance_token: str = "sks"
    initializer_token: str = "photo"

    # Training
    resolution: int = 512
    train_batch_size: int = 1
    max_train_steps: int = 800
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    rank: int = 4
    lora_dropout: float = 0.0

    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # no|fp16|bf16
    seed: int = 42
    dataloader_num_workers: int = 0

    enable_xformers_memory_efficient_attention: bool = False
    gradient_checkpointing: bool = False

    # Optional: train LoRA on text encoder too (official supports it)
    train_text_encoder_lora: bool = False

    # Optional: train the instance token embedding (one row) to match PMG masking scheme
    train_instance_token_embedding: bool = True

    # Logging/output
    output_dir: str = "{SER_DATASET_BASE_PATH}/dreambooth_lora_sd15"
    checkpointing_steps: int = 0  # 0 disables periodic saving
    log_interval: int = 50


def _seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_add_token(tokenizer, text_encoder, token: str, initializer_token: str) -> int:
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


def _build_dataset(cfg: TrainConfig):
    ds = cfg.dataset.lower()
    if ds == "ser":
        data_cfg = SERDreamBoothDataConfig(
            train_json=cfg.train_json,
            image_size=cfg.resolution,
            target_token=cfg.target_token,
            instance_token=cfg.instance_token,
        )
        return SERDreamBoothDataset(data_cfg)

    if ds == "pog":
        if not cfg.masked_captions_file:
            cfg.masked_captions_file = "{POG_BASE_PATH}/POG_captions_sampled_masked.json"
        if not cfg.original_captions_file:
            cfg.original_captions_file = "{POG_BASE_PATH}/POG_captions_sampled.json"
        data_cfg = POGDreamBoothDataConfig(
            train_json=cfg.train_json,
            masked_captions_file=cfg.masked_captions_file,
            original_captions_file=cfg.original_captions_file,
            image_size=cfg.resolution,
            target_token=cfg.target_token,
            instance_token=cfg.instance_token,
        )
        return POGDreamBoothDataset(data_cfg)

    if ds == "flickr_aes":
        if not cfg.masked_captions_file:
            cfg.masked_captions_file = "{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json"
        if not cfg.original_captions_file:
            cfg.original_captions_file = "{FLICKR_AES_BASE_PATH}/FLICKR_captions.json"
        data_cfg = FlickrAESDreamBoothDataConfig(
            train_json=cfg.train_json,
            masked_captions_file=cfg.masked_captions_file,
            original_captions_file=cfg.original_captions_file,
            image_size=cfg.resolution,
            target_token=cfg.target_token,
            instance_token=cfg.instance_token,
        )
        return FlickrAESDreamBoothDataset(data_cfg)

    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _collate_fn(examples, tokenizer):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).float()
    prompts = [ex["prompt"] for ex in examples]
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    batch = {
        "pixel_values": pixel_values,
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask,
    }
    return batch


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="ser", choices=["ser", "pog", "flickr_aes"])

    p.add_argument("--train_json", type=str, default=None)
    p.add_argument("--masked_captions_file", type=str, default=None)
    p.add_argument("--original_captions_file", type=str, default=None)

    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)

    p.add_argument("--target_token", type=str, default="[V]")
    p.add_argument("--instance_token", type=str, default="sks")
    p.add_argument("--initializer_token", type=str, default="photo")

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=800)

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--train_text_encoder_lora", action="store_true")

    p.add_argument("--train_instance_token_embedding", action="store_true")
    p.add_argument("--no_train_instance_token_embedding", action="store_true")

    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--checkpointing_steps", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=50)

    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig()
    cfg.dataset = args.dataset

    if args.train_json is not None:
        cfg.train_json = args.train_json
    else:
        if cfg.dataset == "ser":
            cfg.train_json = "{SER_DATASET_BASE_PATH}/processed_masked/train.json"
            cfg.output_dir = "{SER_DATASET_BASE_PATH}/dreambooth_lora_sd15"
        elif cfg.dataset == "pog":
            cfg.train_json = "{POG_BASE_PATH}/processed_dataset/train.json"
            cfg.output_dir = "{POG_BASE_PATH}/dreambooth_lora_sd15"
        elif cfg.dataset == "flickr_aes":
            cfg.train_json = "{FLICKR_AES_BASE_PATH}/processed_dataset/train.json"
            cfg.output_dir = "{FLICKR_AES_BASE_PATH}/dreambooth_lora_sd15"

    if args.masked_captions_file is not None:
        cfg.masked_captions_file = args.masked_captions_file
    if args.original_captions_file is not None:
        cfg.original_captions_file = args.original_captions_file

    if args.pretrained_model_name_or_path is not None:
        cfg.pretrained_model_name_or_path = args.pretrained_model_name_or_path

    cfg.target_token = args.target_token
    cfg.instance_token = args.instance_token
    cfg.initializer_token = args.initializer_token

    cfg.resolution = args.resolution
    cfg.train_batch_size = args.train_batch_size
    cfg.max_train_steps = 2 if args.dry_run else args.max_train_steps

    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.adam_beta1 = args.adam_beta1
    cfg.adam_beta2 = args.adam_beta2
    cfg.adam_epsilon = args.adam_epsilon
    cfg.max_grad_norm = args.max_grad_norm

    cfg.rank = args.rank
    cfg.lora_dropout = args.lora_dropout

    cfg.gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)
    cfg.mixed_precision = args.mixed_precision
    cfg.seed = args.seed
    cfg.dataloader_num_workers = max(0, args.dataloader_num_workers)

    cfg.enable_xformers_memory_efficient_attention = bool(args.enable_xformers_memory_efficient_attention)
    cfg.gradient_checkpointing = bool(args.gradient_checkpointing)

    cfg.train_text_encoder_lora = bool(args.train_text_encoder_lora)

    if args.no_train_instance_token_embedding:
        cfg.train_instance_token_embedding = False
    elif args.train_instance_token_embedding:
        cfg.train_instance_token_embedding = True

    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    cfg.checkpointing_steps = int(args.checkpointing_steps)
    cfg.log_interval = int(args.log_interval)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    _seed_all(cfg.seed)

    try:
        from accelerate import Accelerator
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.loaders import StableDiffusionLoraLoaderMixin
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from transformers import AutoTokenizer, CLIPTextModel
    except ImportError as e:
        raise ImportError("Need accelerate, diffusers, transformers and peft installed.") from e

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision,
    )

    device = accelerator.device

    # Load components
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if cfg.enable_xformers_memory_efficient_attention:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            raise RuntimeError("Failed to enable xformers attention") from e

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if cfg.train_text_encoder_lora:
            text_encoder.gradient_checkpointing_enable()

    # dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # Add instance token
    instance_token_id = _maybe_add_token(tokenizer, text_encoder, cfg.instance_token, cfg.initializer_token)

    # Add LoRA adapters (PEFT)
    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        lora_dropout=cfg.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if cfg.train_text_encoder_lora:
        text_lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.rank,
            lora_dropout=cfg.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    # Optionally train only one token embedding row
    token_embed_param = None
    grad_mask = None
    if cfg.train_instance_token_embedding:
        embeds = text_encoder.get_input_embeddings().weight
        embeds.requires_grad_(True)
        token_embed_param = embeds
        grad_mask = torch.zeros_like(embeds)
        grad_mask[instance_token_id, :] = 1.0
        grad_mask = grad_mask.to(device)

    # Collect params to optimize
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if cfg.train_text_encoder_lora:
        params_to_optimize += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    if token_embed_param is not None:
        params_to_optimize += [token_embed_param]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
    )

    # If training token embedding row, avoid AdamW weight decay drifting whole embedding matrix
    if token_embed_param is not None:
        for group in optimizer.param_groups:
            if token_embed_param in group["params"]:
                group["weight_decay"] = 0.0

    dataset = _build_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=lambda xs: _collate_fn(xs, tokenizer),
    )

    unet, text_encoder, optimizer, dataloader = accelerator.prepare(unet, text_encoder, optimizer, dataloader)

    global_step = 0
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)

    unet.train()
    if cfg.train_text_encoder_lora or token_embed_param is not None:
        text_encoder.train()

    while global_step < cfg.max_train_steps:
        for batch in dataloader:
            if global_step >= cfg.max_train_steps:
                break

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    return_dict=False,
                )[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if token_embed_param is not None:
                    with torch.no_grad():
                        grad = text_encoder.get_input_embeddings().weight.grad
                        if grad is not None:
                            grad *= grad_mask

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, cfg.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if accelerator.is_local_main_process and (global_step % cfg.log_interval == 0 or global_step == cfg.max_train_steps):
                    progress_bar.set_postfix(loss=float(loss.item()))

                if accelerator.is_main_process and cfg.checkpointing_steps and global_step % cfg.checkpointing_steps == 0:
                    _save_outputs(
                        accelerator,
                        unet,
                        text_encoder,
                        tokenizer,
                        cfg,
                        instance_token_id,
                        convert_state_dict_to_diffusers,
                        get_peft_model_state_dict,
                        StableDiffusionLoraLoaderMixin,
                        step_suffix=f"step_{global_step}",
                    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_outputs(
            accelerator,
            unet,
            text_encoder,
            tokenizer,
            cfg,
            instance_token_id,
            convert_state_dict_to_diffusers,
            get_peft_model_state_dict,
            StableDiffusionLoraLoaderMixin,
            step_suffix=None,
        )

    accelerator.end_training()


def _unwrap(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = getattr(model, "_orig_mod", model)
    return model


def _save_outputs(
    accelerator,
    unet,
    text_encoder,
    tokenizer,
    cfg: TrainConfig,
    instance_token_id: int,
    convert_state_dict_to_diffusers,
    get_peft_model_state_dict,
    StableDiffusionLoraLoaderMixin,
    *,
    step_suffix: str | None,
):
    out_dir = cfg.output_dir if step_suffix is None else os.path.join(cfg.output_dir, step_suffix)
    os.makedirs(out_dir, exist_ok=True)

    unet_ = _unwrap(accelerator, unet).to(torch.float32)
    unet_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_))

    text_lora_state = None
    if cfg.train_text_encoder_lora:
        te_ = _unwrap(accelerator, text_encoder).to(torch.float32)
        text_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(te_))

    StableDiffusionLoraLoaderMixin.save_lora_weights(
        save_directory=out_dir,
        unet_lora_layers=unet_lora_state,
        text_encoder_lora_layers=text_lora_state,
    )

    # Save token embedding (optional)
    token_out = os.path.join(out_dir, "learned_token.bin")
    with torch.no_grad():
        te_ = _unwrap(accelerator, text_encoder)
        vec = te_.get_input_embeddings().weight[instance_token_id].detach().cpu()
    torch.save({cfg.instance_token: vec}, token_out)

    with open(os.path.join(out_dir, "tokenizer_info.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_token": cfg.instance_token, "instance_token_id": int(instance_token_id)}, f, ensure_ascii=False, indent=2)

    accelerator.print(f"Saved LoRA weights to: {out_dir}")


if __name__ == "__main__":
    main()
