import json
import os
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from method.dreambooth.core.config import TrainConfig
from method.dreambooth.core.pipeline import maybe_add_instance_token
from method.dreambooth.data.datasets import HistoryItemsDreamBoothDataset


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(examples, tokenizer):
    pixel_values = torch.stack([x["pixel_values"] for x in examples]).float()
    prompts = [x["prompt"] for x in examples]
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask,
    }


def run_training(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    seed_all(cfg.seed)
    t_train_start = time.perf_counter()

    try:
        from accelerate import Accelerator
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.loaders import StableDiffusionLoraLoaderMixin
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from transformers import AutoTokenizer, CLIPTextModel
    except ImportError as exc:
        raise ImportError("Need accelerate, diffusers, transformers and peft installed.") from exc

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision,
    )
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if cfg.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if cfg.train_text_encoder_lora:
            text_encoder.gradient_checkpointing_enable()

    infer_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        infer_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        infer_dtype = torch.bfloat16

    unet.to(device, dtype=torch.float32)
    text_encoder.to(device, dtype=torch.float32)
    vae.to(device, dtype=infer_dtype)

    instance_token_id = maybe_add_instance_token(tokenizer, text_encoder, cfg.instance_token, cfg.initializer_token)

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

    token_embed_param = None
    grad_mask = None
    if cfg.train_instance_token_embedding:
        embeds = text_encoder.get_input_embeddings().weight
        embeds.requires_grad_(True)
        token_embed_param = embeds
        grad_mask = torch.zeros_like(embeds)
        grad_mask[instance_token_id, :] = 1.0
        grad_mask = grad_mask.to(device)

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

    if token_embed_param is not None:
        for group in optimizer.param_groups:
            if token_embed_param in group["params"]:
                group["weight_decay"] = 0.0

    dataset = HistoryItemsDreamBoothDataset(
        train_json=cfg.train_json,
        image_size=cfg.resolution,
        target_token=cfg.target_token,
        instance_token=cfg.instance_token,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=lambda xs: collate_fn(xs, tokenizer),
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
                pixel_values = batch["pixel_values"].to(device, dtype=infer_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with accelerator.autocast():
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
                if accelerator.is_local_main_process and (
                    global_step % cfg.log_interval == 0 or global_step == cfg.max_train_steps
                ):
                    progress_bar.set_postfix(loss=float(loss.item()))

    accelerator.wait_for_everyone()
    t_train_end = time.perf_counter()
    if accelerator.is_main_process:
        save_outputs(
            accelerator=accelerator,
            unet=unet,
            text_encoder=text_encoder,
            cfg=cfg,
            instance_token_id=instance_token_id,
            convert_state_dict_to_diffusers=convert_state_dict_to_diffusers,
            get_peft_model_state_dict=get_peft_model_state_dict,
            StableDiffusionLoraLoaderMixin=StableDiffusionLoraLoaderMixin,
        )
        # Lightweight training timing summary for efficiency comparisons.
        try:
            elapsed_s = float(t_train_end - t_train_start)
            summary = {
                "max_train_steps": int(cfg.max_train_steps),
                "train_batch_size": int(cfg.train_batch_size),
                "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                "mixed_precision": str(cfg.mixed_precision),
                "elapsed_s": elapsed_s,
                "sec_per_step": (elapsed_s / max(1, int(cfg.max_train_steps))),
            }
            with open(os.path.join(cfg.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    accelerator.end_training()


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = getattr(model, "_orig_mod", model)
    return model


def save_outputs(
    accelerator,
    unet,
    text_encoder,
    cfg: TrainConfig,
    instance_token_id: int,
    convert_state_dict_to_diffusers,
    get_peft_model_state_dict,
    StableDiffusionLoraLoaderMixin,
):
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    unet_ = unwrap_model(accelerator, unet).to(torch.float32)
    unet_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_))

    text_lora_state = None
    if cfg.train_text_encoder_lora:
        te_ = unwrap_model(accelerator, text_encoder).to(torch.float32)
        text_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(te_))

    StableDiffusionLoraLoaderMixin.save_lora_weights(
        save_directory=out_dir,
        unet_lora_layers=unet_lora_state,
        text_encoder_lora_layers=text_lora_state,
    )

    if cfg.train_instance_token_embedding:
        with torch.no_grad():
            te_ = unwrap_model(accelerator, text_encoder)
            vec = te_.get_input_embeddings().weight[instance_token_id].detach().cpu()
        torch.save({cfg.instance_token: vec}, os.path.join(out_dir, "learned_token.bin"))

    with open(os.path.join(out_dir, "tokenizer_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"instance_token": cfg.instance_token, "instance_token_id": int(instance_token_id)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    accelerator.print(f"Saved LoRA weights to: {out_dir}")
