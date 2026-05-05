import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from method.dreambooth.data.adapters.userpref import HistorySelectConfig, UserPrefDatasetBuilder, ensure_dir


@dataclass
class TITrainConfig:
    sd15_path: str
    output_dir: str
    train_json: str
    target_token: str = "[V]"
    initializer_token: str = "style"
    num_vectors: int = 8
    image_size: int = 512
    train_batch_size: int = 2
    max_train_steps: int = 3000
    lr: float = 5e-4
    save_interval: int = 200
    seed: int = 42
    # NOTE: NFS + multi-process DataLoader is a common source of "hangs" on clusters.
    # Default to 0 for reliability; increase only if your filesystem is local/fast.
    num_workers: int = 0
    resume_from: str = ""


class UserPrefTIDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        placeholder_tokens_str: str,
        target_token_symbol: str,
        image_size: int = 512,
    ):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)
        self.image_size = int(image_size)
        self.placeholder_str = str(placeholder_tokens_str)
        self.target_symbol = str(target_token_symbol)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Avoid deep recursion on bad samples; bounded resampling across the tiny manifest.
        max_attempts = 64
        n = len(self.data)
        for _ in range(max_attempts):
            sample = self.data[idx % n]
            history_items = sample.get("history_items_info") or []
            if not history_items:
                idx = random.randint(0, n - 1)
                continue

            hist = random.choice(history_items)
            image_path = str(hist.get("image_path") or "")
            if not image_path or (not os.path.exists(image_path)):
                idx = random.randint(0, n - 1)
                continue

            raw_prompt = str(hist.get("masked_caption") or hist.get("caption") or "").strip()
            if not raw_prompt:
                idx = random.randint(0, n - 1)
                continue

            train_prompt = raw_prompt.replace(self.target_symbol, self.placeholder_str)
            if self.placeholder_str not in train_prompt:
                train_prompt = f"{train_prompt}, {self.placeholder_str}".strip(" ,")

            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
            image = np.array(image).astype(np.float32) / 255.0
            image = (image * 2.0) - 1.0
            image = torch.from_numpy(image).permute(2, 0, 1)

            return {"pixel_values": image, "prompt": train_prompt}

        raise RuntimeError("Failed to sample a valid training example after many attempts")


def _latest_step_ckpt(output_dir: Path) -> Tuple[str, int]:
    pats = list(output_dir.glob("learned_embeds_step_*.bin"))
    if not pats:
        return "", 0
    best = None
    best_step = -1
    for p in pats:
        m = re.search(r"learned_embeds_step_(\d+)\.bin$", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best = p
    if best is None:
        return "", 0
    return str(best), best_step


def train_one_user(cfg: TITrainConfig):
    from diffusers import StableDiffusionPipeline

    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    t_train_start = time.perf_counter()

    pipe = StableDiffusionPipeline.from_pretrained(cfg.sd15_path, safety_checker=None)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    pipe.to(device)
    text_encoder.to(dtype=torch.float32)
    vae.to(dtype=torch.float16)
    unet.to(dtype=torch.float16)

    placeholder_tokens = [f"<v_{i}>" for i in range(int(cfg.num_vectors))]
    placeholder_tokens_str = " ".join(placeholder_tokens)
    print(f"Training placeholder tokens: {placeholder_tokens_str}")

    num_added = tokenizer.add_tokens(placeholder_tokens)
    if num_added != int(cfg.num_vectors):
        raise ValueError(f"Failed to add placeholder tokens: expected {cfg.num_vectors}, got {num_added}")
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    initializer_token_ids = tokenizer.encode(str(cfg.initializer_token), add_special_tokens=False)
    if len(initializer_token_ids) == 0:
        raise ValueError(f"Initializer token '{cfg.initializer_token}' not found in tokenizer")
    init_token_id = int(initializer_token_ids[0])

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        init_vec = token_embeds[init_token_id].clone()
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = init_vec

    # Resume (weights only; optimizer restarts)
    global_step = 0
    resume_path = str(cfg.resume_from).strip()
    if not resume_path:
        resume_path, global_step = _latest_step_ckpt(Path(cfg.output_dir))
    if resume_path:
        print(f"[RESUME] loading embeddings from {resume_path}")
        obj = torch.load(resume_path, map_location="cpu")
        learned = obj.get(cfg.target_token)
        if learned is None:
            # tolerate any single-key dict
            if isinstance(obj, dict) and len(obj) == 1:
                learned = next(iter(obj.values()))
        if learned is None:
            raise ValueError(f"[RESUME] invalid embed file: {resume_path}")
        learned = learned.to(dtype=token_embeds.dtype)
        if learned.shape[0] != len(placeholder_token_ids):
            raise ValueError(f"[RESUME] num_vectors mismatch: got {learned.shape}, expected {len(placeholder_token_ids)} vectors")
        with torch.no_grad():
            for j, token_id in enumerate(placeholder_token_ids):
                token_embeds[token_id] = learned[j].clone()
        m = re.search(r"step_(\d+)\.bin$", resume_path)
        if m:
            global_step = int(m.group(1))
        print(f"[RESUME] starting from global_step={global_step}")

    embeds = text_encoder.get_input_embeddings().weight
    embeds.requires_grad_(True)
    optimizer = torch.optim.AdamW([embeds], lr=float(cfg.lr))

    grad_mask = torch.zeros_like(embeds)
    grad_mask[placeholder_token_ids, :] = 1.0
    grad_mask = grad_mask.to(device)

    dataset = UserPrefTIDataset(
        json_path=str(cfg.train_json),
        placeholder_tokens_str=placeholder_tokens_str,
        target_token_symbol=str(cfg.target_token),
        image_size=int(cfg.image_size),
    )
    ds_len = len(dataset)
    if ds_len <= 0:
        raise ValueError(f"Empty dataset: {cfg.train_json}")

    eff_bs = int(cfg.train_batch_size)
    if eff_bs > ds_len:
        print(
            f"[TI] train_batch_size={eff_bs} > dataset_len={ds_len}; "
            f"clamping batch_size to {ds_len} (otherwise drop_last=True yields zero batches).",
            flush=True,
        )
        eff_bs = ds_len

    # Critical: with tiny manifests, drop_last=True can produce an empty iterator forever.
    drop_last = ds_len >= eff_bs * 2

    dataloader = DataLoader(
        dataset,
        batch_size=int(eff_bs),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=bool(drop_last),
        pin_memory=False,
        persistent_workers=False if int(cfg.num_workers) == 0 else True,
    )
    try:
        dl_len = len(dataloader)
    except TypeError:
        dl_len = -1
    if dl_len == 0:
        raise RuntimeError(
            f"DataLoader is empty (dataset_len={ds_len}, batch_size={eff_bs}, drop_last={drop_last}). "
            f"Fix train_batch_size / drop_last / train.json."
        )
    print(f"[TI] dataloader: len(dataset)={ds_len}, batch_size={eff_bs}, drop_last={drop_last}, len(dataloader)={dl_len}", flush=True)
    noise_scheduler = pipe.scheduler

    print(f"Start TI training. Steps={cfg.max_train_steps} (resume at {global_step})")
    progress_bar = tqdm(
        total=int(cfg.max_train_steps),
        initial=int(global_step),
        mininterval=1.0,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    first_batch = True
    while global_step < int(cfg.max_train_steps):
        for batch in dataloader:
            if global_step >= int(cfg.max_train_steps):
                break

            if first_batch:
                t0 = time.time()
                print("[TI] fetched first training batch; starting forward/backward...", flush=True)

            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            prompts = batch["prompt"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            tokenized = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)
            encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=torch.float16)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight.grad *= grad_mask
            optimizer.step()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=float(loss.item()))

            if first_batch:
                dt = time.time() - t0
                print(f"[TI] finished first optimizer step in {dt:.1f}s (loss={float(loss.item()):.6f})", flush=True)
                first_batch = False

            if global_step % int(cfg.save_interval) == 0:
                current = text_encoder.get_input_embeddings().weight[placeholder_token_ids].detach().cpu()
                save_path = os.path.join(cfg.output_dir, f"learned_embeds_step_{global_step}.bin")
                torch.save({cfg.target_token: current}, save_path)
                print(f"\n[CKPT] saved: {save_path}")

    final_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_ids].detach().cpu()
    final_path = os.path.join(cfg.output_dir, "learned_embeds.bin")
    torch.save({cfg.target_token: final_embeds}, final_path)
    t_train_end = time.perf_counter()
    print(f"Done. Final embeds: {final_path}")
    # Persist training timing summary for efficiency comparisons.
    try:
        elapsed_s = float(t_train_end - t_train_start)
        summary = {
            "max_train_steps": int(cfg.max_train_steps),
            "train_batch_size": int(cfg.train_batch_size),
            "elapsed_s": elapsed_s,
            "sec_per_step": (elapsed_s / max(1, int(cfg.max_train_steps))),
            "resume_from": str(cfg.resume_from),
        }
        with open(os.path.join(cfg.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    data_root = repo_root / "data" / "userpref_v1"

    p = argparse.ArgumentParser(description="Per-user Textual Inversion training for userpref_v1 (SD1.5)")
    p.add_argument("--splits_json", type=str, default=str(data_root / "splits.json"))
    p.add_argument("--test_json", type=str, default=str(data_root / "processed_dataset" / "test.json"))

    p.add_argument("--sd15_path", type=str, required=True)
    p.add_argument("--output_root", type=str, default=str(repo_root / "outputs" / "userpref_v1_textual_inversion"))
    p.add_argument("--per_user_data_root", type=str, default=str(data_root / "textual_inversion" / "per_user"))

    p.add_argument("--history_policy", type=str, default="threshold", choices=["all", "topk", "threshold"])
    p.add_argument("--history_topk", type=int, default=10)
    p.add_argument("--history_threshold", type=float, default=4.0)

    p.add_argument("--target_token", type=str, default="[V]")
    p.add_argument("--initializer_token", type=str, default="style")
    p.add_argument("--num_vectors", type=int, default=8)

    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--user_ids", type=str, default="", help="Comma-separated user ids; empty means all test users")
    p.add_argument("--no_style_mask", action="store_true")
    p.add_argument(
        "--style_mask_terms",
        type=str,
        default=str(repo_root / "method" / "dreambooth" / "style_mask_terms.json"),
    )

    p.add_argument("--resume_from", type=str, default="", help="Optional path to learned_embeds_step_*.bin for one user")
    p.add_argument("--overwrite", action="store_true", help="If set, retrain even if learned_embeds.bin exists")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    data_root = repo_root / "data" / "userpref_v1"

    history_cfg = HistorySelectConfig(policy=args.history_policy, topk=args.history_topk, threshold=args.history_threshold)
    builder = UserPrefDatasetBuilder(
        splits_json=Path(args.splits_json),
        test_json=Path(args.test_json),
        repo_root=repo_root,
        data_root=data_root,
        history_cfg=history_cfg,
        instance_token=str(args.target_token),
        style_mask_terms=str(args.style_mask_terms),
        enable_style_mask=(not bool(args.no_style_mask)),
        user_ids=str(args.user_ids),
    )
    users = builder.list_target_users()
    print(f"Users to train: {len(users)}")

    output_root = Path(args.output_root)
    per_user_data_root = Path(args.per_user_data_root)
    ensure_dir(output_root)
    ensure_dir(per_user_data_root)

    for uid in users:
        user_out_dir = output_root / uid
        ensure_dir(user_out_dir)

        final_path = user_out_dir / "learned_embeds.bin"
        if final_path.exists() and (not bool(args.overwrite)):
            print(f"[SKIP] user={uid}: already has {final_path}")
            continue

        train_json = builder.build_train_manifest(user_id=uid, output_root=per_user_data_root)
        if train_json is None:
            print(f"[SKIP] user={uid}: no usable history items")
            continue

        cfg = TITrainConfig(
            sd15_path=str(args.sd15_path),
            output_dir=str(user_out_dir),
            train_json=str(train_json),
            target_token=str(args.target_token),
            initializer_token=str(args.initializer_token),
            num_vectors=int(args.num_vectors),
            image_size=int(args.image_size),
            train_batch_size=int(args.train_batch_size),
            max_train_steps=int(args.max_train_steps),
            lr=float(args.lr),
            save_interval=int(args.save_interval),
            seed=int(args.seed),
            num_workers=int(args.num_workers),
            resume_from=str(args.resume_from).strip(),
        )

        print(f"\n=== user {uid} ===")
        print(f"train_json: {train_json}")
        print(f"out_dir: {user_out_dir}")
        train_one_user(cfg)


if __name__ == "__main__":
    main()

