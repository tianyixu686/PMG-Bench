import argparse
from pathlib import Path

from method.dreambooth.core.config import TrainConfig
from method.dreambooth.core.trainer import run_training
from method.dreambooth.data.adapters.userpref import HistorySelectConfig, UserPrefDatasetBuilder, ensure_dir


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    data_root = repo_root / "data" / "userpref_v1"

    p = argparse.ArgumentParser(description="Per-user DreamBooth LoRA training for userpref_v1")
    p.add_argument("--splits_json", type=str, default=str(data_root / "splits.json"))
    p.add_argument("--test_json", type=str, default=str(data_root / "processed_dataset" / "test.json"))

    p.add_argument("--sd15_path", type=str, required=True)
    p.add_argument("--output_root", type=str, default=str(repo_root / "outputs" / "userpref_v1_dreambooth_lora"))
    p.add_argument("--per_user_data_root", type=str, default=str(data_root / "dreambooth" / "per_user"))

    p.add_argument("--history_policy", type=str, default="topk", choices=["all", "topk", "threshold"])
    p.add_argument("--history_topk", type=int, default=30)
    p.add_argument("--history_threshold", type=float, default=4.0)

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_train_steps", type=int, default=800)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    p.add_argument("--user_ids", type=str, default="", help="Comma-separated user ids; empty means all test users")

    p.add_argument("--no_style_mask", action="store_true")
    p.add_argument(
        "--style_mask_terms",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "style_mask_terms.json"),
    )

    p.add_argument("--instance_token", type=str, default="sks")
    p.add_argument("--initializer_token", type=str, default="photo")

    p.add_argument("--train_text_encoder_lora", action="store_true")
    p.add_argument("--train_instance_token_embedding", action="store_true")
    p.add_argument("--no_train_instance_token_embedding", action="store_true")

    p.add_argument("--dry_run", action="store_true", help="Run each user for 2 steps")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    data_root = repo_root / "data" / "userpref_v1"

    history_cfg = HistorySelectConfig(
        policy=args.history_policy,
        topk=args.history_topk,
        threshold=args.history_threshold,
    )

    builder = UserPrefDatasetBuilder(
        splits_json=Path(args.splits_json),
        test_json=Path(args.test_json),
        repo_root=repo_root,
        data_root=data_root,
        history_cfg=history_cfg,
        instance_token=str(args.instance_token),
        style_mask_terms=str(args.style_mask_terms),
        enable_style_mask=(not args.no_style_mask),
        user_ids=str(args.user_ids),
    )
    test_users = builder.list_target_users()

    output_root = Path(args.output_root)
    per_user_data_root = Path(args.per_user_data_root)
    ensure_dir(output_root)
    ensure_dir(per_user_data_root)

    print(f"Test users to train: {len(test_users)}")
    print(f"Per-user data root: {per_user_data_root}")
    print(f"Per-user output root: {output_root}")

    for uid in test_users:
        train_json = builder.build_train_manifest(user_id=uid, output_root=per_user_data_root)

        if train_json is None:
            print(f"[SKIP] user={uid}: no usable history items")
            continue

        user_out_dir = output_root / uid
        ensure_dir(user_out_dir)

        train_cfg = TrainConfig(
            train_json=str(train_json),
            pretrained_model_name_or_path=str(args.sd15_path),
            output_dir=str(user_out_dir),
            instance_token=str(args.instance_token),
            initializer_token=str(args.initializer_token),
            resolution=int(args.resolution),
            train_batch_size=int(args.train_batch_size),
            max_train_steps=(2 if args.dry_run else int(args.max_train_steps)),
            learning_rate=float(args.learning_rate),
            rank=int(args.rank),
            mixed_precision=str(args.mixed_precision),
            gradient_accumulation_steps=max(1, int(args.gradient_accumulation_steps)),
            seed=int(args.seed),
            dataloader_num_workers=max(0, int(args.dataloader_num_workers)),
            train_text_encoder_lora=bool(args.train_text_encoder_lora),
            train_instance_token_embedding=(
                False if args.no_train_instance_token_embedding else bool(args.train_instance_token_embedding)
            ),
        )

        print(f"\n=== user {uid} ===")
        run_training(train_cfg)

    print("Done.")


if __name__ == "__main__":
    main()
