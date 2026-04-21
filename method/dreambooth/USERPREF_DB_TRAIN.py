import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from style_mask import mask_prompt_to_instance_token


@dataclass
class HistorySelectConfig:
    policy: str  # all|topk|threshold
    topk: int
    threshold: float


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _select_history_items(history_items: List[dict], cfg: HistorySelectConfig) -> List[dict]:
    items = list(history_items or [])
    if not items:
        return []

    def score(it: dict) -> Tuple[float, float]:
        # prefer preference_score, then quality_score
        pref = it.get("preference_score")
        qual = it.get("quality_score")
        try:
            pref_f = float(pref)
        except Exception:
            pref_f = float("nan")
        try:
            qual_f = float(qual)
        except Exception:
            qual_f = float("nan")
        # nan treated as very small
        if pref_f != pref_f:
            pref_f = -1e9
        if qual_f != qual_f:
            qual_f = -1e9
        return pref_f, qual_f

    if cfg.policy == "all":
        return items

    if cfg.policy == "threshold":
        out = []
        for it in items:
            pref = it.get("preference_score")
            try:
                pref_f = float(pref)
            except Exception:
                continue
            if pref_f >= float(cfg.threshold):
                out.append(it)
        # fallback: if too few, take topk
        if len(out) < max(1, min(cfg.topk, len(items))):
            items_sorted = sorted(items, key=score, reverse=True)
            return items_sorted[: max(1, min(cfg.topk, len(items_sorted)))]
        return out

    if cfg.policy == "topk":
        items_sorted = sorted(items, key=score, reverse=True)
        k = max(1, min(int(cfg.topk), len(items_sorted)))
        return items_sorted[:k]

    raise ValueError(f"Unknown history policy: {cfg.policy}")


def _group_samples_by_user(test_data: List[dict]) -> Dict[str, dict]:
    """Return representative sample per user (history is identical across A/B)."""
    out: Dict[str, dict] = {}
    for s in test_data or []:
        uid = str(s.get("worker_id") or "").strip()
        if not uid:
            continue
        if uid not in out:
            out[uid] = s
    return out


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "userpref_v1"

    def resolve_userpref_image_path(p: str) -> str:
        if not p:
            return ""
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        s = str(p).replace("\\", "/")
        # Prefer data_root-relative paths like images/history/...
        cand1 = (data_root / pp).resolve()
        if cand1.exists() or s.startswith("images/"):
            return str(cand1)
        # Backward compat: repo-root relative (e.g. data/userpref_v1/images/...)
        cand2 = (repo_root / pp).resolve()
        return str(cand2)

    p = argparse.ArgumentParser(description="Per-user DreamBooth(LoRA) training for userpref_v1 test users")
    p.add_argument("--splits_json", type=str, default=str(data_root / "splits.json"))
    p.add_argument("--test_json", type=str, default=str(data_root / "processed_dataset" / "test.json"))

    p.add_argument("--sd15_path", type=str, required=True, help="Base SD1.5 model path")

    p.add_argument("--output_root", type=str, default=str(repo_root / "outputs" / "userpref_v1_dreambooth_lora"))
    p.add_argument("--per_user_data_root", type=str, default=str(data_root / "dreambooth" / "per_user"))

    p.add_argument("--history_policy", type=str, default="topk", choices=["all", "topk", "threshold"])
    p.add_argument("--history_topk", type=int, default=30)
    p.add_argument("--history_threshold", type=float, default=4.0)

    # Training hyperparams (forwarded)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_train_steps", type=int, default=800)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    p.add_argument("--user_ids", type=str, default="", help="Comma-separated user ids to train (subset). Empty = all test_users")

    p.add_argument(
        "--no_style_mask",
        action="store_true",
        help="Disable style masking for DreamBooth prompts (default: enabled).",
    )
    p.add_argument(
        "--style_mask_terms",
        type=str,
        default=str(Path(__file__).resolve().parent / "style_mask_terms.json"),
        help="JSON file with style masking terms; used when style mask is enabled.",
    )

    p.add_argument(
        "--train_instance_token_embedding",
        action="store_true",
        help="If set, allow inner trainer to also train the instance token embedding. Default: off (DreamBooth/LoRA trains network params).",
    )

    p.add_argument(
        "--instance_token",
        type=str,
        default="sks",
        help="DreamBooth instance token used in masked prompts (must be a single token; will be added to tokenizer by inner trainer).",
    )

    p.add_argument("--run", action="store_true", help="Actually run training (otherwise only print commands)")
    p.add_argument("--dry_run", action="store_true", help="Pass --dry_run to inner trainer (2 steps)")

    args = p.parse_args()

    splits = _load_json(Path(args.splits_json))
    test_users = [str(u) for u in (splits.get("test_users") or [])]

    test_data = _load_json(Path(args.test_json))
    user2sample = _group_samples_by_user(test_data)

    if args.user_ids.strip():
        allow = {x.strip() for x in args.user_ids.split(",") if x.strip()}
        test_users = [u for u in test_users if u in allow]

    # Only keep users that exist in test.json
    test_users = [u for u in test_users if u in user2sample]

    cfg = HistorySelectConfig(policy=args.history_policy, topk=args.history_topk, threshold=args.history_threshold)

    output_root = Path(args.output_root)
    per_user_data_root = Path(args.per_user_data_root)
    _ensure_dir(output_root)
    _ensure_dir(per_user_data_root)

    inner_trainer = Path(__file__).resolve().parent / "train_dreambooth_lora_sd15_diffusers.py"

    print(f"Test users to train: {len(test_users)}")
    print(f"Per-user data root: {per_user_data_root}")
    print(f"Per-user output root: {output_root}")

    for uid in test_users:
        sample = user2sample[uid]
        history_items = sample.get("history_items_info", [])
        selected_raw = _select_history_items(history_items, cfg)
        selected = []
        for it in selected_raw:
            it2 = dict(it)
            it2["image_path"] = resolve_userpref_image_path(str(it2.get("image_path") or ""))
            if not args.no_style_mask:
                cap = str(it2.get("caption") or "")
                it2["masked_caption"] = mask_prompt_to_instance_token(
                    cap,
                    instance_token=str(args.instance_token),
                    terms_path=args.style_mask_terms,
                    enabled=True,
                )
            selected.append(it2)
        if not selected:
            print(f"[SKIP] user={uid}: no history items")
            continue

        # Build per-user train json compatible with SERDreamBoothDataset
        user_data_dir = per_user_data_root / uid
        _ensure_dir(user_data_dir)
        train_json = user_data_dir / "train.json"
        with train_json.open("w", encoding="utf-8") as f:
            json.dump([{ "history_items_info": selected }], f, ensure_ascii=False, indent=2)

        user_out_dir = output_root / uid
        _ensure_dir(user_out_dir)

        cmd = [
            "python",
            str(inner_trainer),
            "--dataset",
            "ser",
            "--train_json",
            str(train_json),
            "--pretrained_model_name_or_path",
            str(args.sd15_path),
            "--output_dir",
            str(user_out_dir),
            "--instance_token",
            str(args.instance_token),
            "--resolution",
            str(args.resolution),
            "--max_train_steps",
            str(args.max_train_steps),
            "--learning_rate",
            str(args.learning_rate),
            "--rank",
            str(args.rank),
            "--mixed_precision",
            str(args.mixed_precision),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--train_batch_size",
            str(args.train_batch_size),
            "--seed",
            str(args.seed),
            "--dataloader_num_workers",
            str(args.dataloader_num_workers),
        ]
        if not args.train_instance_token_embedding:
            cmd.append("--no_train_instance_token_embedding")
        if args.dry_run:
            cmd.append("--dry_run")

        print("\n=== user", uid, "===")
        print(" ".join(cmd))

        if args.run:
            subprocess.run(cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
