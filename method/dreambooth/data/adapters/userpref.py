import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from method.dreambooth.data.base import DatasetBuilder
from method.dreambooth.utils.style_mask import mask_prompt_to_instance_token


@dataclass
class HistorySelectConfig:
    policy: str = "topk"
    topk: int = 30
    threshold: float = 4.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_rank_score(item: dict) -> Tuple[float, float]:
    try:
        pref = float(item.get("preference_score"))
    except Exception:
        pref = float("nan")
    try:
        qual = float(item.get("quality_score"))
    except Exception:
        qual = float("nan")
    if pref != pref:
        pref = -1e9
    if qual != qual:
        qual = -1e9
    return pref, qual


def select_history_items(history_items: List[dict], cfg: HistorySelectConfig) -> List[dict]:
    items = list(history_items or [])
    if not items:
        return []

    if cfg.policy == "all":
        return items

    if cfg.policy == "topk":
        items_sorted = sorted(items, key=parse_rank_score, reverse=True)
        k = max(1, min(int(cfg.topk), len(items_sorted)))
        return items_sorted[:k]

    if cfg.policy == "threshold":
        picked = []
        for item in items:
            try:
                pref = float(item.get("preference_score"))
            except Exception:
                continue
            if pref >= float(cfg.threshold):
                picked.append(item)

        min_required = max(1, min(int(cfg.topk), len(items)))
        if len(picked) >= min_required:
            return picked

        items_sorted = sorted(items, key=parse_rank_score, reverse=True)
        return items_sorted[:min_required]

    raise ValueError(f"Unknown history policy: {cfg.policy}")


def resolve_image_path(raw_path: str, *, repo_root: Path, data_root: Path) -> str:
    if not raw_path:
        return ""
    p = Path(str(raw_path))
    if p.is_absolute():
        return str(p)

    candidate1 = (data_root / p).resolve()
    if candidate1.exists() or str(raw_path).replace("\\", "/").startswith("images/"):
        return str(candidate1)

    candidate2 = (repo_root / p).resolve()
    return str(candidate2)


def group_samples_by_user(test_data: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for sample in test_data:
        uid = str(sample.get("worker_id") or "").strip()
        if not uid:
            continue
        if uid not in out:
            out[uid] = sample
    return out


class UserPrefDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        *,
        splits_json: Path,
        test_json: Path,
        repo_root: Path,
        data_root: Path,
        history_cfg: HistorySelectConfig,
        instance_token: str,
        style_mask_terms: str,
        enable_style_mask: bool,
        user_ids: str = "",
    ):
        self.splits_json = splits_json
        self.test_json = test_json
        self.repo_root = repo_root
        self.data_root = data_root
        self.history_cfg = history_cfg
        self.instance_token = instance_token
        self.style_mask_terms = style_mask_terms
        self.enable_style_mask = enable_style_mask
        self.user_ids = user_ids

        self.splits = load_json(self.splits_json)
        self.test_data = load_json(self.test_json)
        self.user2sample = group_samples_by_user(self.test_data)

    def list_target_users(self) -> List[str]:
        users = [str(x) for x in (self.splits.get("test_users") or [])]
        if self.user_ids.strip():
            allow = {x.strip() for x in self.user_ids.split(",") if x.strip()}
            users = [u for u in users if u in allow]
        users = [u for u in users if u in self.user2sample]
        return users

    def build_train_manifest(self, user_id: str, output_root: Path) -> Path | None:
        sample = self.user2sample.get(str(user_id))
        if sample is None:
            return None

        selected_raw = select_history_items(sample.get("history_items_info", []), self.history_cfg)
        selected = []
        for item in selected_raw:
            item_new = dict(item)
            item_new["image_path"] = resolve_image_path(
                str(item_new.get("image_path") or ""),
                repo_root=self.repo_root,
                data_root=self.data_root,
            )
            if self.enable_style_mask:
                caption = str(item_new.get("caption") or "")
                item_new["masked_caption"] = mask_prompt_to_instance_token(
                    caption,
                    instance_token=self.instance_token,
                    terms_path=self.style_mask_terms,
                    enabled=True,
                )
            selected.append(item_new)

        if not selected:
            return None

        user_dir = output_root / str(user_id)
        ensure_dir(user_dir)
        train_json = user_dir / "train.json"
        with train_json.open("w", encoding="utf-8") as f:
            json.dump([{"history_items_info": selected}], f, ensure_ascii=False, indent=2)
        return train_json

    def get_test_samples(self) -> List[Dict]:
        return list(self.test_data)
