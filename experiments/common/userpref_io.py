"""Load userpref_v1 processed JSON and resolve image paths (aligned with tools/eval_userpref_outputs.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .repo import repo_root as _repo_root


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(raw_path: str, *, repo_root: Path, data_root: Path) -> str:
    if not raw_path:
        return ""
    p = Path(str(raw_path))
    if p.is_absolute():
        return str(p)
    cand1 = (data_root / p).resolve()
    if cand1.exists() or str(raw_path).replace("\\", "/").startswith("images/"):
        return str(cand1)
    return str((repo_root / p).resolve())


def select_target_from_candidates(sample: dict, *, repo_root: Path, data_root: Path) -> str:
    candidates = sample.get("stage2_candidates") or []
    best_path = ""
    best_score = -1e18
    for item in candidates:
        path = resolve_image_path(str(item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)
        if not path or not os.path.exists(path):
            continue
        try:
            score = float(item.get("preference_score"))
        except Exception:
            score = float("nan")
        if score == score and score > best_score:
            best_score = score
            best_path = path
    if best_path:
        return best_path
    target_item = sample.get("target_item_info") or {}
    return resolve_image_path(str(target_item.get("image_path") or ""), repo_root=repo_root, data_root=data_root)


def data_root_from_split_json(split_json: Path) -> Path:
    if split_json.name.lower() == "test.json" and split_json.parent.name == "processed_dataset":
        return split_json.parents[1]
    return _repo_root() / "data" / "userpref_v1"


def load_split(split_json: Path) -> List[dict]:
    data = load_json(split_json)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {split_json}")
    return data


def sample_key(sample: dict) -> str:
    uid = str(sample.get("worker_id") or sample.get("user_id") or "").strip()
    qv = str((sample.get("target_item_info") or {}).get("query_variant") or "").strip()
    return f"{uid}_{qv}" if qv else uid


def load_concat_splits(
    processed_dir: Path, names: Tuple[str, ...] = ("train.json", "val.json", "test.json")
) -> Tuple[List[dict], Dict[str, Any]]:
    """合并 train/val/test 的 processed 列表。"""
    merged: List[dict] = []
    meta: Dict[str, Any] = {"files": {}}
    for nm in names:
        p = processed_dir / nm
        if not p.is_file():
            meta["files"][nm] = {"exists": False}
            continue
        part = load_split(p)
        merged.extend(part)
        meta["files"][nm] = {"exists": True, "n_rows": len(part)}
    meta["n_rows_total_concat"] = len(merged)
    return merged, meta
