import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


@dataclass
class Sources:
    rating_stage12_json: Path
    bench_add_user_target_images_dir: Path
    merged_data_with_simple_json: Path


_PROMPT_LINE_RE = re.compile(r"^(?P<key>[^:]+):\s*(?P<val>.*)$")


def _resolve_path(repo_root: Path, p: str) -> Path:
    path = Path(str(p))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_sources(repo_root: Path, sources_json: Path) -> Tuple[Sources, Path]:
    with sources_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    sources = Sources(
        rating_stage12_json=_resolve_path(repo_root, raw["rating_stage12_json"]),
        bench_add_user_target_images_dir=_resolve_path(repo_root, raw["bench_add_user_target_images_dir"]),
        merged_data_with_simple_json=_resolve_path(repo_root, raw["merged_data_with_simple_json"]),
    )
    out_root = _resolve_path(repo_root, raw.get("output_root", "data/userpref_v1"))
    return sources, out_root


def _read_prompts_txt(prompts_txt: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not prompts_txt.exists():
        return out
    with prompts_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line.strip():
                continue
            m = _PROMPT_LINE_RE.match(line)
            if not m:
                continue
            key = m.group("key").strip()
            val = m.group("val").strip()
            if key and val:
                out[key] = val
    return out


def _load_merged_items_map(merged_json: Path) -> Dict[str, Dict[str, str]]:
    """Map image_id -> {"image_path": <str>, "prompt_simple": <str>, "canonical_image_id": <str>}.

    Notes:
    - If `image_path` contains a filename like `0017713.jpg`, we treat `0017713` as canonical id (keeps leading zeros).
    - We also add an alias key with leading zeros stripped (e.g. `17713`) to support numeric ids from rating JSON.
    """
    with merged_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, Dict[str, str]] = {}
    if isinstance(data, list):
        for it in data:
            image_path = it.get("image_path")
            prompt_simple = it.get("prompt_simple")

            image_id_raw = str(it.get("image_id", "") or "").strip()
            canonical_id = image_id_raw

            # Prefer canonical id from filename stem to keep leading zeros.
            file_stem = ""
            if isinstance(image_path, str) and image_path:
                m = re.search(r"(?:^|/)(\d+)\.(?:jpg|jpeg|png)$", image_path, flags=re.IGNORECASE)
                if m:
                    file_stem = m.group(1)
                    canonical_id = file_stem

            if not canonical_id:
                continue
            if image_path is None and prompt_simple is None:
                continue

            row = {
                "image_path": str(image_path) if image_path is not None else "",
                "prompt_simple": str(prompt_simple) if prompt_simple is not None else "",
                "canonical_image_id": canonical_id,
            }

            def _put(key: str):
                key = str(key).strip()
                if not key:
                    return
                # keep first occurrence if duplicates exist
                if key not in out:
                    out[key] = row

            # Primary key: whatever the merged file carries.
            _put(image_id_raw or canonical_id)
            # Canonical id from filename (keeps leading zeros).
            _put(canonical_id)
            # Alias: strip leading zeros for numeric ids in rating JSON.
            if canonical_id.isdigit():
                _put(str(int(canonical_id)))
    return out


def _norm_int_str(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    if s.isdigit():
        return str(int(s))
    return s


def _safe_user_id(u) -> str:
    s = str(u)
    return s.strip()


def _split_users(user_ids: List[str], seed: int = 2026, train_ratio: float = 0.8, val_ratio: float = 0.1):
    rng = random.Random(seed)
    ids = list(user_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)

    train_users = ids[:n_train]
    val_users = ids[n_train:n_train + n_val]
    test_users = ids[n_train + n_val:]
    return train_users, val_users, test_users


def _to_rel_history_path(image_id: str) -> str:
    # 约定：你后续把历史图片放到 data/userpref_v1/images/history 下
    return f"images/history/{image_id}.jpg"


def _to_rel_stage2_path(user_id: str, image_name: str) -> str:
    return f"images/stage2/{user_id}/{image_name}"


def _candidate_key_from_image_name(user_id: str, image_name: str) -> str:
    # e.g. "0_positive_A_B.png" -> "positive_A_B"
    prefix = f"{user_id}_"
    key = image_name
    if key.startswith(prefix):
        key = key[len(prefix):]
    key = re.sub(r"\.(png|jpg|jpeg)$", "", key, flags=re.IGNORECASE)
    return key


def _pick_reference_candidate(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None

    def k(x: dict):
        return (
            float(x.get("preference_score", -1e9)),
            float(x.get("task_match_score", -1e9)),
            float(x.get("quality_score", -1e9)),
            -float(x.get("initial_rank", 1e9)),
        )

    return sorted(candidates, key=k, reverse=True)[0]


def build():
    repo_root = Path(__file__).resolve().parents[1]
    sources_path = repo_root / "data" / "userpref_v1" / "sources.json"
    if not sources_path.exists():
        raise FileNotFoundError(
            "Missing data/userpref_v1/sources.json. Copy data/userpref_v1/sources.example.json to sources.json and edit paths."
        )
    sources, out_root = _load_sources(repo_root, sources_path)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "processed_dataset").mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)

    merged_map = _load_merged_items_map(sources.merged_data_with_simple_json)

    with sources.rating_stage12_json.open("r", encoding="utf-8") as f:
        rating_data = json.load(f)

    # user list
    user_ids = [_safe_user_id(x.get("user_id")) for x in rating_data]
    user_ids = [u for u in user_ids if u]

    train_users, val_users, test_users = _split_users(user_ids)
    splits = {"train_users": train_users, "val_users": val_users, "test_users": test_users}
    with (out_root / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    def user_split(u: str) -> str:
        if u in train_users:
            return "train"
        if u in val_users:
            return "val"
        return "test"

    processed = {"train": [], "val": [], "test": []}

    stage2_jsonl = (out_root / "manifests" / "stage2_queries.jsonl")
    stage2_f = stage2_jsonl.open("w", encoding="utf-8")

    # Optional manifests
    users_jsonl = (out_root / "manifests" / "users.jsonl").open("w", encoding="utf-8")

    for row in rating_data:
        user_id = _safe_user_id(row.get("user_id"))
        if not user_id:
            continue

        # history
        history_image_ids = [str(x) for x in (row.get("image_ids") or [])]
        pref_scores = row.get("preference_scores") or []
        qual_scores = row.get("quality_scores") or []

        history_items_info = []
        for idx, image_id in enumerate(history_image_ids):
            key = str(image_id).strip()
            m = merged_map.get(key) or merged_map.get(_norm_int_str(key)) or {}
            canonical_id = (m.get("canonical_image_id") or key).strip()
            caption = (m.get("prompt_simple") or "").strip()
            history_items_info.append(
                {
                    "item_name": canonical_id,
                    "image_path": _to_rel_history_path(canonical_id),
                    "caption": caption,
                    "preference_score": pref_scores[idx] if idx < len(pref_scores) else None,
                    "quality_score": qual_scores[idx] if idx < len(qual_scores) else None,
                }
            )

        # stage2 prompts
        prompts_txt = sources.bench_add_user_target_images_dir / user_id / "prompts.txt"
        prompt_map = _read_prompts_txt(prompts_txt)
        raw_A = prompt_map.get("raw_prompt_simple_A", "")
        raw_B = prompt_map.get("raw_prompt_simple_B", "")
        raw_cn_A = prompt_map.get("raw_prompt_simple_cn_A", "")
        raw_cn_B = prompt_map.get("raw_prompt_simple_cn_B", "")

        # stage2 candidates from rating_1.json
        stage2_A = list(row.get("target_images") or [])
        stage2_B = list(row.get("target_images_B") or [])

        # write user manifest row (lightweight)
        users_jsonl.write(
            json.dumps(
                {
                    "user_id": user_id,
                    "split": user_split(user_id),
                    "age": row.get("age"),
                    "gender": row.get("gender"),
                    "job": row.get("job"),
                    "interest": row.get("interest"),
                    "history_count": len(history_items_info),
                    "has_stage2": bool(stage2_A or stage2_B),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        for variant, raw_prompt, raw_prompt_cn, cand_list in [
            ("A", raw_A, raw_cn_A, stage2_A),
            ("B", raw_B, raw_cn_B, stage2_B),
        ]:
            candidates = []
            for c in cand_list:
                image_name = str(c.get("image_name") or "").strip()
                if not image_name:
                    continue
                key = _candidate_key_from_image_name(user_id, image_name)
                cand_prompt = prompt_map.get(key, "")
                candidates.append(
                    {
                        "image_name": image_name,
                        "image_path": _to_rel_stage2_path(user_id, image_name),
                        "prompt": cand_prompt,
                        "initial_rank": c.get("initial_rank"),
                        "quality_score": c.get("quality_score"),
                        "preference_score": c.get("preference_score"),
                        "task_match_score": c.get("task_match_score"),
                    }
                )

            ref = _pick_reference_candidate(candidates)
            if ref is None:
                continue

            sample = {
                "worker_id": user_id,
                "history_items_info": history_items_info,
                "target_item_info": {
                    "item_name": ref["image_name"],
                    "image_path": ref["image_path"],
                    "caption": raw_prompt,
                    "caption_cn": raw_prompt_cn,
                    "query_variant": variant,
                },
                "stage2_candidates": candidates,
            }

            processed[user_split(user_id)].append(sample)

            # stage2 manifest (for evaluation)
            stage2_f.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "split": user_split(user_id),
                        "query_variant": variant,
                        "raw_prompt_simple": raw_prompt,
                        "raw_prompt_simple_cn": raw_prompt_cn,
                        "candidates": candidates,
                        "reference_image_name": ref["image_name"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stage2_f.close()
    users_jsonl.close()

    for split_name in ["train", "val", "test"]:
        out_path = out_root / "processed_dataset" / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(processed[split_name], f, ensure_ascii=False, indent=2)

    print("Wrote:", out_root / "splits.json")
    print("Wrote:", out_root / "manifests" / "stage2_queries.jsonl")
    print("Wrote:", out_root / "manifests" / "users.jsonl")
    print("Wrote:", out_root / "processed_dataset" / "train.json")
    print("Wrote:", out_root / "processed_dataset" / "val.json")
    print("Wrote:", out_root / "processed_dataset" / "test.json")


if __name__ == "__main__":
    build()
