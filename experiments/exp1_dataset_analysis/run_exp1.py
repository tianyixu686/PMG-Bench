#!/usr/bin/env python3
"""
实验一：UserPref-Bench 数据集统计（用户数/样本数/图像计数/样本结构说明/用户画像/评分分布）
与目标图在 CLIP 风格空间的分布（热力图 + 聚类覆盖度）。

输出 dataset_report.json 含：sample_unit、counts_this_split、samples_per_user、query_variant 分布、
manifests/users.jsonl 画像汇总、history 与 stage2 分数统计、stage2 三维 Pearson 相关；
图含 hist_stage2_*、hist_history_*、scatter_stage2_*、scatter3d_stage2_scores、heatmap 等。

用法（在 PMG-Bench 仓库根目录）:
  python -m experiments.exp1_dataset_analysis.run_exp1 \\
    --split_json data/userpref_v1/processed_dataset/test.json \\
    --out_dir outputs/experiments/exp1_dataset_analysis/run01

  # 全库（train+val+test 合并列表）计数:
  python -m experiments.exp1_dataset_analysis.run_exp1 ... --include_all_splits

可选：若已安装 umap-learn，加 --use_umap 得到非线性 2D；否则默认 PCA。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common.clip_embed import embed_image_paths
from experiments.common.overall_scores import (
    collect_overall_three_scores,
    plot_fig4_1_three_distributions,
    summarize_1d,
)
from experiments.common.repo import repo_root
from experiments.common.userpref_io import (
    data_root_from_split_json,
    load_concat_splits,
    load_split,
    resolve_image_path,
    select_target_from_candidates,
    sample_key,
)


def _entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log(p)).sum())


def _gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    if n == 0:
        return 0.0
    return float((2 * np.dot(np.arange(1, n + 1), x)) / (n * x.sum()) - (n + 1) / n)


def collect_stage2_scores(samples: List[dict]) -> Dict[str, List[float]]:
    out = {"quality_score": [], "preference_score": [], "task_match_score": []}
    for s in samples:
        for c in s.get("stage2_candidates") or []:
            for k in out:
                v = c.get(k)
                if v is None:
                    continue
                try:
                    out[k].append(float(v))
                except Exception:
                    pass
    return out


def collect_history_scores(samples: List[dict]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"preference_score": [], "quality_score": []}
    for s in samples:
        for h in s.get("history_items_info") or []:
            for k in out:
                v = h.get(k)
                if v is None:
                    continue
                try:
                    out[k].append(float(v))
                except Exception:
                    pass
    return out


def count_entities(samples: List[dict], repo: Path, data_root: Path) -> Dict[str, Any]:
    """图像/引用层计数：去重 id、路径存在性（尽力而为）。"""
    history_item_names: set = set()
    history_paths_resolved: set = set()
    stage2_keys: set = set()
    target_paths: set = set()
    n_history_rows = 0
    n_stage2_rows = 0
    hist_file_exists = 0
    st2_file_exists = 0
    tgt_file_exists = 0

    for s in samples:
        for h in s.get("history_items_info") or []:
            n_history_rows += 1
            name = str(h.get("item_name") or "").strip()
            if name:
                history_item_names.add(name)
            rp = resolve_image_path(str(h.get("image_path") or ""), repo_root=repo, data_root=data_root)
            if rp:
                history_paths_resolved.add(rp)
                if Path(rp).is_file():
                    hist_file_exists += 1
        for c in s.get("stage2_candidates") or []:
            n_stage2_rows += 1
            rp = resolve_image_path(str(c.get("image_path") or ""), repo_root=repo, data_root=data_root)
            nm = str(c.get("image_name") or "").strip()
            sid = ("path", rp) if rp else ("name", nm) if nm else ("empty", str(id(c)))
            stage2_keys.add(sid)
            if rp and Path(rp).is_file():
                st2_file_exists += 1
        ti = s.get("target_item_info") or {}
        rp = resolve_image_path(str(ti.get("image_path") or ""), repo_root=repo, data_root=data_root)
        if rp:
            target_paths.add(rp)
            if Path(rp).is_file():
                tgt_file_exists += 1

    return {
        "n_samples": len(samples),
        "n_unique_users_worker_id": len({str(s.get("worker_id") or "").strip() for s in samples if str(s.get("worker_id") or "").strip()}),
        "n_history_item_rows": n_history_rows,
        "n_unique_history_item_name": len(history_item_names),
        "n_unique_history_resolved_path": len(history_paths_resolved),
        "n_history_files_found_on_disk": hist_file_exists,
        "n_stage2_candidate_rows": n_stage2_rows,
        "n_unique_stage2_candidate_images": len(stage2_keys),
        "n_unique_target_image_path": len(target_paths),
        "n_target_files_found_on_disk": tgt_file_exists,
    }


def samples_per_user_stats(samples: List[dict]) -> Dict[str, Any]:
    c = Counter(str(s.get("worker_id") or "").strip() for s in samples if str(s.get("worker_id") or "").strip())
    counts = list(c.values())
    hist = {str(k): int(v) for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:50]}
    return {
        "n_users_with_at_least_one_sample": len(c),
        "samples_per_user_histogram": dict(Counter(counts)),
        "samples_per_user_min": int(min(counts)) if counts else 0,
        "samples_per_user_max": int(max(counts)) if counts else 0,
        "samples_per_user_mean": float(np.mean(counts)) if counts else 0.0,
        "top_users_by_sample_count": hist,
    }


def query_variant_counts(samples: List[dict]) -> Dict[str, int]:
    out = Counter()
    for s in samples:
        qv = str((s.get("target_item_info") or {}).get("query_variant") or "").strip() or "unknown"
        out[qv] += 1
    return dict(out)


def collect_stage2_triplets(samples: List[dict]) -> List[Tuple[float, float, float]]:
    triples: List[Tuple[float, float, float]] = []
    for s in samples:
        for c in s.get("stage2_candidates") or []:
            try:
                p = float(c.get("preference_score"))
                q = float(c.get("quality_score"))
                t = float(c.get("task_match_score"))
            except Exception:
                continue
            if p == p and q == q and t == t:
                triples.append((p, q, t))
    return triples


def load_users_manifest(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        return {}
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("user_id") or row.get("worker_id") or "").strip()
            if uid:
                out[uid] = row
    return out


def _top_categories(values: List[Any], k: int = 20) -> List[Dict[str, Any]]:
    c = Counter(str(v) if v is not None else "(missing)" for v in values)
    return [{"value": a, "count": b} for a, b in c.most_common(k)]


def user_profile_distribution(manifest: Dict[str, dict], user_ids: set) -> Dict[str, Any]:
    if not manifest or not user_ids:
        return {"source": "manifests/users.jsonl", "matched_users": 0, "note": "manifest 不存在或当前 split 无匹配 user_id"}
    ages: List[Any] = []
    genders: List[Any] = []
    jobs: List[Any] = []
    interests: List[Any] = []
    matched = 0
    for uid in user_ids:
        row = manifest.get(uid)
        if not row:
            continue
        matched += 1
        ages.append(row.get("age"))
        genders.append(row.get("gender"))
        jobs.append(row.get("job"))
        interests.append(row.get("interest"))

    return {
        "source": "manifests/users.jsonl",
        "n_users_requested": len(user_ids),
        "n_users_matched_in_manifest": matched,
        "age_top": _top_categories(ages, 25),
        "gender_top": _top_categories(genders, 15),
        "job_top": _top_categories(jobs, 25),
        "interest_top": _top_categories(interests, 25),
    }


def history_stats(samples: List[dict]) -> Dict[str, Any]:
    lens = []
    pref_m = []
    qual_m = []
    for s in samples:
        h = s.get("history_items_info") or []
        lens.append(len(h))
        prefs = [float(x["preference_score"]) for x in h if x.get("preference_score") is not None]
        quals = [float(x["quality_score"]) for x in h if x.get("quality_score") is not None]
        if prefs:
            pref_m.append(float(np.mean(prefs)))
        if quals:
            qual_m.append(float(np.mean(quals)))
    return {
        "history_len_per_sample": {
            "min": int(np.min(lens)) if lens else 0,
            "max": int(np.max(lens)) if lens else 0,
            "mean": float(np.mean(lens)) if lens else 0.0,
            "p50": float(np.percentile(lens, 50)) if lens else 0.0,
        },
        "per_user_history_mean_preference": {"mean": float(np.mean(pref_m)), "std": float(np.std(pref_m))} if pref_m else {},
        "per_user_history_mean_quality": {"mean": float(np.mean(qual_m)), "std": float(np.std(qual_m))} if qual_m else {},
    }


def load_splits_users(splits_path: Path) -> Dict[str, Any]:
    if not splits_path.is_file():
        return {}
    with splits_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_overall_three_dimension_report(
    score_samples: List[dict], scope: str
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    scores, counts = collect_overall_three_scores(score_samples)
    return (
        {
            "scope": scope,
            "source_counts": counts,
            "quality_score": summarize_1d(scores["quality_score"]),
            "task_match_score": summarize_1d(scores["task_match_score"]),
            "aesthetic_preference_score": summarize_1d(scores["aesthetic_preference_score"]),
            "note_zh": (
                "质量分、审美偏好分：合并自 history_items_info 与 stage2_candidates 中所有有效分值。"
                "任务匹配分：合并自上述两处中的 task_match_score（当前构建数据以 stage2 为主；history 若有则一并计入）。"
            ),
        },
        scores,
    )


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 3 or b.size < 3:
        return None
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    m = np.corrcoef(a, b)
    return float(m[0, 1])


def run(
    split_json: Path,
    out_dir: Path,
    *,
    n_clusters: int,
    use_umap: bool,
    random_state: int,
    image_size: int,
    include_all_splits: bool,
) -> None:
    repo = repo_root()
    data_root = data_root_from_split_json(split_json)
    samples = load_split(split_json)

    # 图 4-1 与「整体三维分布」统计：默认合并 train+val+test（与论文全库口径一致）
    score_samples = samples
    overall_scores_scope = "current_split_json_only"
    if split_json.parent.name == "processed_dataset":
        merged_all, _concat_meta_scores = load_concat_splits(split_json.parent)
        if merged_all:
            score_samples = merged_all
            overall_scores_scope = "merged_train_val_test_json"

    splits_path = data_root / "splits.json"
    if not splits_path.is_file():
        splits_path = repo / "data" / "userpref_v1" / "splits.json"
    splits = load_splits_users(splits_path)

    user_ids_in_split = {str(s.get("worker_id") or "").strip() for s in samples if str(s.get("worker_id") or "").strip()}
    counts_split = count_entities(samples, repo, data_root)
    manifest_users = load_users_manifest(data_root / "manifests" / "users.jsonl")

    full_processed_extra: Dict[str, Any] = {}
    if include_all_splits:
        merged, concat_meta = load_concat_splits(split_json.parent)
        full_processed_extra = {
            "concat_json_meta": concat_meta,
            "counts_concat_train_val_test": count_entities(merged, repo, data_root),
            "n_unique_users_concat": len({str(s.get("worker_id") or "").strip() for s in merged if str(s.get("worker_id") or "").strip()}),
            "samples_per_user_concat": samples_per_user_stats(merged),
        }

    target_paths: List[str] = []
    meta_rows: List[dict] = []
    for idx, s in enumerate(samples):
        tp = select_target_from_candidates(s, repo_root=repo, data_root=data_root)
        target_paths.append(tp)
        ti = s.get("target_item_info") or {}
        meta_rows.append(
            {
                "sample_idx": idx,
                "worker_id": str(s.get("worker_id") or ""),
                "query_variant": str(ti.get("query_variant") or ""),
                "sample_key": sample_key(s),
                "target_path": tp,
                "target_exists": bool(tp and Path(tp).is_file()),
            }
        )

    device_str = "cuda"
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = str(device)
    except Exception:
        device = None  # type: ignore

    emb_np = None
    valid_for_emb: List[int] = []
    if device is not None:
        import torch

        emb, valid_idx = embed_image_paths(
            target_paths, device=torch.device(device), image_size=image_size, batch_size=32
        )
        if emb is not None:
            emb_np = emb.numpy()
            valid_for_emb = valid_idx

    overall_three_report, overall_three_scores = build_overall_three_dimension_report(score_samples, overall_scores_scope)

    triples = collect_stage2_triplets(samples)
    triple_stats: Dict[str, Any] = {"n_triplets_all_three_scores": len(triples)}
    if triples:
        T = np.array(triples, dtype=np.float64)
        triple_stats["mean_preference"] = float(T[:, 0].mean())
        triple_stats["mean_quality"] = float(T[:, 1].mean())
        triple_stats["mean_task_match"] = float(T[:, 2].mean())
        triple_stats["pearson_preference_quality"] = _pearson_1d(T[:, 0], T[:, 1])
        triple_stats["pearson_preference_task_match"] = _pearson_1d(T[:, 0], T[:, 2])
        triple_stats["pearson_quality_task_match"] = _pearson_1d(T[:, 1], T[:, 2])

    hist_scores = collect_history_scores(samples)
    hist_score_stats = {
        k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        for k, v in hist_scores.items()
        if v
    }

    report: Dict[str, Any] = {
        "split_json": str(split_json),
        "data_root": str(data_root),
        "sample_unit": {
            "description_zh": (
                "processed_dataset 中每条记录为一个评测样本，对应唯一键 (worker_id, query_variant)。"
                "包含：阶段1 history_items_info（多张历史图及 preference/quality 评分）；"
                "阶段2 stage2_candidates（该 query 下多张候选图及 quality/preference/task_match 三维评分）；"
                "target_item_info 为该 query 的伪参考图与任务文本 caption（通常取 preference 最高的候选）。"
            ),
            "unique_key": ["worker_id", "target_item_info.query_variant"],
            "top_level_fields": ["worker_id", "history_items_info", "target_item_info", "stage2_candidates"],
            "note_history_scores": "history 条目仅有 preference_score 与 quality_score，无 task_match_score。",
        },
        "n_samples": len(samples),
        "n_unique_users_in_this_split_json": len(user_ids_in_split),
        "counts_this_split": counts_split,
        "samples_per_user_this_split": samples_per_user_stats(samples),
        "query_variant_counts_this_split": query_variant_counts(samples),
        "user_profile_distribution": user_profile_distribution(manifest_users, user_ids_in_split),
        "splits_json": str(splits_path) if splits_path.is_file() else None,
        "split_user_counts": {
            "train": len(splits.get("train_users") or []),
            "val": len(splits.get("val_users") or []),
            "test": len(splits.get("test_users") or []),
            "total_unique_users_train_plus_val_plus_test": (
                len(splits.get("train_users") or [])
                + len(splits.get("val_users") or [])
                + len(splits.get("test_users") or [])
            ),
        }
        if splits
        else None,
        "history": history_stats(samples),
        "history_score_stats_pooled_items": hist_score_stats,
        "stage2_triplet_stats": triple_stats,
        "overall_three_dimension_scores": overall_three_report,
        "device": device_str,
    }
    if full_processed_extra:
        report["full_processed_train_val_test"] = full_processed_extra

    s2 = collect_stage2_scores(samples)
    report["stage2_score_stats_pooled_candidates"] = {
        k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
        for k, v in s2.items()
        if v
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 图 4-1：质量 / 任务匹配 / 审美偏好 整体分布（三张 + 组合）---
    report["fig4_1_output_paths"] = plot_fig4_1_three_distributions(out_dir, overall_three_scores)

    # --- 1D histograms: stage2 (三维分项) + history (二维分项) ---
    for name, values in s2.items():
        if len(values) < 2:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=40, color="steelblue", alpha=0.85)
        plt.xlabel(name)
        plt.ylabel("count")
        plt.title(f"Stage2 candidates: {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_stage2_{name}.png", dpi=150)
        plt.close()

    for name, values in hist_scores.items():
        if len(values) < 2:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=40, color="sienna", alpha=0.85)
        plt.xlabel(name)
        plt.ylabel("count")
        plt.title(f"History items (pooled): {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_history_{name}.png", dpi=150)
        plt.close()

    # --- pairwise scatter (preference vs quality etc.) ---
    if len(s2.get("preference_score") or []) > 10:
        pq = list(zip(s2["preference_score"], s2["quality_score"]))
        if len(pq) > 10:
            p, q = zip(*pq[:5000])
            plt.figure(figsize=(5, 5))
            plt.scatter(p, q, s=8, alpha=0.35, c="darkgreen")
            plt.xlabel("preference_score")
            plt.ylabel("quality_score")
            plt.title("Stage2 candidates (subsample up to 5000 pairs)")
            plt.tight_layout()
            plt.savefig(out_dir / "scatter_stage2_preference_vs_quality.png", dpi=150)
            plt.close()

    if len(triples) > 30:
        T = np.array(triples, dtype=np.float64)
        cap = min(8000, len(T))
        T = T[:cap]
        plt.figure(figsize=(5, 5))
        plt.scatter(T[:, 0], T[:, 2], s=8, alpha=0.35, c="navy")
        plt.xlabel("preference_score")
        plt.ylabel("task_match_score")
        plt.title(f"Stage2 candidates (n={cap})")
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_stage2_preference_vs_task_match.png", dpi=150)
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.scatter(T[:, 1], T[:, 2], s=8, alpha=0.35, c="purple")
        plt.xlabel("quality_score")
        plt.ylabel("task_match_score")
        plt.title(f"Stage2 candidates (n={cap})")
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_stage2_quality_vs_task_match.png", dpi=150)
        plt.close()

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(T[:, 0], T[:, 1], T[:, 2], s=6, alpha=0.25, c=T[:, 2], cmap="viridis")
        ax.set_xlabel("preference")
        ax.set_ylabel("quality")
        ax.set_zlabel("task_match")
        ax.set_title("Stage2: 3D score distribution (subsample)")
        plt.tight_layout()
        plt.savefig(out_dir / "scatter3d_stage2_scores.png", dpi=150)
        plt.close()

    cluster_labels = np.full(len(samples), -1, dtype=np.int32)
    z2 = None
    if emb_np is not None and len(emb_np) >= max(n_clusters, 2):
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            try:
                km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
            except TypeError:
                km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels_fit = km.fit_predict(emb_np)
            for j, global_idx in enumerate(valid_for_emb):
                cluster_labels[global_idx] = int(labels_fit[j])

            counts = np.bincount(labels_fit, minlength=n_clusters)
            report["style_clusters"] = {
                "method": "kmeans_clip_vit_b32",
                "n_clusters": n_clusters,
                "n_embedded_targets": int(len(emb_np)),
                "entropy": _entropy(counts),
                "effective_clusters_exp_entropy": float(math.exp(_entropy(counts))),
                "min_cluster_frac": float(counts.min() / counts.sum()),
                "gini_cluster_sizes": _gini(counts),
            }

            if use_umap:
                try:
                    import umap

                    reducer = umap.UMAP(n_components=2, random_state=random_state)
                    z2 = reducer.fit_transform(emb_np)
                    report["embedding_2d"] = "umap"
                except Exception:
                    z2 = PCA(n_components=2, random_state=random_state).fit_transform(emb_np)
                    report["embedding_2d"] = "pca_fallback_umap_missing"
            else:
                z2 = PCA(n_components=2, random_state=random_state).fit_transform(emb_np)
                report["embedding_2d"] = "pca"

            # heatmap in 2D style space
            plt.figure(figsize=(7, 6))
            h = plt.hist2d(z2[:, 0], z2[:, 1], bins=40, cmap="magma")
            plt.colorbar(h[3], label="target count / bin")
            plt.xlabel("dim-1")
            plt.ylabel("dim-2")
            plt.title("Target images: density in 2D (PCA or UMAP on CLIP embeddings)")
            plt.tight_layout()
            plt.savefig(out_dir / "heatmap_target_style_space.png", dpi=160)
            plt.close()

            plt.figure(figsize=(7, 6))
            sc = plt.scatter(z2[:, 0], z2[:, 1], c=labels_fit, cmap="tab10", s=22, alpha=0.85)
            plt.colorbar(sc, label="cluster id")
            plt.xlabel("dim-1")
            plt.ylabel("dim-2")
            plt.title("Target images colored by KMeans cluster")
            plt.tight_layout()
            plt.savefig(out_dir / "scatter_targets_by_cluster.png", dpi=160)
            plt.close()
        except Exception as e:
            report["style_clusters_error"] = str(e)

    for i, row in enumerate(meta_rows):
        row["cluster_id"] = int(cluster_labels[i])

    with (out_dir / "dataset_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with (out_dir / "sample_table.json").open("w", encoding="utf-8") as f:
        json.dump(meta_rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_dir / 'dataset_report.json'}")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


def main():
    ap = argparse.ArgumentParser(description="Exp1: dataset stats + target style-space heatmap")
    ap.add_argument("--split_json", type=str, default=str(_ROOT / "data/userpref_v1/processed_dataset/test.json"))
    ap.add_argument("--out_dir", type=str, default=str(_ROOT / "outputs/experiments/exp1_dataset_analysis/default_run"))
    ap.add_argument("--n_clusters", type=int, default=12)
    ap.add_argument("--use_umap", action="store_true")
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument(
        "--include_all_splits",
        action="store_true",
        help="同时读取同目录下 train.json/val.json/test.json 合并统计「全库」用户数与图像计数",
    )
    args = ap.parse_args()
    run(
        Path(args.split_json),
        Path(args.out_dir),
        n_clusters=args.n_clusters,
        use_umap=args.use_umap,
        random_state=args.random_state,
        image_size=args.image_size,
        include_all_splits=bool(args.include_all_splits),
    )


if __name__ == "__main__":
    main()
