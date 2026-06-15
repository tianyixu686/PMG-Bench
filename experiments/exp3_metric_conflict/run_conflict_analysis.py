#!/usr/bin/env python3
"""
实验三延伸：指标冲突分析 — Spearman 相关矩阵、CLIP-T 高且 LPIPS 差的样本、按风格簇的冲突率。

输入: run_eval_benchmark.py 产出的 JSON（含 per_sample.metrics）。

可选: --cluster_table 指向实验一的 sample_table.json（含 sample_idx 与 cluster_id）。

用法:
  python -m experiments.exp3_metric_conflict.run_conflict_analysis \\
    --benchmark_json outputs/.../ip_adapter_eval.json \\
    --cluster_table outputs/experiments/exp1_dataset_analysis/run01/sample_table.json \\
    --out_dir outputs/experiments/exp3_metric_conflict/run01

  # 合并多方法、仅三指标相关矩阵 + CLIP-I 高且 LPIPS 差:
  python -m experiments.exp3_metric_conflict.run_conflict_analysis \\
    --benchmark_json a/ip_eval.json --benchmark_json a/db_eval.json \\
    --corr_metrics clip_i,dino_i,lpips_target \\
    --conflict_clip_metric clip_i \\
    --out_dir outputs/.../exp3_conflict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("Please install pandas: pip install pandas") from e

from experiments.common.plot_locale import metric_axis_label, setup_matplotlib_chinese

_CLIP_LABEL_ZH = {
    "clip_i": "CLIP-I（相对目标图，越高越好）",
    "clip_t": "CLIP-T（相对 caption，越高越好）",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rows_from_benchmark(obj: dict, method: str = "") -> pd.DataFrame:
    rows = []
    for item in obj.get("per_sample") or []:
        m = item.get("metrics") or {}
        row = {
            "sample_idx": item.get("sample_idx"),
            "user_id": item.get("user_id"),
            "query_variant": item.get("query_variant"),
            "gen_path": item.get("gen_path"),
            "target_path": item.get("target_path"),
            "clip_i": m.get("clip_i"),
            "clip_t": m.get("clip_t"),
            "dino_i": m.get("dino_i"),
            "lpips_target": m.get("lpips_target"),
            "gram_target": m.get("gram_target"),
            "hpsv2": m.get("hpsv2"),
        }
        if method:
            row["method"] = method
        rows.append(row)
    return pd.DataFrame(rows)


def merge_benchmark_rows(paths: List[Path]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for p in paths:
        obj = load_json(p)
        tag = p.stem.replace("_eval", "").replace(".json", "")
        parts.append(rows_from_benchmark(obj, method=tag))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark_json",
        type=str,
        action="append",
        default=None,
        help="可重复传入；多文件时合并 per_sample（建议四方法各一份 eval.json）",
    )
    ap.add_argument(
        "--benchmark_json_legacy",
        type=str,
        default="",
        help="兼容旧用法：单文件，等价于一次 --benchmark_json",
    )
    ap.add_argument("--cluster_table", type=str, default="", help="实验一输出的 sample_table.json")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--clip_quantile", type=float, default=0.8, help="CLIP 高于该分位视为「高」")
    ap.add_argument("--lpips_quantile", type=float, default=0.8, help="LPIPS 高于该分位视为「差」")
    ap.add_argument(
        "--conflict_clip_metric",
        type=str,
        choices=("clip_i", "clip_t"),
        default="clip_i",
        help="与 LPIPS 联合判定冲突时使用的 CLIP 通道（默认 clip_i = 生成 vs 目标图）",
    )
    ap.add_argument(
        "--corr_metrics",
        type=str,
        default="clip_i,clip_t,dino_i,lpips_target,gram_target,hpsv2",
        help="参与 Spearman 矩阵的列，逗号分隔。论文「三指标」可用 clip_i,dino_i,lpips_target",
    )
    args = ap.parse_args()

    paths: List[Path] = []
    if args.benchmark_json:
        paths.extend(Path(p).expanduser().resolve() for p in args.benchmark_json)
    if str(args.benchmark_json_legacy).strip():
        paths.append(Path(args.benchmark_json_legacy).expanduser().resolve())
    if not paths:
        raise SystemExit("请提供 --benchmark_json（可多次）或 --benchmark_json_legacy")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_chinese()

    if len(paths) == 1:
        obj = load_json(paths[0])
        df = rows_from_benchmark(obj)
    else:
        df = merge_benchmark_rows(paths)

    all_metric_cols = ["clip_i", "clip_t", "dino_i", "lpips_target", "gram_target", "hpsv2"]
    corr_cols = [c.strip() for c in str(args.corr_metrics).split(",") if c.strip()]
    for c in corr_cols:
        if c not in all_metric_cols:
            raise SystemExit(f"Unknown corr metric {c!r}; allowed: {all_metric_cols}")
    mdf = df[all_metric_cols].apply(pd.to_numeric, errors="coerce")
    corr = mdf[corr_cols].corr(method="spearman")

    corr.to_csv(out_dir / "spearman_correlation.csv")
    plt.figure(figsize=(max(6, 0.9 * len(corr_cols)), max(5, 0.85 * len(corr_cols))))
    im = plt.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_labels = [metric_axis_label(str(c)) for c in corr.columns]
    ytick_labels = [metric_axis_label(str(i)) for i in corr.index]
    plt.xticks(range(len(corr.columns)), tick_labels, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), ytick_labels)
    plt.title("Spearman 相关系数（自动评测指标）")
    plt.tight_layout()
    plt.savefig(out_dir / "spearman_correlation.png", dpi=160)
    plt.close()

    clip_col = args.conflict_clip_metric
    clip_s = mdf[clip_col].dropna()
    lp = mdf["lpips_target"].dropna()
    if len(clip_s) > 5 and len(lp) > 5:
        c_thr = float(clip_s.quantile(args.clip_quantile))
        l_thr = float(lp.quantile(args.lpips_quantile))
        conflict = (
            (mdf[clip_col] >= c_thr)
            & (mdf["lpips_target"] >= l_thr)
            & mdf[clip_col].notna()
            & mdf["lpips_target"].notna()
        )
        df["conflict_clip_high_lpips_bad"] = conflict.astype(int)
        n_conf = int(conflict.sum())
        n_valid = int((mdf[clip_col].notna() & mdf["lpips_target"].notna()).sum())
        report = {
            "n_samples_metrics": int(len(df)),
            "n_valid_clip_lpips": n_valid,
            "conflict_count": n_conf,
            "conflict_rate": float(n_conf / n_valid) if n_valid else 0.0,
            "conflict_clip_metric": clip_col,
            "clip_threshold": c_thr,
            "lpips_threshold": l_thr,
            "clip_quantile": args.clip_quantile,
            "lpips_quantile": args.lpips_quantile,
        }
    else:
        report = {"error": "not enough data for clip vs lpips conflict rule"}
        df["conflict_clip_high_lpips_bad"] = 0

    cluster_path = Path(args.cluster_table).expanduser().resolve() if str(args.cluster_table).strip() else None
    if cluster_path and cluster_path.is_file():
        ct = pd.DataFrame(load_json(cluster_path))
        if "sample_idx" in ct.columns and "cluster_id" in ct.columns:
            df = df.merge(ct[["sample_idx", "cluster_id"]], on="sample_idx", how="left")
            if "conflict_clip_high_lpips_bad" in df.columns:
                g = df.dropna(subset=["cluster_id"]).groupby("cluster_id")["conflict_clip_high_lpips_bad"]
                per_cluster = g.agg(["sum", "count", "mean"]).reset_index()
                per_cluster.rename(columns={"sum": "conflicts", "count": "n", "mean": "conflict_rate"}, inplace=True)
                per_cluster.to_csv(out_dir / "conflict_rate_by_cluster.csv", index=False)
                report["per_cluster_conflict"] = per_cluster.to_dict(orient="records")

                plt.figure(figsize=(max(8, per_cluster["cluster_id"].nunique() * 0.4), 4))
                plt.bar(per_cluster["cluster_id"].astype(str), per_cluster["conflict_rate"], color="coral")
                plt.xlabel("风格簇 ID")
                plt.ylabel("冲突率")
                plt.title(f"{metric_axis_label(clip_col)} 高且 LPIPS 高：各风格簇冲突率")
                plt.tight_layout()
                plt.savefig(out_dir / "conflict_rate_by_cluster.png", dpi=150)
                plt.close()

    if len(clip_s) > 5 and len(lp) > 5:
        conflict_rows = df[df["conflict_clip_high_lpips_bad"] == 1]
        conflict_rows.to_csv(out_dir / "conflict_samples.csv", index=False)
        plt.figure(figsize=(6, 5))
        ok = mdf[clip_col].notna() & mdf["lpips_target"].notna()
        plt.scatter(
            mdf.loc[ok & ~conflict, clip_col],
            mdf.loc[ok & ~conflict, "lpips_target"],
            s=16,
            alpha=0.45,
            c="gray",
            label="其他样本",
        )
        plt.scatter(
            mdf.loc[ok & conflict, clip_col],
            mdf.loc[ok & conflict, "lpips_target"],
            s=28,
            alpha=0.85,
            c="red",
            label="冲突样本",
        )
        plt.axvline(c_thr, color="k", ls="--", lw=0.8, alpha=0.5)
        plt.axhline(l_thr, color="k", ls="--", lw=0.8, alpha=0.5)
        plt.xlabel(_CLIP_LABEL_ZH.get(clip_col, metric_axis_label(clip_col)))
        plt.ylabel("相对目标图的 LPIPS（越高越不相似）")
        plt.legend(loc="upper left")
        plt.title("指标冲突区域（右上象限）")
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_clip_vs_lpips.png", dpi=150)
        plt.close()
    else:
        conflict_rows = df.iloc[0:0]
        conflict_rows.to_csv(out_dir / "conflict_samples.csv", index=False)

    with (out_dir / "conflict_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Wrote:", out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
