#!/usr/bin/env python3
"""仅生成图 4-1（不跑 CLIP/UMAP）。默认合并 train+val+test。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.common.overall_scores import (
    collect_overall_three_scores,
    plot_fig4_1_three_distributions,
    summarize_1d,
)
from experiments.common.repo import repo_root
from experiments.common.userpref_io import load_concat_splits


def main():
    repo = repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed_dir",
        type=str,
        default=str(repo / "data/userpref_v1/processed_dataset"),
        help="含 train.json / val.json / test.json 的目录",
    )
    ap.add_argument("--out_dir", type=str, default=str(repo / "outputs/experiments/fig4_1_overall_scores"))
    ap.add_argument("--chinese_labels", action="store_true", help="轴标题用中文（需系统有可用的 CJK 字体）")
    args = ap.parse_args()
    proc = Path(args.processed_dir)
    merged, meta = load_concat_splits(proc)
    if not merged:
        raise SystemExit(f"No merged samples under {proc}")
    scores, counts = collect_overall_three_scores(merged)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = plot_fig4_1_three_distributions(out, scores, use_chinese_labels=bool(args.chinese_labels))
    summary = {
        "concat_meta": meta,
        "source_counts": counts,
        "quality_score": summarize_1d(scores["quality_score"]),
        "task_match_score": summarize_1d(scores["task_match_score"]),
        "aesthetic_preference_score": summarize_1d(scores["aesthetic_preference_score"]),
        "figures": paths,
    }
    with (out / "fig4_1_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
