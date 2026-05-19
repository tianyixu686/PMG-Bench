"""
全库「三维」打分整体分布：质量分、任务匹配分、审美偏好分（preference）。
- 质量 / 审美偏好：合并 history_items_info 与 stage2_candidates 中所有有效分值。
- 任务匹配：仅 stage2_candidates（history 一般无 task_match_score；若有则一并并入整体分布）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def collect_overall_three_scores(samples: List[dict]) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    aesthetic_preference: List[float] = []
    quality: List[float] = []
    task_match: List[float] = []
    n_pref_h = n_pref_s2 = n_qual_h = n_qual_s2 = n_tm_s2 = n_tm_h = 0

    for s in samples:
        for h in s.get("history_items_info") or []:
            v = h.get("preference_score")
            if v is not None:
                try:
                    aesthetic_preference.append(float(v))
                    n_pref_h += 1
                except (TypeError, ValueError):
                    pass
            v = h.get("quality_score")
            if v is not None:
                try:
                    quality.append(float(v))
                    n_qual_h += 1
                except (TypeError, ValueError):
                    pass
            v = h.get("task_match_score")
            if v is not None:
                try:
                    fv = float(v)
                    if fv == fv:
                        task_match.append(fv)
                        n_tm_h += 1
                except (TypeError, ValueError):
                    pass

        for c in s.get("stage2_candidates") or []:
            v = c.get("preference_score")
            if v is not None:
                try:
                    aesthetic_preference.append(float(v))
                    n_pref_s2 += 1
                except (TypeError, ValueError):
                    pass
            v = c.get("quality_score")
            if v is not None:
                try:
                    quality.append(float(v))
                    n_qual_s2 += 1
                except (TypeError, ValueError):
                    pass
            v = c.get("task_match_score")
            if v is not None:
                try:
                    fv = float(v)
                    if fv == fv:
                        task_match.append(fv)
                        n_tm_s2 += 1
                except (TypeError, ValueError):
                    pass

    counts = {
        "preference_from_history": n_pref_h,
        "preference_from_stage2": n_pref_s2,
        "quality_from_history": n_qual_h,
        "quality_from_stage2": n_qual_s2,
        "task_match_from_history": n_tm_h,
        "task_match_from_stage2": n_tm_s2,
    }
    return (
        {
            "aesthetic_preference_score": aesthetic_preference,
            "quality_score": quality,
            "task_match_score": task_match,
        },
        counts,
    )


def summarize_1d(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    a = np.array(values, dtype=np.float64)
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
    }


def _pick_font_preference() -> List[str]:
    import matplotlib.font_manager as fm

    prefer: List[str] = []
    for f in fm.fontManager.ttflist:
        n = (f.name or "").lower()
        if any(x in n for x in ("cjk", "noto sans cjk", "noto serif cjk", "simhei", "yahei", "wenquanyi", "zen hei")):
            if f.name not in prefer:
                prefer.append(f.name)
    return prefer + ["DejaVu Sans"]


def _setup_plot_font() -> None:
    import matplotlib

    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = _pick_font_preference()


def plot_fig4_1_three_distributions(
    out_dir,
    scores: Dict[str, List[float]],
    *,
    dpi: int = 150,
    bins_discrete: bool = True,
    use_chinese_labels: bool = False,
) -> List[str]:
    """输出三张独立分布图 + 一张 1×3 组合图（论文图 4-1）。"""
    from pathlib import Path

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_font()

    def bins_for(xs: List[float]):
        if not xs:
            return 40
        lo, hi = min(xs), max(xs)
        if bins_discrete and hi - lo <= 9 and lo >= 0:
            return np.arange(int(lo) - 0.5, int(hi) + 1.5, 1.0)
        return 40

    if use_chinese_labels:
        specs = [
            ("quality_score", "质量分", "#2c7fb8"),
            ("task_match_score", "任务匹配分", "#7fbc41"),
            ("aesthetic_preference_score", "审美偏好分", "#f46d43"),
        ]
        supt = "图 4-1  质量分、任务匹配分与审美偏好分整体分布"
        ylab = "频数"
        sub = "（整体分布）"
    else:
        specs = [
            ("quality_score", "Quality score", "#2c7fb8"),
            ("task_match_score", "Task match score", "#7fbc41"),
            ("aesthetic_preference_score", "Aesthetic preference", "#f46d43"),
        ]
        supt = "Fig. 4-1  Overall distributions (pooled history + stage2)"
        ylab = "Count"
        sub = "(overall)"
    written: List[str] = []

    for key, title_zh, color in specs:
        xs = scores.get(key) or []
        if len(xs) < 1:
            continue
        plt.figure(figsize=(5.2, 3.8))
        b = bins_for(xs)
        if isinstance(b, np.ndarray):
            plt.hist(xs, bins=b, color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        else:
            plt.hist(xs, bins=b, color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        plt.xlabel(title_zh)
        plt.ylabel(ylab)
        plt.title(f"{title_zh} {sub}".strip())
        plt.tight_layout()
        fn = out_dir / f"fig4_1_{key}.png"
        plt.savefig(fn, dpi=dpi)
        plt.close()
        written.append(str(fn))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (key, title_zh, color) in zip(axes, specs):
        xs = scores.get(key) or []
        if len(xs) < 1:
            ax.set_visible(False)
            continue
        b = bins_for(xs)
        if isinstance(b, np.ndarray):
            ax.hist(xs, bins=b, color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        else:
            ax.hist(xs, bins=b, color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        ax.set_xlabel(title_zh)
        ax.set_ylabel(ylab)
    fig.suptitle(supt, fontsize=12, y=1.02)
    fig.tight_layout()
    comb = out_dir / "fig4_1_three_scores_combined.png"
    fig.savefig(comb, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(str(comb))
    return written
