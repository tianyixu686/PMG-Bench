#!/usr/bin/env python3
"""
论文级可视化：读取若干 `run_eval_benchmark.py` 输出的 JSON，生成
  - 3.2 维度指标表（Markdown）
  - 各方法 CLIP(I-T) 总体均值（clip_t）
  - 图：DINO-Sim vs LPIPS（每点=某用户在该方法上的样本均值）
  - Gram 总体表
  - 表 4-8：HPS v2 vs FID
  - 雷达图（CLIP、DINO、LPIPS、Gram、HPS；LPIPS/Gram 转为越大越好后归一化）
  - 案例对比拼图（fig_4_8_case_grid：仅 Target + 各方法，避免单图过大）
  - 另附两张图：历史 preference/quality 分数柱（多用户分面板）、高/低偏好历史图墙（≥4 vs ≤2，少用户多缩略图）

用法:
  python experiments/exp3_bench_metrics/plot_paper_figs.py \\
    --test_json data/userpref_v1/processed_dataset/test.json \\
    --out_dir outputs/experiments/bench_paper_figs \\
    --eval_specs textual_inversion=... dreambooth=... ip_adapter=... pmg=... \\
    --case_users 2,11,25 \\
    --case_users_history 2,11,25,78,82,100 \\
    --case_users_history_moodboard 2,11,25 \\
    --data_root data/userpref_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(np.mean(np.array(xs, dtype=np.float64)))


def _parse_eval_specs(specs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"Bad --eval_specs entry (want name=path): {s}")
        k, v = s.split("=", 1)
        out[k.strip()] = Path(v.strip()).expanduser().resolve()
    return out


def _per_user_means(per_sample: List[dict], key: str) -> Dict[str, List[float]]:
    """user_id -> list of sample-level metric values (non-null)."""
    bu: Dict[str, List[float]] = defaultdict(list)
    for row in per_sample:
        uid = str(row.get("user_id") or "").strip()
        m = row.get("metrics") or {}
        v = m.get(key)
        if v is None:
            continue
        try:
            bu[uid].append(float(v))
        except Exception:
            continue
    return bu


def _norm_radar(values_by_method: Dict[str, Dict[str, float]], *, higher_better: Dict[str, bool]) -> Dict[str, Dict[str, float]]:
    """Per-axis min-max normalize to [0,1] across methods."""
    methods = list(values_by_method.keys())
    axes = list(next(iter(values_by_method.values())).keys())
    out: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    for ax in axes:
        hb = higher_better.get(ax, True)
        raw = [values_by_method[m][ax] for m in methods if ax in values_by_method[m] and not math.isnan(values_by_method[m][ax])]
        if not raw:
            for m in methods:
                out[m][ax] = float("nan")
            continue
        lo, hi = min(raw), max(raw)
        for m in methods:
            v = values_by_method[m].get(ax, float("nan"))
            if v != v:
                out[m][ax] = float("nan")
                continue
            if hi == lo:
                out[m][ax] = 0.5
            elif hb:
                out[m][ax] = (v - lo) / (hi - lo)
            else:
                out[m][ax] = (hi - v) / (hi - lo)
    return out


def _write_md_table(path: Path, rows: List[Tuple[str, str, str]]):
    """rows: (dimension, metric, meaning)"""
    lines = [
        "| 维度 | 指标 | 含义 |",
        "|------|------|------|",
    ]
    for a, b, c in rows:
        lines.append(f"| {a} | {b} | {c} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


from experiments.common.plot_locale import (
    cjk_font_properties,
    method_label as _method_label,
    metric_axis_label,
    setup_matplotlib_chinese,
)


def plot_scatter_dino_lpips(
    series: Dict[str, Dict[str, Tuple[float, float]]],
    out_path: Path,
):
    import matplotlib.pyplot as plt

    setup_matplotlib_chinese()
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(series))))
    for i, (name, pts) in enumerate(series.items()):
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        ax.scatter(
            xs,
            ys,
            s=64,
            alpha=0.78,
            label=_method_label(name),
            color=colors[i % 10],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel("DINO-I（与目标图余弦相似度，越高越好）", fontsize=12)
    ax.set_ylabel("相对目标图的 LPIPS（越低越好）", fontsize=12)
    ax.set_title("各用户均值：DINO-I 与 LPIPS（语义 vs 感知匹配）", fontsize=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.92)
    ax.grid(True, alpha=0.35)
    ax.text(
        0.02,
        0.98,
        "理想区域：右下方\n（高 DINO-I，低 LPIPS）",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# 若某方法在多数轴上归一化后接近 0，极坐标上会塌成「一根线」；对半径做线性抬底，不改变各方法在每条轴上的排序。
_RADAR_RADIUS_FLOOR = 0.22


def _radar_radius_for_plot(r: float) -> float:
    if r != r:
        return _RADAR_RADIUS_FLOOR
    r = float(max(0.0, min(1.0, r)))
    return _RADAR_RADIUS_FLOOR + (1.0 - _RADAR_RADIUS_FLOOR) * r


def plot_radar(norm: Dict[str, Dict[str, float]], axes_order: List[str], out_path: Path):
    import matplotlib.pyplot as plt

    setup_matplotlib_chinese()
    labels = [metric_axis_label(k) for k in axes_order]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.0), subplot_kw=dict(polar=True))
    for name, vals in norm.items():
        row = [vals.get(k, float("nan")) for k in axes_order]
        row = [_radar_radius_for_plot(x) if x == x else _RADAR_RADIUS_FLOOR for x in row]
        row += row[:1]
        ax.plot(angles, row, linewidth=2.2, label=_method_label(name))
        ax.fill(angles, row, alpha=0.09)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_title(
        f"多指标雷达图（方法间 min–max；LPIPS/Gram 已反向；"
        f"半径线性映射至 [{_RADAR_RADIUS_FLOOR:.2f}, 1]）",
        fontsize=10,
        pad=16,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=9, framealpha=0.92)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _thumb(path: Optional[str], size: int) -> "Image.Image":
    from PIL import Image

    if not path or not Path(path).is_file():
        im = Image.new("RGB", (size, size), color=(200, 200, 200))
        return im
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    w, h = im.size
    canvas.paste(im, ((size - w) // 2, (size - h) // 2))
    return canvas


def _default_data_root(test_json: Path) -> Path:
    """test.json 通常在 processed_dataset/ 下，图像路径相对于 userpref_v1 根目录。"""
    parent = test_json.parent
    if parent.name == "processed_dataset":
        return parent.parent
    return parent


def _uid_to_first_sample_index(test: List[dict]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for idx, s in enumerate(test):
        uid = str(s.get("worker_id") or "").strip()
        if uid and uid not in m:
            m[uid] = int(idx)
    return m


def _history_series(sample: dict) -> Tuple[List[int], List[int], List[str]]:
    hist = sample.get("history_items_info") or []
    prefs: List[int] = []
    quals: List[int] = []
    rels: List[str] = []
    for h in hist:
        try:
            prefs.append(int(h.get("preference_score")))
        except Exception:
            prefs.append(0)
        try:
            quals.append(int(h.get("quality_score")))
        except Exception:
            quals.append(0)
        rels.append(str(h.get("image_path") or ""))
    return prefs, quals, rels


def _resolve_under_root(data_root: Path, rel: str) -> Optional[str]:
    if not rel:
        return None
    p = (data_root / rel).resolve()
    return str(p) if p.is_file() else None


def _plot_history_bars_on_ax(ax, sample: dict, *, title: str = "", legend: bool = False, fontsize: int = 10):
    """单面板：历史每条 preference / quality 并列柱。"""
    prefs, quals, _ = _history_series(sample)
    if not prefs:
        ax.text(0.5, 0.5, "无历史记录", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
        return
    n = len(prefs)
    x = np.arange(n)
    ax.bar(x - 0.18, prefs, width=0.36, color="steelblue", alpha=0.88, label="偏好分")
    ax.bar(x + 0.18, quals, width=0.36, color="darkorange", alpha=0.78, label="质量分")
    ax.set_ylim(0.5, 5.5)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks([])
    ax.set_ylabel("分数（1–5）", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize - 1)
    ax.grid(True, axis="y", alpha=0.28)
    if legend:
        ax.legend(loc="upper right", fontsize=fontsize - 2, framealpha=0.92)
    if title:
        ax.set_title(title, fontsize=fontsize + 1)


def plot_history_score_panels(
    *,
    test_json: Path,
    case_users: List[str],
    out_path: Path,
):
    """独立附图：每用户一子图，仅画历史分数条，便于与主案例图分开排版。"""
    import matplotlib.pyplot as plt

    setup_matplotlib_chinese()
    test = _load(test_json)
    if not isinstance(test, list):
        return
    uid_m = _uid_to_first_sample_index(test)
    users = [u.strip() for u in case_users if u.strip() and u.strip() in uid_m]
    n = len(users)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5), squeeze=False)
    for i, u in enumerate(users):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sample = test[uid_m[u]]
        _plot_history_bars_on_ax(ax, sample, title=f"用户 {u}", legend=(i == 0))
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_axis_off()
    fig.suptitle("用户历史：偏好分与质量分（训练池每条交互，1–5 分）", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compose_history_thumb_grid(
    abs_paths: List[str],
    *,
    cell: int,
    ncols: int,
    gap: int = 4,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> "Image.Image":
    """等方格、统一间距；末行不足列数时用白格填满，保证矩形整齐。"""
    from PIL import Image

    paths = [p for p in abs_paths if p and Path(p).is_file()]
    n = len(paths)
    nrows = max(1, (n + ncols - 1) // ncols) if n > 0 else 1
    w = ncols * cell + (ncols - 1) * gap
    h = nrows * cell + (nrows - 1) * gap
    canvas = Image.new("RGB", (w, h), bg)
    blank = Image.new("RGB", (cell, cell), bg)
    total_slots = nrows * ncols
    for idx in range(total_slots):
        r, c = divmod(idx, ncols)
        x = c * (cell + gap)
        y = r * (cell + gap)
        if idx < n:
            im = _thumb(paths[idx], cell)
            canvas.paste(im, (x, y))
        else:
            canvas.paste(blank, (x, y))
    return canvas


def _empty_bin_placeholder(
    *,
    ncols: int,
    cell: int,
    gap: int,
    bg: Tuple[int, int, int] = (248, 248, 248),
) -> "Image.Image":
    """空桶：与单行行高一致，便于与对侧对齐。"""
    from PIL import Image, ImageDraw, ImageFont

    w = ncols * cell + (ncols - 1) * gap
    h = cell
    im = Image.new("RGB", (w, h), bg)
    dr = ImageDraw.Draw(im)
    msg = "（该区间无图像）"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = dr.textbbox((0, 0), msg, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = dr.textsize(msg, font=font)
    dr.text(((w - tw) // 2, (h - th) // 2), msg, fill=(120, 120, 120), font=font)
    return im


def _pad_to_size(
    im: "Image.Image",
    target_w: int,
    target_h: int,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> "Image.Image":
    """左上对齐铺到固定画布；若超出则裁左上角，保证同一行左右栏像素尺寸一致。"""
    from PIL import Image

    out = Image.new("RGB", (target_w, target_h), bg)
    w, h = im.size
    if w <= target_w and h <= target_h:
        out.paste(im, (0, 0))
        return out
    cropped = im.crop((0, 0, min(w, target_w), min(h, target_h)))
    out.paste(cropped, (0, 0))
    return out


def plot_history_pref_high_low_moodboards(
    *,
    test_json: Path,
    data_root: Path,
    case_users: List[str],
    out_path: Path,
    high_min: int = 4,
    low_max: int = 2,
    grid_ncols: int = 4,
    cell: int = 100,
):
    """
    每用户一行：左列 user id，中/右列为高分 / 低分图墙；首行单独列标题。
    同一行左右像素对齐；按行高度分垂直空间；**尽量大图、少页边**（单幅 3 列 GridSpec + tight 裁边）。
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    test = _load(test_json)
    if not isinstance(test, list):
        return
    uid_m = _uid_to_first_sample_index(test)
    users = [u.strip() for u in case_users if u.strip() and u.strip() in uid_m]
    nu = len(users)
    if nu == 0:
        return

    gap = 3
    setup_matplotlib_chinese()
    cjk = cjk_font_properties()
    title_hi = f"高分图（偏好 ≥ {high_min}）"
    title_lo = f"低分图（偏好 ≤ {low_max}）"

    row_pairs: List[Tuple[object, object, str, int]] = []
    for u in users:
        prefs, _, rels = _history_series(test[uid_m[u]])
        high_abs: List[str] = []
        low_abs: List[str] = []
        for pr, rel in zip(prefs, rels):
            ap = _resolve_under_root(data_root, rel)
            if not ap:
                continue
            if pr >= high_min:
                high_abs.append(ap)
            if pr <= low_max:
                low_abs.append(ap)

        if high_abs:
            im_hi = _compose_history_thumb_grid(high_abs, cell=cell, ncols=grid_ncols, gap=gap)
        else:
            im_hi = _empty_bin_placeholder(ncols=grid_ncols, cell=cell, gap=gap)
        if low_abs:
            im_lo = _compose_history_thumb_grid(low_abs, cell=cell, ncols=grid_ncols, gap=gap)
        else:
            im_lo = _empty_bin_placeholder(ncols=grid_ncols, cell=cell, gap=gap)

        wr = max(im_hi.size[0], im_lo.size[0])
        hr = max(im_hi.size[1], im_lo.size[1])
        im_hi = _pad_to_size(im_hi, wr, hr)
        im_lo = _pad_to_size(im_lo, wr, hr)
        row_pairs.append((im_hi, im_lo, u, hr))

    sum_px_h = sum(r[3] for r in row_pairs)
    # 更高英寸 / 更宽画布 → 子图里缩略图被放得更大；tight 只裁外圈白边
    fig_w = 16.8
    fig_h = min(27.0, max(6.2, 0.28 + sum_px_h / 62.0))
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    title_row = 0.14
    hratios = [title_row] + [float(max(r[3], 48)) for r in row_pairs]
    gs = GridSpec(
        nu + 1,
        3,
        figure=fig,
        height_ratios=hratios,
        width_ratios=[0.11, 1.0, 1.0],
        wspace=0.01,
        hspace=0.065,
        left=0.008,
        right=0.998,
        top=0.93,
        bottom=0.018,
    )

    ax_h0 = fig.add_subplot(gs[0, 0])
    ax_h0.set_axis_off()
    ax_h1 = fig.add_subplot(gs[0, 1])
    ax_h1.set_axis_off()
    tkw = {"fontproperties": cjk, "fontsize": 12, "ha": "center", "va": "center"} if cjk else {"fontsize": 12, "ha": "center", "va": "center"}
    ax_h1.text(0.5, 0.5, title_hi, transform=ax_h1.transAxes, **tkw)
    ax_h2 = fig.add_subplot(gs[0, 2])
    ax_h2.set_axis_off()
    ax_h2.text(0.5, 0.5, title_lo, transform=ax_h2.transAxes, **tkw)

    for row, (im_hi, im_lo, u, _hr) in enumerate(row_pairs):
        r = row + 1
        ax_u = fig.add_subplot(gs[r, 0])
        ax_u.set_axis_off()
        ukw: Dict[str, object] = {
            "transform": ax_u.transAxes,
            "ha": "right",
            "va": "center",
            "fontsize": 17,
            "fontweight": "bold",
            "linespacing": 1.15,
        }
        if cjk:
            ukw["fontproperties"] = cjk
        ax_u.text(0.98, 0.5, f"用户\n{u}", **ukw)

        ax1 = fig.add_subplot(gs[r, 1])
        ax1.imshow(np.asarray(im_hi), interpolation="nearest")
        ax1.set_axis_off()

        ax2 = fig.add_subplot(gs[r, 2])
        ax2.imshow(np.asarray(im_lo), interpolation="nearest")
        ax2.set_axis_off()

    st = "UserPref 历史：高/低偏好图像对照（训练池）"
    st_kw: Dict[str, object] = {"fontsize": 13, "y": 0.995}
    if cjk:
        st_kw["fontproperties"] = cjk
    fig.suptitle(st, **st_kw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_case_grid(
    *,
    test_json: Path,
    eval_by_method: Dict[str, dict],
    method_order: List[str],
    case_users: List[str],
    out_path: Path,
    thumb: int = 192,
):
    """每行一用户：Target + 各方法（仅此内容，历史信息见独立附图）。"""
    import matplotlib.pyplot as plt

    setup_matplotlib_chinese()
    test = _load(test_json)
    uid_to_first_idx = _uid_to_first_sample_index(test)

    col_labels = ["目标图"] + [_method_label(m) for m in method_order]
    rows_imgs: List[List] = []
    row_user_labels: List[str] = []
    for u in case_users:
        sid = uid_to_first_idx.get(u.strip())
        if sid is None:
            continue
        row: List = []
        tgt = None
        for data in eval_by_method.values():
            for ps in data.get("per_sample", []):
                if int(ps.get("sample_idx", -1)) != sid:
                    continue
                tgt = ps.get("target_path")
                break
            if tgt:
                break
        row.append(_thumb(str(tgt) if tgt else None, thumb))
        for mk in method_order:
            gen_p = None
            data = eval_by_method.get(mk, {})
            for ps in data.get("per_sample", []):
                if int(ps.get("sample_idx", -1)) == sid:
                    gen_p = ps.get("gen_path")
                    break
            row.append(_thumb(str(gen_p) if gen_p else None, thumb))
        rows_imgs.append(row)
        row_user_labels.append(f"用户 {u.strip()}")

    if not rows_imgs:
        return

    nrows = len(rows_imgs)
    ncols = len(rows_imgs[0])
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.15, nrows * 2.25 + 0.9),
        squeeze=False,
    )
    for ri in range(nrows):
        for ci in range(ncols):
            ax = axes[ri][ci]
            im = rows_imgs[ri][ci]
            ax.imshow(np.asarray(im))
            ax.axis("off")
            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=10, pad=6)
            if ci == 0:
                ax.set_ylabel(row_user_labels[ri], fontsize=10, rotation=90, va="center", labelpad=12)
    fig.suptitle("定性对比（目标图与各方法生成结果）", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--eval_specs",
        nargs="+",
        required=True,
        help='e.g. textual_inversion=path/to/ti_eval.json dreambooth=... ip_adapter=... pmg=...',
    )
    ap.add_argument(
        "--case_users",
        type=str,
        default="2,11,25",
        help="主案例图 fig_4_8_case_grid：worker_id 列表（建议 3 行以内便于阅读）",
    )
    ap.add_argument(
        "--case_users_history",
        type=str,
        default="2,11,25,78,82,100",
        help="历史 preference/quality 柱形图附图的 worker_id",
    )
    ap.add_argument(
        "--case_users_history_moodboard",
        type=str,
        default="2,11,25",
        help="高/低偏好历史图墙（≥4 vs ≤2）的 worker_id，宜 2–3 人、每行左右两栏多缩略图",
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default="",
        help="userpref 数据根目录（含 images/）；默认同目录 test.json 的上一级 userpref_v1",
    )
    ap.add_argument(
        "--custom_diffusion_eval",
        type=str,
        default="",
        help="Optional path to Custom Diffusion run_eval_benchmark JSON (若无可留空，案例图该列为灰底)",
    )
    args = ap.parse_args()

    test_json = Path(args.test_json).expanduser().resolve()
    dr = str(args.data_root).strip()
    data_root = Path(dr).expanduser().resolve() if dr else _default_data_root(test_json)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_paths = _parse_eval_specs(list(args.eval_specs))
    data_by_name: Dict[str, dict] = {k: _load(p) for k, p in eval_paths.items()}

    # ---- 3.2 表（固定文字，数值另表）----
    dim_rows = [
        ("语义保真", "CLIP Score (I-T)", "生成图与 prompt 的文本-图像对齐度（本实现为 OpenAI CLIP ViT-B/32 的 cos-sim）"),
        ("主体保真", "DINO-Sim / LPIPS", "与目标图像在语义（DINO）与感知（LPIPS）上的接近程度"),
        ("风格保真", "Gram Matrix Distance", "VGG-19 多层 Gram 矩阵 Frobenius 差之和；越小越接近目标风格统计"),
        ("通用质量", "HPS v2 / FID", "人类偏好分与生成-目标分布的 FID（配对池；小样本需谨慎解释）"),
    ]
    _write_md_table(out_dir / "table_3_2_dimensions.md", dim_rows)

    # ---- 各方法 summary 表 + CLIP 总体均值 ----
    lines = ["# 各方法评测汇总（来自 eval JSON）", ""]
    clip_means = []
    clip_i_means: List[Tuple[str, Optional[float], Optional[int]]] = []
    fid_rows = []
    gram_means = []
    for name, data in data_by_name.items():
        s = data.get("summary", {})
        clip_t = s.get("clip_t", {})
        clip_i = s.get("clip_i", {})
        clip_means.append((name, clip_t.get("mean"), clip_t.get("n")))
        clip_i_means.append((name, clip_i.get("mean"), clip_i.get("n")))
        fid = data.get("fid_vs_target_distribution")
        hps = s.get("hpsv2", {})
        fid_rows.append((name, hps.get("mean"), hps.get("n"), fid))
        g = s.get("gram_target", {})
        gram_means.append((name, g.get("mean"), g.get("n")))
        lines.append(f"## {name}")
        lines.append("```json")
        lines.append(json.dumps(s, ensure_ascii=False, indent=2))
        lines.append("```\n")
    (out_dir / "per_method_summary.md").write_text("\n".join(lines), encoding="utf-8")

    with (out_dir / "clip_it_overall_mean.txt").open("w", encoding="utf-8") as f:
        f.write("method\tclip_t_mean\tn\n")
        for name, mean, n in clip_means:
            f.write(f"{name}\t{mean}\t{n}\n")

    with (out_dir / "clip_i_t_overall_mean.txt").open("w", encoding="utf-8") as f:
        f.write("method\tclip_i_mean\tclip_t_mean\tn\n")
        for (name, mi, ni), (_, mt, nt) in zip(clip_i_means, clip_means):
            n_use = ni if ni is not None else nt
            f.write(f"{name}\t{mi}\t{mt}\t{n_use}\n")

    with (out_dir / "table_clip_overall.md").open("w", encoding="utf-8") as f:
        f.write("| 方法 | CLIP-I mean（与目标图） | CLIP-T mean（与 caption） | n |\n")
        f.write("|------|--------------------------|-----------------------------|---|\n")
        for (name, mi, ni), (_, mt, nt) in zip(clip_i_means, clip_means):
            n_use = ni if ni is not None else nt
            f.write(f"| {_method_label(name)} | {mi} | {mt} | {n_use} |\n")

    with (out_dir / "table_4_8_hps_fid.md").open("w", encoding="utf-8") as f:
        f.write("| 方法 | HPS v2 (mean) | n | FID (paired gen vs target) |\n")
        f.write("|------|---------------|---|-----------------------------|\n")
        for name, hmean, hn, fid in fid_rows:
            f.write(f"| {name} | {hmean} | {hn} | {fid} |\n")

    with (out_dir / "gram_overall.md").open("w", encoding="utf-8") as f:
        f.write("| 方法 | Gram distance (mean) | n |\n")
        f.write("|------|------------------------|---|\n")
        for name, gm, gn in gram_means:
            f.write(f"| {name} | {gm} | {gn} |\n")

    # ---- DINO vs LPIPS scatter (per user mean) ----
    scatter_series: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, data in data_by_name.items():
        bu_d = _per_user_means(data.get("per_sample", []), "dino_i")
        bu_l = _per_user_means(data.get("per_sample", []), "lpips_target")
        pts: Dict[str, Tuple[float, float]] = {}
        users = sorted(set(bu_d.keys()) & set(bu_l.keys()))
        for u in users:
            d = _mean(bu_d[u])
            lp = _mean(bu_l[u])
            if d is None or lp is None:
                continue
            pts[u] = (d, lp)
        scatter_series[name] = pts
    plot_scatter_dino_lpips(scatter_series, out_dir / "fig_4_5_dino_vs_lpips.png")

    # ---- Radar ----
    axes_order = ["clip_t", "dino_i", "lpips_target", "gram_target", "hpsv2"]
    higher_better = {"clip_t": True, "dino_i": True, "lpips_target": False, "gram_target": False, "hpsv2": True}
    vals: Dict[str, Dict[str, float]] = {}
    for name, data in data_by_name.items():
        s = data.get("summary", {})
        vals[name] = {}
        for ax in axes_order:
            item = s.get(ax) if isinstance(s.get(ax), dict) else None
            m = item.get("mean") if item else None
            vals[name][ax] = float(m) if m is not None and m == m else float("nan")
    # 若某轴全 NaN（如未跑 HPS），雷达图跳过该轴
    active_axes = [ax for ax in axes_order if any(vals[m][ax] == vals[m][ax] for m in vals)]
    if len(active_axes) >= 3:
        vals_f = {m: {ax: vals[m][ax] for ax in active_axes} for m in vals}
        hb = {ax: higher_better[ax] for ax in active_axes}
        norm = _norm_radar(vals_f, higher_better=hb)
        for m in norm:
            for ax in norm[m]:
                if norm[m][ax] != norm[m][ax]:
                    norm[m][ax] = 0.0
        plot_radar(norm, active_axes, out_dir / "fig_4_7_radar.png")

    # ---- 图 4-8：主图仅 Target + 方法；历史信息拆成两张附图 ----
    case_users = [x.strip() for x in str(args.case_users).split(",") if x.strip()]
    case_hist = [x.strip() for x in str(args.case_users_history).split(",") if x.strip()]
    case_mood = [x.strip() for x in str(args.case_users_history_moodboard).split(",") if x.strip()]
    method_order = [k for k in ["textual_inversion", "dreambooth", "ip_adapter", "pmg"] if k in data_by_name]
    cd_path = str(args.custom_diffusion_eval).strip()
    if cd_path and Path(cd_path).is_file():
        data_by_name["custom_diffusion"] = _load(Path(cd_path))
        # 案例图列顺序：Target, TI, DB, CD, IP, PMG
        method_order = [
            k
            for k in ["textual_inversion", "dreambooth", "custom_diffusion", "ip_adapter", "pmg"]
            if k in data_by_name
        ]
    plot_case_grid(
        test_json=test_json,
        eval_by_method=data_by_name,
        method_order=method_order,
        case_users=case_users,
        out_path=out_dir / "fig_4_8_case_grid.png",
        thumb=192,
    )
    plot_history_score_panels(
        test_json=test_json,
        case_users=case_hist,
        out_path=out_dir / "fig_4_8_history_scores.png",
    )
    plot_history_pref_high_low_moodboards(
        test_json=test_json,
        data_root=data_root,
        case_users=case_mood,
        out_path=out_dir / "fig_4_8_history_pref_high_low.png",
    )

    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
