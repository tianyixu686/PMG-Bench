import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float], int]:
    if not values:
        return None, None, 0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(m), float(math.sqrt(var)), int(len(values))


def _extract_from_our_eval(obj: dict) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """
    tools/eval_userpref_outputs.py output:
      { summary: {metric: {mean,std,n}} }
    """
    summary = obj.get("summary")
    if not isinstance(summary, dict):
        return None
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k, v in summary.items():
        if not isinstance(v, dict):
            continue
        out[k] = {
            "mean": _safe_float(v.get("mean")),
            "std": _safe_float(v.get("std")),
            "n": int(v.get("n") or 0),
        }
    return out


def _extract_from_dreambooth_eval(obj: dict) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """
    method/dreambooth/tasks/eval_userpref.py output:
      { average_metrics: {metric: mean}, per_sample_metrics: [...] }
    """
    avg = obj.get("average_metrics")
    if not isinstance(avg, dict):
        return None
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k, mean in avg.items():
        out[k] = {"mean": _safe_float(mean), "std": None, "n": int(obj.get("evaluated_samples") or 0)}
    return out


def _extract_from_ip_adapter_results(obj: dict) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """
    method/ip-adater/personalized_generation_userpref.py output:
      { results: [ { metrics: {...} } ], ... }
    """
    rows = obj.get("results")
    if not isinstance(rows, list):
        return None
    metric_values: Dict[str, List[float]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        m = r.get("metrics")
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            metric_values.setdefault(k, []).append(fv)

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k, vals in metric_values.items():
        mean, std, n = _mean_std(vals)
        out[k] = {"mean": mean, "std": std, "n": n}
    return out


def extract_summary(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    obj = _read_json(path)
    if isinstance(obj, dict):
        for fn in (_extract_from_our_eval, _extract_from_ip_adapter_results, _extract_from_dreambooth_eval):
            out = fn(obj)
            if out is not None:
                return out
    raise ValueError(f"Unrecognized metrics JSON schema: {path}")


def plot_bars(
    *,
    out_path: Path,
    title: str,
    run_names: List[str],
    means: List[Optional[float]],
    stds: List[Optional[float]],
    ylabel: str,
):
    import matplotlib.pyplot as plt

    x = list(range(len(run_names)))
    y = [m if m is not None else float("nan") for m in means]
    e = [s if s is not None else 0.0 for s in stds]

    plt.figure(figsize=(max(6, len(run_names) * 1.2), 4))
    plt.bar(x, y, yerr=e, capsize=3)
    plt.xticks(x, run_names, rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Visualize/compare metrics across multiple userpref_v1 runs")
    p.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help='Run specs: "name=/abs/or/rel/path/to/metrics.json" (supports our eval json, ip-adapter generation_results.json, dreambooth eval metrics.json)',
    )
    p.add_argument("--output_dir", type=str, required=True, help="Where to write plots and a summary CSV")
    p.add_argument(
        "--metrics",
        type=str,
        default="clip_i,dino_i,clip_t,lpips_target,ssim_target",
        help="Comma-separated metric keys to plot (missing metrics will be skipped)",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_names: List[str] = []
    run_paths: List[Path] = []
    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"Invalid run spec (expect name=path): {spec}")
        name, path_s = spec.split("=", 1)
        run_names.append(name.strip())
        run_paths.append(Path(path_s).expanduser().resolve())

    summaries = [extract_summary(pth) for pth in run_paths]

    metric_keys = [m.strip() for m in str(args.metrics).split(",") if m.strip()]

    # Write a wide CSV (run x metrics)
    csv_path = out_dir / "compare_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["run", "source_path"]
        for mk in metric_keys:
            header += [f"{mk}.mean", f"{mk}.std", f"{mk}.n"]
        writer.writerow(header)

        for name, path, summ in zip(run_names, run_paths, summaries):
            row: List[Any] = [name, str(path)]
            for mk in metric_keys:
                m = summ.get(mk) or {}
                row.append(m.get("mean"))
                row.append(m.get("std"))
                row.append(m.get("n"))
            writer.writerow(row)

    # Plots per-metric
    for mk in metric_keys:
        means: List[Optional[float]] = []
        stds: List[Optional[float]] = []
        present = False
        for summ in summaries:
            d = summ.get(mk)
            if d is None:
                means.append(None)
                stds.append(None)
            else:
                means.append(d.get("mean"))
                stds.append(d.get("std"))
                present = True
        if not present:
            continue
        plot_bars(
            out_path=out_dir / "plots" / f"{mk}.png",
            title=f"{mk} (mean ± std; missing shown as NaN)",
            run_names=run_names,
            means=means,
            stds=stds,
            ylabel=mk,
        )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots under: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()

