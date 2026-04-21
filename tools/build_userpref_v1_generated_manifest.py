import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _load_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_test_json = repo_root / "data" / "userpref_v1" / "processed_dataset" / "test.json"

    p = argparse.ArgumentParser(description="Build generated manifest for userpref_v1 outputs")
    p.add_argument("--test_json", type=str, default=str(default_test_json))
    p.add_argument(
        "--method_output",
        type=str,
        action="append",
        default=[],
        help="Repeatable. Format: method_name=path/to/eval_output_dir (contains sample_0000/gen_0.jpg)",
    )
    p.add_argument(
        "--out_jsonl",
        type=str,
        default=str(repo_root / "outputs" / "userpref_v1_manifests" / "generated_manifest.jsonl"),
    )

    args = p.parse_args()

    test_json = Path(args.test_json)
    test_data: List[dict] = _load_json(test_json)

    method2dir: Dict[str, Path] = {}
    for mo in args.method_output:
        if "=" not in mo:
            raise SystemExit(f"Bad --method_output: {mo}. Expected method=dir")
        method, out_dir = mo.split("=", 1)
        method = method.strip()
        out_dir = out_dir.strip().strip('"')
        if not method or not out_dir:
            raise SystemExit(f"Bad --method_output: {mo}. Expected method=dir")
        method2dir[method] = Path(out_dir)

    if not method2dir:
        raise SystemExit("No --method_output provided")

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for method, out_dir in method2dir.items():
            for idx, sample in enumerate(test_data):
                sample_dir = out_dir / f"sample_{idx:04d}"
                gen0 = sample_dir / "gen_0.jpg"
                meta_path = sample_dir / "meta.json"
                meta = _load_meta(meta_path)

                target_item_info = sample.get("target_item_info", {})
                row = {
                    "dataset": "userpref_v1",
                    "method": method,
                    "sample_idx": int(idx),
                    "user_id": sample.get("user_id") or sample.get("worker_id"),
                    "query_variant": target_item_info.get("query_variant", ""),
                    "prompt": target_item_info.get("caption", ""),
                    "generated_image": _try_relpath(gen0, repo_root),
                    "sample_dir": _try_relpath(sample_dir, repo_root),
                    "meta": meta,
                    "target_item_info": target_item_info,
                    "history_items_info": sample.get("history_items_info", []),
                    "stage2_candidates": sample.get("stage2_candidates", []),
                    "exists": bool(gen0.exists()),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_jsonl}")


if __name__ == "__main__":
    main()
