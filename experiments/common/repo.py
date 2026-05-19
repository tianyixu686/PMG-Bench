from pathlib import Path


def repo_root() -> Path:
    """PMG-Bench repository root (parent of `experiments/`)."""
    return Path(__file__).resolve().parents[2]
