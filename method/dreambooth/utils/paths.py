from pathlib import Path


def repo_root_from_here(file_path: str) -> Path:
    return Path(file_path).resolve().parents[2]


def default_userpref_data_root(repo_root: Path) -> Path:
    return repo_root / "data" / "userpref_v1"


def resolve_userpref_image_path(raw_path: str, *, repo_root: Path, data_root: Path) -> str:
    if not raw_path:
        return ""

    p = Path(str(raw_path))
    if p.is_absolute():
        return str(p)

    candidate1 = (data_root / p).resolve()
    if candidate1.exists() or str(raw_path).replace("\\", "/").startswith("images/"):
        return str(candidate1)

    candidate2 = (repo_root / p).resolve()
    return str(candidate2)
