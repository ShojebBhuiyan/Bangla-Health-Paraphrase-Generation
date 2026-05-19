"""Project path resolution."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def parquet_split_ready(path: Path) -> bool:
    """True when path is a single parquet file or a Spark output folder with parts."""
    if path.is_file():
        return path.suffix == ".parquet"
    if path.is_dir():
        return any(path.glob("*.parquet"))
    return False


def parquet_data_files(path: Path) -> str:
    """Path/glob string for ``datasets.load_dataset(..., data_files=...)``."""
    if path.is_dir():
        return str(path / "*.parquet")
    return str(path)


def mlflow_tracking_uri(tracking_uri: str, root: Path = PROJECT_ROOT) -> str:
    """Resolve a local MLflow path to a ``file://`` URI (required on Windows)."""
    uri = tracking_uri.strip()
    if "://" in uri:
        return uri
    return (root / uri).resolve().as_uri()


def ensure_dirs() -> None:
    for sub in (
        "raw",
        "interim",
        "processed",
        "augmented",
    ):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    for sub in (
        "checkpoints",
        "logs",
        "metrics",
        "figures",
        "reports",
        "mlruns",
        ".stages",
        "spark_checkpoints",
    ):
        (OUTPUTS_DIR / sub).mkdir(parents=True, exist_ok=True)
