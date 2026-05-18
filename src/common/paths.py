"""Project path resolution."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


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
