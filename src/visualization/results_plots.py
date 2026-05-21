"""Generate comparison and ablation plots from metrics."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT, mlflow_tracking_uri
from src.visualization.plots import (
    plot_bar_comparison,
    plot_combined_metric_comparison,
    plot_training_curves,
)
from src.visualization.tables import export_results

logger = get_logger(__name__)


def plot_metric_comparisons() -> None:
    cfg = load_config()
    metrics_dir = PROJECT_ROOT / cfg.outputs.metrics
    figures_dir = PROJECT_ROOT / cfg.outputs.figures / "comparison"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in metrics_dir.glob("*.json"):
        if path.stem in ("summary",):
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    if not rows:
        logger.warning("No metrics found for plotting")
        return

    df = pd.DataFrame(rows)
    labels = df["model_key"].tolist()
    metric_cols = ["BLEU", "ROUGE-L", "BERTScore", "Distinct-1", "Distinct-2", "semsim_mpnet", "semsim_bnsbert"]

    available = [m for m in metric_cols if m in df.columns]
    combined: dict[str, list[float]] = {}
    for metric in available:
        combined[metric] = df[metric].tolist()
        plot_bar_comparison(
            labels,
            {metric: df[metric].tolist()},
            f"{metric} Comparison",
            metric,
            figures_dir / f"{metric.lower()}_comparison",
        )

    if combined:
        plot_combined_metric_comparison(
            labels,
            combined,
            "Model Comparison — All Metrics",
            figures_dir / "all_metrics_comparison",
        )


def plot_mlflow_loss_curves() -> None:
    cfg = load_config()
    figures_dir = PROJECT_ROOT / cfg.outputs.figures / "training"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri(cfg.mlflow.tracking_uri))
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if not experiment:
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        name = run.get("tags.mlflow.runName", run_id)
        history = mlflow.tracking.MlflowClient().get_metric_history(run_id, "loss")
        if not history:
            continue
        steps = [m.step for m in history]
        values = [m.value for m in history]
        plot_training_curves(steps, values, None, f"Training Loss - {name}", figures_dir / f"loss_{name}")


def generate_all_plots() -> None:
    export_results()
    plot_metric_comparisons()
    try:
        plot_mlflow_loss_curves()
    except Exception as exc:
        logger.warning("Could not plot MLflow curves: %s", exc)
