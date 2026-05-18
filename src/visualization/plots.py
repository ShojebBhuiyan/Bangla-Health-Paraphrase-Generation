"""Publication-quality plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

DPI = 300
STYLE = {"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11}


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, base_path: Path) -> tuple[Path, Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_histogram(values: list[float], title: str, xlabel: str, out: Path) -> tuple[Path, Path]:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50, color="#2E86AB", edgecolor="white", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    return save_figure(fig, out)


def plot_bar_comparison(
    labels: list[str],
    values: dict[str, list[float]],
    title: str,
    ylabel: str,
    out: Path,
) -> tuple[Path, Path]:
    setup_style()
    metrics = list(values.keys())
    x = range(len(labels))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2) * width + width / 2
        ax.bar([xi + offset for xi in x], values[metric], width=width, label=metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    return save_figure(fig, out)


def plot_training_curves(
    steps: list[int],
    train_loss: list[float],
    val_loss: list[float] | None,
    title: str,
    out: Path,
) -> tuple[Path, Path]:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_loss, label="Train loss", color="#2E86AB")
    if val_loss:
        ax.plot(steps[: len(val_loss)], val_loss, label="Val loss", color="#A23B72")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    return save_figure(fig, out)
