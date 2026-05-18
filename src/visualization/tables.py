"""Result tables and LaTeX export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT

logger = get_logger(__name__)


def build_summary_table() -> pd.DataFrame:
    cfg = load_config()
    metrics_dir = PROJECT_ROOT / cfg.outputs.metrics
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(data)
    return pd.DataFrame(rows)


def export_latex(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, float_format="%.4f")
    out_path.write_text(latex, encoding="utf-8")
    logger.info("LaTeX table saved to %s", out_path)
    return out_path


def export_results(config=None) -> tuple[Path, Path]:
    cfg = config or load_config()
    df = build_summary_table()
    csv_path = PROJECT_ROOT / cfg.outputs.metrics / "summary.csv"
    tex_path = PROJECT_ROOT / cfg.outputs.reports / "results_table.tex"
    df.to_csv(csv_path, index=False)
    export_latex(df, tex_path)
    return csv_path, tex_path
