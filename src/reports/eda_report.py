"""Render EDA HTML report from Spark stats."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT
from src.spark.eda import compute_eda_stats

logger = get_logger(__name__)


def render_eda_report(config: AppConfig | None = None, force: bool = False) -> Path:
    cfg = config or load_config()
    if not force and is_done("eda_report", cfg):
        out = PROJECT_ROOT / cfg.outputs.reports / "eda_report.html"
        logger.info("EDA report already exists: %s", out)
        return out

    stats = compute_eda_stats(cfg)
    template_dir = PROJECT_ROOT / "configs" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("eda_report.html.j2")

    figures = [
        {"title": "Source Length Distribution", "path": "../figures/eda/src_len_dist.png"},
        {"title": "Target Length Distribution", "path": "../figures/eda/tgt_len_dist.png"},
        {"title": "Length Ratio Distribution", "path": "../figures/eda/len_ratio_dist.png"},
        {"title": "Word Overlap Distribution", "path": "../figures/eda/word_overlap_dist.png"},
    ]

    html = template.render(
        experiment_name=cfg.experiment.name,
        figures=figures,
        **stats,
    )

    out_path = PROJECT_ROOT / cfg.outputs.reports / "eda_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    mark_done("eda_report", {"path": str(out_path)}, cfg)
    logger.info("EDA report saved to %s", out_path)
    return out_path
