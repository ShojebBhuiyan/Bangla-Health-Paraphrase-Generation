"""Final research report generator."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.common.config import load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT

logger = get_logger(__name__)


def render_final_report() -> Path:
    cfg = load_config()
    metrics_dir = PROJECT_ROOT / cfg.outputs.metrics
    metrics = []
    for path in sorted(metrics_dir.glob("*.json")):
        if path.stem == "summary":
            continue
        metrics.append(json.loads(path.read_text(encoding="utf-8")))

    template_dir = PROJECT_ROOT / "configs" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("final_report.html.j2")

    html = template.render(
        experiment=cfg.experiment.name,
        config_yaml=yaml.dump(cfg.model_dump(), default_flow_style=False),
        metrics=metrics,
        eda_link="eda_report.html",
        results_tex="../reports/results_table.tex",
    )

    out = PROJECT_ROOT / cfg.outputs.reports / "final_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("Final report saved to %s", out)
    return out
