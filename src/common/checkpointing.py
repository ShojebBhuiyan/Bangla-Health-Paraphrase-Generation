"""Stage-based checkpoint sentinels for resumable pipelines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import AppConfig, load_config
from src.common.paths import OUTPUTS_DIR


def _stages_dir(config: AppConfig | None = None) -> Path:
    cfg = config or load_config()
    stages = Path(cfg.outputs.stages)
    if not stages.is_absolute():
        from src.common.paths import PROJECT_ROOT

        stages = PROJECT_ROOT / stages
    stages.mkdir(parents=True, exist_ok=True)
    return stages


def stage_path(stage_name: str, config: AppConfig | None = None) -> Path:
    safe_name = stage_name.replace("/", "_").replace("\\", "_")
    return _stages_dir(config) / f"{safe_name}.ok"


def is_done(
    stage_name: str,
    config: AppConfig | None = None,
    *,
    artifacts: list[Path] | None = None,
) -> bool:
    if not stage_path(stage_name, config).exists():
        return False
    if artifacts is None:
        return True
    return all(p.exists() for p in artifacts)


def mark_done(stage_name: str, metadata: dict | None = None, config: AppConfig | None = None) -> Path:
    path = stage_path(stage_name, config)
    payload = {
        "stage": stage_name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def clear_stage(stage_name: str, config: AppConfig | None = None) -> None:
    path = stage_path(stage_name, config)
    if path.exists():
        path.unlink()


def checkpoint_exists(name: str, config: AppConfig | None = None) -> bool:
    return is_done(name, config)
