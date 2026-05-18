"""Training callbacks for MLflow and checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback

from src.common.logging import get_logger

logger = get_logger(__name__)


class CheckpointStateCallback(TrainerCallback):
    def __init__(self, checkpoint_dir: Path, model_key: str):
        self.checkpoint_dir = checkpoint_dir
        self.model_key = model_key
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        state_path = self.checkpoint_dir / "state.json"
        payload = {
            "model_key": self.model_key,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
        }
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Checkpoint saved at step %d", state.global_step)


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")
