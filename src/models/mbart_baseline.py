"""mBART-large-50 full fine-tuning with 8-bit Adam."""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.training.trainer import train_model

logger = get_logger(__name__)


def build_mbart_baseline(config: AppConfig | None = None):
    cfg = config or load_config()
    spec = next(s for s in cfg.all_model_specs() if s.key == "mbart_baseline")
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec.name)

    if spec.optim == "adamw_bnb_8bit":
        try:
            import bitsandbytes  # noqa: F401

            logger.info("bitsandbytes available for 8-bit Adam")
        except ImportError:
            logger.warning(
                "bitsandbytes not found. Install with: pip install bitsandbytes "
                "or pip install bitsandbytes-windows-webui on Windows. "
                "Falling back to paged_adamw_32bit."
            )
            spec.optim = "paged_adamw_32bit"

    return model, tokenizer, spec


def run_mbart_baseline(config: AppConfig | None = None, force: bool = False):
    model, tokenizer, spec = build_mbart_baseline(config)
    return train_model(model, tokenizer, spec, config, force=force)
