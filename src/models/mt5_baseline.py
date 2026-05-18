"""mT5-small full fine-tuning baseline."""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common.config import AppConfig, load_config
from src.training.trainer import train_model


def build_mt5_baseline(config: AppConfig | None = None):
    cfg = config or load_config()
    spec = next(s for s in cfg.all_model_specs() if s.key == "mt5_baseline")
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec.name)
    return model, tokenizer, spec


def run_mt5_baseline(config: AppConfig | None = None, force: bool = False):
    model, tokenizer, spec = build_mt5_baseline(config)
    return train_model(model, tokenizer, spec, config, force=force)
