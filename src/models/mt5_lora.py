"""mT5-small with LoRA fine-tuning."""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.training.trainer import train_model

logger = get_logger(__name__)


def build_mt5_lora(config: AppConfig | None = None):
    cfg = config or load_config()
    spec = cfg.all_model_specs()[0]
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec.name)

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer, spec


def run_mt5_lora(config: AppConfig | None = None, force: bool = False):
    model, tokenizer, spec = build_mt5_lora(config)
    return train_model(model, tokenizer, spec, config, force=force)
