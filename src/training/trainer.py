"""Generic Seq2Seq training wrapper."""

from __future__ import annotations

import inspect
from pathlib import Path

import mlflow
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, ModelSpec, load_config
from src.common.logging import get_logger
from src.common.paths import PROJECT_ROOT, mlflow_tracking_uri
from src.common.seed import set_seed
from src.data.datamodule import load_tokenized_splits
from src.training.callbacks import CheckpointStateCallback, MetricsLoggerCallback
from src.training.memory_probe import probe_memory
from src.training.schedules import get_scheduler_type

logger = get_logger(__name__)


def build_training_args(
    model_spec: ModelSpec,
    output_dir: Path,
    config: AppConfig,
) -> Seq2SeqTrainingArguments:
    train_cfg = config.training
    optim = model_spec.optim or "adamw_torch"
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.epochs,
        per_device_train_batch_size=model_spec.train_batch_size,
        per_device_eval_batch_size=model_spec.eval_batch_size,
        gradient_accumulation_steps=model_spec.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_steps=train_cfg.warmup_steps,
        lr_scheduler_type=get_scheduler_type(train_cfg.scheduler),
        label_smoothing_factor=train_cfg.label_smoothing,
        fp16=train_cfg.fp16 and torch.cuda.is_available(),
        fp16_full_eval=False,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to=[],
        optim=optim,
        seed=config.experiment.seed,
    )


def train_model(
    model,
    tokenizer,
    model_spec: ModelSpec,
    config: AppConfig | None = None,
    force: bool = False,
) -> Path:
    cfg = config or load_config()
    set_seed(cfg.experiment.seed)
    stage = f"train_{model_spec.key}"

    output_dir = PROJECT_ROOT / cfg.outputs.checkpoints / model_spec.key
    if not force and is_done(stage, cfg):
        logger.info("Training already complete for %s", model_spec.key)
        return output_dir

    splits = load_tokenized_splits(model_spec, cfg)
    args = build_training_args(model_spec, output_dir, cfg)

    mlflow.set_tracking_uri(mlflow_tracking_uri(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    callbacks = [
        CheckpointStateCallback(output_dir, model_spec.key),
        MetricsLoggerCallback(PROJECT_ROOT / cfg.outputs.logs / f"{model_spec.key}_metrics.jsonl"),
    ]
    if cfg.training.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience))

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    sample = {k: torch.tensor(v[:2]) for k, v in splits["train"][:2].items()}
    if torch.cuda.is_available() and not probe_memory(model, sample):
        raise RuntimeError(
            f"Memory probe failed for {model_spec.key}. "
            "Reduce batch size or enable gradient checkpointing."
        )

    tokenizer_kw = (
        "processing_class"
        if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        data_collator=collator,
        callbacks=callbacks,
        **{tokenizer_kw: tokenizer},
    )

    resume = None
    if output_dir.exists() and any(output_dir.glob("checkpoint-*")):
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        resume = str(checkpoints[-1])
        logger.info("Resuming from %s", resume)

    with mlflow.start_run(run_name=model_spec.key):
        mlflow.log_params(
            {
                "model_name": model_spec.name,
                "use_lora": model_spec.use_lora,
                "learning_rate": cfg.training.learning_rate,
                "epochs": cfg.training.epochs,
                "batch_size": model_spec.train_batch_size,
                "grad_accum": model_spec.gradient_accumulation_steps,
            }
        )
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))

        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})

    mark_done(stage, {"model": model_spec.key}, cfg)
    logger.info("Training complete for %s", model_spec.key)
    return output_dir
