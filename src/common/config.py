"""YAML configuration loader with Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.common.paths import CONFIG_PATH


class ExperimentConfig(BaseModel):
    name: str
    seed: int


class DatasetConfig(BaseModel):
    source: str
    raw_csv: str
    train_split: float
    validation_split: float
    test_split: float
    dev_subset: int | None = None
    max_input_length: int
    max_target_length: int
    min_tokens: int = 3
    max_tokens: int = 256


class SparkConfig(BaseModel):
    app_name: str
    shuffle_partitions: int
    driver_memory: str
    executor_memory: str
    enable_arrow: bool
    near_dup_jaccard_threshold: float = 0.95
    near_dup_enabled: bool = True


class ModelSpec(BaseModel):
    name: str
    key: str
    use_lora: bool = False
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    optim: str | None = None


class MainModelConfig(BaseModel):
    name: str
    key: str
    use_lora: bool
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4


class ModelsConfig(BaseModel):
    main: MainModelConfig
    baselines: list[ModelSpec]


class LoRAConfig(BaseModel):
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


class TrainingConfig(BaseModel):
    learning_rate: float
    epochs: int
    weight_decay: float
    warmup_steps: int
    label_smoothing: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 2
    scheduler: str = "cosine"
    use_augmented_train: bool = False
    num_beams: int = 4
    no_repeat_ngram_size: int = 3


class AugmentationConfig(BaseModel):
    synonym_prob: float = 0.1
    swap_prob: float = 0.05
    deletion_prob: float = 0.05
    min_tokens_after_aug: int = 3


class EvaluationConfig(BaseModel):
    metrics: list[str]
    bertscore_model: str
    semantic_models: list[str]
    eda_similarity_sample_size: int = 10000


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class OutputsConfig(BaseModel):
    checkpoints: str
    logs: str
    metrics: str
    figures: str
    reports: str
    stages: str
    spark_checkpoints: str


class AppConfig(BaseModel):
    experiment: ExperimentConfig
    dataset: DatasetConfig
    spark: SparkConfig
    models: ModelsConfig
    lora: LoRAConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    outputs: OutputsConfig

    def output_path(self, key: str) -> Path:
        return Path(getattr(self.outputs, key))

    def all_model_specs(self) -> list[ModelSpec]:
        main = ModelSpec(
            name=self.models.main.name,
            key=self.models.main.key,
            use_lora=self.models.main.use_lora,
            train_batch_size=self.models.main.train_batch_size,
            eval_batch_size=self.models.main.eval_batch_size,
            gradient_accumulation_steps=self.models.main.gradient_accumulation_steps,
        )
        return [main, *self.models.baselines]


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
