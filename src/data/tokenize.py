"""Per-model tokenization with caching."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, ModelSpec, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR, parquet_data_files, parquet_split_ready
from src.common.seed import set_seed

logger = get_logger(__name__)


def _model_prefix(model_name: str) -> tuple[str, dict]:
    name = model_name.lower()
    if "mbart" in name:
        return "", {"src_lang": "bn_IN", "tgt_lang": "bn_IN"}
    return "paraphrase: ", {}


def _load_split_parquet(split: str, config: AppConfig) -> Path:
    if split == "train" and config.training.use_augmented_train:
        aug = DATA_DIR / "augmented" / "train.parquet"
        if aug.exists():
            return aug
    return DATA_DIR / "processed" / f"{split}.parquet"


def tokenize_split(
    model_spec: ModelSpec,
    split: str,
    config: AppConfig | None = None,
    force: bool = False,
) -> Path:
    cfg = config or load_config()
    set_seed(cfg.experiment.seed)
    stage = f"tokenize_{model_spec.key}_{split}"

    out_dir = DATA_DIR / "processed" / "tokenized" / model_spec.key / split
    if not force and is_done(stage, cfg) and out_dir.exists():
        logger.info("Tokenized dataset exists: %s", out_dir)
        return out_dir

    from transformers import AutoTokenizer

    parquet_path = _load_split_parquet(split, cfg)
    if not parquet_split_ready(parquet_path):
        raise FileNotFoundError(
            f"Split not found or empty (expected parquet file or Spark folder): {parquet_path}"
        )

    ds = load_dataset("parquet", data_files=parquet_data_files(parquet_path), split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_spec.name)
    prefix, lang_kwargs = _model_prefix(model_spec.name)

    def preprocess(batch):
        inputs = [prefix + s for s in batch["source_sentence"]]
        return tokenizer(
            inputs,
            text_target=batch["paraphrased_sentence"],
            max_length=cfg.dataset.max_input_length,
            max_target_length=cfg.dataset.max_target_length,
            truncation=True,
            padding="max_length",
            **lang_kwargs,
        )

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(out_dir))
    mark_done(stage, {"model": model_spec.key, "split": split}, cfg)
    logger.info("Tokenized %s/%s -> %s", model_spec.key, split, out_dir)
    return out_dir


def tokenize_all(model_key: str | None = None, config: AppConfig | None = None, force: bool = False) -> None:
    cfg = config or load_config()
    specs = cfg.all_model_specs()
    if model_key:
        specs = [s for s in specs if s.key == model_key]
        if not specs:
            raise ValueError(f"Unknown model key: {model_key}")

    for spec in specs:
        for split in ("train", "val", "test"):
            tokenize_split(spec, split, cfg, force=force)
