"""Bangla token-level EDA augmentation (driver-side; avoids Windows Python worker issues)."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR, parquet_split_ready
from src.common.seed import set_seed

logger = get_logger(__name__)

_wordnet = None


def _get_wordnet():
    global _wordnet
    if _wordnet is None:
        try:
            from bnlp_toolkit import Wordnet

            _wordnet = Wordnet()
        except Exception:
            _wordnet = False
    return _wordnet if _wordnet is not False else None


def _synonym(word: str) -> str:
    wn = _get_wordnet()
    if not wn:
        return word
    try:
        syns = wn.synonyms(word)
        if syns:
            return random.choice(syns)
    except Exception:
        pass
    return word


def _augment_text(
    text: str,
    synonym_prob: float,
    swap_prob: float,
    deletion_prob: float,
    min_tokens: int,
) -> str:
    if not text:
        return text
    tokens = text.split()
    if len(tokens) < min_tokens:
        return text

    new_tokens = []
    for token in tokens:
        r = random.random()
        if r < synonym_prob:
            new_tokens.append(_synonym(token))
        else:
            new_tokens.append(token)

    if len(new_tokens) >= 2 and random.random() < swap_prob:
        i = random.randint(0, len(new_tokens) - 2)
        new_tokens[i], new_tokens[i + 1] = new_tokens[i + 1], new_tokens[i]

    if len(new_tokens) > min_tokens and random.random() < deletion_prob:
        idx = random.randint(0, len(new_tokens) - 1)
        new_tokens.pop(idx)

    return " ".join(new_tokens)


def augment_train(config: AppConfig | None = None, force: bool = False) -> Path:
    cfg = config or load_config()
    set_seed(cfg.experiment.seed)

    out = DATA_DIR / "augmented" / "train.parquet"
    if not force and is_done("augmentation", cfg) and parquet_split_ready(out):
        logger.info("Augmentation already complete: %s", out)
        return out
    if not force and is_done("augmentation", cfg):
        logger.warning("Augmentation sentinel exists but output is missing; re-running")

    from src.spark.session import get_spark

    spark = get_spark(cfg)
    train_path = DATA_DIR / "processed" / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Run preprocessing before augmentation")

    aug = cfg.augmentation
    pdf = spark.read.parquet(str(train_path)).toPandas()
    logger.info("Augmenting %d training rows on driver", len(pdf))

    augment_fn = lambda s: _augment_text(
        s,
        aug.synonym_prob,
        aug.swap_prob,
        aug.deletion_prob,
        aug.min_tokens_after_aug,
    )

    original = pdf.copy()
    original["is_augmented"] = False

    augmented = pdf.copy()
    augmented["source_sentence"] = augmented["source_sentence"].map(augment_fn)
    augmented["is_augmented"] = True

    combined = pd.concat([original, augmented], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    spark.createDataFrame(combined).write.mode("overwrite").parquet(str(out))
    row_count = len(combined)
    mark_done("augmentation", {"rows": row_count}, cfg)
    logger.info("Augmented train set saved: %s (%d rows)", out, row_count)
    return out
