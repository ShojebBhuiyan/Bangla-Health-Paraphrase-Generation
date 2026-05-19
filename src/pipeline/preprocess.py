"""Pipeline orchestration entry points."""

from __future__ import annotations

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR, parquet_split_ready
from src.common.seed import set_seed
from src.spark.clean import clean
from src.spark.deduplicate import deduplicate
from src.spark.feature_engineering import run_feature_engineering
from src.spark.ingest import ingest
from src.spark.validate import profile_raw, validate

logger = get_logger(__name__)

_PREPROCESS_ARTIFACTS = tuple(
    DATA_DIR / "processed" / f"{split}.parquet" for split in ("train", "val", "test")
)


def _preprocess_artifacts_ready() -> bool:
    return all(parquet_split_ready(p) for p in _PREPROCESS_ARTIFACTS)


def run_preprocess(config: AppConfig | None = None, force: bool = False) -> None:
    cfg = config or load_config()
    set_seed(cfg.experiment.seed)

    if not force and is_done("preprocess", cfg) and _preprocess_artifacts_ready():
        logger.info("Preprocess stage already complete; skipping")
        return
    if not force and is_done("preprocess", cfg):
        logger.warning("Preprocess sentinel exists but parquet splits are missing; re-running")

    df = ingest(cfg)
    profile_raw(df, cfg)
    df = clean(df, cfg)
    df = deduplicate(df, cfg)
    validate(df, cfg)
    run_feature_engineering(df, cfg)
    mark_done("preprocess", {"status": "complete"}, cfg)
    logger.info("Preprocessing complete")
