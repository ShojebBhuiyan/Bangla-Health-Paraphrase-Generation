"""Pipeline orchestration entry points."""

from __future__ import annotations

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, load_config
from src.common.logging import get_logger
from src.common.seed import set_seed
from src.spark.clean import clean
from src.spark.deduplicate import deduplicate
from src.spark.feature_engineering import run_feature_engineering
from src.spark.ingest import ingest
from src.spark.validate import validate

logger = get_logger(__name__)


def run_preprocess(config: AppConfig | None = None, force: bool = False) -> None:
    cfg = config or load_config()
    set_seed(cfg.experiment.seed)

    if not force and is_done("preprocess", cfg):
        logger.info("Preprocess stage already complete; skipping")
        return

    df = ingest(cfg)
    validate(df, cfg)
    df = clean(df, cfg)
    df = deduplicate(df, cfg)
    run_feature_engineering(df, cfg)
    mark_done("preprocess", {"status": "complete"}, cfg)
    logger.info("Preprocessing complete")
