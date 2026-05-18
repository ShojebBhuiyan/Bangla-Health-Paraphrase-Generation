"""Training entry point."""

from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.logging import get_logger
from src.data.tokenize import tokenize_all
from src.models.mbart_baseline import run_mbart_baseline
from src.models.mt5_baseline import run_mt5_baseline
from src.models.mt5_lora import run_mt5_lora

logger = get_logger(__name__)

RUNNERS = {
    "mt5_lora": run_mt5_lora,
    "mt5_baseline": run_mt5_baseline,
    "mbart_baseline": run_mbart_baseline,
}


def main():
    parser = argparse.ArgumentParser(description="Train paraphrase model")
    parser.add_argument("--model", required=True, choices=list(RUNNERS.keys()))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if not args.skip_tokenize:
        tokenize_all(args.model, cfg, force=args.force)

    RUNNERS[args.model](cfg, force=args.force)


if __name__ == "__main__":
    main()
