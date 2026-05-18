"""Evaluation entry point."""

import argparse

from src.evaluation.evaluator import evaluate_all, evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.model:
        evaluate_model(args.model, force=args.force)
    else:
        evaluate_all(force=args.force)
