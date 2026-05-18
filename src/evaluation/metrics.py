"""Core evaluation metrics."""

from __future__ import annotations

from collections import Counter

import sacrebleu
from rouge_score import rouge_scorer


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_bleu(predictions, [references]).score


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


def compute_distinct_n(texts: list[str], n: int) -> float:
    total = 0
    unique = 0
    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total += len(ngrams)
        unique += len(set(ngrams))
    return unique / max(total, 1)
