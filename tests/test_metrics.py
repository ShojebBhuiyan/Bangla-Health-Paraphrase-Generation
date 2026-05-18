"""Metrics unit tests."""

from src.evaluation.metrics import compute_bleu, compute_distinct_n, compute_rouge_l


def test_bleu_perfect_match():
    refs = ["hello world"]
    preds = ["hello world"]
    assert compute_bleu(preds, refs) == 100.0


def test_distinct_n():
    texts = ["a b c d", "a b e f"]
    d1 = compute_distinct_n(texts, 1)
    d2 = compute_distinct_n(texts, 2)
    assert 0 < d1 <= 1
    assert 0 < d2 <= 1


def test_rouge_l():
    score = compute_rouge_l(["same text"], ["same text"])
    assert score == 1.0
