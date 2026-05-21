"""Metrics unit tests."""

from unittest.mock import patch

import pytest
import torch

from src.evaluation.bertscore import compute_bertscore
from src.evaluation.metrics import compute_bleu, compute_distinct_n, compute_rouge_l


def test_compute_bertscore_coerces_iterables():
    """HuggingFace Dataset columns must be coerced before bert_score sees them."""

    class ColumnLike(list):
        pass

    with patch("src.evaluation.bertscore.bert_score_fn") as mock_score:
        mock_score.return_value = (None, None, torch.tensor([0.85]))
        score = compute_bertscore(["prediction"], ColumnLike(["reference"]))
        assert score == pytest.approx(0.85)
        mock_score.assert_called_once()


def test_bleu_perfect_match():
    refs = ["the cat sat on the mat"]
    preds = ["the cat sat on the mat"]
    score = compute_bleu(preds, refs)
    assert score >= 99.0


def test_distinct_n():
    texts = ["a b c d", "a b e f"]
    d1 = compute_distinct_n(texts, 1)
    d2 = compute_distinct_n(texts, 2)
    assert 0 < d1 <= 1
    assert 0 < d2 <= 1


def test_rouge_l():
    score = compute_rouge_l(["same text"], ["same text"])
    assert score == 1.0
