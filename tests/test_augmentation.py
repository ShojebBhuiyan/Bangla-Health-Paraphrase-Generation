"""Augmentation unit tests."""

import random

from src.spark.augmentation import _augment_text


def test_augment_text_preserves_min_length():
    random.seed(42)
    text = "ডেঙ্গু রোগ হলো একটি স্বাস্থ্য সমস্যা"
    result = _augment_text(text, 0.0, 0.0, 0.0, min_tokens=3)
    assert len(result.split()) >= 3


def test_augment_text_swap():
    random.seed(0)
    text = "one two three four"
    result = _augment_text(text, 0.0, 1.0, 0.0, min_tokens=2)
    assert len(result.split()) == 4
