"""Scheduler helpers."""

from __future__ import annotations


def get_scheduler_type(name: str) -> str:
    mapping = {
        "cosine": "cosine",
        "linear": "linear",
        "constant": "constant",
    }
    return mapping.get(name.lower(), "cosine")
