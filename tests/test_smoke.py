"""End-to-end smoke test with dev subset."""

import json
from pathlib import Path

import pytest
import yaml

from src.common.config import load_config
from src.common.paths import PROJECT_ROOT


@pytest.fixture
def dev_config(tmp_path: Path):
    cfg_path = PROJECT_ROOT / "configs" / "experiment_config.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["dataset"]["dev_subset"] = 200
    out = tmp_path / "dev_config.yaml"
    out.write_text(yaml.dump(data), encoding="utf-8")
    return load_config(out)


def test_dev_config_loads(dev_config):
    assert dev_config.dataset.dev_subset == 200


def test_config_splits_sum_to_one():
    cfg = load_config()
    total = cfg.dataset.train_split + cfg.dataset.validation_split + cfg.dataset.test_split
    assert abs(total - 1.0) < 1e-6
