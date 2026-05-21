"""Tests for configuration and checkpointing utilities."""

from pathlib import Path

import pytest
import yaml

from src.common.checkpointing import is_done, mark_done, stage_path
from src.common.config import AppConfig, load_config
from src.common.paths import mlflow_tracking_uri, parquet_data_files, parquet_split_ready
from src.common.reload import assert_preprocess_pipeline, reload_preprocess_modules


def test_preprocess_pipeline_order():
    reload_preprocess_modules()
    assert_preprocess_pipeline()


def test_load_config_validates():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.experiment.seed == 42
    assert cfg.dataset.train_split == 0.8
    assert cfg.models.main.key == "mt5_lora"
    assert len(cfg.models.baselines) == 1


def test_all_model_specs():
    cfg = load_config()
    specs = cfg.all_model_specs()
    assert len(specs) == 2
    assert specs[0].key == "mt5_lora"
    assert specs[1].key == "mt5_baseline"


def test_checkpointing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_STAGES", str(tmp_path))
    from src.common import checkpointing as cp

    class FakeOutputs:
        stages = str(tmp_path)

    class FakeConfig:
        outputs = FakeOutputs()

    fake = FakeConfig()
    assert not is_done("test_stage", fake)
    mark_done("test_stage", {"rows": 100}, fake)
    assert is_done("test_stage", fake)
    assert stage_path("test_stage", fake).exists()

    missing = tmp_path / "missing.parquet"
    assert not is_done("test_stage", fake, artifacts=[missing])
    missing.touch()
    assert is_done("test_stage", fake, artifacts=[missing])


def test_parquet_split_helpers(tmp_path: Path):
    spark_dir = tmp_path / "train.parquet"
    spark_dir.mkdir()
    assert not parquet_split_ready(spark_dir)
    part = spark_dir / "part-00000.parquet"
    part.write_bytes(b"x")
    assert parquet_split_ready(spark_dir)
    assert parquet_data_files(spark_dir).endswith("*.parquet")
    assert parquet_data_files(part) == str(part)


def test_config_from_temp_yaml(tmp_path: Path):
    data = {
        "experiment": {"name": "test", "seed": 1},
        "dataset": {
            "source": "x",
            "raw_csv": "datasets/x.csv",
            "train_split": 0.8,
            "validation_split": 0.1,
            "test_split": 0.1,
            "max_input_length": 128,
            "max_target_length": 128,
        },
        "spark": {
            "app_name": "Test",
            "shuffle_partitions": 10,
            "driver_memory": "2g",
            "executor_memory": "2g",
            "enable_arrow": True,
        },
        "models": {
            "main": {
                "name": "google/mt5-small",
                "key": "mt5_lora",
                "use_lora": True,
            },
            "baselines": [],
        },
        "lora": {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["q"]},
        "training": {
            "learning_rate": 1e-5,
            "epochs": 1,
            "weight_decay": 0.01,
            "warmup_steps": 10,
        },
        "augmentation": {},
        "evaluation": {
            "metrics": ["BLEU"],
            "bertscore_model": "xlm-roberta-large",
            "semantic_models": ["model"],
        },
        "mlflow": {"tracking_uri": "outputs/mlruns", "experiment_name": "test"},
        "outputs": {
            "checkpoints": "outputs/checkpoints",
            "logs": "outputs/logs",
            "metrics": "outputs/metrics",
            "figures": "outputs/figures",
            "reports": "outputs/reports",
            "stages": "outputs/.stages",
            "spark_checkpoints": "outputs/spark_checkpoints",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.experiment.name == "test"


def test_mlflow_tracking_uri_local_path(tmp_path: Path):
    root = tmp_path / "proj"
    uri = mlflow_tracking_uri("outputs/mlruns", root=root)
    assert uri.startswith("file://")
    assert uri.endswith("/outputs/mlruns")


def test_mlflow_tracking_uri_passthrough():
    remote = "http://127.0.0.1:5000"
    assert mlflow_tracking_uri(remote) == remote
