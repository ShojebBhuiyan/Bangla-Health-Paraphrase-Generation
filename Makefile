# Bangla Health Paraphrase - Task Runner

.PHONY: setup preprocess eda augment train-mt5-lora train-baselines evaluate test

setup:
	pip install -r requirements.txt
	powershell -ExecutionPolicy Bypass -File scripts/setup_spark_windows.ps1

preprocess:
	python scripts/run_preprocess.py

eda:
	python scripts/run_eda.py

augment:
	python scripts/run_augment.py

train-mt5-lora:
	python scripts/run_train.py --model mt5_lora

train-baselines:
	python scripts/run_train.py --model mt5_baseline
	python scripts/run_train.py --model mbart_baseline

evaluate:
	python scripts/run_evaluate.py

test:
	pytest tests/ -v
