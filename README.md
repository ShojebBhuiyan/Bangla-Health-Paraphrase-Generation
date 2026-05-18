# Bangla Health Paraphrase Generation

Research-grade Bangla healthcare paraphrase generation using PySpark, mT5 with LoRA, and distributed preprocessing.

See [project_details.md](project_details.md) for research goals and [configs/experiment_config.yaml](configs/experiment_config.yaml) for experiment settings.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\scripts\setup_spark_windows.ps1
```

## Run

```powershell
python scripts/run_preprocess.py
python scripts/run_eda.py
python scripts/run_augment.py
python scripts/run_train.py --model mt5_lora
python scripts/run_evaluate.py --model mt5_lora
```

Or use the notebook workflow in `notebooks/`.
