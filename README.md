# Bangla Health Paraphrase Generation

Research-grade Bangla healthcare paraphrase generation using PySpark distributed preprocessing, mT5 with LoRA fine-tuning, and multilingual baseline comparison.

## Research Hypotheses

| ID | Hypothesis | How tested |
|----|-----------|------------|
| H1 | mT5 with LoRA outperforms full-FT baseline on the same backbone | Compare `mt5_lora` vs `mt5_baseline` on test BLEU/ROUGE/BERTScore |
| H2 | Domain-specific preprocessing improves quality | Ablation: train mT5+LoRA on uncleaned data vs cleaned |
| H3 | Data augmentation improves robustness | Toggle `training.use_augmented_train` in config |

## Models

| Key | Model | Training |
|-----|-------|----------|
| `mt5_lora` | google/mt5-small | LoRA (r=16) |
| `mt5_baseline` | google/mt5-small | Full fine-tuning |

## Setup

**Prerequisites:** Java 17+ (`JAVA_HOME`), [Apache Spark 4.1.x](https://spark.apache.org/downloads.html) unpacked to `C:\Spark` (or set `SPARK_HOME`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip uninstall pyspark -y   # if previously installed — use system Spark instead
.\scripts\setup_spark_windows.ps1
```

PySpark is **not** installed via pip. The project uses your system Spark (`SPARK_HOME`) for the JVM and Python bindings; the venv supplies ML deps (torch, transformers, bnlp, etc.). Workers run with `.venv\Scripts\python.exe` via `PYSPARK_PYTHON`.

## Quick Start

```powershell
# Full pipeline
python scripts/run_preprocess.py
python scripts/run_eda.py
python scripts/run_augment.py
python scripts/run_train.py --model mt5_lora
python scripts/run_evaluate.py

# Or via task runner
.\scripts\tasks.ps1 -Task preprocess
.\scripts\tasks.ps1 -Task eda
.\scripts\tasks.ps1 -Task train-mt5-lora
.\scripts\tasks.ps1 -Task evaluate
```

## Dev Subset

Set `dataset.dev_subset: 10000` in `configs/experiment_config.yaml` for fast smoke tests on a GTX 1070.

## Outputs

| Path | Contents |
|------|----------|
| `outputs/reports/eda_report.html` | EDA report |
| `outputs/reports/final_report.html` | Final research report |
| `outputs/metrics/summary.csv` | All model metrics |
| `outputs/figures/` | PNG + PDF plots (300 DPI) |
| `outputs/checkpoints/` | Trained models |
| `outputs/mlruns/` | MLflow experiment tracking |

## Notebook Workflow

Run notebooks in order (`notebooks/00_setup.ipynb` through `07_results_visualization.ipynb`). Each stage checks checkpoint sentinels and skips completed work.

## Dataset

Source: [faisal4590aziz/bangla-health-related-paraphrased-dataset](https://huggingface.co/datasets/faisal4590aziz/bangla-health-related-paraphrased-dataset)

Local copy: `datasets/all_paraphrased_data.csv` (~200K Bangla health paraphrase pairs).

Splits: 80% train / 10% val / 10% test (seed=42).

## Metrics

BLEU, ROUGE-L, BERTScore (xlm-roberta-large), Distinct-1/2, Sentence-BERT cosine similarity (multilingual-mpnet + Bengali-sbert).
