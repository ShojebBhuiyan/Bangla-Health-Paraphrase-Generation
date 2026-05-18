# Coding Agent Prompt

You are a senior ML Engineer, Data Engineer, and Research Engineer
implementing a research-grade NLP system.

## Project

**Title:** Efficient Domain-Specific Bangla Health Paraphrase Generation
using Parameter-Efficient Transformer Fine-Tuning with Distributed Data
Processing

Dataset: `faisal4590aziz/bangla-health-related-paraphrased-dataset`
Dataset can be found inside datasets folder.

## Models

Primary: - google/mt5-small or mt5-base - LoRA fine tuning

Baselines: - t5-small - facebook/bart-base

Metrics: - BLEU - ROUGE-L - BERTScore - Distinct-1 - Distinct-2 -
Semantic Similarity

---

## Python Version

3.12.10

---

## PySpark Requirement

Use PySpark throughout the pipeline:

- Dataset ingestion
- Validation
- Cleaning
- Deduplication
- Feature engineering
- Distributed preprocessing
- EDA
- Data augmentation
- Intermediate persistence
- Experiment processing

Do not use pandas for large processing.

---

## Architecture Requirements

Must implement:

- Modular codebase
- Logging
- Error handling
- Checkpointing
- Resume support
- Automatic plot generation
- Automatic result tables
- Automatic EDA report
- Experiment tracking
- Notebook workflow
- Git commits after every major step

---

## Required folders

configs/ data/ notebooks/ src/ outputs/

Include:

- Spark session manager
- Spark preprocessing
- Spark EDA
- Feature engineering
- LoRA training
- Evaluation
- Visualization
- Utilities

---

## EDA

Generate:

- Sentence length distribution
- Word distribution
- Vocabulary statistics
- Similarity distribution
- Missing values
- Dataset summary

Outputs:

outputs/reports/eda_report.html

---

## Training

Use:

- Seq2SeqTrainer
- LoRA
- mixed precision
- gradient checkpointing
- early stopping
- cosine scheduler
- label smoothing

---

## Checkpointing

Save:

- processed parquet
- tokenized datasets
- model state
- optimizer state
- scheduler state
- metrics
- configs

Notebook execution should skip completed stages.

Example:

```python
if checkpoint_exists("tokenized_dataset"):
    load_tokenized_dataset()
else:
    tokenize()
```

---

## Logging

Log:

- preprocessing
- spark usage
- memory
- training metrics
- checkpoints
- runtime

Use rotating logs.

---

## Visualization

Generate:

- training loss
- validation loss
- BLEU comparison
- ROUGE comparison
- BERTScore comparison
- semantic similarity
- baseline comparison

Save PNG and PDF at 300 DPI.

---

## Git Requirement

After every meaningful implementation:

git add . git commit -m "(meaningful message)"

Never bundle unrelated changes.

---

Execution sequence:

1.  Implement
2.  Test
3.  Validate
4.  Generate artifacts
5.  Generate logs
6.  Commit
7.  Continue
