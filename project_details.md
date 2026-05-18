# Project Details

## Project Title

Efficient Domain-Specific Bangla Health Paraphrase Generation using
Parameter-Efficient Transformer Fine-Tuning with Distributed Data
Processing

## Research Goal

Develop a research-grade Bangla healthcare paraphrase generation system
using transformer architectures and distributed data processing.

## Research Objectives

1.  Build a robust Bangla healthcare paraphrase generator.
2.  Compare mT5 against baseline models.
3.  Evaluate parameter-efficient fine-tuning using LoRA.
4.  Measure semantic preservation and diversity.
5.  Analyze the impact of preprocessing and augmentation.

## Dataset

Source: faisal4590aziz/bangla-health-related-paraphrased-dataset

Approximate characteristics: - \~200K paraphrase pairs - Bangla
language - Healthcare domain

Dataset can be found inside the datasets folder.

## Hypotheses

H1: mT5 with LoRA will outperform T5 and BART.

H2: Domain-specific preprocessing improves quality.

H3: Data augmentation improves robustness.

## Baselines

- t5-small
- facebook/bart-base

## Main Model

- google/mt5-small
- optional: google/mt5-base

## Metrics

Generation quality: - BLEU - ROUGE-L - BERTScore

Diversity: - Distinct-1 - Distinct-2

Semantic: - Sentence-BERT cosine similarity

## Deliverables

- EDA report
- Clean dataset
- Trained models
- Publication-ready plots
- Result tables
- Notebook workflow
- Reproducible experiments
- Research paper assets
