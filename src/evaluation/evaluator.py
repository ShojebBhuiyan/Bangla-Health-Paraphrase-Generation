"""Full model evaluation on test split."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common.checkpointing import is_done, mark_done
from src.common.config import AppConfig, ModelSpec, load_config
from src.common.logging import get_logger
from src.common.paths import DATA_DIR, PROJECT_ROOT, parquet_data_files
from src.evaluation.bertscore import compute_bertscore
from src.evaluation.metrics import compute_bleu, compute_distinct_n, compute_rouge_l
from src.evaluation.semantic import compute_semantic_similarity

logger = get_logger(__name__)


def _load_model(model_spec: ModelSpec, checkpoint_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_spec.name)
    base = AutoModelForSeq2SeqLM.from_pretrained(model_spec.name)
    final_dir = checkpoint_dir / "final"
    if model_spec.use_lora and final_dir.exists():
        model = PeftModel.from_pretrained(base, str(final_dir))
    elif final_dir.exists():
        model = AutoModelForSeq2SeqLM.from_pretrained(str(final_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(final_dir))
    else:
        model = base
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model.config.tie_word_embeddings = False
    return model, tokenizer, device


def _prefix_input(text: str, model_name: str) -> str:
    if "mbart" in model_name.lower():
        return text
    return f"paraphrase: {text}"


def generate_predictions(
    model,
    tokenizer,
    sources: list[str],
    model_name: str,
    config: AppConfig,
    device: str,
) -> list[str]:
    predictions = []
    batch_size = 8
    train_cfg = config.training

    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        inputs = [_prefix_input(s, model_name) for s in batch]
        encoded = tokenizer(
            inputs,
            max_length=config.dataset.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=config.dataset.max_target_length,
                num_beams=train_cfg.num_beams,
                no_repeat_ngram_size=train_cfg.no_repeat_ngram_size,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def evaluate_model(
    model_key: str,
    config: AppConfig | None = None,
    force: bool = False,
) -> dict:
    cfg = config or load_config()
    stage = f"evaluate_{model_key}"
    metrics_path = PROJECT_ROOT / cfg.outputs.metrics / f"{model_key}.json"

    if not force and is_done(stage, cfg) and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    spec = next(s for s in cfg.all_model_specs() if s.key == model_key)
    checkpoint_dir = PROJECT_ROOT / cfg.outputs.checkpoints / model_key

    test_path = DATA_DIR / "processed" / "test.parquet"
    test_ds = load_dataset("parquet", data_files=parquet_data_files(test_path), split="train")
    sources = list(test_ds["source_sentence"])
    references = list(test_ds["paraphrased_sentence"])

    model, tokenizer, device = _load_model(spec, checkpoint_dir)
    predictions = generate_predictions(model, tokenizer, sources, spec.name, cfg, device)

    results = {
        "model_key": model_key,
        "model_name": spec.name,
        "BLEU": compute_bleu(predictions, references),
        "ROUGE-L": compute_rouge_l(predictions, references),
        "BERTScore": compute_bertscore(predictions, references, cfg.evaluation.bertscore_model),
        "Distinct-1": compute_distinct_n(predictions, 1),
        "Distinct-2": compute_distinct_n(predictions, 2),
    }

    for sem_model in cfg.evaluation.semantic_models:
        key = "semsim_mpnet" if "mpnet" in sem_model else "semsim_bnsbert"
        results[key] = compute_semantic_similarity(sources, predictions, sem_model)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    mark_done(stage, results, cfg)
    logger.info("Evaluation complete for %s: %s", model_key, results)
    return results


def evaluate_all(config: AppConfig | None = None, force: bool = False) -> list[dict]:
    cfg = config or load_config()
    all_results = []
    for spec in cfg.all_model_specs():
        all_results.append(evaluate_model(spec.key, cfg, force=force))

    import pandas as pd

    summary_path = PROJECT_ROOT / cfg.outputs.metrics / "summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    logger.info("Summary metrics saved to %s", summary_path)
    return all_results
