"""VRAM probe before training."""

from __future__ import annotations

import torch

from src.common.logging import get_logger

logger = get_logger(__name__)


def probe_memory(model, sample_batch: dict, device: str = "cuda") -> bool:
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; skipping memory probe")
        return True

    model = model.to(device)
    model.train()
    try:
        inputs = {k: v.to(device) for k, v in sample_batch.items() if hasattr(v, "to")}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        torch.cuda.empty_cache()
        logger.info("Memory probe passed on %s", device)
        return True
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error("OOM during memory probe: %s", exc)
            return False
        raise
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
