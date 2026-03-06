"""
HuggingFace model loading helpers.

4-bit NF4 quantized load with optional PeftModel adapter.
https://huggingface.co/docs/transformers/main/quantization/bitsandbytes
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def _compute_dtype() -> torch.dtype:
    """bf16 on Ampere+ GPUs (cc ≥ 8), fp16 otherwise."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_quantized_model(
    model_id: str,
    *,
    token: str | None = None,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load a 4-bit NF4 quantized base model and tokenizer.

    Attaches a PeftModel adapter when adapter_path is given and exists.
    Falls back to bf16/fp16 non-quantized load if bitsandbytes is unavailable.
    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = _compute_dtype()
    device_map: dict | None = {"": 0} if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map=device_map,
            token=token,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            device_map=device_map,
            token=token,
        )

    # Greedy decoding defaults — avoids HF sampling-parameter warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    if adapter_path:
        from peft import PeftModel
        p = Path(adapter_path)
        if p.exists():
            model = PeftModel.from_pretrained(model, p, token=token)

    return model, tok


def resolve_adapter_dir(path_str: str) -> Path:
    """Return the path to the latest adapter checkpoint under path_str."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if (p / "adapter_config.json").exists():
        return p
    ckpts = sorted(
        [x for x in p.glob("checkpoint-*") if (x / "adapter_config.json").exists()],
        key=lambda x: int(re.findall(r"\d+", x.name)[-1]),
    )
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No adapter found under: {p}")


__all__ = ["load_quantized_model", "resolve_adapter_dir"]
