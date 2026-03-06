"""Model loading helpers for the notebook workflows."""

from __future__ import annotations

import gc
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


def build_4bit_quant_config() -> tuple[Any, torch.dtype, bool]:
    """Return (BitsAndBytesConfig, compute_dtype, use_bf16).

    Inspired by the usual QLoRA-style 4-bit NF4 loading setup.
    """
    from transformers import BitsAndBytesConfig

    compute_dtype = _compute_dtype()
    use_bf16 = bool(compute_dtype == torch.bfloat16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return bnb_config, compute_dtype, use_bf16


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

    # Keep generation defaults deterministic.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    if adapter_path:
        from peft import PeftModel
        p = Path(adapter_path)
        if p.exists():
            model = PeftModel.from_pretrained(model, p, token=token)

    return model, tok


def build_trainable_qlora_model(
    *,
    experiment_config: dict[str, Any],
    token: str | None = None,
) -> tuple[Any, Any, Any, torch.dtype, bool]:
    """Load the base model, attach LoRA layers, and return training-ready objects.

    Inspired by the common QLoRA recipe: load a 4-bit base model, prepare it for
    k-bit training, then add LoRA adapters on top.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Runtime -> Change runtime type -> GPU.")

    model_id = experiment_config["model_id"]
    bnb_config, compute_dtype, use_bf16 = build_4bit_quant_config()
    base_model, tok = load_quantized_model(model_id, token=token)
    # This follows the usual "prepare k-bit base first, then add LoRA adapters" order.
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=experiment_config["lora_r"],
        lora_alpha=experiment_config["lora_alpha"],
        lora_dropout=experiment_config["lora_dropout"],
        target_modules=experiment_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tok, bnb_config, compute_dtype, use_bf16


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


def load_eval_adapter_model(
    *,
    model_id: str,
    adapter_path: str,
    bnb_config: Any,
    compute_dtype: torch.dtype,
    token: str | None = None,
    offload_dir: str | Path = "/content/offload",
) -> tuple[Any, Path]:
    """Load a base model plus local adapter for evaluation."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    resolved_adapter = resolve_adapter_dir(adapter_path)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    offload_dir = Path(offload_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)

    eval_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        max_memory={0: "10GiB", "cpu": "64GiB"},
        offload_folder=str(offload_dir),
        token=token if token else True,
    )
    eval_model = PeftModel.from_pretrained(
        eval_base,
        str(resolved_adapter),
        is_trainable=False,
        local_files_only=True,
    )
    eval_model.eval()
    eval_model.generation_config.do_sample = False
    eval_model.generation_config.temperature = 1.0
    eval_model.generation_config.top_p = 1.0
    return eval_model, resolved_adapter


__all__ = [
    "build_trainable_qlora_model",
    "build_4bit_quant_config",
    "load_eval_adapter_model",
    "load_quantized_model",
    "resolve_adapter_dir",
]
