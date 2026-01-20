"""
LLM loading helpers.
Refs: Hugging Face Transformers 4-bit NF4 + BitsAndBytes loading
(https://huggingface.co/docs/transformers/main_classes/quantization) and
PEFT QLoRA examples. Logic is our own thin wrapper to set deterministic decoding
defaults for eval.
"""

from __future__ import annotations

import re
from typing import Any


SQL_RE = re.compile(r"(?is)\\bselect\\b.*?(;|\\Z)")


def extract_first_select(text: str) -> str | None:
    m = SQL_RE.search((text or "").strip())
    if not m:
        return None
    sql = m.group(0).strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def generate_sql_from_messages(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 128,
) -> str:
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[-1] :]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    sql = extract_first_select(gen_text)
    return sql if sql is not None else gen_text
