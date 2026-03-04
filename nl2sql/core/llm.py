"""
Model generation helpers for chat LLMs.

Wraps the transformers generation pipeline for chat-template models.
Handles SQL extraction from raw output.
"""

from __future__ import annotations

import re
from typing import Any


# Find SELECT at line starts, with optional "sql:" prefix.
SQL_START_RE = re.compile(r"(?im)^\s*(?:sql\s*:\s*)?select\b")

# Reject prose-like "FROM the/this/..." candidates.
_PROSE_FROM_STOPWORDS = {"the", "a", "an", "this", "that", "these", "those"}


def _read_from_target(s: str) -> str | None:
    """Return the first token after FROM to check it looks like a table, not prose."""
    s = (s or "").lstrip()
    if not s:
        return None
    if s.startswith("("):
        return "("
    if s[0] in ('`', '"', "["):
        closing = {"`": "`", '"': '"', "[": "]"}[s[0]]
        end = s.find(closing, 1)
        if end == -1:
            return None
        return s[1:end].strip()
    m = re.match(r"[a-zA-Z_][\w$]*(?:\.[a-zA-Z_][\w$]*)*", s)
    return m.group(0) if m else None


def extract_first_select(text: str) -> str | None:
    """Extract the first valid SELECT statement from model output."""
    t = (text or "").strip()
    t = t.replace("```json", "```").replace("```sql", "```")
    t = re.sub(r"```(.*?)```", r"\1", t, flags=re.DOTALL).strip()

    for m in SQL_START_RE.finditer(t):
        start = m.start()
        tail = t[start:]
        semi = tail.find(";")
        stmt = tail if semi == -1 else tail[: semi + 1]
        stmt = re.sub(r"(?im)^\s*sql\s*:\s*", "", stmt, count=1).strip()

        # Keep only table-backed queries.
        from_m = re.search(r"(?is)\bfrom\b", stmt)
        if not from_m:
            continue
        target = _read_from_target(stmt[from_m.end():])
        if not target:
            continue
        if target != "(" and target.lower() in _PROSE_FROM_STOPWORDS:
            continue

        if not stmt.endswith(";"):
            stmt += ";"
        return stmt
    return None


def generate_sql_from_messages(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
    extract_select: bool = False,
    stop_on_semicolon: bool = False,
    **_kwargs: Any,  # compatibility with older call sites
) -> str:
    """Run the model and return a SQL string (or raw text if no SELECT found)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnSemicolon(StoppingCriteria):
        """Stop generation at the first ';' to avoid run-on explanations.

        Implements the HuggingFace StoppingCriteria interface:
        https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.StoppingCriteria
        """
        def __init__(self, tok: Any):
            ids = tok.encode(";", add_special_tokens=False)
            self._semi_id = ids[-1] if ids else None

        def __call__(self, input_ids, scores, **kwargs):
            if self._semi_id is None:
                return False
            return input_ids[0, -1].item() == self._semi_id

    # Build model-native chat tokens and prepare a matching attention mask.
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Suppress HF warnings when greedy decoding is used.
    effective_temperature = float(temperature) if do_sample else 1.0
    effective_top_p = float(top_p) if do_sample else 1.0

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": effective_temperature,
        "top_p": effective_top_p,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if stop_on_semicolon:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnSemicolon(tokenizer)])

    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

    # Drop prompt tokens; keep only newly generated text.
    gen_ids = out[0][input_ids.shape[-1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if extract_select:
        sql = extract_first_select(gen_text)
        return sql if sql is not None else gen_text
    return gen_text
