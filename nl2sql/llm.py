"""
LLM loading helpers.
Refs: Hugging Face Transformers 4-bit NF4 + BitsAndBytes loading
(https://huggingface.co/docs/transformers/main_classes/quantization),
PEFT/QLoRA examples (https://huggingface.co/docs/peft/),
BitsAndBytes docs (https://github.com/TimDettmers/bitsandbytes).

# Used here: consistent 4-bit Llama-3 load with deterministic decoding for eval.
loads Llama‑3‑8B in 4-bit NF4 with BitsAndBytes, deterministic decoding (do_sample=False), and a helper to grab the first SELECT from generated text.
"""

from __future__ import annotations

import re
from typing import Any

from transformers import StoppingCriteria, StoppingCriteriaList


# Regex reference: https://docs.python.org/3/library/re.html
#
# Rationale: try to extract a single executable SELECT from model output while
# tolerating explanatory text before/after.
#
# We bias toward "SQL-ish" statements:
# - the SELECT should start a line (or be prefixed by "SQL:")
# - the statement should contain a FROM clause
#
# This avoids common false positives like "please select ... from ..." in prose.
SQL_START_RE = re.compile(r"(?im)^\s*(?:sql\s*:\s*)?select\b")

# Tiny stopword list to filter obvious prose like "from the ...".
_PROSE_FROM_STOPWORDS = {"the", "a", "an", "this", "that", "these", "those"}


def _read_from_target(s: str) -> str | None:
    """
    Return the first token after FROM:
    - '(' for subqueries: FROM (SELECT ...)
    - unquoted or quoted identifier (optionally schema-qualified)
    """
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
    # Strip common markdown code-fence wrappers before trying to extract SQL.
    # This keeps downstream postprocess deterministic and avoids fence fragments
    # leaking into execution.
    t = (text or "").strip()
    t = t.replace("```json", "```").replace("```sql", "```")
    t = re.sub(r"```(.*?)```", r"\1", t, flags=re.DOTALL).strip()

    for m in SQL_START_RE.finditer(t):
        start = m.start()
        tail = t[start:]
        semi = tail.find(";")
        stmt = tail if semi == -1 else tail[: semi + 1]
        stmt = re.sub(r"(?im)^\s*sql\s*:\s*", "", stmt, count=1).strip()

        # Must look like a table-backed query (ClassicModels items are).
        from_m = re.search(r"(?is)\bfrom\b", stmt)
        if not from_m:
            continue
        target = _read_from_target(stmt[from_m.end() :])
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
) -> str:
    import torch

    class _StopOnSemicolon(StoppingCriteria):
        """Stop generation at the first ';' to reduce run-on explanations."""

        def __init__(self, tok: Any):
            semi = tok.encode(";", add_special_tokens=False)
            self._semi_id = semi[-1] if semi else None

        def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
            if self._semi_id is None:
                return False
            return input_ids[0, -1].item() == self._semi_id

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    # Some tokenizers share pad/eos ids, which prevents generate() from inferring an
    # attention mask reliably. Our prompts are not padded, so an all-ones mask is valid.
    attention_mask = torch.ones_like(input_ids)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            stopping_criteria=StoppingCriteriaList([_StopOnSemicolon(tokenizer)]),
        )

    gen_ids = out[0][input_ids.shape[-1] :]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    sql = extract_first_select(gen_text)
    return sql if sql is not None else gen_text
