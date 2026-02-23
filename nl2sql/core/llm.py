"""
Model-generation helpers for chat LLMs.

How to read this file:
1) `extract_first_select()` cleans raw model text to one SQL statement.
2) `generate_sql_from_messages()` runs chat-template generation with safe defaults.
3) Optional lightweight constraints block non-SELECT DDL/DML tokens.

References:
- Transformers generation docs: https://huggingface.co/docs/transformers/main_classes/text_generation
- Transformers quantization docs: https://huggingface.co/docs/transformers/main_classes/quantization
"""

from __future__ import annotations

import re
from typing import Any, Iterable

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


def debug_extract_first_select(text: str) -> dict[str, Any]:
    """
    Debug companion for extract_first_select().

    Returns candidate-level reasoning so notebooks can show:
    - what SQL-like spans were inspected
    - why candidates were rejected
    - which candidate (if any) was selected
    """
    t = (text or "").strip()
    t = t.replace("```json", "```").replace("```sql", "```")
    t = re.sub(r"```(.*?)```", r"\1", t, flags=re.DOTALL).strip()

    candidates: list[dict[str, Any]] = []
    selected_sql: str | None = None

    for m in SQL_START_RE.finditer(t):
        start = m.start()
        tail = t[start:]
        semi = tail.find(";")
        stmt = tail if semi == -1 else tail[: semi + 1]
        stmt = re.sub(r"(?im)^\s*sql\s*:\s*", "", stmt, count=1).strip()

        candidate: dict[str, Any] = {
            "start_index": start,
            "candidate_sql": stmt,
            "accepted": False,
            "reject_reason": None,
        }

        from_m = re.search(r"(?is)\bfrom\b", stmt)
        if not from_m:
            candidate["reject_reason"] = "missing_from_clause"
            candidates.append(candidate)
            continue
        target = _read_from_target(stmt[from_m.end() :])
        candidate["from_target"] = target
        if not target:
            candidate["reject_reason"] = "missing_from_target"
            candidates.append(candidate)
            continue
        if target != "(" and target.lower() in _PROSE_FROM_STOPWORDS:
            candidate["reject_reason"] = "prose_from_target"
            candidates.append(candidate)
            continue

        if not stmt.endswith(";"):
            stmt += ";"
        candidate["candidate_sql"] = stmt
        candidate["accepted"] = True
        candidates.append(candidate)
        selected_sql = stmt
        break

    return {
        "input_text": text,
        "normalized_text": t,
        "candidates": candidates,
        "selected_sql": selected_sql,
    }


def generate_sql_from_messages(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 128,
    constrained: bool = True,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    return_debug: bool = False,
) -> Any:
    import torch
    try:
        from transformers import BadWordsLogitsProcessor, LogitsProcessorList
    except Exception:  # pragma: no cover - fallback if transformers API shifts
        BadWordsLogitsProcessor = None  # type: ignore
        LogitsProcessorList = None  # type: ignore

    class _StopOnSemicolon(StoppingCriteria):
        """Stop generation at the first ';' to reduce run-on explanations."""

        def __init__(self, tok: Any):
            semi = tok.encode(";", add_special_tokens=False)
            self._semi_id = semi[-1] if semi else None

        def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
            if self._semi_id is None:
                return False
            return input_ids[0, -1].item() == self._semi_id

    def _bad_word_variants(words: Iterable[str]) -> list[str]:
        variants: list[str] = []
        for w in words:
            variants.extend([w, w.upper(), w.lower(), w.capitalize()])
            variants.extend([f" {w}", f" {w.upper()}", f" {w.lower()}", f" {w.capitalize()}"])
        # De-dupe while preserving order.
        seen = set()
        out = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _build_bad_words_ids(tok: Any) -> list[list[int]]:
        # Conservative DDL/DML/transaction keywords to prevent non-SELECT output.
        # This is a light PICARD-style constraint that should not over-block SELECTs.
        bad_words = [
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "truncate",
            "replace",
            "merge",
            "grant",
            "revoke",
            "commit",
            "rollback",
            "begin",
        ]
        bad_ids: list[list[int]] = []
        for w in _bad_word_variants(bad_words):
            ids = tok.encode(w, add_special_tokens=False)
            if ids:
                bad_ids.append(ids)
        return bad_ids

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

    if num_return_sequences and num_return_sequences > 1 and not do_sample:
        do_sample = True

    logits_processor = None
    if constrained and LogitsProcessorList is not None and BadWordsLogitsProcessor is not None:
        bad_words_ids = _build_bad_words_ids(tokenizer)
        if bad_words_ids:
            logits_processor = LogitsProcessorList(
                [BadWordsLogitsProcessor(bad_words_ids=bad_words_ids, eos_token_id=eos_token_id)]
            )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "stopping_criteria": StoppingCriteriaList([_StopOnSemicolon(tokenizer)]),
        "logits_processor": logits_processor,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})
    if num_return_sequences and num_return_sequences > 1:
        gen_kwargs["num_return_sequences"] = num_return_sequences

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    if num_return_sequences and num_return_sequences > 1:
        results: list[str] = []
        seq_debug: list[dict[str, Any]] = []
        for seq in out:
            gen_ids = seq[input_ids.shape[-1] :]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            sql = extract_first_select(gen_text)
            results.append(sql if sql is not None else gen_text)
            if return_debug:
                seq_debug.append(
                    {
                        "raw_generated_text": gen_text,
                        "extract_debug": debug_extract_first_select(gen_text),
                        "final_candidate": sql if sql is not None else gen_text,
                        "used_raw_fallback": sql is None,
                    }
                )
        if return_debug:
            return results, {
                "message_count": len(messages),
                "input_token_count": int(input_ids.shape[-1]),
                "num_return_sequences": int(num_return_sequences),
                "generation_args": {
                    "max_new_tokens": int(max_new_tokens),
                    "constrained": bool(constrained),
                    "do_sample": bool(do_sample),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
                "sequences": seq_debug,
            }
        return results

    gen_ids = out[0][input_ids.shape[-1] :]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    sql = extract_first_select(gen_text)
    final_candidate = sql if sql is not None else gen_text
    if return_debug:
        return final_candidate, {
            "message_count": len(messages),
            "input_token_count": int(input_ids.shape[-1]),
            "generation_args": {
                "max_new_tokens": int(max_new_tokens),
                "constrained": bool(constrained),
                "do_sample": bool(do_sample),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
            "raw_generated_text": gen_text,
            "extract_debug": debug_extract_first_select(gen_text),
            "final_candidate": final_candidate,
            "used_raw_fallback": sql is None,
        }
    return final_candidate
