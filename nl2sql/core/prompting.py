"""Prompt builders for schema-grounded NL-to-SQL generation.

This core-layer module turns schema text, optional exemplars, and the new
question into the chat-message structure expected by the model wrapper.
"""

from __future__ import annotations


# ai note copilot: "prompt template syntax; constraints written by me"
SYSTEM_INSTRUCTIONS = """You are a MySQL analyst.
Write one SQL SELECT query for the user question.

Rules:
- Output only SQL.
- Output exactly one statement starting with SELECT.
- Use only tables and columns in the provided schema details.
- Use ORDER BY and LIMIT only when the question asks for ranking.
"""
# This is the shared prompt contract for both zero-shot and few-shot runs.
def make_few_shot_messages(
    *,
    schema: str,
    exemplars: list[dict],
    nlq: str,
) -> list[dict[str, str]]:
    """Build chat messages from schema text, optional exemplars, and one NLQ."""
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        # Put schema first so every later example and question shares the same grounding context.
        {"role": "user", "content": "Schema Details:\n" + schema},
    ]
    # ai note copilot: "chat-turn list pattern for schema + exemplar messages"
    for ex in exemplars:
        # Use explicit user/assistant turns so the model sees worked examples in chat format.
        msgs.append({"role": "user", "content": f"Example Question: {ex['nlq']}"})
        msgs.append({"role": "assistant", "content": ex["sql"].rstrip(";") + ";"})
    msgs.append({"role": "user", "content": f"Natural Language Question: {nlq}"})
    return msgs


def make_training_example(nlq: str, sql: str, schema: str, tokenizer: object) -> str:
    """Format one ``(NLQ, SQL)`` pair as a chat-template string for SFT training."""
    messages = make_few_shot_messages(schema=schema, exemplars=[], nlq=nlq)
    messages.append({"role": "assistant", "content": sql.rstrip(";") + ";"})
    return tokenizer.apply_chat_template(messages, tokenize=False)  # type: ignore[attr-defined]
