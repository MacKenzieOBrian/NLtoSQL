"""
Prompt builders for the baseline few-shot evaluation.

Assembles schema context, few-shot NLQ→SQL examples, and the target
question into a chat message list for the model.
"""

from __future__ import annotations


# https://huggingface.co/docs/transformers/en/chat_templating
SYSTEM_INSTRUCTIONS = """You are a MySQL analyst.
Write one SQL SELECT query for the user question.

Rules:
- Output only SQL.
- Output exactly one statement starting with SELECT.
- Use only tables and columns in the provided schema details.
- Use ORDER BY and LIMIT only when the question asks for ranking.
"""


# KEY FUNCTION — builds the prompt for every baseline and ReAct evaluation call.
def make_few_shot_messages(
    *,
    schema: str,
    exemplars: list[dict],
    nlq: str,
) -> list[dict[str, str]]:
    """Build a schema-grounded few-shot prompt for chat-template models.

    Schema precedes exemplars so table names are in context before the model reads
    the examples. https://huggingface.co/docs/transformers/en/chat_templating
    """
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": "Schema Details:\n" + schema},
    ]
    for ex in exemplars:
        msgs.append({"role": "user", "content": f"Example Question: {ex['nlq']}"})
        msgs.append({"role": "assistant", "content": ex["sql"].rstrip(";") + ";"})
    msgs.append({"role": "user", "content": f"Natural Language Question: {nlq}"})
    return msgs


def make_training_example(nlq: str, sql: str, schema: str, tokenizer: object) -> str:
    """Format one (NLQ, SQL) pair as a chat-template string for SFT training."""
    messages = make_few_shot_messages(schema=schema, exemplars=[], nlq=nlq)
    messages.append({"role": "assistant", "content": sql.rstrip(";") + ";"})
    return tokenizer.apply_chat_template(messages, tokenize=False)  # type: ignore[attr-defined]
