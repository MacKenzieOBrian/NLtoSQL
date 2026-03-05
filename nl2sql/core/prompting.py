"""
Prompt builders for the baseline few-shot evaluation.

Assembles schema context, few-shot NLQ→SQL examples, and the target
question into a chat message list for the model.
"""

from __future__ import annotations


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

    Schema appears before exemplars so table names are in context when the model
    reads the examples — consistent ordering is required for cross-run comparability.
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
