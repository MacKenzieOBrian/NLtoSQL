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


# primary path: this prompt builder is used for model-only raw runs.
def make_few_shot_messages(
    *,
    schema: str,
    exemplars: list[dict],
    nlq: str,
) -> list[dict[str, str]]:
    """
    Build a schema-grounded few-shot prompt in a single, explainable format.

    Design intent:
    - keep prompt structure fixed for comparability across runs
    - include schema/table context before examples to anchor decoding
    - include curated NLQ->SQL exemplars for in-context adaptation
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
