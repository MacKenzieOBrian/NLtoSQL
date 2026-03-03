"""
Prompt builders.

How to read this file:
1) `SYSTEM_INSTRUCTIONS` is intentionally minimal for easy explanation.
2) `make_few_shot_messages()` assembles schema context + few-shot examples + NLQ.

Reliability-extension rationale:
- Prompt engineering + schema grounding are used to reduce hallucinated SQL.
- Few-shot examples are used as structured in-context support.

Related literature: few-shot prompting [8], fair ICL vs fine-tuning
comparisons [7], ICL behavior without weight updates [4], and schema-grounded
text-to-SQL methods [2, 18, 20, 9].
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
    table_descriptions: str | None = None,
) -> list[dict[str, str]]:
    """
    Build a schema-grounded few-shot prompt in a single, explainable format.

    Design intent:
    - keep prompt structure fixed for comparability across runs
    - include schema/table context before examples to anchor decoding
    - include curated NLQ->SQL exemplars for in-context adaptation
    """
    context_parts = ["Schema Details:\n" + schema]
    if table_descriptions:
        context_parts.append("Table Descriptions:\n" + table_descriptions)

    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": "\n\n".join(context_parts)},
    ]
    for ex in exemplars:
        msgs.append({"role": "user", "content": f"Example Question: {ex['nlq']}"})
        msgs.append({"role": "assistant", "content": ex["sql"].rstrip(";") + ";"})
    msgs.append({"role": "user", "content": f"Natural Language Question: {nlq}"})
    return msgs
