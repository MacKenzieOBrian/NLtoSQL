"""
Prompt builders.

How to read this file:
1) `SYSTEM_INSTRUCTIONS` is intentionally minimal for easy explanation.
2) `make_few_shot_messages()` assembles schema context + few-shot examples + NLQ.

Reliability-extension rationale:
- Prompt engineering + schema grounding are used to reduce hallucinated SQL.
- Few-shot examples are used as structured in-context support.

References (project anchors):
- `REFERENCES.md#ref-brown2020-gpt3`
- `REFERENCES.md#ref-mosbach2023-icl`
- `REFERENCES.md#ref-velasquez2023-prompteng`
- `REFERENCES.md#ref-wang2020-ratsql`
- `REFERENCES.md#ref-lin2020-bridge`
- `REFERENCES.md#ref-li2023-resdsql`
- `REFERENCES.md#ref-zhu2024-survey`
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
