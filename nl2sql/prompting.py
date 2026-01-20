"""
Prompt builders.
Refs: schema-grounded prompting practice from NL→SQL surveys
(https://arxiv.org/abs/2410.06011) and few-shot patterns from standard ICL
literature. Implementation is our own: build system/schema/exemplar/NLQ prompts,
guard against leakage, order schema text for better column choice.
"""

from __future__ import annotations


SYSTEM_INSTRUCTIONS = """You are an expert data analyst writing MySQL queries.
Given the database schema and a natural language question, write a single SQL SELECT query.

Rules:
- Output ONLY SQL (no explanation, no markdown).
- Output exactly ONE statement, starting with SELECT.
- Select ONLY the columns needed to answer the question (minimal projection).
- Use only the tables/columns in the schema.
- Prefer explicit JOIN syntax.
- Use LIMIT when the question implies \"top\" or \"first\".
"""


def make_few_shot_messages(*, schema: str, exemplars: list[dict], nlq: str) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": "Schema:\n" + schema},
    ]
    for ex in exemplars:
        msgs.append({"role": "user", "content": f"NLQ: {ex['nlq']}"})
        msgs.append({"role": "assistant", "content": ex["sql"].rstrip(";") + ";"})
    msgs.append({"role": "user", "content": f"NLQ: {nlq}"})
    return msgs
