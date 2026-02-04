"""
Prompt builders.
Refs: schema-grounded prompting practice from NL->SQL surveys
(https://arxiv.org/abs/2410.06011) and few-shot ICL patterns
(e.g., https://arxiv.org/abs/2005.14165). We build system/schema/exemplar/NLQ
messages, guard leakage, and order schema text for better column choice.

#  builds the chat messages (system + schema + k exemplars + NLQ)
"""

from __future__ import annotations


# The system prompt is deliberately stable across experiments so differences in VA/EX
# can be attributed to the method (prompting vs QLoRA vs agent controls), not prompt drift.
#
# Most rules below are motivated by observed failure modes in early baselines:
# - "Output only SQL" reduces prompt-echo/explanations that cause VA=0.
# - "Minimal projection" reduces EM noise and discourages selecting extra columns.
# - "Use LIMIT/ORDER BY only when asked" prevents spurious ranking/limits.
# - Routing hints reduce schema-linking errors without hardcoding full answers.
SYSTEM_INSTRUCTIONS = """You are an expert data analyst writing MySQL queries.
Given the database schema and a natural language question, write a single SQL SELECT query.

Rules (schema-grounded and minimal):
- Output ONLY SQL (no explanation, no markdown).
- Output exactly ONE statement, starting with SELECT.
- Select ONLY the columns needed to answer the question (minimal projection).
- Use only the tables/columns listed in the schema; do NOT invent columns.
- Prefer explicit JOIN syntax.
- Use LIMIT/ORDER BY only when the NLQ implies ranking (top/highest/lowest/first/last).
- Status literals allowed: 'Shipped', 'Cancelled', 'On Hold', 'Disputed', 'In Process', 'Resolved'.
- Routing hints:
  * country/creditLimit filters -> join customers (orders.customerNumber = customers.customerNumber).
  * productLine/productVendor -> use products (and orderdetails for aggregates).
  * order totals -> SUM(orderdetails.quantityOrdered * orderdetails.priceEach) grouped by orderNumber.
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
