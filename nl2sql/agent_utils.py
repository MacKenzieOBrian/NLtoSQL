"""
Utilities to strengthen the ReAct-style NL→SQL agent without rewriting notebooks.

Provides:
- clean_candidate: strict SELECT-only filter to drop junk “Show SQL…” outputs.
- build_tabular_prompt: alternative prompt that makes the model enumerate tables/joins mentally.
- vanilla_candidate: deterministic few-shot baseline candidate (for fallback/rerank).
- classify_error / error_hint: tiny error taxonomy to drive targeted repair prompts.
- semantic_score: lightweight lexical heuristic to rerank executable candidates.

These helpers are intentionally lightweight and dependency-free so they can be
imported directly in notebooks (`notebooks/03_agentic_eval.ipynb`) or scripts.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from nl2sql.llm import extract_first_select
from nl2sql.prompting import make_few_shot_messages


def clean_candidate(raw: str) -> Optional[str]:
    """
    Keep only a single SELECT ... FROM ... statement; reject markdown/junk/plain text.
    Returns None if no usable SELECT is found.
    """
    sql = extract_first_select(raw)
    if not sql:
        return None

    sql = sql.strip()
    lower = sql.lower()

    # If the first 'SELECT' isn't at the start, realign
    if not lower.startswith("select"):
        idx = lower.find("select")
        if idx == -1:
            return None
        sql = sql[idx:].strip()
        lower = sql.lower()

    # Cut at the first ';' so trailing instructions don't poison the filter
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
        lower = sql.lower()

    # Must contain FROM (allowing newlines/whitespace)
    if not re.search(r"\bfrom\b", lower):
        return None

    # Lightweight junk filters on trimmed SQL
    bad_phrases = (
        "```",
        "answer:",
        "explanation",
    )
    if any(bp in lower for bp in bad_phrases):
        return None

    # Reject instruction echoes like "SELECT statement only"
    if re.search(r"\bselect\s+(query|statement)\b", lower):
        return None

    return sql + ";"


def build_tabular_prompt(nlq: str, schema_text: str) -> str:
    """
    Alternative prompt that nudges the model to reason about tables/joins explicitly.
    """
    return f"""
You are an expert SQL engineer. Follow these steps internally, but only output the final SELECT:
1) Decide which tables are needed.
2) Choose the join keys.
3) Choose the columns to return (only those asked).
4) Write ONE valid MySQL SELECT answer.

Schema:
{schema_text}

Question: {nlq}

Output only the final SQL statement and nothing else.
"""


def vanilla_candidate(
    nlq: str,
    schema_summary,
    tok,
    model,
    exemplars: Optional[Iterable[dict]] = None,
    max_new_tokens: int = 256,
):
    """
    Produce a deterministic few-shot candidate using the baseline prompt.
    Useful as a fallback when the ReAct loop finds no valid SQL.
    """
    from nl2sql.postprocess import guarded_postprocess

    msgs = make_few_shot_messages(schema=schema_summary, exemplars=exemplars or [], nlq=nlq)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    raw_sql = extract_first_select(text) or text
    sql = guarded_postprocess(raw_sql, nlq)
    return clean_candidate(sql)


def classify_error(err: str) -> str:
    """
    Coarse error taxonomy for MySQL execution errors.
    """
    e = err.lower()
    if "unknown column" in e:
        return "unknown_column"
    if "unknown table" in e or "doesn't exist" in e:
        return "unknown_table"
    if "ambiguous column" in e:
        return "ambiguous_column"
    if "group by" in e and "isn't in group by" in e:
        return "group_by"
    if "syntax error" in e or "check the manual" in e:
        return "syntax"
    return "other"


def error_hint(kind: str, err: str) -> str:
    """
    Human-readable hint to feed back into a repair prompt.
    """
    if kind == "unknown_column":
        return "Use only columns that exist in the schema; add the correct joins if a column lives in another table."
    if kind == "unknown_table":
        return "Use only tables that exist in the schema (customers, orders, orderdetails, products, payments, offices, employees, productlines)."
    if kind == "ambiguous_column":
        return "Qualify columns with the correct table alias (e.g., customers.customerNumber vs orders.customerNumber)."
    if kind == "group_by":
        return "All non-aggregated selected columns must appear in GROUP BY."
    if kind == "syntax":
        return "Fix the MySQL syntax (commas/parentheses/keywords)."
    return "Fix the SQL so it is valid MySQL and answers the question."


# Simple schema keyword map for lexical reranking
SCHEMA_KEYWORDS = {
    "country": ["country"],
    "customer": ["customer", "client", "buyer"],
    "order": ["order", "purchase"],
    "product": ["product", "item"],
    "office": ["office", "city", "location"],
    "employee": ["employee", "sales rep", "salesperson"],
    "total": ["total", "sum", "amount", "revenue"],
    "average": ["average", "avg", "mean"],
    "count": ["how many", "number of", "count"],
    "date": ["date", "year", "month"],
}


def count_select_columns(sql: str) -> int:
    lower = sql.lower()
    if "select" not in lower:
        return 99
    if not re.search(r"\bfrom\b", lower):
        return 99
    select_part = re.split(r"\bfrom\b", lower, 1)[0]
    select_part = select_part.split("select", 1)[1]
    return select_part.count(",") + 1


def semantic_score(nlq: str, sql: str) -> float:
    """
    Lightweight lexical score to prefer candidates whose columns/aggregates
    align with the NLQ intent. Not a true semantic parser, but better than
    “fewest columns”.
    """
    nlq_low = nlq.lower()
    sql_low = sql.lower()
    score = 0.0

    for key, aliases in SCHEMA_KEYWORDS.items():
        if any(a in nlq_low for a in aliases) and key in sql_low:
            score += 2.0

    nl_tokens = set(re.findall(r"[a-zA-Z]+", nlq_low))
    sql_tokens = set(re.findall(r"[a-zA-Z]+", sql_low))
    overlap = len(nl_tokens & sql_tokens)
    score += 0.1 * overlap

    if any(w in nlq_low for w in ["total", "sum", "revenue", "amount"]):
        if re.search(r"\b(sum|count|avg|max|min)\s*\(", sql_low, re.IGNORECASE):
            score += 3.0
        else:
            score -= 5.0

    if "each" in nlq_low or "per " in nlq_low:
        if re.search(r"\b(sum|count|avg|max|min)\s*\(", sql_low, re.IGNORECASE):
            score -= 2.0

    return score
