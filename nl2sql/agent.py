"""
ReAct-style Text-to-SQL agent (ClassicModels).

Goal:
- Turn an NLQ into a single executable MySQL SELECT statement.

Core idea (ReAct; Yao et al., 2023):
- Generate candidate actions (SQL strings),
- Act by executing them against the database (QueryRunner),
- Observe success/errors,
- Iterate a bounded number of steps using the observation as feedback.

This implementation is optimized for dissertation explainability:
- bounded loop + bounded repair (auditability)
- deterministic postprocess (no weight changes)
- explicit gates (execution validity + intent constraints)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import re

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .agent_utils import (
    build_schema_subset,
    clean_candidate_with_reason,
    count_select_columns,
    enforce_projection_contract,
    intent_constraints,
    semantic_score,
    vanilla_candidate,
)
from .postprocess import guarded_postprocess
from .query_runner import QueryRunner


@dataclass(frozen=True)
class ReactConfig:
    # Control / cost bounds.
    max_steps: int = 3
    num_cands: int = 6
    max_new_tokens: int = 128
    enable_repair: bool = True
    repair_num_cands: int = 4

    # Candidate diversity.
    do_sample: bool = True
    temperature: float = 0.5
    top_p: float = 0.9

    # Optional prompt variants and controls.
    use_tabular_prompt: bool = True
    use_schema_subset: bool = True
    use_projection_contract: bool = True

    # Scoring.
    column_penalty: float = 0.5


class _StopOnSemicolon(StoppingCriteria):
    """Stop generation at the first ';' to reduce run-on explanations."""

    def __init__(self, tokenizer: Any):
        semi = tokenizer.encode(";", add_special_tokens=False)
        self._semi_id = semi[-1] if semi else None

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
        if self._semi_id is None:
            return False
        return input_ids[0, -1].item() == self._semi_id


def _trim(s: Any, n: int = 500) -> Any:
    if s is None:
        return None
    out = str(s)
    return out if len(out) <= n else out[:n] + "..."


class ReactSqlAgent:
    def __init__(
        self,
        *,
        model: Any,
        tok: Any,
        runner: QueryRunner,
        cfg: Optional[ReactConfig] = None,
        extra_score_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.model = model
        self.tok = tok
        self.runner = runner
        self.cfg = cfg or ReactConfig()
        self.extra_score_fn = extra_score_fn or (lambda _nlq, _sql: 0.0)

        # Ensure we can pad during generation.
        if getattr(self.tok, "pad_token", None) is None and getattr(self.tok, "eos_token", None) is not None:
            self.tok.pad_token = self.tok.eos_token

    # -----------------
    # Prompt builders
    # -----------------
    def _build_react_prompt(self, *, nlq: str, schema_text: str, history: list[dict], observation: str) -> str:
        schema_view = build_schema_subset(schema_text, nlq) if self.cfg.use_schema_subset else schema_text
        history_text = "\n\n".join(
            f"Thought/Action: {h.get('ta','')}\nObservation: {h.get('obs','')}" for h in history
        ) or "None yet."
        return f"""
You are an expert MySQL analyst.

TASK:
- Write exactly ONE valid MySQL SELECT statement.
- Output only SQL (no explanation, no markdown).
- The output must include a FROM clause.
- Use only schema tables/columns.
- Use ORDER BY/LIMIT only if explicitly asked.

Schema:
{schema_view}

Question:
{nlq}

Recent steps:
{history_text}

Last observation:
{observation}

Respond with only the final SQL statement.
""".strip()

    def _build_tabular_prompt(self, *, nlq: str, schema_text: str) -> str:
        schema_view = build_schema_subset(schema_text, nlq) if self.cfg.use_schema_subset else schema_text
        return f"""
You are an expert SQL engineer. Think through tables and join keys, then output one SELECT.

Schema:
{schema_view}

Question:
{nlq}

Output only the final SQL statement and nothing else.
""".strip()

    # -----------------
    # Candidate generation
    # -----------------
    def generate_candidates(self, prompt: str, *, num: int, do_sample: Optional[bool] = None) -> list[str]:
        cfg = self.cfg
        do_sample = cfg.do_sample if do_sample is None else do_sample
        if not do_sample:
            num = 1

        # Prefixing with SELECT reduces non-SQL continuations.
        prompt = prompt.rstrip() + "\nSELECT "
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num,
            "stopping_criteria": StoppingCriteriaList([_StopOnSemicolon(self.tok)]),
            "eos_token_id": getattr(self.tok, "eos_token_id", None),
            "pad_token_id": getattr(self.tok, "pad_token_id", getattr(self.tok, "eos_token_id", None)),
        }
        if do_sample:
            gen_kwargs.update({"temperature": cfg.temperature, "top_p": cfg.top_p})

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        cands: list[str] = []
        prompt_len = inputs.input_ids.shape[-1]
        for i in range(num):
            gen_ids = out[i][prompt_len:]
            gen = self.tok.decode(gen_ids, skip_special_tokens=True)
            # Ensure the candidate starts with SELECT even if the model repeats it.
            gen = re.sub(r"(?is)^\s*select\b\s*", "", gen)
            cands.append("SELECT " + gen)
        return cands

    # -----------------
    # Candidate evaluation
    # -----------------
    def postprocess_sql(self, *, sql: str, nlq: str) -> str:
        # Deterministic postprocess layer (no weight changes).
        cleaned = guarded_postprocess(sql, nlq)
        if self.cfg.use_projection_contract:
            cleaned = enforce_projection_contract(cleaned, nlq)
        return cleaned

    def evaluate_candidate(self, *, nlq: str, raw: str) -> tuple[Optional[tuple[str, float]], dict]:
        sql, reason = clean_candidate_with_reason(raw)
        if not sql:
            return None, {"phase": "clean_reject", "reason": reason, "raw": _trim(raw)}

        sql = self.postprocess_sql(sql=sql, nlq=nlq)

        # Execution gate (Act): must run successfully.
        meta = self.runner.run(sql, capture_df=False)
        if not meta.success:
            return None, {"phase": "exec_fail", "sql": sql, "error": _trim(meta.error)}

        # Intent gate: prevent executable-but-wrong-shape answers.
        ok, why = intent_constraints(nlq, sql)
        if not ok:
            return None, {"phase": "intent_reject", "sql": sql, "reason": why}

        # Simple, auditable scoring.
        s_sem = semantic_score(nlq, sql)
        s_cols = count_select_columns(sql)
        s_extra = float(self.extra_score_fn(nlq, sql))
        score = float(s_sem) - float(self.cfg.column_penalty) * float(s_cols) + s_extra

        return (sql, score), {"phase": "accept", "sql": sql, "score": score, "sem": s_sem, "cols": s_cols, "extra": s_extra}

    # -----------------
    # Repair
    # -----------------
    def repair_sql(self, *, nlq: str, bad_sql: str, error_msg: str, schema_text: str) -> tuple[Optional[str], dict]:
        if not self.cfg.enable_repair:
            return None, {"enabled": False}

        prompt = f"""
You are an expert MySQL engineer.

Schema:
{schema_text}

User question:
{nlq}

Invalid SQL:
{bad_sql}

Database error:
{error_msg}

Fix the SQL so it is valid MySQL and answers the question.
Output ONLY the corrected SELECT statement.
""".strip()

        fixes = self.generate_candidates(prompt, num=self.cfg.repair_num_cands, do_sample=True)
        if not fixes:
            return None, {"enabled": True, "status": "no_fix_generated"}

        last_info: dict = {"enabled": True, "status": "no_valid_fix"}
        for raw_fix in fixes:
            cand, _reason = clean_candidate_with_reason(raw_fix)
            if not cand:
                continue

            sql = self.postprocess_sql(sql=cand, nlq=nlq)
            meta = self.runner.run(sql, capture_df=False)
            if not meta.success:
                last_info = {"enabled": True, "status": "exec_fail", "raw_fix": _trim(raw_fix), "fixed_sql": sql, "exec_error": _trim(meta.error)}
                continue

            ok, why = intent_constraints(nlq, sql)
            if not ok:
                last_info = {"enabled": True, "status": "intent_reject", "raw_fix": _trim(raw_fix), "fixed_sql": sql, "reason": why}
                continue

            return sql, {"enabled": True, "status": "exec_ok", "raw_fix": _trim(raw_fix), "fixed_sql": sql}

        return None, last_info

    # -----------------
    # Main loop
    # -----------------
    def react_sql(
        self,
        *,
        nlq: str,
        schema_text: str,
        schema_summary: Optional[str] = None,
        exemplars: Optional[list[dict]] = None,
    ) -> tuple[str, list[dict]]:
        """Run a bounded ReAct loop and return (best_sql, trace)."""
        cfg = self.cfg
        history: list[dict] = []
        observation = "Start."
        last_failed_sql: Optional[str] = None
        last_error: Optional[str] = None

        for step in range(cfg.max_steps):
            prompts = [self._build_react_prompt(nlq=nlq, schema_text=schema_text, history=history, observation=observation)]
            if cfg.use_tabular_prompt:
                prompts.append(self._build_tabular_prompt(nlq=nlq, schema_text=schema_text))

            per_prompt = max(1, cfg.num_cands // len(prompts))
            raw_cands: list[str] = []
            for p in prompts:
                raw_cands.extend(self.generate_candidates(p, num=per_prompt, do_sample=cfg.do_sample))

            best: Optional[tuple[str, float]] = None

            for raw in raw_cands:
                result, log = self.evaluate_candidate(nlq=nlq, raw=raw)
                log = {"step": step, **log}
                history.append({k: _trim(v) for k, v in log.items()})

                if log.get("phase") == "exec_fail":
                    last_error = log.get("error")
                    last_failed_sql = log.get("sql")

                if result is not None:
                    sql, score = result
                    if best is None or score > best[1]:
                        best = (sql, score)

            if best is not None:
                sql, score = best
                history.append({"step": step, "phase": "final", "sql": sql, "score": score})
                return sql, history

            # Optional repair: use error message as feedback, then re-run gates.
            if cfg.enable_repair and last_error and last_failed_sql:
                repaired, repinfo = self.repair_sql(
                    nlq=nlq,
                    bad_sql=last_failed_sql,
                    error_msg=str(last_error),
                    schema_text=schema_text,
                )
                history.append({"step": step, "phase": "repair", "bad_sql": last_failed_sql, "error": _trim(last_error), **repinfo})
                if repaired:
                    history.append({"step": step, "phase": "final", "sql": repaired, "score": "repair_accept"})
                    return repaired, history
                observation = f"Previous SQL failed: {last_error}. Revise tables/joins and try again."
            else:
                observation = "No executable candidates. Try a simpler join path."

            history.append({"step": step, "phase": "observation", "obs": _trim(observation)})

        # Deterministic fallback (few-shot baseline) if provided.
        if schema_summary is not None:
            fallback = vanilla_candidate(
                nlq=nlq,
                schema_summary=schema_summary,
                tok=self.tok,
                model=self.model,
                exemplars=exemplars or [],
            )
            if fallback:
                history.append({"step": cfg.max_steps, "phase": "fallback", "sql": fallback})
                return fallback, history

        history.append({"step": cfg.max_steps, "phase": "fail", "reason": "No valid SQL found"})
        return "", history

