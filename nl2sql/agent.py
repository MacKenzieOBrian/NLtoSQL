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
    missing_explicit_fields,
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
    # Optional early-stop threshold for multi-step refinement.
    # If None, the loop returns the best executable candidate per step (current behavior).
    accept_score: Optional[float] = None
    # Verbose tracing for debugging the full loop.
    verbose: bool = False


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

    def _debug(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg)

    # -----------------
    # Prompt builders
    # -----------------
    def _format_history_item(self, h: dict) -> str:
        # ReAct alignment: treat each history entry as (Action, Observation).
        # We log rich fields in the trace; this formatter keeps prompts compact and auditable.
        action = h.get("sql") or h.get("raw") or h.get("phase") or ""
        obs = h.get("obs") or h.get("error") or h.get("reason") or ""
        step = h.get("step")
        prefix = f"Step {step} " if step is not None else ""
        return f"{prefix}Action: {_trim(action)}\nObservation: {_trim(obs)}"

    def _build_react_prompt(self, *, nlq: str, schema_text: str, history: list[dict], observation: str) -> str:
        schema_view = build_schema_subset(schema_text, nlq) if self.cfg.use_schema_subset else schema_text
        if self.cfg.verbose:
            table_lines = [ln for ln in schema_view.splitlines() if "(" in ln and ")" in ln]
            self._debug(f"[prompt] react schema_subset={self.cfg.use_schema_subset} tables={len(table_lines)}")
        # Keep only the most recent items to limit prompt size.
        history_text = "\n\n".join(self._format_history_item(h) for h in history[-4:]) or "None yet."
        return f"""
You are an expert MySQL analyst.

TASK:
- Write exactly ONE valid MySQL SELECT statement.
- Output only SQL (no explanation, no markdown).
- The output must include a FROM clause.
- Use only schema tables/columns.
- Use ORDER BY/LIMIT only if explicitly asked.
- Use the Observation to correct mistakes from prior attempts.

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
        if self.cfg.verbose:
            table_lines = [ln for ln in schema_view.splitlines() if "(" in ln and ")" in ln]
            self._debug(f"[prompt] tabular schema_subset={self.cfg.use_schema_subset} tables={len(table_lines)}")
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

        self._debug(
            f"[gen] do_sample={do_sample} num={num} max_new_tokens={cfg.max_new_tokens} "
            f"temp={cfg.temperature} top_p={cfg.top_p} prompt_tokens={inputs.input_ids.shape[-1]}"
        )

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
        if self.cfg.verbose and cleaned != sql:
            self._debug(f"[post] guarded_postprocess changed sql: {_trim(cleaned)}")
        if self.cfg.use_projection_contract:
            contracted = enforce_projection_contract(cleaned, nlq)
            if self.cfg.verbose and contracted != cleaned:
                self._debug(f"[post] projection_contract applied: {_trim(contracted)}")
            cleaned = contracted
        return cleaned

    def evaluate_candidate(self, *, nlq: str, raw: str) -> tuple[Optional[tuple[str, float]], dict]:
        self._debug(f"[eval] raw candidate: {_trim(raw)}")
        sql, reason = clean_candidate_with_reason(raw)
        if not sql:
            self._debug(f"[eval] clean_reject reason={reason}")
            return None, {
                "phase": "clean_reject",
                "reason": reason,
                "raw": _trim(raw),
                "obs": f"Rejected during cleanup: {reason}",
            }

        sql = self.postprocess_sql(sql=sql, nlq=nlq)
        self._debug(f"[eval] postprocess sql: {_trim(sql)}")

        missing_fields = missing_explicit_fields(nlq, sql)
        if missing_fields:
            self._debug(f"[eval] missing explicit fields: {missing_fields}")

        # Execution gate (Act): must run successfully.
        meta = self.runner.run(sql, capture_df=False)
        if not meta.success:
            err = _trim(meta.error)
            self._debug(f"[eval] exec_fail error={err}")
            return None, {
                "phase": "exec_fail",
                "sql": sql,
                "error": err,
                "obs": f"Execution error: {err}",
            }

        # Intent gate: prevent executable-but-wrong-shape answers.
        ok, why = intent_constraints(nlq, sql)
        if not ok:
            self._debug(f"[eval] intent_reject reason={why}")
            return None, {
                "phase": "intent_reject",
                "sql": sql,
                "reason": why,
                "obs": f"Intent mismatch: {why}",
            }

        # Simple, auditable scoring.
        s_sem = semantic_score(nlq, sql)
        s_cols = count_select_columns(sql)
        s_extra = float(self.extra_score_fn(nlq, sql))
        score = float(s_sem) - float(self.cfg.column_penalty) * float(s_cols) + s_extra
        self._debug(f"[eval] accept score={score:.2f} sem={s_sem:.2f} cols={s_cols} extra={s_extra:.2f}")

        log = {"phase": "accept", "sql": sql, "score": score, "sem": s_sem, "cols": s_cols, "extra": s_extra}
        if missing_fields:
            log["missing_fields"] = missing_fields
            log["obs"] = f"Missing requested fields: {', '.join(missing_fields)}"
        return (sql, score), log

    # -----------------
    # Repair
    # -----------------
    def repair_sql(self, *, nlq: str, bad_sql: str, error_msg: str, schema_text: str) -> tuple[Optional[str], dict]:
        if not self.cfg.enable_repair:
            return None, {"enabled": False}

        self._debug(f"[repair] starting: bad_sql={_trim(bad_sql)} error={_trim(error_msg)}")

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
            self._debug("[repair] no fixes generated")
            return None, {"enabled": True, "status": "no_fix_generated"}
        self._debug(f"[repair] generated fixes={len(fixes)}")

        last_info: dict = {"enabled": True, "status": "no_valid_fix"}
        for raw_fix in fixes:
            cand, _reason = clean_candidate_with_reason(raw_fix)
            if not cand:
                self._debug("[repair] clean_reject on fix")
                continue

            sql = self.postprocess_sql(sql=cand, nlq=nlq)
            meta = self.runner.run(sql, capture_df=False)
            if not meta.success:
                self._debug(f"[repair] exec_fail on fix: {_trim(meta.error)}")
                last_info = {"enabled": True, "status": "exec_fail", "raw_fix": _trim(raw_fix), "fixed_sql": sql, "exec_error": _trim(meta.error)}
                continue

            ok, why = intent_constraints(nlq, sql)
            if not ok:
                self._debug(f"[repair] intent_reject on fix: {why}")
                last_info = {"enabled": True, "status": "intent_reject", "raw_fix": _trim(raw_fix), "fixed_sql": sql, "reason": why}
                continue

            self._debug(f"[repair] accepted fix: {_trim(sql)}")
            return sql, {"enabled": True, "status": "exec_ok", "raw_fix": _trim(raw_fix), "fixed_sql": sql}

        self._debug("[repair] no valid fix accepted")
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

        self._debug(
            f"[start] steps={cfg.max_steps} num_cands={cfg.num_cands} "
            f"tabular={cfg.use_tabular_prompt} schema_subset={cfg.use_schema_subset} "
            f"projection_contract={cfg.use_projection_contract} accept_score={cfg.accept_score}"
        )

        for step in range(cfg.max_steps):
            self._debug(f"[step {step}] obs={_trim(observation)} history={len(history)}")
            prompts = [self._build_react_prompt(nlq=nlq, schema_text=schema_text, history=history, observation=observation)]
            if cfg.use_tabular_prompt:
                prompts.append(self._build_tabular_prompt(nlq=nlq, schema_text=schema_text))

            self._debug(f"[step {step}] prompts_used={len(prompts)}")
            per_prompt = max(1, cfg.num_cands // len(prompts))
            raw_cands: list[str] = []
            for p in prompts:
                self._debug(f"[step {step}] generating candidates per_prompt={per_prompt}")
                raw_cands.extend(self.generate_candidates(p, num=per_prompt, do_sample=cfg.do_sample))

            self._debug(f"[step {step}] total candidates={len(raw_cands)}")
            best: Optional[tuple[str, float]] = None
            best_info: Optional[dict] = None

            for idx, raw in enumerate(raw_cands, start=1):
                self._debug(f"[step {step}] candidate {idx}/{len(raw_cands)}")
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
                        best_info = log

            if best is not None:
                sql, score = best
                # Optional multi-step refinement: only stop early if the score clears a threshold.
                if cfg.accept_score is None or score >= cfg.accept_score:
                    self._debug(f"[step {step}] accept final score={score:.2f}")
                    history.append({"step": step, "phase": "final", "sql": sql, "score": score})
                    return sql, history
                missing_note = ""
                if best_info and best_info.get("missing_fields"):
                    missing_note = f" Missing fields: {', '.join(best_info['missing_fields'])}."
                observation = _trim(
                    f"Best candidate scored {score:.2f} (below threshold). "
                    f"Re-evaluate joins/filters.{missing_note} Candidate: {sql}"
                )
                history.append({"step": step, "phase": "observation", "obs": observation})
                continue

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
                    self._debug(f"[step {step}] repair accepted")
                    history.append({"step": step, "phase": "final", "sql": repaired, "score": "repair_accept"})
                    return repaired, history
                observation = _trim(
                    f"Previous SQL failed: {last_error}. Revise tables/joins and try again."
                )
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
                self._debug("[fallback] using deterministic baseline candidate")
                history.append({"step": cfg.max_steps, "phase": "fallback", "sql": fallback})
                return fallback, history

        history.append({"step": cfg.max_steps, "phase": "fail", "reason": "No valid SQL found"})
        return "", history
