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
- bounded loop + bounded reflection (auditability)
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
    classify_intent,
    clean_candidate_with_reason,
    count_select_columns,
    enforce_projection_contract,
    intent_constraints,
    missing_explicit_fields,
    semantic_score,
    vanilla_candidate,
    _extract_value_hints,
)
from .postprocess import guarded_postprocess
from .query_runner import QueryRunner


@dataclass(frozen=True)
class ReactConfig:
    # Control / cost bounds.
    max_steps: int = 1
    num_cands: int = 12
    max_new_tokens: int = 96
    enable_reflection: bool = True
    reflection_num_cands: int = 4

    # Candidate diversity.
    do_sample: bool = True
    temperature: float = 0.3
    top_p: float = 0.9

    # Optional prompt controls.
    use_schema_subset: bool = True
    use_projection_contract: bool = True

    # Scoring.
    column_penalty: float = 0.5
    # Intent alignment: hard gate or soft penalty.
    enforce_intent_constraints: bool = False
    intent_penalty: float = 1.0
    # Explicit field/value gates (projection + literal filters).
    enforce_explicit_fields: bool = True
    enforce_value_hints: bool = True
    # Optional prefilter: limit how many candidates are executed per step.
    max_exec_cands: Optional[int] = 8
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
    # Schema validation
    # -----------------
    def _parse_schema_text(self, schema_text: str) -> tuple[set[str], dict[str, set[str]]]:
        tables: set[str] = set()
        table_cols: dict[str, set[str]] = {}
        if not schema_text:
            return tables, table_cols
        for line in schema_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(?is)^([a-zA-Z_][\w$]*)\s*\((.*)\)\s*$", line)
            if not m:
                continue
            table = m.group(1).strip().lower()
            cols_raw = m.group(2)
            cols = [c.strip().lower() for c in cols_raw.split(",") if c.strip()]
            tables.add(table)
            table_cols[table] = set(cols)
        return tables, table_cols

    def _schema_validate(self, *, sql: str, schema_index: tuple[set[str], dict[str, set[str]]]) -> tuple[bool, str]:
        tables, table_cols = schema_index
        if not tables:
            return True, "no_schema"

        sql_low = sql.lower()
        # Rationale: early runs failed on misspelled tables/columns; this check makes
        # those errors explicit before execution so the loop can repair them.
        # Validate explicit table names in FROM/JOIN (skip subqueries).
        for m in re.finditer(r"(?is)\b(from|join)\s+([a-zA-Z_][\w$]*)", sql_low):
            table = m.group(2)
            # Skip if this is followed by a "(" (derived table).
            after = sql_low[m.end() : m.end() + 1]
            if after == "(":
                continue
            if table not in tables:
                return False, f"unknown_table:{table}"

        # Validate qualified columns table.column when table is known.
        for m in re.finditer(r"(?is)\b([a-zA-Z_][\w$]*)\.([a-zA-Z_][\w$]*)\b", sql_low):
            table = m.group(1)
            col = m.group(2)
            if table in table_cols and col not in table_cols[table]:
                return False, f"unknown_column:{table}.{col}"

        return True, "ok"

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

    def _format_exemplars(self, exemplars: Optional[list[dict]], max_ex: int = 2) -> str:
        if not exemplars:
            return ""
        lines: list[str] = []
        for ex in exemplars[:max_ex]:
            nlq = (ex.get("nlq") or "").strip()
            sql = (ex.get("sql") or "").strip()
            if nlq and sql:
                lines.append(f"Example\nQuestion: {nlq}\nSQL: {sql}")
        return "\n\n".join(lines)

    def _build_react_prompt(self, *, nlq: str, schema_text: str, history: list[dict], observation: str) -> str:
        schema_view = build_schema_subset(schema_text, nlq) if self.cfg.use_schema_subset else schema_text
        if self.cfg.verbose:
            table_lines = [ln for ln in schema_view.splitlines() if "(" in ln and ")" in ln]
            self._debug(f"[prompt] react schema_subset={self.cfg.use_schema_subset} tables={len(table_lines)}")
        # Keep only the most recent items to limit prompt size.
        # Rationale: long histories dilute the latest error signal and increase prompt cost.
        history_text = "\n\n".join(self._format_history_item(h) for h in history[-4:]) or "None yet."
        exemplars_text = self._format_exemplars(getattr(self, "_prompt_exemplars", None))
        exemplars_block = f"Examples:\n{exemplars_text}\n\n" if exemplars_text else ""
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

{exemplars_block}Question:
{nlq}

Recent steps:
{history_text}

Last observation:
{observation}

Respond with only the final SQL statement.
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
        # Rationale: the model is less likely to start with explanations when anchored.
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
        # Rationale: keeps behavior inspectable and consistent across runs.
        if self.cfg.verbose:
            self._debug("[guard] calling guarded_postprocess")
        cleaned = guarded_postprocess(sql, nlq)
        if self.cfg.verbose and cleaned != sql:
            self._debug(f"[post] guarded_postprocess changed sql: {_trim(cleaned)}")
        if self.cfg.use_projection_contract:
            if self.cfg.verbose:
                self._debug("[guard] calling enforce_projection_contract")
            contracted = enforce_projection_contract(cleaned, nlq)
            if self.cfg.verbose and contracted != cleaned:
                self._debug(f"[post] projection_contract applied: {_trim(contracted)}")
            cleaned = contracted
        return cleaned

    def evaluate_candidate(
        self,
        *,
        nlq: str,
        raw: str,
        schema_index: tuple[set[str], dict[str, set[str]]],
    ) -> tuple[Optional[tuple[str, float]], dict]:
        if self.cfg.verbose:
            self._debug("[guard] calling clean_candidate_with_reason")
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

        # Rationale: normalize/guard before any schema or execution checks so
        # errors are attributable to the SQL itself, not formatting noise.
        sql = self.postprocess_sql(sql=sql, nlq=nlq)
        self._debug(f"[eval] postprocess sql: {_trim(sql)}")

        missing_fields = missing_explicit_fields(nlq, sql)
        if missing_fields:
            self._debug(f"[eval] missing explicit fields: {missing_fields}")
            if self.cfg.enforce_explicit_fields:
                return None, {
                    "phase": "explicit_fields_reject",
                    "sql": sql,
                    "reason": "missing_required_field",
                    "missing_fields": missing_fields,
                    "obs": f"Missing required fields: {', '.join(missing_fields)}",
                }

        if self.cfg.enforce_value_hints:
            value_hints = _extract_value_hints(nlq)
            if value_hints and not any(v in (sql or "").lower() for v in value_hints):
                return None, {
                    "phase": "value_hint_reject",
                    "sql": sql,
                    "reason": "missing_value_hint",
                    "value_hints": value_hints,
                    "obs": "Missing required value hint(s)",
                }

        ok_schema, why_schema = self._schema_validate(sql=sql, schema_index=schema_index)
        if not ok_schema:
            self._debug(f"[eval] schema_reject reason={why_schema}")
            return None, {
                "phase": "schema_reject",
                "sql": sql,
                "reason": why_schema,
                "obs": f"Schema mismatch: {why_schema}",
            }

        # Execution gate (Act): must run successfully.
        # Rationale: invalid SQL should not be scored as "correct" in EX/TS.
        if self.cfg.verbose:
            self._debug("[guard] calling runner.run (execution gate)")
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
        # Rationale: a runnable query can still be semantically wrong (e.g., missing COUNT).
        if self.cfg.verbose:
            self._debug("[guard] calling intent_constraints")
        ok, why = intent_constraints(nlq, sql)
        intent_penalty = 0.0
        if not ok:
            if self.cfg.enforce_intent_constraints:
                self._debug(f"[eval] intent_reject reason={why}")
                return None, {
                    "phase": "intent_reject",
                    "sql": sql,
                    "reason": why,
                    "obs": f"Intent mismatch: {why}",
                }
            intent_penalty = float(self.cfg.intent_penalty)
            self._debug(f"[eval] intent_soft reason={why} penalty={intent_penalty:.2f}")

        # Simple, auditable scoring.
        # Rationale: lightweight heuristics beat "first candidate wins" without hiding logic.
        s_sem = semantic_score(nlq, sql)
        s_cols = count_select_columns(sql)
        s_extra = float(self.extra_score_fn(nlq, sql)) - float(intent_penalty)
        score = float(s_sem) - float(self.cfg.column_penalty) * float(s_cols) + s_extra
        self._debug(f"[eval] accept score={score:.2f} sem={s_sem:.2f} cols={s_cols} extra={s_extra:.2f}")

        log = {
            "phase": "accept",
            "sql": sql,
            "score": score,
            "sem": s_sem,
            "cols": s_cols,
            "extra": s_extra,
            "intent_ok": ok,
            "intent_reason": why,
        }
        if missing_fields:
            log["missing_fields"] = missing_fields
        obs_parts = []
        if missing_fields:
            obs_parts.append(f"Missing requested fields: {', '.join(missing_fields)}")
        if not ok and not self.cfg.enforce_intent_constraints:
            log["intent_penalty"] = intent_penalty
            obs_parts.append(f"Intent mismatch (soft): {why}")
        if obs_parts:
            log["obs"] = "; ".join(obs_parts)
        return (sql, score), log

    def _prefilter_candidates(self, *, nlq: str, raw_cands: list[str]) -> list[str]:
        cfg = self.cfg
        if not raw_cands:
            return raw_cands
        if cfg.max_exec_cands is None or cfg.max_exec_cands <= 0:
            return raw_cands
        if len(raw_cands) <= cfg.max_exec_cands:
            return raw_cands

        # Rationale: limit expensive execution checks to plausible candidates.
        scored: list[tuple[float, int, str]] = []
        for idx, raw in enumerate(raw_cands):
            sql, reason = clean_candidate_with_reason(raw)
            if not sql:
                scored.append((-1e9, idx, raw))
                continue
            # Use a lightweight, execution-free score to rank candidates.
            # Postprocess is deterministic and keeps ranking consistent with later evaluation.
            sql = self.postprocess_sql(sql=sql, nlq=nlq)
            s_sem = semantic_score(nlq, sql)
            s_cols = count_select_columns(sql)
            pre_score = float(s_sem) - float(self.cfg.column_penalty) * float(s_cols)
            scored.append((pre_score, idx, raw))

        scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        keep = [raw for _, _, raw in scored[: cfg.max_exec_cands]]
        return keep

    # -----------------
    # Reflection
    # -----------------
    def reflect_sql(self, *, nlq: str, bad_sql: str, error_msg: str, schema_text: str, validation: Optional[dict] = None) -> tuple[Optional[str], dict]:
        if not self.cfg.enable_reflection:
            return None, {"enabled": False}

        self._debug(f"[reflect] starting: bad_sql={_trim(bad_sql)} error={_trim(error_msg)}")
        intent = classify_intent(nlq)
        schema_index = self._parse_schema_text(schema_text)

        err_hint = ""
        # Rationale: MySQL error codes give targeted repair hints without guessing semantics.
        m = re.search(r"\((\d+),", str(error_msg or ""))
        if m:
            code = m.group(1)
            if code == "1064":
                err_hint = "Fix SQL syntax only; remove stray keywords or duplicated clauses."
            elif code == "1054":
                err_hint = "Unknown column: use only columns present in the schema tables."
            elif code == "1146":
                err_hint = "Unknown table: use only tables present in the schema."
            elif code == "1052":
                err_hint = "Ambiguous column: qualify with table alias."
            elif code == "1055":
                err_hint = "GROUP BY mismatch: either aggregate or include all non-aggregated columns in GROUP BY."

        validation_block = ""
        if validation:
            phase = validation.get("phase")
            reason = validation.get("reason") or validation.get("error") or ""
            obs = validation.get("obs") or ""
            validation_block = f"""
Validation findings:
- phase: {phase}
- detail: {reason}
- observation: {obs}
""".rstrip()

        prompt = f"""
You are an expert MySQL engineer.

Schema:
{schema_text}

User question:
{nlq}

Detected intent (heuristic):
{intent}

Invalid SQL:
{bad_sql}

Database error:
{error_msg}

Fix guidance:
{err_hint if err_hint else "Resolve the error while preserving the question intent."}

{validation_block}

Fix the SQL so it is valid MySQL and answers the question.
Output ONLY the corrected SELECT statement.
""".strip()

        fixes = self.generate_candidates(prompt, num=self.cfg.reflection_num_cands, do_sample=True)
        if not fixes:
            self._debug("[reflect] no fixes generated")
            return None, {"enabled": True, "status": "no_fix_generated"}
        self._debug(f"[reflect] generated fixes={len(fixes)}")

        last_info: dict = {"enabled": True, "status": "no_valid_fix"}
        for raw_fix in fixes:
            cand, _reason = clean_candidate_with_reason(raw_fix)
            if not cand:
                self._debug("[reflect] clean_reject on fix")
                continue

            sql = self.postprocess_sql(sql=cand, nlq=nlq)
            ok_schema, why_schema = self._schema_validate(sql=sql, schema_index=schema_index)
            if not ok_schema:
                self._debug(f"[reflect] schema_reject on fix: {why_schema}")
                last_info = {"enabled": True, "status": "schema_reject", "raw_fix": _trim(raw_fix), "fixed_sql": sql, "reason": why_schema}
                continue
            meta = self.runner.run(sql, capture_df=False)
            if not meta.success:
                self._debug(f"[reflect] exec_fail on fix: {_trim(meta.error)}")
                last_info = {"enabled": True, "status": "exec_fail", "raw_fix": _trim(raw_fix), "fixed_sql": sql, "exec_error": _trim(meta.error)}
                continue

            ok, why = intent_constraints(nlq, sql)
            if not ok:
                self._debug(f"[reflect] intent_reject on fix: {why}")
                last_info = {"enabled": True, "status": "intent_reject", "raw_fix": _trim(raw_fix), "fixed_sql": sql, "reason": why}
                continue

            self._debug(f"[reflect] accepted fix: {_trim(sql)}")
            return sql, {"enabled": True, "status": "exec_ok", "raw_fix": _trim(raw_fix), "fixed_sql": sql}

        self._debug("[reflect] no valid fix accepted")
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
        last_failure: Optional[dict] = None
        last_failure_rank = -1
        # Rationale: prioritize failures that are most informative for repair.
        failure_rank = {"exec_fail": 3, "schema_reject": 2, "intent_reject": 1}
        best_overall: Optional[tuple[str, float]] = None
        best_overall_info: Optional[dict] = None
        # Store exemplars on the agent so prompt builders can include them.
        # Rationale: a small number of examples improves structure without overfitting.
        self._prompt_exemplars = exemplars or []
        schema_index = self._parse_schema_text(schema_text)

        self._debug(
            f"[start] steps={cfg.max_steps} num_cands={cfg.num_cands} "
            f"schema_subset={cfg.use_schema_subset} "
            f"projection_contract={cfg.use_projection_contract} accept_score={cfg.accept_score} "
            f"intent_gate={'hard' if cfg.enforce_intent_constraints else 'soft'} "
            f"intent_penalty={cfg.intent_penalty} max_exec_cands={cfg.max_exec_cands}"
        )

        for step in range(cfg.max_steps):
            self._debug(f"[step {step}] obs={_trim(observation)} history={len(history)}")
            prompt = self._build_react_prompt(
                nlq=nlq,
                schema_text=schema_text,
                history=history,
                observation=observation,
            )
            per_prompt = max(1, cfg.num_cands)
            raw_cands: list[str] = []
            self._debug(f"[step {step}] generating greedy candidates per_prompt=1")
            raw_cands.extend(self.generate_candidates(prompt, num=1, do_sample=False))
            if cfg.do_sample and per_prompt > 1:
                self._debug(f"[step {step}] generating sampled candidates per_prompt={per_prompt - 1}")
                raw_cands.extend(self.generate_candidates(prompt, num=per_prompt - 1, do_sample=True))

            self._debug(f"[step {step}] total candidates={len(raw_cands)}")
            prefiltered = self._prefilter_candidates(nlq=nlq, raw_cands=raw_cands)
            if len(prefiltered) != len(raw_cands):
                self._debug(f"[step {step}] prefilter kept={len(prefiltered)}")
            raw_cands = prefiltered
            best: Optional[tuple[str, float]] = None
            best_info: Optional[dict] = None

            for idx, raw in enumerate(raw_cands, start=1):
                self._debug(f"[step {step}] candidate {idx}/{len(raw_cands)}")
                result, log = self.evaluate_candidate(nlq=nlq, raw=raw, schema_index=schema_index)
                log = {"step": step, **log}
                history.append({k: _trim(v) for k, v in log.items()})

                phase = log.get("phase")
                if phase in failure_rank:
                    rank = failure_rank[phase]  # type: ignore[index]
                    if rank >= last_failure_rank:
                        last_failure_rank = rank
                        last_failure = log

                if result is not None:
                    sql, score = result
                    if best is None or score > best[1]:
                        best = (sql, score)
                        best_info = log
                    if best_overall is None or score > best_overall[1]:
                        best_overall = (sql, score)
                        best_overall_info = log

            if best is not None:
                sql, score = best
                # Optional multi-step refinement: only stop early if the score clears a threshold.
                # Rationale: gives the loop a chance to correct near-miss candidates.
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

            # Optional reflection: use validation feedback, then re-run gates.
            if cfg.enable_reflection and last_failure and last_failure.get("sql"):
                error_msg = last_failure.get("error") or last_failure.get("reason") or last_failure.get("obs") or ""
                reflected, repinfo = self.reflect_sql(
                    nlq=nlq,
                    bad_sql=last_failure["sql"],
                    error_msg=str(error_msg),
                    schema_text=schema_text,
                    validation=last_failure,
                )
                history.append({"step": step, "phase": "reflection", "bad_sql": last_failure["sql"], "error": _trim(error_msg), **repinfo})
                if reflected:
                    self._debug(f"[step {step}] reflection accepted")
                    history.append({"step": step, "phase": "final", "sql": reflected, "score": "reflection_accept"})
                    return reflected, history
                observation = _trim(
                    f"Previous SQL failed validation: {error_msg}. Revise tables/joins and try again."
                )
            else:
                observation = "No executable candidates. Try a simpler join path."

            history.append({"step": step, "phase": "observation", "obs": _trim(observation)})

        # Deterministic fallback (few-shot baseline) if provided.
        if best_overall is not None:
            sql, score = best_overall
            note = "best_overall_below_threshold"
            if best_overall_info and best_overall_info.get("missing_fields"):
                note += f" missing_fields={','.join(best_overall_info['missing_fields'])}"
            history.append({"step": cfg.max_steps, "phase": "final", "sql": sql, "score": score, "note": note})
            return sql, history

        if schema_summary is not None:
            fallback = vanilla_candidate(
                nlq=nlq,
                schema_summary=schema_summary,
                tok=self.tok,
                model=self.model,
                exemplars=exemplars or [],
            )
            if fallback:
                result, log = self.evaluate_candidate(nlq=nlq, raw=fallback, schema_index=schema_index)
                log = {"step": cfg.max_steps, "source": "fallback", **log}
                history.append({k: _trim(v) for k, v in log.items()})
                if result is not None:
                    sql, score = result
                    self._debug("[fallback] using validated baseline candidate")
                    history.append({"step": cfg.max_steps, "phase": "final", "sql": sql, "score": score, "source": "fallback"})
                    return sql, history

        history.append({"step": cfg.max_steps, "phase": "fail", "reason": "No valid SQL found"})
        return "", history
