#!/usr/bin/env python3
"""
End-to-end runner for baseline prompting, QLoRA adapter eval, and a small ReAct
sanity check. This mirrors the notebook flows (02/05/03) but as a CLI.

Refs/inspiration:
- HF Transformers + BitsAndBytes 4-bit NF4 load:
  https://huggingface.co/docs/transformers/main_classes/quantization
- PEFT/QLoRA adapter loading:
  https://huggingface.co/docs/peft/
- Cloud SQL connector + SQLAlchemy creator:
  https://cloud.google.com/sql/docs/mysql/connect-run
  https://docs.sqlalchemy.org/en/20/core/engines.html#custom-dbapi-connect
- ReAct pattern (Yao et al., 2023): simple Thought/Action/Obs loop for SQL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from nl2sql.db import create_engine_with_connector
from nl2sql.eval import eval_run, execution_accuracy
from nl2sql.postprocess import guarded_postprocess, normalize_sql
from nl2sql.prompting import make_few_shot_messages
from nl2sql.query_runner import QueryRunner
from nl2sql.schema import build_schema_summary
from nl2sql.agent_utils import (
    clean_candidate,
    build_tabular_prompt,
    vanilla_candidate,
    classify_error,
    error_hint,
    semantic_score,
    count_select_columns,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline + QLoRA + ReAct evals.")
    p.add_argument("--mode", choices=["all", "baseline", "qlora", "react"], default="all")
    p.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter-path", default="results/adapters/qlora_classicmodels")
    p.add_argument("--test-set", default="data/classicmodels_test_200.json")
    p.add_argument("--db-name", default="classicmodels")
    p.add_argument("--limit", type=int, default=200, help="Limit for baseline/QLoRA eval.")
    p.add_argument("--react-limit", type=int, default=5, help="Small slice for ReAct sanity check.")
    p.add_argument("--k", type=int, default=3, help="Few-shot k for baseline/QLoRA.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit load (uses full precision).")
    return p.parse_args()


def load_test_set(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in test set, got {type(data)}")
    return data


def load_engine(db_name: str):
    icn = os.getenv("INSTANCE_CONNECTION_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    if not all([icn, user, password]):
        raise RuntimeError("Set INSTANCE_CONNECTION_NAME, DB_USER, DB_PASS env vars.")
    engine, connector = create_engine_with_connector(
        instance_connection_name=icn,
        user=user,
        password=password,
        db_name=db_name,
    )
    return engine, connector


def build_schema(engine, db_name: str) -> str:
    return build_schema_summary(engine, db_name=db_name)


def load_model_and_tok(model_id: str, *, load_in_4bit: bool, device_map: str):
    bnb_config = None
    dtype = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(model_id, token=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=dtype,
        token=True,
    )
    # Deterministic decoding for eval
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    return model, tok


def run_baseline(
    *,
    test_set: list[dict[str, Any]],
    schema_summary: str,
    engine,
    model,
    tok,
    limit: int,
    seed: int,
    k: int,
) -> None:
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    run_metadata = {"method": "baseline", "model_id": model.name_or_path}
    eval_run(
        test_set=test_set,
        exemplar_pool=test_set,
        k=0,
        limit=limit,
        seed=seed,
        engine=engine,
        model=model,
        tokenizer=tok,
        schema_summary=schema_summary,
        save_path="results/baseline/results_zero_shot.json",
        run_metadata=run_metadata,
        avoid_exemplar_leakage=True,
    )
    eval_run(
        test_set=test_set,
        exemplar_pool=test_set,
        k=k,
        limit=limit,
        seed=seed,
        engine=engine,
        model=model,
        tokenizer=tok,
        schema_summary=schema_summary,
        save_path=f"results/baseline/results_few_shot_k{k}.json",
        run_metadata=run_metadata,
        avoid_exemplar_leakage=True,
    )


def wrap_peft(base_model, adapter_path: str):
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    return PeftModel.from_pretrained(base_model, adapter_path)


def run_qlora_eval(
    *,
    test_set: list[dict[str, Any]],
    schema_summary: str,
    engine,
    model,
    tok,
    limit: int,
    seed: int,
    k: int,
    adapter_path: str,
) -> None:
    Path("results/qlora").mkdir(parents=True, exist_ok=True)
    eval_model = wrap_peft(model, adapter_path)
    run_metadata = {
        "method": "qlora",
        "model_id": model.name_or_path,
        "adapter_path": adapter_path,
    }
    eval_run(
        test_set=test_set,
        exemplar_pool=test_set,
        k=0,
        limit=limit,
        seed=seed,
        engine=engine,
        model=eval_model,
        tokenizer=tok,
        schema_summary=schema_summary,
        save_path="results/qlora/results_zero_shot.json",
        run_metadata=run_metadata,
        avoid_exemplar_leakage=True,
    )
    eval_run(
        test_set=test_set,
        exemplar_pool=test_set,
        k=k,
        limit=limit,
        seed=seed,
        engine=engine,
        model=eval_model,
        tokenizer=tok,
        schema_summary=schema_summary,
        save_path=f"results/qlora/results_few_shot_k{k}.json",
        run_metadata=run_metadata,
        avoid_exemplar_leakage=True,
    )


# --- ReAct (lightweight, small-slice sanity check) ---
PROMPT_INSTR = (
    "You are an expert SQL agent. Only output a single SELECT statement "
    "against the ClassicModels schema below. No DDL/DML, no comments. "
    "Use the previous observation to fix errors. Reply with SQL only."
)


def build_react_prompt(nlq: str, schema_text: str, history: list[dict[str, str]], observation: str) -> str:
    history_text = "\n".join([f"Thought/Action: {h['ta']}\nObservation: {h['obs']}" for h in history]) or "None."
    return f"""
You are an expert MySQL analyst.

TASK:
- Output ONE and ONLY ONE valid MySQL SELECT statement.
- Do NOT explain or comment.
- The output MUST start with SELECT.
- If unsure, still output your best single SELECT.

Schema:
{schema_text}

Question: {nlq}

Previous trace:
{history_text}
Last observation: {observation}

Return only the final SELECT statement.
"""


def _generate_candidates(prompt: str, model, tok, num: int = 2, do_sample: bool = True) -> list[str]:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=192,
        do_sample=do_sample,
        temperature=0.5,
        top_p=0.9,
        num_return_sequences=num,
    )
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    cands = []
    for i in range(num):
        gen_ids = out[i][inputs.input_ids.shape[-1]:]
        gen = tok.decode(gen_ids, skip_special_tokens=True)
        cands.append(gen)
    return cands


def react_sql(
    nlq: str,
    schema_text: str,
    model,
    tok,
    runner: QueryRunner,
    max_steps: int = 3,
    num_cands: int = 4,
    exemplars: list[dict] | None = None,
):
    history: list[dict[str, str]] = []
    observation = "Start."
    best_sql = None
    best_score = float("-inf")

    for _ in range(max_steps):
        # diversify prompts
        raw_cands = []
        raw_cands += _generate_candidates(
            build_react_prompt(nlq, schema_text, history, observation),
            model,
            tok,
            num=max(1, num_cands // 2),
            do_sample=True,
        )
        raw_cands += _generate_candidates(
            build_tabular_prompt(nlq, schema_text),
            model,
            tok,
            num=num_cands - len(raw_cands),
            do_sample=True,
        )

        ranked: list[str] = []
        for raw in raw_cands:
            sql = clean_candidate(raw)
            if not sql:
                history.append({"ta": raw, "obs": "Rejected: not a pure SELECT"})
                continue
            sql = guarded_postprocess(sql, nlq)
            ranked.append(sql)

        step_success = False
        last_error = None

        for sql in ranked:
            try:
                meta = runner.run(sql)
                if not meta.success:
                    raise ValueError(meta.error or "exec failed")
                score = semantic_score(nlq, sql) - 0.2 * count_select_columns(sql)
                if score > best_score:
                    best_score = score
                    best_sql = sql
                history.append({"ta": sql, "obs": "SUCCESS"})
                step_success = True
            except Exception as e:
                last_error = str(e)
                kind = classify_error(last_error)
                history.append({"ta": sql, "obs": f"ERROR ({kind}): {last_error}"})
                # single repair attempt
                hint = error_hint(kind, last_error)
                repair_prompt = build_react_prompt(
                    nlq,
                    schema_text,
                    history,
                    f"{last_error}. {hint}",
                )
                repair_raw = _generate_candidates(repair_prompt, model, tok, num=1, do_sample=True)[0]
                repaired = clean_candidate(repair_raw)
                if repaired:
                    try:
                        meta2 = runner.run(repaired)
                        if meta2.success:
                            score = semantic_score(nlq, repaired) - 0.2 * count_select_columns(repaired)
                            if score > best_score:
                                best_score = score
                                best_sql = repaired
                            history.append({"ta": repaired, "obs": "SUCCESS (repair)"})
                            step_success = True
                            continue
                    except Exception as e2:
                        history.append({"ta": repaired, "obs": f"Repair ERROR: {e2}"})

        if step_success and best_sql:
            observation = "SUCCESS"
            break
        else:
            observation = f"ERROR: {last_error or 'all candidates failed'}"

    # deterministic fallback if nothing worked
    if best_sql is None:
        fallback_sql = vanilla_candidate(nlq, schema_text, tok, model, exemplars=exemplars or [])
        if fallback_sql:
            best_sql = fallback_sql
            best_score = semantic_score(nlq, best_sql)
            history.append({"ta": "fallback:few-shot", "obs": "USED"})

    return best_sql, history


def run_react_small(
    *,
    test_set: list[dict[str, Any]],
    schema_summary: str,
    engine,
    model,
    tok,
    limit: int,
    adapter_path: str | None,
) -> None:
    Path("results/agent").mkdir(parents=True, exist_ok=True)
    eval_model = model if adapter_path is None else wrap_peft(model, adapter_path)
    runner = QueryRunner(engine, max_rows=1000)

    items = test_set[:limit]
    results = []
    for i, sample in enumerate(items, start=1):
        nlq = sample["nlq"]
        gold_sql = sample["sql"]
        pred_sql, trace = react_sql(
            nlq,
            schema_summary,
            eval_model,
            tok,
            runner,
            max_steps=3,
            num_cands=4,
            exemplars=test_set[:3],
        )
        va = 0
        em = int(normalize_sql(pred_sql) == normalize_sql(gold_sql))
        ex = 0
        try:
            runner.run(pred_sql)
            va = 1
            ex = int(execution_accuracy(engine=engine, pred_sql=pred_sql, gold_sql=gold_sql)[0])
        except Exception:
            va = 0
        results.append(
            {
                "nlq": nlq,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "va": va,
                "em": em,
                "ex": ex,
                "trace": trace,
            }
        )
        print(f"[ReAct] {i}/{len(items)} VA={va} EM={em}")

    save_path = Path("results/agent/results_react_small.json")
    save_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", save_path)


def main() -> None:
    args = parse_args()
    test_set = load_test_set(args.test_set)
    engine, connector = load_engine(args.db_name)
    schema_summary = build_schema(engine, args.db_name)

    load_in_4bit = not args.no_4bit
    model, tok = load_model_and_tok(args.model_id, load_in_4bit=load_in_4bit, device_map=args.device_map)

    try:
        if args.mode in ("all", "baseline"):
            print("Running baseline...")
            run_baseline(
                test_set=test_set,
                schema_summary=schema_summary,
                engine=engine,
                model=model,
                tok=tok,
                limit=args.limit,
                seed=args.seed,
                k=args.k,
            )

        if args.mode in ("all", "qlora"):
            print("Running QLoRA eval...")
            run_qlora_eval(
                test_set=test_set,
                schema_summary=schema_summary,
                engine=engine,
                model=model,
                tok=tok,
                limit=args.limit,
                seed=args.seed,
                k=args.k,
                adapter_path=args.adapter_path,
            )

        if args.mode in ("all", "react"):
            print("Running ReAct small-slice sanity check...")
            run_react_small(
                test_set=test_set,
                schema_summary=schema_summary,
                engine=engine,
                model=model,
                tok=tok,
                limit=args.react_limit,
                adapter_path=args.adapter_path if args.mode != "baseline" else None,
            )
    finally:
        connector.close()


if __name__ == "__main__":
    main()
