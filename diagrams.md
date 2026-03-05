# System Diagrams

Six Mermaid diagrams for the dissertation technical explanation.
Paste each fenced block into your LaTeX/Word tool or a Mermaid live renderer.

---

## 1. ReAct Loop — `run_react_pipeline`

The Reason→Act→Observe cycle from Yao et al. (2023).
Each box is one traced step recorded in the JSON trace list.
`repairs_used` is a single shared budget across validation and execution repairs (max_repairs=2).

```mermaid
flowchart TD
    START([NLQ input]) --> SCHEMA[Step 1: get_schema\nensure_schema_text from AgentContext]
    SCHEMA --> GEN[Step 2: generate_sql\nfew-shot k=3 prompt + LLM + extract SELECT]

    GEN --> LOOP{step < max_steps=8?}
    LOOP -- No --> EXHAUST([STOP: max_steps_exhausted\nreturn current_sql])

    LOOP -- Yes --> VSQL[validate_sql\nSELECT present? table names in schema?]

    VSQL -- invalid --> VREPAIR{repairs_used\n< max_repairs=2?}
    VREPAIR -- No --> VSTOP([STOP: validation_failed\nreturn current_sql])
    VREPAIR -- Yes --> REP1[repair_sql\nzero-shot: SQL_REPAIR_SYSTEM_PROMPT\n+ schema + error hint]
    REP1 --> INC1[repairs_used++]
    INC1 --> LOOP

    VSQL -- valid --> RUN[run_sql\nctx.runner.run — live DB execution]

    RUN -- success --> SUCCESS([STOP: success\nreturn current_sql])

    RUN -- runtime error --> RREPAIR{repairs_used\n< max_repairs=2?}
    RREPAIR -- No --> RSTOP([STOP: execution_failed\nreturn current_sql])
    RREPAIR -- Yes --> REP2[repair_sql\nexecution-guided repair\nDIN-SQL approach]
    REP2 --> INC2[repairs_used++]
    INC2 --> LOOP
```

---

## 2. Baseline Evaluation Pipeline — `eval_run`

One full pass over the 200-item benchmark for a single condition (model × k × seed).

```mermaid
flowchart TD
    INIT([model · tokenizer · k · seed\ntest_set · schema_summary · EvalRunConfig]) --> LOOP[For each of 200 items]

    LOOP --> POOL[_build_item_pool\nexclude current NLQ to prevent leakage]
    POOL --> SAMPLE{k > 0?}
    SAMPLE -- Yes\nfew-shot --> DRAW[_sample_exemplars\nglobal sequential RNG seeded per run]
    SAMPLE -- No\nzero-shot --> EMPTY[exemplars = empty list]
    DRAW --> MSGS[make_few_shot_messages\nsystem prompt + k examples + NLQ]
    EMPTY --> MSGS

    MSGS --> GEN[generate_sql_from_messages\napply_chat_template + model.generate\ndecode + raw_sql]

    GEN --> LAYER{EvalRunConfig\nreliability flags?}
    LAYER -- all False\nprimary path --> PRED[pred_sql = raw_sql]
    LAYER -- extension only --> OPT[_clean_sql\nsql_guardrails + guarded_postprocess]
    OPT --> PRED

    PRED --> VA[qr.run pred_sql\nValidation Accuracy]
    PRED --> EM[normalize_sql comparison\nExact Match]
    PRED --> EX[execution_accuracy\nCounter bag comparison\npred rows == gold rows?]

    VA -- success AND ts_enabled --> TS[test_suite_accuracy_for_item\nN perturbed databases]
    VA -- no TS --> TSN[ts = None]

    VA --> ITEM[EvalItem frozen dataclass\nva · em · ex · ts · raw_sql · pred_sql · error]
    EM --> ITEM
    EX --> ITEM
    TS --> ITEM
    TSN --> ITEM

    ITEM --> NEXT{More items?}
    NEXT -- Yes --> LOOP
    NEXT -- No --> AGG[_summarize_eval\nAggregate VA · EM · EX · TS rates\nSerialise to JSON]
```

---

## 3. Module Architecture

Static dependency map — arrows show imports.

```mermaid
graph LR
    subgraph Notebooks
        NB02[02_baseline_prompting_eval]
        NB03[03_agentic_eval]
        NB05[05_qlora_train_eval]
        NB06[06_research_comparison]
    end

    subgraph scripts
        GEN[generate_research_comparison.py]
    end

    subgraph nl2sql/evaluation
        EVAL[eval.py\neval_run · execution_accuracy\ntest_suite_accuracy_for_item\nEvalRunConfig · EvalItem]
    end

    subgraph nl2sql/agent
        REACT[react_pipeline.py\nrun_react_pipeline · generate_sql\nrepair_sql · evaluate_react_ablation]
        AT[agent_tools.py\nAgentContext · get_agent_context\nensure_schema_text · schema_to_text]
        PROMPTS[prompts.py\nSQL_GENERATOR_SYSTEM_PROMPT\nSQL_REPAIR_SYSTEM_PROMPT]
    end

    subgraph nl2sql/core
        LLM[llm.py\ngenerate_sql_from_messages\nextract_first_select]
        DB[db.py\nsafe_connection\ncreate_engine_with_connector]
        SCHEMA[schema.py\nbuild_schema_summary]
        PROMPT[prompting.py\nmake_few_shot_messages]
        POST[postprocess.py\nguarded_postprocess · normalize_sql\nfirst_select_only]
        VAL[validation.py\nvalidate_sql · parse_schema_text]
        QR[query_runner.py\nQueryRunner.run · check_sql_safety\nDEFAULT_FORBIDDEN_TOKENS · now_utc_iso]
        GUARD[sql_guardrails.py\nclean_candidate_with_reason]
    end

    subgraph data
        RESULTS[(results/**/*.json)]
        MYSQL[(ClassicModels MySQL)]
    end

    NB02 --> EVAL
    NB03 --> REACT
    NB05 --> EVAL
    NB06 --> GEN

    EVAL --> LLM
    EVAL --> PROMPT
    EVAL --> QR
    EVAL --> POST
    EVAL --> GUARD
    EVAL --> DB

    REACT --> LLM
    REACT --> VAL
    REACT --> POST
    REACT --> PROMPT
    REACT --> PROMPTS
    REACT --> AT
    REACT --> EVAL

    AT --> SCHEMA
    AT --> QR

    GUARD --> LLM
    GUARD --> QR

    POST --> LLM

    SCHEMA --> DB
    QR --> DB
    DB --> MYSQL

    EVAL --> RESULTS
    REACT --> RESULTS
    GEN --> RESULTS
```

---

## 4. Statistical Analysis Pipeline — `generate_research_comparison.py`

Run discovery → per-item metrics → hypothesis tests → corrected decisions.

```mermaid
flowchart TD
    A1[results/baseline/runs\nresults_k*_seed*.json] --> DISC[discover_runs\nrglob · parse metadata\ninfer model_tag · method · k · seed]
    A2[results/agent/runs\nresults_react_200.json] --> RDISC[discover_react_runs\ncondition_id = llama_react_k3\nor qwen_react_k3]

    DISC --> DEDUP[Deduplication\nkeep newest RunSpec per condition_id + seed]
    RDISC --> DEDUP

    DEDUP --> BUILD[build_tables_from_runs\nexpand items list per RunSpec\nextract va · em · ex · ts per example_id]

    BUILD --> PCSV[per_item_metrics_primary_raw.csv\none row per item per run\nrun_id · condition_id · seed · i · va · em · ex · ts]
    BUILD --> MCSV[run_manifest.csv\none row per run · aggregate rates]

    PCSV --> STATS[compute_mean_median\ngroupby run_id · metric\nmean · median · std · Shapiro-Wilk W + p]
    STATS --> SCSV[stats_mean_median_shapiro.csv]

    PCSV --> PAIRS[12 predefined hypothesis pairs\ne.g. llama_base_k0 vs llama_base_k3]

    PAIRS --> JOIN[_join_for_pair\nalign by example_id + seed\nn >= 600 paired observations]

    JOIN --> WILCOX[Wilcoxon signed-rank\nprimary · zero_method=wilcox]
    JOIN --> TTEST[Paired t-test\ncorroborating · CLT valid at n>=600]
    JOIN --> CI[95% CI on mean difference\nt-distribution df=n-1]
    JOIN --> COHEN[Cohen's dz\nmean_diff / std_paired_diff]

    WILCOX --> BH[BH-FDR correction\n12 comparisons per metric family]
    TTEST --> BH

    BH --> TCSV[stats_paired_ttests.csv\nwilcoxon_p · ttest_p · bh_adj_p\nci_lower · ci_upper · cohens_d\ndecision_bh_fdr_alpha_0_05]
    CI --> TCSV
    COHEN --> TCSV
```

---

## 5. Metric Scoring and Failure Taxonomy

How VA, EM, EX, and TS are computed for each prediction.

```mermaid
flowchart TD
    SQL([pred_sql]) --> RUN[qr.run pred_sql\nQueryRunner — live DB]

    RUN -- runtime error --> VA0[VA = 0]
    VA0 --> FAIL1[[Syntax / Runtime Fail\nVA=0, EM=0, EX=0]]

    RUN -- success --> VA1[VA = 1]

    VA1 --> NRM[normalize_sql pred\nvs normalize_sql gold]
    NRM -- equal --> EM1[EM = 1]
    NRM -- not equal --> EM0[EM = 0]

    VA1 --> EXC[execution_accuracy\npred rows vs gold rows\nCounter bag equality — order-insensitive]
    EXC -- bags match --> EX1[EX = 1]
    EXC -- bags differ --> EX0[EX = 0]

    EM1 --> EXACT[[Exact Match\nEM=1, EX=1]]
    EX1 & EM0 --> NEAR[[Near Miss\nEX=1, EM=0\nsemantically correct, different syntax]]
    EX0 --> SEMFAIL[[Semantic Fail\nVA=1, EX=0\nexecutes but wrong result]]

    VA1 --> TSCOND{ts_enabled AND\nts_suite_db_names set?}
    TSCOND -- Yes --> TSRUN[test_suite_accuracy_for_item\npred + gold on N perturbed DBs\nfraction where results match]
    TSCOND -- No --> TSNONE[TS = None]
    TSRUN --> TSVAL[TS = 0 or 1]
```

---

## 6. Single NLQ Data Flow

End-to-end journey of one test item through the baseline evaluation path.

```mermaid
flowchart LR
    NLQ([NLQ string\ngold_sql]) --> POOL[_build_item_pool\nfilter exemplar candidates]
    POOL --> SAMP[_sample_exemplars\nrng.sample pool k=3 times]

    SAMP --> MSGS[make_few_shot_messages\nsystem_prompt + schema\n+ 3 NLQ-to-SQL examples + question]

    MSGS --> TEMPL[tokenizer.apply_chat_template\nconvert to token ids]
    TEMPL --> MODEL[model.generate\ngreedy · max_new_tokens=128\nstop on semicolon]
    MODEL --> DECODE[tokenizer.decode\nstrip prompt tokens to raw_sql]

    DECODE --> PRED[pred_sql\nno post-processing in primary runs]

    PRED --> VA{qr.run\npred_sql}
    VA -- OK --> VA1[VA = 1]
    VA -- Error --> VA0[VA = 0]

    PRED --> EM{normalize_sql\nequality check}
    EM -- equal --> EM1[EM = 1]
    EM -- not equal --> EM0[EM = 0]

    PRED --> EX{execute_fetch\npred + gold\nCounter comparison}
    EX -- match --> EX1[EX = 1]
    EX -- differ --> EX0[EX = 0]

    VA1 & EM1 & EX1 --> ITEM([EvalItem frozen dataclass\ni · nlq · raw_sql · pred_sql\nva · em · ex · ts · error])
    VA0 & EM0 & EX0 --> ITEM
```
