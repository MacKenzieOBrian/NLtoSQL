# System Diagrams

Five Mermaid diagrams for the dissertation technical explanation.
Paste each fenced block into your LaTeX/Word tool or a Mermaid live renderer.

---

## 1. ReAct Loop — `run_react_pipeline`

The Action→Observation cycle from Yao et al. (2023).
Each box is one traced step recorded in the JSON trace list.
Repair budget is shared across all repair types (max_repairs=2).

```mermaid
flowchart TD
    START([NLQ input]) --> SCHEMA[ACTION: get_schema\nensure_schema_text from agent context]
    SCHEMA --> GEN[ACTION: generate_sql\nfew-shot k=3 prompt → LLM → extract SELECT]

    GEN --> LOOP{step < max_steps?}
    LOOP -- No --> EXHAUST([STOP: max_steps_exhausted\nreturn best sql])

    LOOP -- Yes --> VSQL[ACTION: validate_sql\ncheck SELECT present · table names · schema]

    VSQL -- invalid --> VREPAIR{repairs_used\n< max_repairs=2?}
    VREPAIR -- No --> VSTOP([STOP: validation_failed\nreturn current sql])
    VREPAIR -- Yes --> REP1[ACTION: repair_sql\nZero-shot: SQL_REPAIR_SYSTEM_PROMPT\n+ schema + error hint]
    REP1 --> INC1[repairs_used++]
    INC1 --> LOOP

    VSQL -- valid --> RUN[ACTION: run_sql\nctx.runner.run — live DB execution]

    RUN -- success --> SUCCESS([STOP: success ✓\nreturn sql])

    RUN -- runtime error --> RREPAIR{repairs_used\n< max_repairs=2?}
    RREPAIR -- No --> RSTOP([STOP: execution_failed\nreturn current sql])
    RREPAIR -- Yes --> REP2[ACTION: repair_sql\nExecution-guided repair\nDIN-SQL approach]
    REP2 --> INC2[repairs_used++]
    INC2 --> LOOP
```

---

## 2. Baseline Evaluation Pipeline — `evaluate_run`

One full pass over the 200-item benchmark for a single condition (model × k × seed).

```mermaid
flowchart TD
    INIT([model · tokenizer · k · seed\ntest_set · schema_summary]) --> LOOP[For each of 200 items]

    LOOP --> POOL[Build exemplar pool\nexclude current NLQ to prevent leakage]
    POOL --> SAMPLE{k > 0?}
    SAMPLE -- Yes\nfew-shot --> DRAW[rng.sample pool k times\nglobal sequential RNG]
    SAMPLE -- No\nzero-shot --> EMPTY[exemplars = empty list]
    DRAW --> MSGS[make_few_shot_messages\nsystem prompt + k examples + NLQ]
    EMPTY --> MSGS

    MSGS --> GEN[generate_sql_from_messages\napply_chat_template → model.generate\ndecode → raw_sql]

    GEN --> LAYER{Optional reliability layer\ndisabled in primary runs}
    LAYER -- model_only_raw\nprimary --> PRED[pred_sql = raw_sql]
    LAYER -- extension only --> OPT[sql_guardrails + guarded_postprocess]
    OPT --> PRED

    PRED --> VA[qr.run pred_sql\nValidation Accuracy\ndoes query execute?]
    PRED --> EM[normalize_sql comparison\nExact Match\nnormalised text equality]
    PRED --> EX[execution_accuracy\nCounter bag comparison\npred rows == gold rows]

    VA -- success AND ts_enabled --> TS[test_suite_accuracy_for_item\nN perturbed databases]
    VA -- no TS --> TSN[ts = None]

    VA --> ITEM[EvalItem\nva · em · ex · ts · error · trace]
    EM --> ITEM
    EX --> ITEM
    TS --> ITEM
    TSN --> ITEM

    ITEM --> NEXT{More items?}
    NEXT -- Yes --> LOOP
    NEXT -- No --> AGG[Aggregate VA · EM · EX · TS rates\nSerialise results to JSON\neval_profile = model_only_raw]
```

---

## 3. Module Architecture

Static dependency map across the `nl2sql` package, notebooks, and scripts.

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
        EVAL[eval.py\nevaluate_run\nexecution_accuracy\ntest_suite_accuracy_for_item]
    end

    subgraph nl2sql/agent
        REACT[react_pipeline.py\nrun_react_pipeline\ngenerate_sql · repair_sql]
        AT[agent_tools.py\nget_agent_context\nensure_schema_text]
        PROMPTS[prompts.py\nSQL_GENERATOR_SYSTEM_PROMPT\nSQL_REPAIR_SYSTEM_PROMPT]
    end

    subgraph nl2sql/core
        LLM[llm.py\ngenerate_sql_from_messages\nextract_first_select]
        DB[db.py\nsafe_connection]
        SCHEMA[schema.py\nbuild_schema_summary]
        PROMPT[prompting.py\nmake_few_shot_messages]
        POST[postprocess.py\nguarded_postprocess · normalize_sql]
        VAL[validation.py\nvalidate_sql · validate_constraints\nparse_schema_text]
        QR[query_runner.py\nQueryRunner.run]
        GUARD[sql_guardrails.py\nclean_candidate_with_reason]
    end

    subgraph data
        RESULTS[(results\n**\/*.json)]
        DB2[(ClassicModels\nMySQL)]
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

    REACT --> LLM
    REACT --> VAL
    REACT --> POST
    REACT --> PROMPT
    REACT --> PROMPTS
    REACT --> AT

    AT --> SCHEMA
    AT --> DB
    AT --> QR

    QR --> DB2
    DB --> DB2
    EVAL --> DB
    GEN --> RESULTS
    EVAL --> RESULTS
    REACT --> RESULTS
```

---

## 4. Statistical Analysis Pipeline — `generate_research_comparison.py`

Run discovery → per-item metrics → hypothesis tests → corrected decisions.

```mermaid
flowchart TD
    A1[results/baseline/runs\nresults_k*_seed*.json] --> DISC[discover_runs\nrglob · parse metadata\ninfer model_tag · method · k · seed]
    A2[results/agent/runs\nresults_react_200.json] --> RDISC[discover_react_runs\ncondition_id = llama_react_k3\nor qwen_react_k3]

    DISC --> DEDUP[Deduplication\nkeep newest RunSpec\nper condition_id + seed key]
    RDISC --> DEDUP

    DEDUP --> BUILD[build_tables_from_runs\nfor each RunSpec: expand items list\nextract va · em · ex · ts per example_id]

    BUILD --> PCSV[per_item_metrics_primary_raw.csv\none row per item per run\nrun_id · condition_id · seed · i · va · em · ex · ts]
    BUILD --> MCSV[run_manifest.csv\none row per run\nmetadata + aggregate rates]

    PCSV --> STATS[compute_mean_median\ngroupby run_id · metric\nmean · median · std]
    STATS --> SCSV[stats_mean_median_std.csv]

    PCSV --> PAIRS[12 predefined hypothesis pairs\ne.g. llama_base_k0 vs llama_base_k3\nllama_base_k3 vs llama_qlora_k3\netc.]

    PAIRS --> JOIN[_join_for_pair\nalign by example_id + seed\nn ≥ 600 paired observations]

    JOIN --> WILCOX[Wilcoxon signed-rank\nprimary test\nzero_method=wilcox]
    JOIN --> TTEST[Paired t-test\ncorroborating\nCLT valid at n≥600]
    JOIN --> CI[95% CI on mean difference\nt-distribution df=n-1]
    JOIN --> COHEN[Cohen's dz\nmean_diff / std_paired_diff]

    WILCOX --> BH[BH-FDR correction\n12 comparisons per metric\np_adj = rank/12 × 0.05]
    TTEST --> BH

    BH --> TCSV[stats_paired_ttests.csv\nwilcoxon_p · ttest_p · bh_adj_p\nci_lower · ci_upper · cohens_d\ndecision_bh_fdr_alpha_0_05]
    CI --> TCSV
    COHEN --> TCSV
```

---

## 5. Metric Scoring and Failure Taxonomy

How VA, EM, EX, and TS are computed for each prediction, and the resulting failure classes.

```mermaid
flowchart TD
    SQL([pred_sql]) --> RUN[qr.run pred_sql\nQueryRunner — live DB]

    RUN -- runtime error --> VA0[VA = 0]
    VA0 --> FAIL1[[Syntax / Runtime Fail\nVA=0, EM=0, EX=0]]

    RUN -- success --> VA1[VA = 1]

    VA1 --> NRM[normalize_sql pred\nvs normalize_sql gold]
    NRM -- equal --> EM1[EM = 1]
    NRM -- not equal --> EM0[EM = 0]

    VA1 --> EXC[execution_accuracy\npred rows run on prod DB\ngold rows run on prod DB\nCounter bag equality]
    EXC -- bags match --> EX1[EX = 1]
    EXC -- bags differ --> EX0[EX = 0]

    EM1 --> EXACT[[Exact Match\nEM=1, EX=1]]
    EX1 & EM0 --> NEAR[[Near Miss\nEX=1, EM=0\nsemantically correct\ndifferent syntax]]
    EX0 --> SEMFAIL[[Semantic Fail\nVA=1, EX=0\nexecutes but wrong result]]

    VA1 --> TSCOND{ts_enabled AND\nts_suite_db_names set?}
    TSCOND -- Yes --> TSRUN[test_suite_accuracy_for_item\nrun gold + pred on N perturbed DBs\nfraction where results match]
    TSCOND -- No --> TSNONE[TS = None]
    TSRUN --> TSVAL[TS = 0.0 – 1.0]
```
