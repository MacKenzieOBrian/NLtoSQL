# System Diagrams

Six Mermaid diagrams that explain the project in a simple way.
Paste each fenced block into your document tool or a Mermaid live renderer.

---

## 1. ReAct Loop — `run_react_pipeline`

This is the main loop the agent follows for one question.
Each box is one step saved into the JSON trace.
`repairs_used` is one shared repair budget for both validation and execution fixes.

```mermaid
flowchart TD
    START([NLQ input]) --> SCHEMA[Step 1: get_schema\nensure_schema_text from AgentContext]
    SCHEMA --> GEN[Step 2: generate_sql\nfew-shot prompt + model call + extract SELECT]

    GEN --> LOOP{step < max_steps=8?}
    LOOP -- No --> EXHAUST([STOP: max_steps_exhausted\nreturn current_sql])

    LOOP -- Yes --> VSQL[validate_sql\nSELECT present? table names in schema?]

    VSQL -- invalid --> VREPAIR{repairs_used\n< max_repairs=2?}
    VREPAIR -- No --> VSTOP([STOP: validation_failed\nreturn current_sql])
    VREPAIR -- Yes --> REP1[repair_sql\nrepair prompt\n+ schema + error hint]
    REP1 --> INC1[repairs_used++]
    INC1 --> LOOP

    VSQL -- valid --> RUN[run_sql\nctx.runner.run — live DB execution]

    RUN -- success --> SUCCESS([STOP: success\nreturn current_sql])

    RUN -- runtime error --> RREPAIR{repairs_used\n< max_repairs=2?}
    RREPAIR -- No --> RSTOP([STOP: execution_failed\nreturn current_sql])
    RREPAIR -- Yes --> REP2[repair_sql\nuse execution error to try again]
    REP2 --> INC2[repairs_used++]
    INC2 --> LOOP
```

---

## 2. Evaluation Pipeline — `eval_run`

One full pass over the benchmark items for one setting (model × k × seed).

```mermaid
flowchart TD
    INIT([model · tokenizer · k · seed\ntest_set · schema_summary · EvalRunConfig]) --> LOOP[For each selected item]

    LOOP --> POOL[_build_item_pool\nexclude current NLQ to prevent leakage]
    POOL --> SAMPLE{k > 0?}
    SAMPLE -- Yes\nfew-shot --> DRAW[_sample_exemplars\nsample k examples from the run RNG]
    SAMPLE -- No\nzero-shot --> EMPTY[exemplars = empty list]
    DRAW --> MSGS[make_few_shot_messages\nsystem prompt + k examples + NLQ]
    EMPTY --> MSGS

    MSGS --> GEN[generate_sql_from_messages\nbuild tokens + run model + decode\noptional extract_select / stop_on_semicolon]

    GEN --> LAYER{guardrails or\npostprocess enabled?}
    LAYER -- No --> PRED[pred_sql = raw_sql]
    LAYER -- Yes --> OPT[_clean_sql\nsql_guardrails + guarded_postprocess]
    OPT --> PRED

    PRED --> VA[qr.run pred_sql\ncheck if the SQL runs]
    PRED --> EM[normalize_sql comparison\nExact Match]
    PRED --> EX[execution_accuracy\ncompare predicted rows with gold rows]

    VA -- success AND ts_enabled --> TS[test_suite_accuracy_for_item\nN perturbed databases]
    VA -- no TS --> TSN[ts = None]

    VA --> ITEM[EvalItem\nva · em · ex · ts · raw_sql · pred_sql · error]
    EM --> ITEM
    EX --> ITEM
    TS --> ITEM
    TSN --> ITEM

    ITEM --> NEXT{More items?}
    NEXT -- Yes --> LOOP
    NEXT -- No --> AGG[_summarize_eval + _build_eval_payload\nwork out final rates\nsave JSON report]
```

---

## 3. Module Architecture

High-level file map. Arrows show the main imports.

```mermaid
graph LR
    subgraph Notebooks
        NB02[02_baseline_prompting_eval]
        NB03[03_agentic_eval]
        NB04[04_build_training_set]
        NB05[05_qlora_train_eval]
        NB06[06_research_comparison]
    end

    subgraph scripts
        SETUP[colab_setup.sh\nColab dependency install]
        GENCLI[generate_research_comparison.py\nthin CLI wrapper]
    end

    subgraph nl2sql/infra
        IDB[infra/db.py\nconnect_notebook_db\ncreate_engine_with_connector]
        IEXP[infra/experiment_helpers.py\nrun_model_grid_notebook_eval\nrun_react_notebook_eval\nconfigure_react_notebook]
        IMOD[infra/model_loading.py\nload_quantized_model\nbuild_trainable_qlora_model\nload_eval_adapter_model]
        INB[infra/notebook_utils.py\nauth helpers + load_test_set\nload_train_records]
        ITRAIN[infra/training_set.py\ntraining-set validation helpers]
    end

    subgraph nl2sql/evaluation
        EVAL[eval.py\neval_run · execution_accuracy\ntest_suite_accuracy_for_item\nEvalRunConfig · EvalItem]
        GRID[grid_runner.py\nrun_eval_grid]
        RCOMP[research_comparison.py\ngenerate]
        RRUNS[research_runs.py\ndiscover_all_runs\nbuild_tables_from_runs]
        RSTATS[research_stats.py\ncompute_mean_median_std\ncompute_paired_tests]
        APLOT[analysis_plots.py\ncomparison plots + failure breakdown]
    end

    subgraph nl2sql/agent
        REACT[react_pipeline.py\nrun_react_pipeline · generate_sql\nrepair_sql]
        AT[agent_tools.py\nAgentContext · get_agent_context\nensure_schema_text · schema_to_text]
        PROMPTS[prompts.py\nSQL_GENERATOR_SYSTEM_PROMPT\nSQL_REPAIR_SYSTEM_PROMPT]
    end

    subgraph nl2sql/core
        LLM[llm.py\ngenerate_sql_from_messages\nextract_first_select]
        SCHEMA[schema.py\nbuild_schema_summary]
        PROMPT[prompting.py\nmake_few_shot_messages]
        POST[postprocess.py\nguarded_postprocess · normalize_sql]
        VAL[validation.py\nvalidate_sql · parse_schema_text]
        QR[query_runner.py\nQueryRunner.run · check_sql_safety\nDEFAULT_FORBIDDEN_TOKENS · now_utc_iso]
        GUARD[sql_guardrails.py\nclean_candidate_with_reason]
    end

    subgraph data
        RESULTS[(results/**/*.json)]
        MYSQL[(ClassicModels MySQL)]
    end

    NB02 --> SETUP
    NB03 --> SETUP
    NB05 --> SETUP

    NB02 --> INB
    NB02 --> IDB
    NB02 --> IMOD
    NB02 --> IEXP

    NB03 --> INB
    NB03 --> IDB
    NB03 --> IMOD
    NB03 --> IEXP

    NB04 --> INB
    NB04 --> IDB
    NB04 --> ITRAIN

    NB05 --> INB
    NB05 --> IDB
    NB05 --> IMOD
    NB05 --> IEXP

    NB06 --> RCOMP
    NB06 --> APLOT

    GENCLI --> RCOMP

    IEXP --> GRID
    IEXP --> EVAL
    IEXP --> REACT
    IEXP --> QR
    ITRAIN --> INB

    GRID --> EVAL
    GRID --> IDB

    RCOMP --> RRUNS
    RCOMP --> RSTATS

    EVAL --> LLM
    EVAL --> PROMPT
    EVAL --> QR
    EVAL --> POST
    EVAL --> GUARD
    EVAL --> IDB

    REACT --> LLM
    REACT --> VAL
    REACT --> POST
    REACT --> PROMPT
    REACT --> PROMPTS
    REACT --> AT

    AT --> SCHEMA

    GUARD --> LLM
    GUARD --> QR

    SCHEMA --> IDB
    QR --> IDB
    IDB --> MYSQL

    GRID --> RESULTS
    IEXP --> RESULTS
    RCOMP --> RESULTS
```

---

## 4. Statistical Analysis Pipeline — `research_comparison.generate`

How saved result files turn into summary tables and test results.

```mermaid
flowchart TD
    A1[results/baseline/runs + results/qlora/runs\nresults_k*_seed*.json] --> DISC[discover_primary_runs\nparse metadata\ninfer model_tag · method · k · seed\naccept full runs only]
    A2[results/agent/runs\nresults_react_200.json or results_react_eval.json] --> RDISC[discover_react_runs\ninfer model_tag + few_shot_k + seed\naccept full runs only]

    DISC --> DEDUP[Keep newest run\none RunSpec per condition_id + seed]
    RDISC --> DEDUP

    DEDUP --> BUILD[build_tables_from_runs\nexpand items list per RunSpec\nextract va · em · ex · ts per example_id]
    BUILD --> PREP[prepare_per_item_table\nturn va · em · ex · ts into numeric columns]

    PREP --> PCSV[per_item_metrics_primary_raw.csv\none row per item per run\nrun_id · condition_id · seed · i · va · em · ex · ts]
    BUILD --> MCSV[run_manifest.csv\none row per run\ncondition_id · seed · source_json]

    PREP --> STATS[compute_mean_median_std\ngroup by run_id and metric\nmean · median · std · Shapiro-Wilk W + p]
    STATS --> SCSV[stats_mean_median_std.csv]

    PREP --> PAIRS[planned_comparisons\nkeep only the comparison pairs that exist]

    PAIRS --> JOIN[_join_pair\nmatch by seed + example_id\nfallback: seed + nlq]

    JOIN --> WILCOX[Wilcoxon signed-rank\nmain hypothesis test]
    JOIN --> TTEST[Paired t-test\nextra check on the same paired differences]
    JOIN --> CI[95% CI on mean difference\nt-distribution df=n-1]
    JOIN --> COHEN[Cohen's dz\nmean_diff / std_paired_diff]

    WILCOX --> BH[BH-FDR correction\nadjust p-values inside each metric family]

    BH --> TCSV[stats_paired_ttests.csv\nwilcoxon_p · ttest_p · bh_adj_p\nci_lower · ci_upper · cohens_d\ndecision_bh_fdr_alpha_0_05]
    TTEST --> TCSV
    CI --> TCSV
    COHEN --> TCSV
```

---

## 5. Metric Scoring and Failure Taxonomy

How VA, EM, EX, and TS are worked out for one prediction.

```mermaid
flowchart TD
    SQL([pred_sql]) --> RUN[qr.run pred_sql\nrun the SQL on the live DB]

    RUN -- runtime error --> VA0[VA = 0]
    VA0 --> FAIL1[[Syntax / Runtime Fail\nVA=0, EM=0, EX=0]]

    RUN -- success --> VA1[VA = 1]

    VA1 --> NRM[normalize_sql pred\nvs normalize_sql gold]
    NRM -- equal --> EM1[EM = 1]
    NRM -- not equal --> EM0[EM = 0]

    VA1 --> EXC[execution_accuracy\ncompare predicted rows with gold rows\nusually order-insensitive]
    EXC -- bags match --> EX1[EX = 1]
    EXC -- bags differ --> EX0[EX = 0]

    EM1 --> EXACT[[Exact Match\nEM=1, EX=1]]
    EX1 & EM0 --> NEAR[[Near Miss\nEX=1, EM=0\nright result, different SQL form]]
    EX0 --> SEMFAIL[[Semantic Fail\nVA=1, EX=0\nexecutes but wrong result]]

    VA1 --> TSCOND{ts_enabled AND\nts_suite_db_names set?}
    TSCOND -- Yes --> TSRUN[test_suite_accuracy_for_item\npred + gold on N perturbed DBs\nTS=1 only if all checked replicas match]
    TSCOND -- No --> TSNONE[TS = None]
    TSRUN --> TSVAL[TS = 0 or 1]
```

---

## 6. Single NLQ Data Flow

What happens to one benchmark question in the shared baseline/QLoRA evaluation path.

```mermaid
flowchart LR
    NLQ([NLQ string\ngold_sql]) --> POOL[_build_item_pool\nfilter exemplar candidates]
    POOL --> SAMP[_sample_exemplars\nrng.sample pool k=3 times]

    SAMP --> MSGS[make_few_shot_messages\nsystem prompt + schema\n+ 3 example pairs + question]

    MSGS --> TEMPL[tokenizer.apply_chat_template\nturn prompt into token ids]
    TEMPL --> MODEL[model.generate\ngreedy decode\nstop on semicolon]
    MODEL --> DECODE[tokenizer.decode\nturn output tokens into raw_sql]

    DECODE --> CLEAN{guardrails or\npostprocess enabled?}
    CLEAN -- No --> PRED[pred_sql = raw_sql]
    CLEAN -- Yes --> OPT[_clean_sql\nsql_guardrails + guarded_postprocess]
    OPT --> PRED

    PRED --> VA{qr.run\npred_sql}
    VA -- OK --> VA1[VA = 1]
    VA -- Error --> VA0[VA = 0]

    PRED --> EM{normalize_sql\nequality check}
    EM -- equal --> EM1[EM = 1]
    EM -- not equal --> EM0[EM = 0]

    PRED --> EX{execute_fetch\npred + gold\ncompare result rows}
    EX -- match --> EX1[EX = 1]
    EX -- differ --> EX0[EX = 0]

    VA1 --> TSCOND{TS enabled?}
    TSCOND -- Yes --> TSRUN[test_suite_accuracy_for_item\nN perturbed DBs]
    TSCOND -- No --> TSNONE[TS = None]

    VA0 --> ITEM([EvalItem\ni · nlq · raw_sql · pred_sql\nva · em · ex · ts · error])
    EM1 --> ITEM
    EM0 --> ITEM
    EX1 --> ITEM
    EX0 --> ITEM
    TSRUN --> ITEM
    TSNONE --> ITEM
```
