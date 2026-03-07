# System Diagrams

Six Mermaid diagrams written to be easy to explain in a viva or demo.
They keep the current code structure, but use plainer labels.

---

## 1. ReAct Loop

This is the main ReAct-style loop for one question.
The simple story is: get context, try SQL, check it, run it, repair if needed.

```mermaid
flowchart TD
    START([Natural-language question]) --> SCHEMA[Read schema context]
    SCHEMA --> STEP{Steps left?}

    STEP -- No --> STOPMAX([Stop: step limit reached])
    STEP -- Yes --> GEN[Generate SQL guess]

    GEN --> VALID[Check SQL before running it]
    VALID -- Invalid --> FIX1{Repair budget left?}
    FIX1 -- No --> STOPBAD([Stop: validation failed])
    FIX1 -- Yes --> REPAIR1[Ask model to repair SQL]
    REPAIR1 --> STEP

    VALID -- Valid --> RUN[Run SQL on the database]
    RUN -- Success --> DONE([Stop: success])
    RUN -- Error --> FIX2{Repair budget left?}
    FIX2 -- No --> STOPERR([Stop: execution failed])
    FIX2 -- Yes --> REPAIR2[Repair SQL using the error message]
    REPAIR2 --> STEP
```

---

## 2. Evaluation Flow

This is one full evaluation run for one condition, such as one model with one `k` and one seed.

```mermaid
flowchart TD
    SETUP([Model + tokenizer + schema + test set + config]) --> LOOP[Pick next benchmark item]

    LOOP --> POOL[Build exemplar pool]
    POOL --> SHOTS{Few-shot or zero-shot?}
    SHOTS -- Few-shot --> SAMPLE[Sample k examples]
    SHOTS -- Zero-shot --> EMPTY[Use no examples]

    SAMPLE --> PROMPT[Build prompt from schema + examples + question]
    EMPTY --> PROMPT

    PROMPT --> MODEL[Generate raw model output]
    MODEL --> CLEAN{Optional cleanup on?}
    CLEAN -- No --> SQL[Use raw SQL]
    CLEAN -- Yes --> FIXSQL[Apply guardrails and cleanup]
    FIXSQL --> SQL

    SQL --> SCORE[Score the prediction]
    SCORE --> VA[VA: does it run?]
    SCORE --> EM[EM: does the SQL text match?]
    SCORE --> EX[EX: do the result rows match?]
    VA --> TS{TS enabled and query valid?}
    TS -- Yes --> TSRUN[Check perturbed databases]
    TS -- No --> TSNONE[TS = none]

    VA --> SAVE[Save scored item]
    EM --> SAVE
    EX --> SAVE
    TSRUN --> SAVE
    TSNONE --> SAVE

    SAVE --> MORE{More items left?}
    MORE -- Yes --> LOOP
    MORE -- No --> REPORT[Save final JSON report]
```

---

## 3. Project Structure

This is the repo at a high level.
The main mental model is: scripts are the official rerun path, notebooks mirror that path for walkthroughs, and one build script turns the manual final pack into the final CSV tables.

```mermaid
graph LR
    subgraph Notebooks
        NB[Runnable notebooks]
    end

    subgraph Infra
        INFRA[infra/
DB setup
model loading
training helpers
ReAct setup]
    end

    subgraph Core
        CORE[core/
prompting
generation
validation
safe SQL run]
    end

    subgraph Agent
        AGENT[agent/
ReAct loop]
    end

    subgraph Evaluation
        EVAL[evaluation/
scoring
fixed grid
manual final pack
simple stats]
    end

    subgraph Scripts
        SCRIPTS[scripts/
fixed run scripts
build_final_analysis.py]
    end

    subgraph Data
        DB[(ClassicModels DB)]
        RESULTS[(Saved JSON runs and CSV outputs)]
    end

    NB --> INFRA
    INFRA --> CORE
    INFRA --> AGENT
    INFRA --> EVAL
    CORE --> DB
    AGENT --> DB
    EVAL --> RESULTS
    SCRIPTS --> EVAL
    SCRIPTS --> RESULTS
    NB --> RESULTS
```

---

## 4. Two-Stage Analysis

This is the final analysis path.
The key idea is: choose the official JSON files by hand, then let one script build the final CSV tables.

```mermaid
flowchart TD
    RUNS[Saved run JSON files from fixed scripts] --> PICK[Manually copy the official files]
    PICK --> PACK[results/final_pack/]
    PACK --> BUILD[python scripts/build_final_analysis.py]

    BUILD --> MANIFEST[manifest.csv]
    BUILD --> PERITEM[per_item.csv]
    BUILD --> SUMMARY[summary_by_condition.csv]
    BUILD --> TESTS[pairwise_tests.csv]
```

---

## 5. Metric Logic

This shows how one prediction is judged.
VA is the easiest check, EM is the strict text check, EX is the main semantic check, and TS is the stricter robustness check.

```mermaid
flowchart TD
    SQL([Predicted SQL]) --> RUN[Try to run the query]

    RUN -- Error --> VA0[VA = 0]
    VA0 --> FAIL[[Syntax or runtime failure]]

    RUN -- Success --> VA1[VA = 1]
    VA1 --> EM[Compare predicted SQL text with gold SQL]
    VA1 --> EX[Compare predicted result rows with gold result rows]

    EM --> EM1[EM = 1 or 0]
    EX --> EX1[EX = 1 or 0]

    VA1 --> TS{TS enabled?}
    TS -- Yes --> TSRUN[Compare on perturbed databases]
    TS -- No --> TSNONE[TS = none]

    EM1 --> OUTCOME[Final scored item]
    EX1 --> OUTCOME
    TSRUN --> OUTCOME
    TSNONE --> OUTCOME
```

---

## 6. Single Question Path

This is the baseline or QLoRA path for one benchmark question.
The short version is: build prompt, get SQL, clean it if needed, score it, save it.

```mermaid
flowchart LR
    Q([Question + gold SQL]) --> EXAMPLES[Choose examples if k > 0]
    EXAMPLES --> PROMPT[Build prompt from schema + examples + question]
    PROMPT --> MODEL[Run model]
    MODEL --> RAW[Get raw model text]
    RAW --> SQL[Extract SQL]

    SQL --> CLEAN{Optional cleanup on?}
    CLEAN -- No --> FINALSQL[Use SQL as-is]
    CLEAN -- Yes --> FIXED[Apply guardrails and cleanup]
    FIXED --> FINALSQL

    FINALSQL --> RUN[Run and score]
    RUN --> SAVE[Save EvalItem]
```
