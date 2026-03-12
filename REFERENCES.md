# References

Reference map for the dissertation codebase. Numbers match the current
bibliography `[1]`-`[24]`.

This file is intentionally not a second bibliography. Its job is to show where
the code has:

- a **Direct pattern** match to a cited method or official implementation
- an **Adapted implementation** of the cited idea
- **Background only** relevance, where the paper explains the design direction
  but the code does not implement that method directly


## Core Implementation Matches

| Ref | Paper / source | Official repo / docs | Code area | Match type | Note |
| --- | --- | --- | --- | --- | --- |
| [17] | ReAct | [ysymyth/ReAct](https://github.com/ysymyth/ReAct) | `nl2sql/agent/react_pipeline.py`, `nl2sql/agent/prompts.py` | Adapted implementation | The live loop keeps the ReAct `reason -> act -> observe -> continue` structure, but adapts the original flat prompt string into chat-format message history for instruct models. |
| [24] | Transformers | [Transformers repo](https://github.com/huggingface/transformers) · [chat templating docs](https://huggingface.co/docs/transformers/chat_templating) · [generation docs](https://huggingface.co/docs/transformers/en/internal/generation_utils) | `nl2sql/core/llm.py`, `nl2sql/agent/react_pipeline.py` | Direct pattern | The code directly uses `apply_chat_template`, `StoppingCriteria`, and `model.generate` in the way the official library documents them. |
| [10] | LoRA | [PEFT repo](https://github.com/huggingface/peft) · [PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora) | `nl2sql/infra/model_loading.py` | Adapted implementation | The method claim comes from LoRA, while the actual adapter attachment and training stack are implemented through PEFT. |
| [14] | QLoRA | [PEFT repo](https://github.com/huggingface/peft) · [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/main/en/index) | `nl2sql/infra/model_loading.py` | Adapted implementation | The loading path mirrors the QLoRA-style 4-bit NF4 setup, but it is a practical Hugging Face / bitsandbytes implementation rather than a copy of one research training script. |
| [19] | Distilled test-suite evaluation | [test-suite-sql-eval repo](https://github.com/taoyds/test-suite-sql-eval) | `nl2sql/evaluation/eval.py` | Adapted implementation | The local helper follows the same evaluation idea, but uses a simplified project-specific function rather than the exact benchmark toolkit. |
| [20] | Spider | [Spider repo](https://github.com/taoyds/spider) · [Spider site](https://yale-lily.github.io/spider) | `nl2sql/evaluation/eval.py` | Background only | The project uses Spider-style execution-focused evaluation logic, but it does not run the official Spider benchmark scripts unchanged. |
| [15] | Qwen2.5 technical report | [Qwen repo](https://github.com/QwenLM/Qwen) · [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | notebooks and run scripts using `Qwen/Qwen2.5-7B-Instruct` | Direct pattern | This is the exact model family and model ID used for the Qwen baseline and ReAct conditions. |
| [23] | Llama 3 technical report | [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | notebooks and run scripts using `meta-llama/Meta-Llama-3-8B-Instruct` | Direct pattern | This is the exact model family and model ID used for the Llama baseline and QLoRA conditions. |


## Evaluation and Analysis

| Ref | Paper / source | Official repo / docs | Code area | Match type | Note |
| --- | --- | --- | --- | --- | --- |
| [21] | Gao et al. benchmark evaluation | — | evaluation design and write-up | Background only | Useful for benchmarking context and for positioning prompt-based text-to-SQL comparisons, but not directly implemented as code. |
| [22] | Dror et al. significance testing | — | `nl2sql/evaluation/simple_stats.py` | Adapted implementation | The code uses Dror's broader principle of matching the test to the data structure, but the specific Mann-Whitney choice is a project simplification rather than a method copied from the paper. |
| [3] | BIRD benchmark | [BIRD site](https://bird-bench.github.io/) | benchmarking discussion and write-up | Background only | This is benchmark context for LLM-era text-to-SQL performance, not a directly implemented benchmark in the repo. |


## Text-to-SQL Design Context

| Ref | Paper / source | Official repo / docs | Code area | Match type | Note |
| --- | --- | --- | --- | --- | --- |
| [1] | Deep learning text-to-SQL survey | — | dissertation framing, task background | Background only | Survey context for the task and its major design pressures. |
| [2] | BRIDGE | [BRIDGE repo](https://github.com/salesforce/TabularSemanticParsing) | `nl2sql/core/schema.py` | Background only | The compact schema summary is motivated by the importance of schema and value grounding, but the code does not implement BRIDGE's value-grounding architecture. |
| [4] | DIN-SQL | [DIN-SQL repo](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting) | `nl2sql/agent/react_pipeline.py`, dissertation design discussion | Background only | Relevant for decomposition and self-correction ideas, but the live controller is a ReAct-style loop rather than DIN-SQL's staged prompting pipeline. |
| [5] | ExCoT | — | `nl2sql/agent/react_pipeline.py`, dissertation design discussion | Background only | Supports execution feedback as a useful reasoning signal, but the project does not implement ExCoT's exact prompting method. |
| [8] | LLM text-to-SQL survey | — | dissertation literature review | Background only | Survey context for prompt design, agents, and LLM-era trade-offs. |
| [11] | LLM-based text-to-SQL survey | — | dissertation literature review | Background only | Background survey used to position open-model and agentic design choices. |
| [12] | Ojuri et al. agents + text-to-SQL | — | dissertation design discussion | Background only | Supports the relevance of agent-based text-to-SQL systems, but the project's controller and evaluation design are different. |
| [13] | PICARD | [PICARD repo](https://github.com/ServiceNow/picard) | `nl2sql/core/validation.py`, `nl2sql/core/postprocess.py` | Background only | The project shares PICARD's concern for guarded SQL generation, but it does not implement grammar-constrained decoding or incremental parsing. |
| [16] | RAT-SQL | [RAT-SQL repo](https://github.com/microsoft/rat-sql) | `nl2sql/core/schema.py` | Background only | The code keeps schema structure explicit in the prompt, but it does not implement RAT-SQL's relation-aware encoder. |
| [18] | RESDSQL | [RESDSQL repo](https://github.com/RUCKBReasoning/RESDSQL) | dissertation design discussion | Background only | Useful for explaining decomposition in text-to-SQL, but not directly implemented in the repo. |


## Prompting and Fine-Tuning Context

| Ref | Paper / source | Official repo / docs | Code area | Match type | Note |
| --- | --- | --- | --- | --- | --- |
| [6] | Few-shot fine-tuning vs ICL | — | experimental design and dissertation discussion | Background only | Supports the fairness of comparing prompting and fine-tuning under one evaluation setup. |
| [7] | Language Models are Few-Shot Learners | — | `nl2sql/evaluation/eval.py`, `nl2sql/agent/react_pipeline.py` | Background only | Foundational support for few-shot exemplar use; the code is a project-specific chat-format adaptation rather than a replication of GPT-3 prompting. |
| [9] | LoRA Learns Less and Forgets Less | — | dissertation discussion of PEFT trade-offs | Background only | Used to discuss possible behavior changes under PEFT, not to justify a specific code path. |


## Notes on Attribution

- **Direct pattern** means the code uses the same interface or control pattern as
  the cited paper or official implementation.
- **Adapted implementation** means the project keeps the core method idea but
  changes the carrier, environment, or evaluation setup to fit this repo.
- **Background only** means the source is used for design rationale or
  dissertation framing, not because the repo directly implements that system.

- The strongest direct implementation links in this repo are:
  - ReAct `[17]` -> `nl2sql/agent/react_pipeline.py`
  - Transformers `[24]` -> `nl2sql/core/llm.py`
  - PEFT-backed LoRA/QLoRA `[10]`, `[14]` -> `nl2sql/infra/model_loading.py`
  - distilled test-suite evaluation `[19]` -> `nl2sql/evaluation/eval.py`

- The weakest matches should stay in markdown only:
  - BRIDGE `[2]`
  - RAT-SQL `[16]`
  - PICARD `[13]`
  - RESDSQL `[18]`

Those papers explain why the project made certain design choices, but the code
does not claim to implement their full architectures.
