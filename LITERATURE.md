# State of the Art

**Proprietary systems.** GPT-4–class models deliver strong text-to-SQL with agentic chains (ReAct) [16], [10], but closed weights and cost limit reproducibility.

**Open-source LLMs.** Smaller instruction models (e.g., Llama-3-8B) are viable when adapted to schema and domain via PEFT/QLoRA [12], [5], [10].

**Prompting vs fine-tuning.** In-context learning (Brown et al., 2020 [6]) is simple but inconsistent for SQL; fine-tuning typically wins on domain tasks (Mosbach et al., 2023 [3]).

**Execution-feedback agents.** ReAct/ExCoT-style loops improve semantic correctness by iterating over execution traces [2], [16]; Ojuri et al. (2025) [10] report gains on ClassicModels.

**Benchmarks and metrics.** Execution accuracy and test-suite accuracy (Zhong et al., 2020 [18]) provide stronger semantic checks than string match; large-scale NL→SQL evaluations include [1], [8], [9], [20].

**Resource-aware PEFT.** QLoRA enables 7B+ models on 8–12GB VRAM by 4-bit quantization + adapters [12], [5], making open-source replication feasible on Colab-class GPUs.

Takeaway: Agentic open-source LLMs with PEFT and execution-grounded evaluation offer a reproducible path toward proprietary-level NL→SQL quality under realistic constraints.
