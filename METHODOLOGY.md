# Methodology (Open-Source Replica of Ojuri et al., 2025)

We mirror Ojuri et al.’s (2025) structure—model adaptation, (planned) agentic execution feedback, and VA/EX/TS evaluation—using only open-source components and Colab-class hardware.

## Domain and Data
- ClassicModels MySQL schema; 200-item held-out test set for VA/EX; separate training JSONL (~200 NL→SQL pairs).
- Rationale: business-style joins/aggregations, reproducible schema, and comparability to Ojuri et al. (2025) and execution-grounded benchmarks [1], [18].

## Adaptation Strategies
- Prompting baselines (zero-/few-shot) for floor/ceiling checks [6], [3].
- QLoRA fine-tuning (PEFT) to internalize schema patterns under VRAM limits [12], [5].
- Planned agentic refinement (ReAct/ExCoT style) to iteratively correct SQL via execution feedback [2], [16].

## Training Configuration (current best practice)
- Base: Llama-3-8B-Instruct, 4-bit NF4 quantization.
- LoRA: r=32, α=64, dropout=0.05; adapters only (base frozen).
- Optim: `paged_adamw_8bit`, LR ≈ 1e-4, warmup 0.05, epochs ≈ 3, batch 1, grad_accum 8.
- Dtype/device: auto bf16 on Ampere, else fp16; `device_map={"":0}` to avoid CPU/disk offload.

## Evaluation Criteria (Ojuri-aligned)
- VA: executability (syntactic/engine validity).
- EX: execution accuracy (result-set equivalence) [18].
- EM: exact string match (strict baseline).
- TS: planned semantic test-suite across perturbed DBs [18].

## Justification
- Few-shot vs fine-tuning: fine-tuning typically improves domain consistency [3]; prompting remains a strong baseline for comparison.
- PEFT/QLoRA: enables 7B+ models on 8–12GB VRAM, avoiding full FT (>60GB) [4], [12].
- Agentic loop: ReAct-style execution feedback reduces logical errors beyond single-shot decoding [2], [16], [10].
