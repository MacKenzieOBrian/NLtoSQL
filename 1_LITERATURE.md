# Literature Review

This review is focused on LLM-based Text-to-SQL systems, with emphasis on evaluation reliability and agentic execution feedback. It is written to justify the choices made in this project, not to catalogue every paper in the area.

---

## Scope and Framing

Text-to-SQL is hard because the same question can map to many correct SQL strings. That means surface-level metrics (exact string match) are brittle and can understate real semantic correctness. The review therefore centers on evaluation that checks execution behavior, not just syntax.

Key sources:
- Execution-based and test-suite evaluation: `REFERENCES.md#ref-zhong2020-ts`, `REFERENCES.md#ref-yu2018-spider`
- Surveys of LLM-based Text-to-SQL: `REFERENCES.md#ref-zhu2024-survey`, `REFERENCES.md#ref-hong2025-survey`

---

## Evaluation Beyond Exact Match

Exact Match (EM) was common in early benchmarks, but it does not capture semantic equivalence. Execution Accuracy (EX) and test-suite-style evaluation address this by running SQL and comparing outputs. Zhong et al. motivate test suites for semantic checks; Yu et al. show why EM alone is insufficient in complex SQL settings.

Key sources:
- `REFERENCES.md#ref-zhong2020-ts`
- `REFERENCES.md#ref-yu2018-spider`

---

## Prompting and In-Context Learning

In-context learning provides a simple baseline that is widely used for Text-to-SQL. It is still an essential comparison point because it isolates the effect of prompting from learned weights. Mosbach et al. argue for fair comparisons between ICL and fine-tuning, which is a design constraint here.

Key sources:
- `REFERENCES.md#ref-brown2020-gpt3`
- `REFERENCES.md#ref-mosbach2023-icl`

---

## Parameter-Efficient Fine-Tuning (PEFT / QLoRA)

PEFT methods make fine-tuning feasible under limited VRAM. They are not a replacement for robust evaluation, but they enable a controlled test of whether training data improves SQL generation beyond prompting. Ding et al. and Goswami et al. provide the foundation for using LoRA/QLoRA in this context.

Key sources:
- `REFERENCES.md#ref-ding2023-peft`
- `REFERENCES.md#ref-goswami2024-peft`

---

## Agentic Execution Feedback (ReAct / ExCoT / Agent-Mediated NL→SQL)

Execution feedback is the main theoretical justification for the agent loop. ReAct formalizes an action-observation loop, ExCoT shows execution feedback can improve Text-to-SQL reasoning, and Ojuri et al. explicitly frame agent-mediated NL→SQL workflows with validation and remediation steps. This motivates a bounded, explainable loop rather than unstructured self-reflection.

Key sources:
- `REFERENCES.md#ref-yao2023-react`
- `REFERENCES.md#ref-zhai2025-excot`
- `REFERENCES.md#ref-ojuri2025-agents`

---

## Schema Linking as a Bottleneck

Multiple surveys and model papers identify schema linking as a dominant error source. This project uses heuristic schema subset selection to reduce prompt noise, explicitly acknowledging that it is not a learned linker and is therefore limited in coverage.

Key sources:
- `REFERENCES.md#ref-li2023-resdsql`
- `REFERENCES.md#ref-zhu2024-survey`
- `REFERENCES.md#ref-hong2025-survey`

---

## Positioning of This Project

The project is intentionally iterative and evaluation-driven:
- Start with a strong ICL baseline.
- Add PEFT (QLoRA) to test whether training improves SQL generation.
- Add an execution-guided agent loop to correct runnable-but-wrong SQL.

This mirrors the methodological ladder in the experiments and keeps claims limited to what the evaluation harness can support.
