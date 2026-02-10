# Literature Review

This review is scoped to one practical question: what improves NL->SQL quality under open-source and limited-compute constraints, and what improves robustness only.

## Framing

Text-to-SQL has a many-to-one mapping problem: multiple SQL strings can express the same semantics. For this reason, execution-based metrics are treated as primary, while exact match is diagnostic.

Key references:
- Spider benchmark framing: `REFERENCES.md#ref-yu2018-spider`
- Test-suite semantic evaluation: `REFERENCES.md#ref-zhong2020-ts`
- Recent LLM Text-to-SQL surveys: `REFERENCES.md#ref-zhu2024-survey`, `REFERENCES.md#ref-hong2025-survey`

## Replication Anchor

The dissertation is methodologically anchored to Ojuri et al. (2025) (`REFERENCES.md#ref-ojuri2025-agents`) but scoped to an open-source local LLM workflow. The literature role of the anchor paper is to define the comparison structure (few-shot vs fine-tuning + agent support), while this project evaluates whether those comparative trends persist when proprietary infrastructure is removed.

## Prompting vs Fine-Tuning

Few-shot prompting is a strong low-cost baseline and must be reported before any adaptation. However, literature repeatedly shows that prompting gains can plateau on compositional joins and aggregation semantics.

Relevant grounding:
- In-context learning baseline logic: `REFERENCES.md#ref-brown2020-gpt3`
- Fair ICL vs fine-tuning comparisons: `REFERENCES.md#ref-mosbach2023-icl`

QLoRA/PEFT allows adaptation without full-model training and is the right tool for compute-constrained experimentation. In this dissertation, fine-tuning is evaluated as a semantic mapping intervention, not as a deployment optimization story.

Relevant grounding:
- PEFT foundations: `REFERENCES.md#ref-ding2023-peft`
- Applied PEFT evidence under constraints: `REFERENCES.md#ref-goswami2024-peft`

## Execution-Guided Agents: What They Solve

ReAct-style and execution-guided methods improve observability and validity by forcing explicit checks, tool calls, and repair-on-failure behavior. They are best interpreted here as evaluation infrastructure and robustness support.

Relevant grounding:
- ReAct action-observation loop: `REFERENCES.md#ref-yao2023-react`
- Execution-guided decoding: `REFERENCES.md#ref-wang2018-eg-decoding`
- ExCoT execution-feedback reasoning: `REFERENCES.md#ref-zhai2025-excot`
- Agent-mediated NL->SQL workflows: `REFERENCES.md#ref-ojuri2025-agents`

This literature also makes a negative point that matters for dissertation claims: execution feedback can raise validity without fully resolving semantic alignment.

## Schema and Value Linking Bottlenecks

Schema selection and value grounding remain persistent error sources in cross-domain Text-to-SQL. Relation-aware linking and decoupled linking/generation approaches motivate explicit linking stages even in lightweight systems.

Relevant grounding:
- Relation-aware schema linking: `REFERENCES.md#ref-wang2020-ratsql`
- Decoupled schema linking and parsing: `REFERENCES.md#ref-li2023-resdsql`
- Value linking (text -> table value correspondence): `REFERENCES.md#ref-lin2020-bridge`

## Claim-Level Literature Anchors

The dissertation argument is anchored to five claim classes.

1. Execution-based metrics should dominate interpretation.
- Basis: Spider evaluation context + test-suite semantics (`ref-yu2018-spider`, `ref-zhong2020-ts`).

2. Prompting and fine-tuning must be compared directly under shared conditions.
- Basis: fair ICL vs fine-tuning framing (`ref-mosbach2023-icl`).

3. QLoRA is a compute-feasible way to test semantic adaptation.
- Basis: PEFT/QLoRA evidence (`ref-ding2023-peft`, `ref-goswami2024-peft`).

4. ReAct-style loops are appropriate for bounded execution feedback and traceability.
- Basis: ReAct and execution-guided reasoning (`ref-yao2023-react`, `ref-zhai2025-excot`, `ref-wang2018-eg-decoding`).

5. Remaining errors are expected to concentrate in linking and composition.
- Basis: schema/value linking literature (`ref-wang2020-ratsql`, `ref-li2023-resdsql`, `ref-lin2020-bridge`).

## Evidence Expectations Derived from Literature

To keep claims defensible, literature implies this evidence standard:
- controlled cross-method comparisons on the same test items,
- uncertainty reporting (confidence intervals),
- paired significance for method deltas,
- error taxonomy explaining where gains came from,
- explicit boundary between primary method claims and execution infrastructure claims.

## Position of This Dissertation

The contribution is intentionally narrow and defensible:
- Primary research: controlled comparison of prompting and QLoRA under fixed resources.
- Secondary infrastructure: minimal execution-guided ReAct loop to stabilize validity and make errors auditable.
- Primary claim style: what changed, by how much, with what uncertainty, and which error categories remain.

This framing reduces explanation cost and aligns with reproducible, examiner-facing methodology.
