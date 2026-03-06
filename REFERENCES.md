# References

Full reference list with code and design anchors.
Numbers match the dissertation bibliography [1]–[29].


## Evaluation Metrics

**[22]** T. Yu et al., 'Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain
Semantic Parsing and Text-to-SQL Task', EMNLP 2018. doi: 10.18653/v1/D18-1425.
→ Defines the EX (execution accuracy) metric. Motivates `Counter(pred_rows) == Counter(gold_rows)`
  in `nl2sql/evaluation/eval.py`.

**[21]** R. Zhong, T. Yu, and D. Klein, 'Semantic Evaluation for Text-to-SQL with Distilled Test
Suites', EMNLP 2020. doi: 10.18653/v1/2020.emnlp-main.29.
→ Motivates `test_suite_accuracy_for_item` in `nl2sql/evaluation/eval.py`.

**[24]** R. Dror et al., 'The Hitchhiker's Guide to Testing Statistical Significance in Natural
Language Processing', ACL 2018. doi: 10.18653/v1/P18-1128.
→ Justifies the choice of non-parametric Wilcoxon test over parametric t-test as primary,
  and BH-FDR correction for multiple comparisons, in `nl2sql/evaluation/research_stats.py`.

**[29]** P. Virtanen et al., 'SciPy 1.0: Fundamental Algorithms for Scientific Computing in
Python', Nature Methods, vol. 17, pp. 261–272, 2020. doi: 10.1038/s41592-020-0772-5.
→ Provides `shapiro`, `wilcoxon`, and `ttest_rel` used in
  `nl2sql/evaluation/research_stats.py`.


## Text-to-SQL Surveys and Benchmarks

**[1]** G. Katsogiannis-Meimarakis and G. Koutrika, 'A survey on deep learning approaches for
text-to-SQL', The VLDB Journal, vol. 32, no. 4, pp. 905–936, 2023. doi: 10.1007/s00778-022-00776-8.
→ Background motivation for the NL-to-SQL task and evaluation setup.

**[9]** X. Zhu et al., 'Large Language Model Enhanced Text-to-SQL Generation: A Survey', Oct. 2024,
arXiv: arXiv:2410.06011.
→ Positions LLM-based approaches (few-shot, fine-tuning, agents) within the broader field.

**[12]** Z. Hong et al., 'Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL',
Sep. 2025, arXiv: arXiv:2406.08426.
→ Contextualises the agentic (ReAct) pipeline as part of next-generation interfaces.

**[3]** J. Li et al., 'Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale
Database Grounded Text-to-SQLs', arXiv:2305.03111, 2023. doi: 10.48550/arXiv.2305.03111.
→ BIRD benchmark — provides broader context for LLM text-to-SQL performance claims.

**[23]** D. Gao et al., 'Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation',
Nov. 2023, arXiv: arXiv:2308.15363. doi: 10.48550/arXiv.2308.15363.
→ Benchmark comparison of few-shot prompting strategies; supports the k=3 few-shot design.

**[13]** S. Ojuri et al., 'Optimizing text-to-SQL conversion techniques through the integration of
intelligent agents and large language models', Information Processing & Management, vol. 62,
no. 5, 2025. doi: 10.1016/j.ipm.2025.104136.
→ Supports the agent-based extension (ReAct pipeline) as a current research direction.


## Prompting and In-Context Learning

**[8]** T. Brown et al., 'Language Models are Few-Shot Learners', NeurIPS 2020, vol. 33,
pp. 1877–1901.
→ Foundational motivation for k-shot exemplar sampling in `nl2sql/agent/react_pipeline.py`
  and `nl2sql/evaluation/eval.py`.

**[4]** Q. Yin et al., 'Deeper Insights Without Updates: The Power of In-Context Learning Over
Fine-Tuning', Oct. 2024, arXiv: arXiv:2410.04691. doi: 10.48550/arXiv.2410.04691.
→ Directly supports the core dissertation finding that ICL (k=3) outperforms QLoRA fine-tuning.

**[7]** M. Mosbach et al., 'Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and
Evaluation', May 2023, arXiv: arXiv:2305.16938. doi: 10.48550/arXiv.2305.16938.
→ Motivates the experimental design comparing few-shot ICL against fine-tuning.


## Agentic / ReAct

**[19]** S. Yao et al., 'ReAct: Synergizing Reasoning and Acting in Language Models', ICLR 2023,
arXiv: arXiv:2210.03629. doi: 10.48550/arXiv.2210.03629.
→ Core motivation for `run_react_pipeline` in `nl2sql/agent/react_pipeline.py`.
  The Thought→Action→Observation loop maps directly to the validate→run→repair cycle.

**[26]** Z. Xi et al., 'The rise and potential of large language model based agents: a survey',
Science China Information Sciences, vol. 68, no. 2, 2025. doi: 10.1007/s11432-024-4222-0.
→ Broader context for the agentic pipeline as an LLM-based agent system.

**[5]** M. Pourreza and D. Rafiei, 'DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with
Self-Correction', Nov. 2023, arXiv: arXiv:2304.11015. doi: 10.48550/arXiv.2304.11015.
→ Motivates execution-guided SQL repair in `repair_sql` in `nl2sql/agent/react_pipeline.py`.

**[6]** B. Zhai et al., 'ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback',
Mar. 2025, arXiv: arXiv:2503.19988. doi: 10.48550/arXiv.2503.19988.
→ Supports execution feedback as a repair signal in the ReAct loop.


## Fine-Tuning Methods

**[11]** E. J. Hu et al., 'LoRA: Low-Rank Adaptation of Large Language Models', Oct. 2021,
arXiv: arXiv:2106.09685. doi: 10.48550/arXiv.2106.09685.
→ Foundational method underlying QLoRA adapter training in `notebooks/05_qlora_train_eval.ipynb`.

**[16]** T. Dettmers et al., 'QLoRA: Efficient Finetuning of Quantized LLMs', May 2023,
arXiv: arXiv:2305.14314. doi: 10.48550/arXiv.2305.14314.
→ Motivates 4-bit quantised fine-tuning under Colab GPU memory constraints.

**[10]** D. Biderman et al., 'LoRA Learns Less and Forgets Less', Sep. 2024,
arXiv: arXiv:2405.09673. doi: 10.48550/arXiv.2405.09673.
→ Supports the finding that QLoRA may suppress ICL benefit (Llama QLoRA k0→k3 non-significant).

**[14]** J. Goswami et al., 'Parameter-efficient fine-tuning large language model approach for
hospital discharge paper summarization', Applied Soft Computing, vol. 157, 2024.
doi: 10.1016/j.asoc.2024.111531.
→ PEFT context — supports QLoRA as a parameter-efficient alternative to full fine-tuning.

**[28]** Y. Wang et al., 'Two-stage LLM Fine-tuning with Less Specialization and More Generalization',
Mar. 2024, arXiv: arXiv:2211.00635. doi: 10.48550/arXiv.2211.00635.
→ Contextualises the trade-off between fine-tuning specialisation and generalisation.


## NL-to-SQL Prior Systems (Related Work)

**[2]** X. V. Lin, R. Socher, and C. Xiong, 'Bridging Textual and Tabular Data for Cross-Domain
Text-to-SQL Semantic Parsing', EMNLP Findings 2020. doi: 10.18653/v1/2020.findings-emnlp.438.
→ BRIDGE system — schema-linking approach; motivates schema summary design in `nl2sql/core/schema.py`.

**[18]** B. Wang et al., 'RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL
Parsers', ACL 2020. doi: 10.18653/v1/2020.acl-main.677.
→ Schema-linking prior work; contextualises the schema summary prompt design.

**[20]** H. Li et al., 'RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL',
AAAI 2023. doi: 10.1609/aaai.v37i11.26535.
→ Decoupled generation approach; contextualises why constrained decoding was tested and deprioritised.

**[15]** T. Scholak, N. Schucher, and D. Bahdanau, 'PICARD: Parsing Incrementally for Constrained
Auto-Regressive Decoding from Language Models', Sep. 2021, arXiv: arXiv:2109.05093.
doi: 10.48550/arXiv.2109.05093.
→ Constrained decoding for SQL; provides background for why grammar-constrained decoding was considered
  during design, even though the final primary implementation uses raw-model and optional reliability-layer
  profiles rather than a grammar-server path.


## Base Models and Libraries

**[25]** A. Grattafiori et al., 'The Llama 3 Herd of Models', Nov. 2024, arXiv: arXiv:2407.21783.
doi: 10.48550/arXiv.2407.21783.
→ Meta-Llama-3-8B-Instruct — primary model for Llama conditions.

**[17]** Qwen et al., 'Qwen2.5 Technical Report', Jan. 2025, arXiv: arXiv:2412.15115.
doi: 10.48550/arXiv.2412.15115.
→ Qwen2.5-7B-Instruct — primary model for Qwen conditions.

**[27]** T. Wolf et al., 'Transformers: State-of-the-Art Natural Language Processing', EMNLP Demos
2020. doi: 10.18653/v1/2020.emnlp-demos.6.
→ HuggingFace Transformers library — `apply_chat_template`, `StoppingCriteria`, `model.generate`
  in `nl2sql/core/llm.py`. Also covers PEFT/TRL used in QLoRA training.
