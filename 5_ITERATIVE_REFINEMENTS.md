# Iterative Refinements Engineering Log (Reformatted)

Each decision is presented in a four-part explanation format.

---

### Decision 5.1 - Stage-gated ablation of the agent loop

**Plain-language**  
When many controls were enabled at once, it was unclear which change improved EX. Stage gating isolates effects.

**Technical description**  
Controls are enabled incrementally via explicit configuration toggles (e.g., repair on/off, schema subset on/off, projection contract on/off). This supports ablation-style runs where each change can be evaluated in isolation.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig`)  
`notebooks/03_agentic_eval.ipynb` (cell `# 6) Agent implementation (imported)` sets config values)

**Justification**  
Ablation is standard in NL->SQL evaluation (Zhu et al., 2024). The trade-off is additional run time and complexity.

---

### Decision 5.2 - Stop-on-semicolon and prompt-echo trimming

**Plain-language**  
Valid SQL was often followed by extra text, causing syntax errors and VA=0. Trimming fixes this without changing model weights.

**Technical description**  
The cleaner cuts at the first semicolon and removes prompt echo or trailing instructions before execution.

**Code locations**  
`nl2sql/agent.py` (`_StopOnSemicolon` in `ReactSqlAgent.generate_candidates`)  
`nl2sql/agent_utils.py` (`clean_candidate_with_reason`)  
`nl2sql/llm.py` (`extract_first_select`)

**Justification**  
Constrained decoding ideas (Scholak et al., 2021) motivate lightweight filtering. The trade-off is rare cases where valid semicolons are trimmed incorrectly.

---

### Decision 5.3 - Relax EX to row-only comparison

**Plain-language**  
EX was failing on correct results when column names or order differed. Comparing rows only better reflects semantic equivalence.

**Technical description**  
`execution_accuracy` compares row multisets using a Counter and does not require column name equality.

**Code locations**  
`nl2sql/eval.py` (`execution_accuracy`)  
`notebooks/03_agentic_eval.ipynb` (evaluation loop)

**Justification**  
Zhong et al. (2020) highlight surface-level variability. The trade-off is that column semantics can be ignored if values coincide.

---

### Decision 5.4 - Port TS harness into the core eval module

**Plain-language**  
Keeping TS only in the notebook made it harder to reuse and audit. Moving it into `nl2sql.eval` improves traceability.

**Technical description**  
TS functions are now in `nl2sql/eval.py` and imported by the notebook and scripts.

**Code locations**  
`nl2sql/eval.py` (`test_suite_accuracy_for_item`)  
`notebooks/03_agentic_eval.ipynb` (TS evaluation)

**Justification**  
Test-suite evaluation is a core metric (Zhong et al., 2020). The trade-off is that this TS remains suite-based, not fully distilled.
