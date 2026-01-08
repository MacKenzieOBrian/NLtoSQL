This folder is where Colab notebooks write evaluation outputs (JSON runs, tables, figures).

By default, `results/` is gitignored to avoid accidentally committing large outputs.
If you want to version key outputs for the dissertation (e.g., baseline `results_*.json` or figures),
remove or adjust the `results/` rule in `.gitignore`.

## Recommended dissertation outputs to keep

- Baseline JSONs: `results/baseline/results_zero_shot_200.json`, `results/baseline/results_few_shot_k3_200.json`
- QLoRA JSONs: `results/qlora/results_zero_shot_200.json`, `results/qlora/results_few_shot_k3_200.json`
- Key figures (PNG) used in the write-up (e.g., bar charts for VA/EM/EX)

## Metric naming note

This project reports:
- `VA`: executability/validity (did it run?)
- `EM`: strict normalized string match
- `EX`: execution accuracy (result set comparison; Ojuri-style)

If you have older baseline JSONs from before EX was redefined as execution accuracy, do not compare them directly; rerun the baselines using the current harness.

Why there are multiple JSONs per method:
- `results_*zero_shot*` (`k=0`) isolates the method’s performance without exemplars.
- `results_*few_shot*` (`k=3`) shows whether adding exemplars helps on top of that method (baseline or QLoRA).
