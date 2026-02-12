# Run Index

This file is the canonical map of active and archived results.

## Active Inputs (used by scripts)

- Baseline k=0: `results/baseline/baseline_k0.json`
- Baseline k=3: `results/baseline/baseline_k3.json`
- QLoRA k=0: `results/qlora/qlora_k0.json`
- QLoRA k=3: `results/qlora/qlora_k3.json`
- ReAct (current comparison input): `results/agent/react_infra_n20_v9.json`

## Run Catalog

| Run ID | Type | File | n | VA | EM | EX | TS | Commit | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2026-01-21_baseline_commit_6dffe02 | baseline | `results/archive/baseline_commit_6dffe02/results_zero_shot_200.json` | 200 | 0.810 | 0.000 | 0.075 |  | 6dffe02 | legacy root baseline before refresh |
| 2026-01-21_baseline_commit_6dffe02 | baseline | `results/archive/baseline_commit_6dffe02/results_few_shot_k3_200.json` | 200 | 0.855 | 0.325 | 0.370 |  | 6dffe02 | legacy root baseline before refresh |
| 2026-02-10_baseline_commit_d2eaa50 | baseline | `results/baseline/baseline_k0.json` | 200 | 0.875 | 0.000 | 0.485 |  | d2eaa50 | current active baseline |
| 2026-02-10_baseline_commit_d2eaa50 | baseline | `results/baseline/baseline_k3.json` | 200 | 0.895 | 0.355 | 0.610 |  | d2eaa50 | current active baseline |
| 2026-02-10_qlora_commit_501a24a | qlora | `results/qlora/qlora_k0.json` | 200 | 0.865 | 0.030 | 0.455 |  | 501a24a | current active qlora |
| 2026-02-10_qlora_commit_501a24a | qlora | `results/qlora/qlora_k3.json` | 200 | 0.870 | 0.350 | 0.550 |  | 501a24a | current active qlora |
| 2026-02-01_react_full_n200 | react | `results/agent/results_react_200.json` | 200 | 0.805 | 0.105 | 0.130 |  |  | full 200-item react run |
| 2026-02-05_react_slice_n20_v3 | react | `results/agent/results_react_200 (3).json` | 20 | 0.750 | 0.200 | 0.350 | 0.350 |  | 20-item slice run |
| 2026-02-09_react_slice_n20_v9 | react | `results/agent/react_infra_n20_v9.json` | 20 | 0.950 | 0.350 | 0.550 | 0.579 |  | 20-item slice run used in comparison script |

## Archived Duplicates / Raw Logs

- `results/archive/baseline_commit_d2eaa50_duplicate/` contains duplicate baseline JSONs that previously lived at `results/results_* (1).json`.
- `results/agent/results_react_200 (2).json` is a debug text log (not valid JSON). Copy preserved at `results/runs/2026-02-04_react_debug_log_v2/agent/results_react_200.log`.

## Naming Rule (from now on)

Use `results/runs/<YYYY-MM-DD>_<method>_<commit or tag>/` and keep only one active file per method in `results/baseline/`, `results/qlora/`, and `results/agent/` for scripts.
