## Logbook (Condensed)

This file records key project changes and why they were made (for dissertation traceability). More detailed notes live in `LOGBOOK_REFLECTIONS.md`.

### 2026-01-08 — Training + validation workflow simplified

- Added a starter QLoRA training set at `data/train/classicmodels_train_200.jsonl` (200 NLQ→SQL pairs) with strict train/test separation from `data/classicmodels_test_200.json`.
- Remade `notebooks/04_build_training_set.ipynb` into a **validator** notebook (leakage check + SELECT-only + executability/VA check on the live ClassicModels DB).
- Updated documentation to reflect the new workflow: curate/edit the JSONL directly, validate it, then run QLoRA (`notebooks/05_qlora_train_eval.ipynb`).

### Baseline checkpoint (n=200)

- Zero-shot: `VA≈0.81`, `EX≈0.00`
- Few-shot (k=3): `VA≈0.855`, `EX≈0.325`

### 2026-01-09 — Colab QLoRA fixes

- Bumped `peft` to `0.17.x` and left `torch` to Colab’s CUDA build to avoid `torch.xpu` import errors.
- Notebook now auto-selects `bf16` (Ampere+) vs `fp16` (T4) and prints GPU capability to prevent `bf16` validation crashes.
- Forced 4-bit model loading onto GPU (`device_map={"":0}`) to stop bitsandbytes from offloading layers to CPU/disk (avoids `quantizer_bnb_4bit` errors).
- Pushes: `f6292b9`, `8c632cb`, `886ee34`.

### 2026-01-09 — QLoRA run results (ClassicModels-200)

- Training (1 epoch, r=16, 4-bit): adapters saved to `results/adapters/qlora_classicmodels`.
- Eval (k=0): `VA=0.730`, `EM=0.005`, `EX=0.030` (adapter alone underperforms baseline VA, EX still low).
- Eval (k=3): `VA=0.860`, `EM=0.260`, `EX=0.305` (few-shot + adapters lifts VA/EX vs k=0, roughly baseline-level EX).
- Takeaway: current QLoRA config did not beat the prompt-only baseline; most gain still comes from few-shot prompting. Next steps: try 2–3 epochs, lower LR (e.g., 1e-4), warmup, and a clean exemplar pool separate from the test set.

### 2026-01-12 — QLoRA second run (ClassicModels-200)

- Training: increased LoRA capacity and steps (r=32, α=64, 3 epochs, warmup) on the same 200-pair train set.
- Eval (k=0): `VA=0.865`, `EM=0.000`, `EX=0.065` (saved to `results/qlora/results_zero_shot_200.json`).
- Eval (k=3): `VA=0.875`, `EM=0.305`, `EX=0.380` (saved to `results/qlora/results_few_shot_k3_200.json`).
- Takeaway: adapter-only VA improved vs run 1; EX remains low without exemplars. With k=3, EX now surpasses the prompt-only few-shot baseline (~0.325→0.380). Next: consider more steps or agentic/TS evaluation to lift semantic accuracy, especially at k=0.
