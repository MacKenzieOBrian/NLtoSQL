# Research Journey

This dissertation follows one practical question: how far can an open-source NL-to-SQL stack go under constrained hardware when evaluated with a fixed, auditable protocol. The project is a transparent replication pathway inspired by Ojuri-style study structure, but implemented with local open-source models and QLoRA adaptation.

The journey began with a baseline-first phase. I built a stable prompt-to-evaluation pipeline on a fixed benchmark so there was a defensible reference before adaptation. This created a clear control condition for later comparisons.

The second phase introduced QLoRA. The goal was not maximum scale, but whether parameter-efficient adaptation improves semantic behavior under realistic compute limits. The benchmark and evaluator contract stayed fixed so observed differences could be interpreted as method effects.

The third phase was scope reduction for dissertation validity. Earlier exploratory infrastructure (agent-style loops and extra helper layers) was removed from the final claim path. The final dissertation protocol uses model-only raw SQL evaluation so the hypothesis is isolated and statistically testable.

The fourth phase focused on evidence quality. Analysis was simplified to the supervisor-required outputs: mean/median summaries, Shapiro-Wilk checks, and paired t-tests. This makes claims easier to defend because every result maps directly to a reproducible file and script.

The contribution is therefore methodological and evidential: a reproducible open-source comparison workflow (base vs QLoRA, k=0 vs k=3) with explicit scope boundaries and statistically grounded interpretation.
