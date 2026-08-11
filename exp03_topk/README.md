# Full NetAttn Top-K

Inference-only ranking controls for the retained seed-42 Full NetAttn checkpoint.

`run_netattn_ranking_control_seed42.py` reproduces the paper's Full, learned Top-7, equal-mass Top-7, structure-matched Random-7, and Bottom-7 comparison. The seven learned coefficients are retained without renormalization. Generated CSV, JSON, and Markdown results are included.

Older notebooks that trained a Top-K model or renormalized truncated coefficients were removed because they do not implement the final paper method.

Path strings are displayed from `L1.B0` through `L4.B1` (shallow to deep),
while the stable internal index remains checkpoint-compatible. The corrected
path identities and analytical cost are documented in
[`../paper/NetAttn/ERRATUM.md`](../paper/NetAttn/ERRATUM.md).
