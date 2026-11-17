# Result validity

Existing checkpoint-derived metrics and figures predate the 2026-11 causal
Transformer/RETAIN fixes. They are historical artifacts, not valid comparisons
with the corrected implementations. Do not cite the old baseline rankings or
subgroup superiority claims. The original files are retained for reproducibility.

Required before reporting updated comparisons: retrain affected baselines on the
same patient split, rerun evaluation/subgroup analyses, regenerate figures from
those new outputs, and record seeds, code SHA and the causal protocol. No new
clinical performance values were fabricated during the security remediation.
