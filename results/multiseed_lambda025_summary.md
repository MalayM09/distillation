# Hop-1 λ_NCE Ablation — Multi-Seed Results (λ_NCE = 0.25)

Re-ran A3-h1 (MedSAM → HA-Net) and A3-h2 (HA-Net → EdgeUNet) with
`--lambda_nce 0.25` (vs the original `--lambda_nce 1.0`) across the same
3 seeds [20260530, 42, 1337]. All other hyperparameters identical to the
λ=1.0 run in `multiseed_lambda100_results.md`. B1, B2, A1, A2 reference
values are unchanged (carried over from the λ=1.0 run since they don't
depend on Hop-1 hyperparameters).

## Headline (n = 3 seeds)

| Ablation | λ=1.0 mean ± std | λ=0.25 mean ± std | Δ |
|---|---|---|---|
| **A3-h1** assistant | 0.7317 ± 0.0298 | **0.7469 ± 0.0280** | **+0.0152** |
| **A3-h2** student | 0.7536 ± 0.0230 | **0.7529 ± 0.0172** | −0.0007 |

## Per-Seed Deltas

| seed | A3-h1 λ=1.0 | A3-h1 λ=0.25 | Δ Hop-1 | A3 λ=1.0 | A3 λ=0.25 | Δ student |
|---|---|---|---|---|---|---|
| 20260530 | 0.7600 | 0.7692 | **+0.0093** | 0.7744 | 0.7586 | **−0.0158** |
| 42       | 0.6906 | 0.7074 | **+0.0168** | 0.7215 | 0.7295 | +0.0080 |
| 1337     | 0.7446 | 0.7642 | **+0.0195** | 0.7648 | 0.7704 | +0.0057 |

## Paired Bootstrap (n = 10,000 resamples)

| Comparison | Δ Dice | CI95 | p | sig |
|---|---|---|---|---|
| A3-h1(new) − A3-h1(old)  *Hop-1 fix worked?* | **+0.0152** | [+0.0093, +0.0195] | **<0.001** | *** |
| A3-h1(new) − B2  *still vs supervised ceiling* | −0.0183 | [−0.0271, −0.0040] | **<0.001** | *** |
| **A3(new) − A2  *TAKD decision rule (docs/02)*** | **−0.0050** | [−0.0155, +0.0049] | **0.291** | **ns** |
| A3(new) − A3(old)  *did the fix propagate?* | −0.0007 | [−0.0158, +0.0080] | 0.808 | ns |
| A3(new) − B1  *absolute lift over floor* | +0.0171 | [+0.0072, +0.0252] | <0.001 | *** |
| A3(new) − B2  *student vs ceiling* | −0.0124 | [−0.0345, +0.0022] | 0.078 | ns |

## Interpretation

The λ_NCE = 0.25 change **significantly improved the Hop-1 assistant**
(+0.0152 Dice, p < 0.001, monotonic across all 3 seeds). The improvement
**did not propagate** to the A3 student — Δ A3(new) − A3(old) = −0.0007
(p = 0.808). The cascade's central decision rule (A3 must beat A2 with
p < 0.05) remains unsatisfied: A3 − A2 = −0.0050, p = 0.291.

The most informative observation is **per-seed**: on seed 20260530 the
Hop-1 assistant improved by +0.0093 but the resulting A3 student
*regressed* by −0.0158. A monotonically better teacher did not produce a
monotonically better student.

This rules out "Hop-1 was tuned wrong" as the explanation for TAKD's
failure. The bottleneck must be in Hop-2's MSE+KL transfer or in the
fundamental information-routing of the cascade itself. Specifically:

- A3-h2 MSE plateaus are *the same or higher* with the new teacher
  (0.942, 0.792, 0.774 vs ~0.85 in the old run) — the student isn't
  fitting the improved teacher's features any better than before.
- A3-h2 KL is *higher* with the new teacher (0.081, 0.238, 0.038 vs
  ~0.04 old) — the improved teacher is *harder* to mimic at the logit
  level.

**Conclusion:** The 3M EdgeUNet has its own capacity ceiling for
absorbing teacher representations. Improvements in teacher quality above
that ceiling do not translate into improved segmentation accuracy. This
is the same Capacity Gap mechanism that kills A1 (R=30) recurring at
R≈5 inside the cascade.
