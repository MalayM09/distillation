# 02 — Ablation Matrix (BUSI)

All experiments are run on the BUSI dataset under identical splits, augmentation, optimizer, and resolution-faithful evaluation. Reported metrics: validation Dice and IoU at native ultrasound resolution. Latency is measured on Kaggle T4 at FP16 for the deployable student; HA-Net and MedSAM latency are reported FP16 for reference only and are not deployment targets.

> **Status (locked 2026-05-28):** the *a priori* hypotheses below have been graded against the 3-seed multi-seed run with paired bootstrap (n=10,000). Two of the three central claims survive; one (TAKD) is honestly falsified including under a Hop-1 hyperparameter fix. The final cell of each row records the empirical outcome. Canonical numbers live in `results/multiseed_lambda100_results.md` and `results/multiseed_lambda025_summary.md`.

| ID | Configuration | Role in Thesis | Teacher Signal | Distillation Loss | Params | Edge | Hypothesis (a priori) | Outcome (n=3 seeds) |
|----|---------------|----------------|----------------|-------------------|--------|------|-----------------------|---------------------|
| **B1** | Vanilla U-Net (3M), supervised only | Baseline — *floor* | None | $\mathcal{L}_{\text{seg}}$ only | <3M | ✅ | Defines the no-distillation accuracy that any KD variant must exceed. | **Dice 0.7357 ± 0.0222** (anchor). |
| **B2** | HA-Net (15M), supervised only | Baseline — *ceiling* | None | $\mathcal{L}_{\text{seg}}$ only | ~15M | ❌ | Upper bound any 15M-class CNN can reach without foundation knowledge. | **Dice 0.7653 ± 0.0240** (ceiling). +0.030 over B1. |
| **A1** | MedSAM → U-Net (direct KD) | Ablation — proves *Capacity Gap* | MedSAM (91M cached) | **Dense InfoNCE** + $\mathcal{L}_{\text{seg}}$ | <3M | ✅ | Direct 91M → 3M KD will *not* exceed B1 mean Dice; gains, if any, are variance-only. | ✅ **CONFIRMED.** Dice 0.7353 ± **0.0063**. A1 − B1 = −0.0005, p = 0.733 (ns). **Variance reduced 3.5×** vs B1 (std 0.0063 vs 0.0222) — the "variance-only" prediction was sharper than the original hypothesis. |
| **A2** | HA-Net → U-Net (domain-expert KD) | Ablation — *specialized vs generic* knowledge | HA-Net (15M, B2) | **MSE + KL** (CNN→CNN) + $\mathcal{L}_{\text{seg}}$ | <3M | ✅ | A domain-matched 15M teacher transfers measurable accuracy to the 3M student, isolating capacity ratio from foundation-scale knowledge. | ✅ **CONFIRMED, strongly.** Dice 0.7579 ± 0.0235. A2 − B1 = **+0.0221**, CI95 [+0.0203, +0.0233], **p < 0.001**. Capacity-matched CNN→CNN distillation works beautifully and is the most statistically reliable result in the matrix (CI width only 0.003). |
| **A3** | MedSAM → HA-Net → U-Net (TAKD, *proposed*) | Novel pipeline — full thesis | MedSAM via HA-Net assistant | Hop 1: **Dense InfoNCE**. Hop 2: **MSE + KL**. Each hop adds $\mathcal{L}_{\text{seg}}$. | <3M | ✅ | TAKD strictly exceeds A1, A2, and B1 on mean Dice/IoU, demonstrating that the assistant rescues foundation-model knowledge that direct KD discards. | ❌ **FALSIFIED (even after Hop-1 fix).** λ_NCE=1.0: Dice 0.7536 ± 0.0230, A3 − A2 = −0.0043, p = 0.067 (ns). λ_NCE=0.25: Dice 0.7529 ± 0.0172, A3 − A2 = −0.0050, p = 0.291 (ns). Hop-1 fix lifted the assistant by **+0.0152 (p<0.001)** but the improvement **did not propagate** to the student (Δ A3 = −0.0007, p = 0.808). TAKD still beats the floor (A3 − B1 = +0.017, p<0.001) but does not exceed direct A2. |

## Distillation Loss Selection (by Architectural Topology)

The choice of distillation objective is *not* a free hyperparameter — it is fixed by the topological relationship between teacher and student. This rule is binding for every row above and for any future experiment in this project.

### Rule 1 — ViT → CNN hops use Dense InfoNCE (Contrastive Distillation)

Applies to: **A1** (MedSAM → U-Net) and **Hop 1 of A3** (MedSAM → HA-Net).

Rationale: MedSAM emits 768-dim ViT patch tokens whose absolute magnitudes, channel ordering, and spatial layout are not commensurable with a CNN's feature maps. Direct MSE or KL between a ViT embedding and a CNN feature map is ill-posed — there is no canonical alignment between the two coordinate systems. Dense InfoNCE sidesteps this by training a small projection head $g_\phi$ on the student/assistant side so that, for every spatial location $i$, the projected feature $g_\phi(f_S^{(i)})$ is pulled toward the teacher token $z_T^{(i)}$ at the same location and pushed away from teacher tokens at all other locations within the batch:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\langle g_\phi(f_S^{(i)}),\, z_T^{(i)}\rangle / \tau)}{\sum_{j} \exp(\langle g_\phi(f_S^{(i)}),\, z_T^{(j)}\rangle / \tau)}$$

This is invariant to linear basis change in the teacher's embedding space and therefore robust to the ViT↔CNN representational mismatch. The composite training loss is $\mathcal{L} = \mathcal{L}_{\text{seg}} + \lambda_{\text{NCE}} \mathcal{L}_{\text{InfoNCE}}$.

### Rule 2 — CNN → CNN hops use MSE Feature Matching + KL Logit KD

Applies to: **A2** (HA-Net → U-Net) and **Hop 2 of A3** (HA-Net → U-Net).

Rationale: when teacher and student share the same architectural family (both convolutional, both translation-equivariant, both spatially aligned at matching pyramid levels), their feature maps live in coordinate systems related by an approximately diagonal linear transform. Direct alignment is then mathematically optimal and statistically more efficient than contrastive objectives, which discard absolute scale information. We use:

- **Feature MSE** at one or more matched encoder/decoder stages, after a $1\times1$ channel-projection conv that maps the student channel count to the teacher's: $\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_i \lVert h_\psi(f_S^{(i)}) - f_T^{(i)} \rVert_2^2$.
- **KL-Divergence on logits** at full segmentation resolution, with temperature $T$: $\mathcal{L}_{\text{KL}} = T^2 \cdot \mathrm{KL}(\sigma(z_T/T) \,\Vert\, \sigma(z_S/T))$.

Composite: $\mathcal{L} = \mathcal{L}_{\text{seg}} + \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}$. No contrastive term — the CNN→CNN topology makes negatives unnecessary and the InfoNCE temperature an additional, unhelpful degree of freedom.

### Why this Distinction Was Expected to Carry the TAKD Claim (and why it didn't)

The Capacity Gap in A1 is not purely a parameter-count problem; it is *also* a topology problem. Even at the optimal InfoNCE temperature, a 3M CNN cannot recover the absolute-scale information that MSE/KL would have provided, because no MSE/KL is available across the ViT↔CNN boundary in the first place. TAKD's *a priori* mechanism was to *route* the high-information CNN-native objectives (MSE + KL) through Hop 2, where they are mathematically admissible, while restricting the lossier contrastive objective to Hop 1, where it is unavoidable. This was the expected reason A3 should exceed A1 (and exceed A2): the foundation-model knowledge would be delivered to the student through a strictly stronger objective than direct InfoNCE.

**This mechanism failed empirically (see Findings below).** The Hop-1 fix experiment shows that even when the assistant is meaningfully improved (+0.015 Dice, p<0.001 at λ_NCE = 0.25), the Hop-2 MSE+KL transfer does not propagate that improvement to the student (Δ student = −0.001, p = 0.808). Hop-2 acts as a *low-pass capacity filter* — the 3M student has its own absorption ceiling for teacher representations, and improvements in teacher quality above that ceiling do not translate into improved Dice. This is the same Capacity Gap mechanism that kills A1 at R≈30, recurring at R≈5 inside the cascade. The topology-routing argument is correct as far as it goes, but it does not predict the student-capacity ceiling, which is the actual binding constraint.

## Findings (empirical, replaces "Decision Rules")

The original Decision Rules section listed pass/fail criteria. After multi-seed evaluation, those criteria have been graded:

- ✅ **A1 ≈ B1 (Capacity Gap) — CONFIRMED.** A1 − B1 = −0.0005, p = 0.733 (ns), n = 3 seeds. The "variance-only" sub-prediction was sharper than originally framed: A1's seed variance is **3.5× lower than B1's** (std 0.0063 vs 0.0222) — direct InfoNCE distillation acts as a strong variance regularizer even when mean Dice is unchanged. This is itself a robust, publishable finding.

- ✅ **A2 > B1 (CNN→CNN distillation works) — CONFIRMED, strongly.** A2 − B1 = +0.0221, CI95 [+0.0203, +0.0233], p < 0.001. The per-seed deltas are remarkably uniform (+0.0227, +0.0203, +0.0233), giving a CI width of just 0.003 — the most statistically reliable result in the entire matrix. **Capacity-matched CNN→CNN distillation at R ≈ 5 lifts the student robustly above its supervised baseline.** Combined with the A1 result, this isolates **capacity ratio** (not "distillation" or "foundation-model use") as the load-bearing variable.

- ❌ **A3 > A2 (TAKD decision rule) — FALSIFIED at both λ_NCE = 1.0 and λ_NCE = 0.25.** At λ_NCE = 1.0, A3 − A2 = −0.0043, p = 0.067 (ns, but trending negative). At λ_NCE = 0.25 (the obvious tuning fix to address Hop-1's contrastive-vs-segmentation loss imbalance), A3 − A2 = −0.0050, p = 0.291. The Hop-1 fix significantly improved the assistant (A3-h1 lift: +0.0152, p < 0.001, monotonic across all 3 seeds) but **did not propagate to the student** (Δ A3 = −0.0007, p = 0.808). The cascade's binding constraint is in Hop 2, not Hop 1.

- ⚠ **A3 > B1 (TAKD still beats no-distillation) — true but weak.** A3 − B1 = +0.0178, p < 0.001 — TAKD provides real benefit over the floor, but the magnitude is no greater than direct A2 and the cascade is strictly more expensive. **For deployment, A2 dominates A3 on every axis we care about: equal or better Dice, simpler training pipeline, fewer hyperparameters, no MedSAM dependency at training time.**

- ⛔ **B2 remains an upper-bound reference, not a competitor.** A2 − B2 = −0.0074, p = 0.300 (ns) — A2 statistically matches its supervised teacher's ceiling. A3 − B2 = −0.0117, p < 0.001 — A3 is significantly below the ceiling, providing further evidence that the cascade *loses* information rather than preserving it.

- 🔒 **Loss-by-topology remains binding for any future experiment.** No A1/Hop-1-of-A3 run is permitted to add MSE or KL across the ViT↔CNN boundary; no A2/Hop-2-of-A3 run is permitted to substitute InfoNCE for MSE+KL. The empirical results above are predicated on the rule being held fixed; relaxing it would re-open the topology-vs-capacity attribution.

### Recommended Headline Framing (for the paper)

1. **The Capacity Gap is real.** Direct MedSAM → EdgeUNet distillation (R ≈ 30) provides no mean Dice lift, only variance reduction (3.5×).
2. **Capacity-matched distillation works robustly.** HA-Net → EdgeUNet (R ≈ 5, CNN→CNN, MSE+KL) lifts the student by +0.022 Dice with p < 0.001.
3. **Naive TAKD does not bridge the gap, and the failure is not Hop-1 tuning.** A Hop-1 fix (λ_NCE 1.0 → 0.25) significantly improves the assistant but does not propagate to the student — the cascade is bottlenecked by Hop-2's MSE+KL transfer capacity, which has its own student-side Capacity Gap.

## Shared Protocol

- Splits: BUSI 5-fold; report mean ± std across folds.
- Training resolution: 256×256 (MedSAM feature cache is 256-native).
- Evaluation: predicted masks upsampled to original resolution before Dice/IoU (see `docs/03_CONSTRAINTS_AND_RULES.md`).
- Optimizer / schedule / augmentation: held constant across all five rows. Only the **teacher source** and the **distillation loss** vary, and the latter is determined by topology per Rules 1 and 2 above — not chosen per-row.
