# 02 — Ablation Matrix (BUSI)

All experiments are run on the BUSI dataset under identical splits, augmentation, optimizer, and resolution-faithful evaluation. Reported metrics: validation Dice and IoU at native ultrasound resolution. Latency is measured on Kaggle T4 at FP16 for the deployable student; HA-Net and MedSAM latency are reported FP16 for reference only and are not deployment targets.

| ID | Configuration | Role in Thesis | Teacher Signal | Distillation Loss | Params | Edge-Deployable | Primary Hypothesis |
|----|---------------|----------------|----------------|-------------------|--------|-----------------|--------------------|
| **B1** | Vanilla U-Net (3M), supervised only | Baseline — establishes the *floor* | None | $\mathcal{L}_{\text{seg}}$ only (Dice + BCE) | <3M | ✅ | Defines the no-distillation accuracy that any KD variant must exceed. |
| **B2** | HA-Net (15M), supervised only | Baseline — establishes the domain-specific *ceiling* | None | $\mathcal{L}_{\text{seg}}$ only (Dice + BCE) | ~15M | ❌ | Defines the upper bound any 15M-class CNN can reach on BUSI without foundation knowledge. |
| **A1** | MedSAM → U-Net (direct KD) | Ablation — proves the *Capacity Gap* failure | MedSAM (91M, cached features) | **Dense InfoNCE** (cross-architecture contrastive) + $\mathcal{L}_{\text{seg}}$ | <3M | ✅ | Direct 91M → 3M KD will *not* exceed B1 mean Dice; gains, if any, are variance-only. |
| **A2** | HA-Net → U-Net (domain-expert KD) | Ablation — tests *specialized vs. generic* knowledge | HA-Net (15M, trained per B2) | **MSE feature matching + KL logit KD** (CNN→CNN) + $\mathcal{L}_{\text{seg}}$ | <3M | ✅ | A domain-matched 15M teacher transfers measurable accuracy to the 3M student, isolating the role of capacity ratio from the role of foundation-scale knowledge. |
| **A3** | MedSAM → HA-Net → U-Net (TAKD, *proposed*) | Novel pipeline — full thesis | MedSAM via HA-Net assistant | Hop 1: **Dense InfoNCE** (MedSAM → HA-Net, ViT→CNN). Hop 2: **MSE + KL** (HA-Net → U-Net, CNN→CNN). Each hop adds $\mathcal{L}_{\text{seg}}$. | <3M | ✅ | TAKD strictly exceeds A1, A2, and B1 on mean Dice/IoU, demonstrating that the assistant rescues foundation-model knowledge that direct KD discards. |

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

### Why this Distinction Carries the TAKD Claim

The Capacity Gap in A1 is not purely a parameter-count problem; it is *also* a topology problem. Even at the optimal InfoNCE temperature, a 3M CNN cannot recover the absolute-scale information that MSE/KL would have provided, because no MSE/KL is available across the ViT↔CNN boundary in the first place. TAKD's mechanism is to *route* the high-information CNN-native objectives (MSE + KL) through Hop 2, where they are mathematically admissible, while restricting the lossier contrastive objective to Hop 1, where it is unavoidable. This is the precise reason A3 is expected to exceed A1: the same foundation-model knowledge is delivered to the student through a strictly stronger objective.

## Decision Rules

- **A3 must beat A2 by a statistically significant margin** (paired bootstrap, p < 0.05) for the TAKD claim to hold; otherwise the assistant is acting only as a regularizer and the paper's central claim is unsupported.
- **A1 ≈ B1 (within noise)** is a *required* result, not a failure — it is the empirical anchor for the Capacity Gap argument in §3 of the narrative.
- **B2 is an upper-bound reference, not a competitor.** It violates the edge constraint and is reported only to bound the headroom available to the 3M student.
- **Loss-by-topology is binding.** No A1/Hop-1-of-A3 run is permitted to add MSE or KL across the ViT↔CNN boundary; no A2/Hop-2-of-A3 run is permitted to substitute InfoNCE for MSE+KL. Mixing these would confound the topology-vs-capacity attribution that the matrix is designed to isolate.

## Shared Protocol

- Splits: BUSI 5-fold; report mean ± std across folds.
- Training resolution: 256×256 (MedSAM feature cache is 256-native).
- Evaluation: predicted masks upsampled to original resolution before Dice/IoU (see `docs/03_CONSTRAINTS_AND_RULES.md`).
- Optimizer / schedule / augmentation: held constant across all five rows. Only the **teacher source** and the **distillation loss** vary, and the latter is determined by topology per Rules 1 and 2 above — not chosen per-row.
