"""
src/losses.py — CAISc 2026

Loss heads used across all phases of the heterogeneous distillation pipeline.

- SoftDiceLoss               : standard 1 - soft Dice on raw logits
- BCEDiceLoss                : BCEWithLogitsLoss + SoftDiceLoss (Phase 2a baseline)
- HeterogeneousDistillationLoss : Phase 2b. BCE + Dice + dense InfoNCE against
                               MedSAM teacher features.

The distillation loss is the paper's central contribution. Key design points:

1. **Dense InfoNCE per spatial token.** For each foreground anchor in the
   student's projected bottleneck, the positive is the MedSAM-encoder feature
   at the *same spatial position*; the negatives are MedSAM-encoder features
   at background positions within the *same image*. This is the "CLIP-style
   contrastive at the bottleneck" pitch translated into a per-token objective.

2. **Within-image negatives only.** Cross-image negatives are tempting (more
   negatives per anchor → tighter InfoNCE), but they introduce semantic
   collisions — another patient's lesion is still a lesion, and using it as
   a negative would push the projector to memorize patient identity rather
   than lesion semantics. Pure intra-image speckle is the cleaner negative.

3. **Stochastic anchor/negative sampling.** Mask sizes vary 10× across cases,
   so we cap anchors/negatives per image at fixed counts (defaults 64 / 256)
   to keep the loss compute bounded and the gradient magnitudes consistent
   across the batch. Sampling is with replacement to handle the rare case of
   <n_anchors foreground tokens.

4. **fp32 inside the contrastive head.** Even when AMP runs the network at
   fp16, we cast student/teacher to fp32 here. Cosine similarities near 1.0
   lose ~3 bits of mantissa in fp16, which destabilizes the InfoNCE log-sum-exp
   for the temperature τ=0.07 regime.

5. **No learnable transformation on the teacher side.** The teacher features
   are merely L2-normalized at loss time. This preserves the "we align to
   MedSAM's encoder geometry" interpretability claim that motivates the
   paper's novelty over BYOL/SimSiam-style asymmetric KD.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    """1 - mean per-sample soft Dice. Operates on raw logits (applies sigmoid)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCEWithLogitsLoss + SoftDiceLoss with configurable weights."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Heterogeneous distillation loss
# ---------------------------------------------------------------------------

@dataclass
class DistillationLossOutput:
    """Bundle returned by HeterogeneousDistillationLoss so the trainer can log components."""
    total: torch.Tensor
    seg: torch.Tensor
    contrastive: torch.Tensor
    alpha: float
    n_active_images: int     # how many images in batch had at least one FG and BG anchor


class HeterogeneousDistillationLoss(nn.Module):
    """
    Phase 2b combined loss: BCE + Dice + α · dense-InfoNCE.

    Forward signature:
        logits        : (B, 1, H, W)         student raw logits
        targets       : (B, 1, H, W) ∈ {0,1} GT binary mask
        student_proj  : (B, C, h, w)         projector output, L2-normalized
        teacher_emb   : (B, C, h, w)         MedSAM ViT encoder output (raw, NOT normalized)
        mask_64       : (B, 1, h, w)         GT mask downsampled to teacher grid
        alpha         : float                contrastive weight for THIS step

    Returns DistillationLossOutput. The trainer applies alpha here (not at the
    callsite) so the per-step log line can report both the raw contrastive
    magnitude AND the alpha-applied total.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        temperature: float = 0.07,
        n_anchors: int = 64,
        n_negatives: int = 256,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.temperature = temperature
        self.n_anchors = n_anchors
        self.n_negatives = n_negatives

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        student_proj: torch.Tensor,
        teacher_emb: torch.Tensor,
        mask_64: torch.Tensor,
        alpha: float,
    ) -> DistillationLossOutput:
        # --- Segmentation supervision (same as Phase 2a baseline) -------------
        l_seg = self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)

        # --- Contrastive distillation -----------------------------------------
        # fp32 cast for numerical stability of cosine sims under AMP fp16.
        s = student_proj.float()
        t = F.normalize(teacher_emb.float(), p=2, dim=1)
        l_contrast, n_active = self._dense_infonce(s, t, mask_64)

        total = l_seg + alpha * l_contrast
        return DistillationLossOutput(
            total=total, seg=l_seg.detach(), contrastive=l_contrast.detach(),
            alpha=alpha, n_active_images=n_active,
        )

    def _dense_infonce(
        self,
        student: torch.Tensor,    # (B, C, h, w) — L2-normalized along C
        teacher: torch.Tensor,    # (B, C, h, w) — L2-normalized along C
        mask_64: torch.Tensor,    # (B, 1, h, w) — binary
    ) -> tuple[torch.Tensor, int]:
        """
        Dense InfoNCE with per-image stochastic anchor / negative sampling.
        """
        B, C, H, W = student.shape
        HW = H * W

        s_flat = student.view(B, C, HW)                  # (B, C, HW)
        t_flat = teacher.view(B, C, HW)                  # (B, C, HW)
        m_flat = (mask_64.view(B, HW) > 0.5)             # (B, HW) bool

        per_image_losses: list[torch.Tensor] = []
        device = student.device

        for i in range(B):
            fg_idx = m_flat[i].nonzero(as_tuple=True)[0]                 # (n_fg,)
            bg_idx = (~m_flat[i]).nonzero(as_tuple=True)[0]              # (n_bg,)
            if fg_idx.numel() == 0 or bg_idx.numel() == 0:
                # No FG (lesion vanished in 64-grid downsample) or all-FG image — skip.
                continue

            a_pick = fg_idx[torch.randint(0, fg_idx.numel(), (self.n_anchors,), device=device)]
            n_pick = bg_idx[torch.randint(0, bg_idx.numel(), (self.n_negatives,), device=device)]

            s_a = s_flat[i, :, a_pick].t()                                # (Na, C)
            t_p = t_flat[i, :, a_pick].t()                                # (Na, C)  positive @ same spatial position
            t_n = t_flat[i, :, n_pick].t()                                # (Nn, C)

            # Cosine similarities (both inputs unit-norm → dot = cosine).
            pos = (s_a * t_p).sum(dim=1, keepdim=True) / self.temperature     # (Na, 1)
            neg = (s_a @ t_n.t()) / self.temperature                          # (Na, Nn)

            # InfoNCE = cross-entropy with target index 0 (the positive column).
            logits_cat = torch.cat([pos, neg], dim=1)                          # (Na, 1 + Nn)
            labels = torch.zeros(self.n_anchors, dtype=torch.long, device=device)
            per_image_losses.append(F.cross_entropy(logits_cat, labels))

        if not per_image_losses:
            # Degenerate batch — return a differentiable zero so the graph stays intact.
            return student.sum() * 0.0, 0
        return torch.stack(per_image_losses).mean(), len(per_image_losses)


# ---------------------------------------------------------------------------
# Alpha schedule — exposed at module level so the trainer can import directly
# ---------------------------------------------------------------------------

def alpha_for_epoch(
    epoch: int,
    warmup_end: int = 10,
    ramp_end: int = 40,
    alpha_start: float = 0.1,
    alpha_max: float = 1.0,
    scale: float = 1.0,
    static: float | None = None,
) -> float:
    """
    Three-phase α schedule for the contrastive distillation weight:

        epoch ≤ warmup_end          → 0.0       (pure segmentation warmup)
        warmup_end < epoch ≤ ramp_end → linear ramp [alpha_start, alpha_max]
        epoch > ramp_end            → alpha_max  (held constant)

    If `static` is not None, it overrides the schedule entirely (for the
    {0.1, 0.5, 1.0, 2.0} ablation table). `scale` multiplies the scheduled
    output (lets you re-run the schedule at half/double magnitude without
    breaking the shape).

    Note: epoch is 1-indexed in the trainer; this function accepts whatever
    convention the caller uses as long as it's consistent.
    """
    if static is not None:
        return float(static)
    if epoch <= warmup_end:
        return 0.0
    if epoch <= ramp_end:
        frac = (epoch - warmup_end) / max(1, (ramp_end - warmup_end))
        return scale * (alpha_start + frac * (alpha_max - alpha_start))
    return scale * alpha_max
