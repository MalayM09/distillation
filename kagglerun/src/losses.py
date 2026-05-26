"""
src/losses.py — CAISc 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


@dataclass
class DistillationLossOutput:
    total: torch.Tensor
    seg: torch.Tensor
    contrastive: torch.Tensor
    alpha: float
    n_active_images: int


class HeterogeneousDistillationLoss(nn.Module):
    """Dense per-token InfoNCE distillation (the paper's main method)."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, temperature=0.07,
                 n_anchors=64, n_negatives=256):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.temperature = temperature
        self.n_anchors = n_anchors
        self.n_negatives = n_negatives

    def forward(self, logits, targets, student_proj, teacher_emb, mask_64, alpha):
        l_seg = self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)
        s = student_proj.float()
        t = F.normalize(teacher_emb.float(), p=2, dim=1)
        l_contrast, n_active = self._dense_infonce(s, t, mask_64)
        total = l_seg + alpha * l_contrast
        return DistillationLossOutput(
            total=total, seg=l_seg.detach(), contrastive=l_contrast.detach(),
            alpha=alpha, n_active_images=n_active,
        )

    def _dense_infonce(self, student, teacher, mask_64):
        B, C, H, W = student.shape
        HW = H * W
        s_flat = student.view(B, C, HW)
        t_flat = teacher.view(B, C, HW)
        m_flat = (mask_64.view(B, HW) > 0.5)
        per_image_losses = []
        device = student.device

        for i in range(B):
            fg_idx = m_flat[i].nonzero(as_tuple=True)[0]
            bg_idx = (~m_flat[i]).nonzero(as_tuple=True)[0]
            if fg_idx.numel() == 0 or bg_idx.numel() == 0:
                continue
            a_pick = fg_idx[torch.randint(0, fg_idx.numel(), (self.n_anchors,), device=device)]
            n_pick = bg_idx[torch.randint(0, bg_idx.numel(), (self.n_negatives,), device=device)]
            s_a = s_flat[i, :, a_pick].t()
            t_p = t_flat[i, :, a_pick].t()
            t_n = t_flat[i, :, n_pick].t()
            pos = (s_a * t_p).sum(dim=1, keepdim=True) / self.temperature
            neg = (s_a @ t_n.t()) / self.temperature
            logits_cat = torch.cat([pos, neg], dim=1)
            labels = torch.zeros(self.n_anchors, dtype=torch.long, device=device)
            per_image_losses.append(F.cross_entropy(logits_cat, labels))

        if not per_image_losses:
            return student.sum() * 0.0, 0
        return torch.stack(per_image_losses).mean(), len(per_image_losses)


class HintMSEDistillationLoss(nn.Module):
    """
    Hint-based MSE distillation baseline (Reviewer-2 ablation).

    Same projector, same L2-normalized 256-d space, same data path — the
    only thing changing vs. the InfoNCE variant is the alignment objective:
    uniform per-position MSE pull, no negative push. This isolates the
    contribution of the contrastive negatives.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets, student_proj, teacher_emb, mask_64, alpha):
        l_seg = self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)
        s = student_proj.float()
        t = F.normalize(teacher_emb.float(), p=2, dim=1)
        l_mse = F.mse_loss(s, t)
        total = l_seg + alpha * l_mse
        return DistillationLossOutput(
            total=total, seg=l_seg.detach(), contrastive=l_mse.detach(),
            alpha=alpha, n_active_images=s.shape[0],
        )


def alpha_for_epoch(epoch, warmup_end=10, ramp_end=40,
                    alpha_start=0.1, alpha_max=1.0, scale=1.0, static=None):
    if static is not None:
        return float(static)
    if epoch <= warmup_end:
        return 0.0
    if epoch <= ramp_end:
        frac = (epoch - warmup_end) / max(1, (ramp_end - warmup_end))
        return scale * (alpha_start + frac * (alpha_max - alpha_start))
    return scale * alpha_max
