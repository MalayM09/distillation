"""
src/loss_hop2.py — Hop-2 distillation loss (CNN -> CNN, Rule 2 of docs/02)

Implements the binding loss for HA-Net (15M, CNN) -> EdgeUNet (<3M, CNN)
transfer, which is the second hop of the TAKD cascade (A3) and the entire
loss for the A2 ablation. Per docs/02 Rule 2, the CNN<->CNN topology makes
direct feature/logit alignment mathematically optimal:

    L = L_seg + lambda_mse * L_MSE + lambda_kl * L_KL

    L_seg : BCEWithLogits + Soft Dice on (student_logits, ground_truth_mask)
    L_MSE : MSE on (proj(student_bottleneck), teacher_bottleneck)
    L_KL  : Bernoulli KL with temperature T between (student_logits, teacher_logits)

Channel-projection adapter
--------------------------
EdgeUNet emits a 192-channel bottleneck at H/32; HA-Net emits a 512-channel
bottleneck at the same stride. Hop-2 MSE requires shape-matched tensors.
A *trainable* 1x1 conv (192 -> 512) lives inside this module and MUST be
included in the training optimizer alongside the student parameters. The
`projector_parameters()` helper exposes them; the typical wiring is:

    loss_fn = Hop2DistillationLoss(...)
    optim = torch.optim.AdamW(
        list(student.parameters()) + list(loss_fn.projector_parameters()),
        lr=...,
    )

AMP / FP16
----------
KL-divergence and MSE distillation terms are explicitly upcast to FP32
before computation, per the PyTorch AMP recipe for distillation losses:
log/exp operations and large-tensor reductions lose precision in FP16.
The segmentation loss (BCEWithLogits + Soft Dice) is AMP-safe in FP16 and
runs in the autocast dtype.
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hop2DistillationLoss(nn.Module):
    """Composite Hop-2 loss with built-in 192->512 channel projector.

    Parameters
    ----------
    student_bottleneck_channels : int
        Channel count of the student's bottleneck tensor. Default 192 = EdgeUNet.
    teacher_bottleneck_channels : int
        Channel count of the teacher's bottleneck tensor. Default 512 = HA-Net.
    temperature : float
        Softening temperature T applied to both logits before KL. The KL term
        is scaled by T^2 so its gradient magnitude is comparable across T's
        (Hinton et al. 2015 distillation convention).
    lambda_mse, lambda_kl : float
        Loss weights. Defaults give equal weight to all three terms (1.0).
    dice_smooth : float
        Smoothing constant for Soft Dice. 1.0 is standard for BUSI-scale masks.
    """

    def __init__(
        self,
        student_bottleneck_channels: int = 192,
        teacher_bottleneck_channels: int = 512,
        temperature: float = 2.0,
        lambda_mse: float = 1.0,
        lambda_kl: float = 1.0,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()

        # Trainable 1x1 channel projector (the shape adapter).
        self.projector = nn.Conv2d(
            in_channels=student_bottleneck_channels,
            out_channels=teacher_bottleneck_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.kaiming_normal_(self.projector.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.projector.bias)

        self.T = float(temperature)
        self.lambda_mse = float(lambda_mse)
        self.lambda_kl = float(lambda_kl)
        self.dice_smooth = float(dice_smooth)

        self.bce = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------ #
    # Optimizer-wiring helper                                            #
    # ------------------------------------------------------------------ #

    def projector_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the trainable projector parameters for inclusion in the optimizer."""
        return self.projector.parameters()

    # ------------------------------------------------------------------ #
    # Sub-losses                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _soft_dice_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        smooth: float,
    ) -> torch.Tensor:
        """1 - Soft Dice over (B, 1, H, W). Reduction over batch and spatial dims."""
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersect = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersect + smooth) / (denom + smooth)
        return 1.0 - dice.mean()

    def _binary_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Bernoulli KL( p_teacher || p_student ) with temperature scaling.

        Computed entirely in FP32 for numerical stability in autocast contexts.
        `F.logsigmoid` is the stable form of log(sigmoid(.)) and log(1-sigmoid(.))
        and avoids the catastrophic cancellation that bare log(sigmoid) suffers.
        """
        T = self.T
        # Force FP32 — KL is sensitive to log/exp precision and the reduction
        # is over (B, 1, 256, 256) ~ 65k elements per sample.
        s = student_logits.float() / T
        t = teacher_logits.float() / T

        p_t = torch.sigmoid(t)

        log_p_s = F.logsigmoid(s)
        log_1m_p_s = F.logsigmoid(-s)
        log_p_t = F.logsigmoid(t)
        log_1m_p_t = F.logsigmoid(-t)

        kl = p_t * (log_p_t - log_p_s) + (1.0 - p_t) * (log_1m_p_t - log_1m_p_s)
        # The T^2 scaling restores gradient magnitudes to ~unit-T scale so this
        # term is comparable across temperature sweeps (Hinton et al. 2015).
        return (T * T) * kl.mean()

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_bottleneck: torch.Tensor,
        teacher_bottleneck: torch.Tensor,
        ground_truth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the composite Hop-2 loss.

        Parameters
        ----------
        student_logits        : (B, 1, H, W)  EdgeUNet raw output.
        teacher_logits        : (B, 1, H, W)  HA-Net raw output (no_grad).
        student_bottleneck    : (B, 192, H/32, W/32)
        teacher_bottleneck    : (B, 512, H/32, W/32)
        ground_truth_mask     : (B, 1, H, W)  binary in {0, 1}.

        Returns
        -------
        total : scalar tensor with grad enabled.
        parts : dict of detached scalar tensors for logging:
                {"total", "seg", "bce", "dice", "mse", "kl"}.
        """
        # ----- L_seg: BCEWithLogits + Soft Dice -----
        l_bce = self.bce(student_logits, ground_truth_mask)
        l_dice = self._soft_dice_loss(student_logits, ground_truth_mask, self.dice_smooth)
        l_seg = l_bce + l_dice

        # ----- L_MSE: project student bottleneck to teacher channels -----
        projected = self.projector(student_bottleneck)
        # Upcast to FP32 for stable accumulation over (B, 512, 8, 8).
        l_mse = F.mse_loss(projected.float(), teacher_bottleneck.float())

        # ----- L_KL: Bernoulli KL with temperature, FP32-safe -----
        l_kl = self._binary_kl(student_logits, teacher_logits)

        # ----- Combined -----
        total = l_seg + self.lambda_mse * l_mse + self.lambda_kl * l_kl

        parts: Dict[str, torch.Tensor] = {
            "total": total.detach(),
            "seg":   l_seg.detach(),
            "bce":   l_bce.detach(),
            "dice":  l_dice.detach(),
            "mse":   l_mse.detach(),
            "kl":    l_kl.detach(),
        }
        return total, parts


# --------------------------------------------------------------------------- #
# Validation block                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, W = 2, 256, 256
    student_logits = torch.randn(B, 1, H, W, requires_grad=True)
    teacher_logits = torch.randn(B, 1, H, W)
    student_bottleneck = torch.randn(B, 192, H // 32, W // 32, requires_grad=True)
    teacher_bottleneck = torch.randn(B, 512, H // 32, W // 32)
    gt = (torch.rand(B, 1, H, W) > 0.85).float()

    loss_fn = Hop2DistillationLoss(
        student_bottleneck_channels=192,
        teacher_bottleneck_channels=512,
        temperature=2.0,
        lambda_mse=1.0,
        lambda_kl=1.0,
        dice_smooth=1.0,
    )

    proj_params = sum(p.numel() for p in loss_fn.projector_parameters())

    total, parts = loss_fn(
        student_logits, teacher_logits,
        student_bottleneck, teacher_bottleneck,
        gt,
    )
    total.backward()

    line = "=" * 68
    print(line)
    print("Hop2DistillationLoss — validation")
    print(line)
    print(f"Projector parameters    : {proj_params:>10,d}   "
          f"(expected {192 * 512 + 512:,d} = 192*512 + 512 bias)")
    print(f"Temperature T           : {loss_fn.T}")
    print(f"Loss components         :")
    for k in ("total", "seg", "bce", "dice", "mse", "kl"):
        print(f"  {k:<8s}              : {parts[k].item(): .6f}")
    print(f"Student-logit grad      : "
          f"{'OK' if student_logits.grad is not None else 'MISSING'}")
    print(f"Student-bottleneck grad : "
          f"{'OK' if student_bottleneck.grad is not None else 'MISSING'}")
    proj_grad_ok = all(p.grad is not None for p in loss_fn.projector_parameters())
    print(f"Projector grad          : {'OK' if proj_grad_ok else 'MISSING'}")
    print(line)
