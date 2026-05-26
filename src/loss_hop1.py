"""
src/loss_hop1.py — Hop-1 distillation loss (ViT -> CNN, Rule 1 of docs/02)

Implements the binding loss for MedSAM (91M, ViT) -> {HA-Net | EdgeUNet} (CNN)
transfer. This is the first hop of the TAKD cascade (A3) and the *entire*
distillation loss for the A1 ablation (direct MedSAM -> EdgeUNet). Per docs/02
Rule 1, MSE/KL across the ViT<->CNN boundary is ill-posed because the absolute
embedding scales and channel orderings of MedSAM's ViT tokens are not
commensurable with a CNN feature map. Dense InfoNCE sidesteps this by training
a small projector so that, at every spatial location, the projected student
embedding is pulled toward the MedSAM token at the same location and pushed
away from MedSAM tokens at all background locations of the same image.

Composite loss
--------------
    L = L_seg + lambda_nce * L_InfoNCE

    L_seg     : BCEWithLogits + Soft Dice on (student_logits, ground_truth_mask)
    L_InfoNCE : per-image, per-foreground-anchor contrastive loss against MedSAM
                tokens at the same image, with all background tokens as negatives,
                temperature tau = 0.07.

Projector
---------
ContrastiveProjector: 1x1 conv (student_channels -> 256) + bilinear upsample
(8x8 -> 64x64). The 64x64 grid matches MedSAM's image-encoder output stride
(1024-input ViT, 16x16 patches, 64x64 patch grid). The 256 channel count
matches MedSAM's neck-projected encoder dimension. The projector is the only
trainable module *inside* this loss; the optimizer must include its parameters,
exposed by `projector_parameters()`.

AMP / FP16
----------
Per user spec, *all contrastive math runs in FP32*. The projector conv may
execute in FP16 inside an autocast context, but its output is immediately
upcast via `.float()` before L2 normalization, dot products, and logsumexp.
This is the recommended AMP recipe for contrastive losses: cosine similarity
and logsumexp are both numerically fragile in FP16 due to (a) the small
dynamic range of sub-unit cosine values and (b) the exponential's
sensitivity to small temperature scaling.

The segmentation loss runs in the autocast dtype (FP16 is safe for BCE +
soft Dice on binary segmentation logits, matching the Hop-2 convention).
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# ContrastiveProjector: channel + spatial adapter (CNN bottleneck -> ViT grid) #
# --------------------------------------------------------------------------- #

class ContrastiveProjector(nn.Module):
    """1x1 channel conv + fixed bilinear upsample to MedSAM's 64x64 token grid.

    The conv is the only learnable component; the bilinear upsample is
    parameter-free and produces a static ONNX Resize node (the projector is
    discarded at deployment, but the convention matches docs/03's ONNX rules
    for the student).

    Parameters
    ----------
    in_channels : int
        Channel count of the student bottleneck. HA-Net: 512; EdgeUNet: 192.
    out_channels : int
        Channel count of the MedSAM features being matched against. Default 256
        matches MedSAM's image-encoder neck output.
    target_size : int
        Spatial grid to which the projected feature is upsampled. Default 64
        matches MedSAM's 1024-input / 16-patch encoder grid.
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        target_size: int = 64,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.conv.bias)
        self.target_size = int(target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H_bot, W_bot)  e.g. (B, 512, 8, 8) for HA-Net
        z = self.conv(x)
        z = F.interpolate(
            z,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        return z


# --------------------------------------------------------------------------- #
# Hop1ContrastiveLoss                                                         #
# --------------------------------------------------------------------------- #

class Hop1ContrastiveLoss(nn.Module):
    """Hop-1 composite loss: segmentation + dense InfoNCE against MedSAM tokens.

    Parameters
    ----------
    student_channels : int
        Bottleneck channels emitted by the student. HA-Net: 512; EdgeUNet: 192.
    teacher_channels : int
        MedSAM encoder channels in the cached feature tensor. Default 256.
    target_grid : int
        Spatial grid for both student-projection and MedSAM features. Default 64.
    temperature : float
        InfoNCE temperature tau. Default 0.07 (SimCLR / standard CL convention).
    lambda_nce : float
        Weight on the InfoNCE term. Default 1.0.
    dice_smooth : float
        Soft-Dice smoothing constant. Default 1.0.

    Optimizer wiring
    ----------------
    The 1x1 projector inside this module is trainable and MUST be added to the
    optimizer alongside the student parameters:

        loss_fn = Hop1ContrastiveLoss(student_channels=512)
        optim = torch.optim.AdamW(
            list(student.parameters()) + list(loss_fn.projector_parameters()),
            lr=...,
        )
    """

    def __init__(
        self,
        student_channels: int = 512,
        teacher_channels: int = 256,
        target_grid: int = 64,
        temperature: float = 0.07,
        lambda_nce: float = 1.0,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.projector = ContrastiveProjector(
            in_channels=student_channels,
            out_channels=teacher_channels,
            target_size=target_grid,
        )

        self.target_grid = int(target_grid)
        self.tau = float(temperature)
        self.lambda_nce = float(lambda_nce)
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
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        return 1.0 - dice.mean()

    def _dense_info_nce(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        mask_lowres: torch.Tensor,
    ) -> torch.Tensor:
        """Per-image dense InfoNCE; mean of per-image anchor losses.

        Parameters
        ----------
        student_feat : (B, C, G, G)  FP32, L2-normalized along dim=1.
        teacher_feat : (B, C, G, G)  FP32, L2-normalized along dim=1.
        mask_lowres  : (B, 1, G, G)  binary in {0, 1}; 1 = lesion (anchor pool).

        For each image b:
            * fg_b = {i : mask_b[i] == 1}  — anchor locations
            * bg_b = {j : mask_b[j] == 0}  — negative pool (same image only)
            * For anchor i in fg_b:
                pos_logit = <s_b[i], t_b[i]> / tau
                neg_logit_j = <s_b[i], t_b[j]> / tau for j in bg_b
                loss_i = logsumexp([pos_logit, *neg_logits]) - pos_logit
            * loss_b = mean over anchors

        Images with empty fg or empty bg are skipped (degenerate InfoNCE).
        If every image in the batch is degenerate, returns 0 with the autograd
        graph intact (via `student_feat.sum() * 0.0`).
        """
        B, C, G, _ = student_feat.shape
        N = G * G

        s_flat = student_feat.reshape(B, C, N)  # (B, C, N)
        t_flat = teacher_feat.reshape(B, C, N)  # (B, C, N)
        m_flat = (mask_lowres.reshape(B, N) > 0.5)  # (B, N) bool

        tau = self.tau
        per_image_losses = []

        for b in range(B):
            fg = m_flat[b]                # (N,) bool
            bg = ~fg
            n_fg = int(fg.sum().item())
            n_bg = int(bg.sum().item())
            if n_fg == 0 or n_bg == 0:
                # Degenerate: no anchors or no negatives in this image — skip.
                continue

            s_fg = s_flat[b, :, fg].t()   # (n_fg, C)  — anchors
            t_pos = t_flat[b, :, fg].t()  # (n_fg, C)  — positive at same location
            t_neg = t_flat[b, :, bg].t()  # (n_bg, C)  — negatives (background only)

            # Cosine similarities (features are already L2-normalized -> dot = cos).
            pos_logit = (s_fg * t_pos).sum(dim=1) / tau          # (n_fg,)
            neg_logit = (s_fg @ t_neg.t()) / tau                 # (n_fg, n_bg)

            # InfoNCE per anchor:
            #   L_i = -log( exp(pos_i) / [exp(pos_i) + sum_j exp(neg_ij)] )
            #       = logsumexp([pos_i, neg_i1, ..., neg_iK]) - pos_i
            all_logit = torch.cat([pos_logit.unsqueeze(1), neg_logit], dim=1)  # (n_fg, 1+n_bg)
            denom = torch.logsumexp(all_logit, dim=1)                          # (n_fg,)
            per_image_losses.append((denom - pos_logit).mean())

        if not per_image_losses:
            # Preserve the autograd graph so optimizer.step() doesn't choke.
            return student_feat.sum() * 0.0

        return torch.stack(per_image_losses).mean()

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        student_logits: torch.Tensor,
        student_bottleneck: torch.Tensor,
        medsam_features: torch.Tensor,
        ground_truth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the composite Hop-1 loss.

        Parameters
        ----------
        student_logits     : (B, 1, H, W)  raw student segmentation logits.
        student_bottleneck : (B, C_s, H_bot, W_bot)  e.g. (B, 512, 8, 8) for HA-Net.
        medsam_features    : (B, 256, 64, 64)  cached MedSAM encoder output (no_grad).
        ground_truth_mask  : (B, 1, H, W)  binary in {0, 1}.

        Returns
        -------
        total : scalar tensor with grad.
        parts : dict of detached scalar tensors for logging
                {"total", "seg", "bce", "dice", "nce"}.
        """
        # ----- L_seg: BCEWithLogits + Soft Dice -----
        l_bce = self.bce(student_logits, ground_truth_mask)
        l_dice = self._soft_dice_loss(student_logits, ground_truth_mask, self.dice_smooth)
        l_seg = l_bce + l_dice

        # ----- Project student bottleneck onto MedSAM's (256, 64, 64) grid -----
        # The projector conv may execute in FP16 under autocast; that is fine.
        # We then upcast to FP32 for ALL downstream contrastive math, per the
        # AMP recipe for contrastive losses (cosine sim + logsumexp need FP32).
        projected = self.projector(student_bottleneck)  # (B, 256, 64, 64)

        s = projected.float()
        t = medsam_features.float()

        # Hard shape contract: projector must land exactly on MedSAM's grid.
        if s.shape != t.shape:
            raise ValueError(
                f"projected student shape {tuple(s.shape)} does not match "
                f"medsam_features shape {tuple(t.shape)}; check student_channels, "
                f"teacher_channels, and target_grid in Hop1ContrastiveLoss."
            )

        # L2-normalize along channel dim so dot products are cosine similarities.
        s = F.normalize(s, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)

        # Nearest-neighbor downsample of GT mask to the contrastive grid.
        mask_lowres = F.interpolate(
            ground_truth_mask.float(),
            size=(self.target_grid, self.target_grid),
            mode="nearest",
        )

        # ----- L_InfoNCE -----
        l_nce = self._dense_info_nce(s, t, mask_lowres)

        # ----- Combined -----
        total = l_seg + self.lambda_nce * l_nce

        parts: Dict[str, torch.Tensor] = {
            "total": total.detach(),
            "seg":   l_seg.detach(),
            "bce":   l_bce.detach(),
            "dice":  l_dice.detach(),
            "nce":   l_nce.detach(),
        }
        return total, parts


# --------------------------------------------------------------------------- #
# Validation block                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, W = 2, 256, 256
    C_student_bot = 512  # HA-Net case
    C_medsam = 256
    G = 64

    student_logits = torch.randn(B, 1, H, W, requires_grad=True)
    student_bottleneck = torch.randn(B, C_student_bot, 8, 8, requires_grad=True)
    medsam_features = torch.randn(B, C_medsam, G, G)  # cached, no grad needed
    # Plausible BUSI-style mask: ~15% foreground in a centered blob region
    gt = torch.zeros(B, 1, H, W)
    gt[:, :, 80:170, 90:180] = 1.0
    gt = (gt > 0.5).float()

    loss_fn = Hop1ContrastiveLoss(
        student_channels=C_student_bot,
        teacher_channels=C_medsam,
        target_grid=G,
        temperature=0.07,
        lambda_nce=1.0,
    )

    proj_params = sum(p.numel() for p in loss_fn.projector_parameters())
    expected_proj = C_student_bot * C_medsam + C_medsam  # 1x1 conv + bias

    total, parts = loss_fn(
        student_logits=student_logits,
        student_bottleneck=student_bottleneck,
        medsam_features=medsam_features,
        ground_truth_mask=gt,
    )
    total.backward()

    line = "=" * 68
    print(line)
    print("Hop1ContrastiveLoss — validation (HA-Net student configuration)")
    print(line)
    print(f"Projector parameters    : {proj_params:>10,d}   "
          f"(expected {expected_proj:,d} = {C_student_bot}*{C_medsam} + {C_medsam} bias)")
    print(f"Temperature tau         : {loss_fn.tau}")
    print(f"Target grid             : {G}x{G}")
    print(f"Loss components         :")
    for k in ("total", "seg", "bce", "dice", "nce"):
        print(f"  {k:<8s}              : {parts[k].item(): .6f}")
    print(f"Student-logit grad      : "
          f"{'OK' if student_logits.grad is not None else 'MISSING'}")
    print(f"Student-bottleneck grad : "
          f"{'OK' if student_bottleneck.grad is not None else 'MISSING'}")
    proj_grad_ok = all(p.grad is not None for p in loss_fn.projector_parameters())
    print(f"Projector grad          : {'OK' if proj_grad_ok else 'MISSING'}")
    print(line)

    # Degenerate edge case: an image with zero foreground should not break the
    # backward pass; the loss should still produce a gradient (zero on NCE,
    # nonzero on seg).
    gt_empty = torch.zeros(B, 1, H, W)
    student_logits2 = torch.randn(B, 1, H, W, requires_grad=True)
    student_bot2 = torch.randn(B, C_student_bot, 8, 8, requires_grad=True)
    total2, parts2 = loss_fn(
        student_logits=student_logits2,
        student_bottleneck=student_bot2,
        medsam_features=medsam_features,
        ground_truth_mask=gt_empty,
    )
    total2.backward()
    print("Degenerate (no-foreground) edge case:")
    print(f"  nce term              : {parts2['nce'].item(): .6f}   (expected 0.0)")
    print(f"  total term            : {parts2['total'].item(): .6f}")
    print(f"  student_logits grad   : "
          f"{'OK' if student_logits2.grad is not None else 'MISSING'}")
    print(line)

    # EdgeUNet student configuration (used for A1 ablation): student_channels=192.
    loss_a1 = Hop1ContrastiveLoss(
        student_channels=192,
        teacher_channels=C_medsam,
        target_grid=G,
        temperature=0.07,
        lambda_nce=1.0,
    )
    a1_proj_params = sum(p.numel() for p in loss_a1.projector_parameters())
    a1_logits = torch.randn(B, 1, H, W, requires_grad=True)
    a1_bot = torch.randn(B, 192, 8, 8, requires_grad=True)
    a1_total, a1_parts = loss_a1(
        student_logits=a1_logits,
        student_bottleneck=a1_bot,
        medsam_features=medsam_features,
        ground_truth_mask=gt,
    )
    a1_total.backward()
    print("EdgeUNet student configuration (A1 ablation):")
    print(f"  projector parameters  : {a1_proj_params:>10,d}   "
          f"(expected {192 * 256 + 256:,d})")
    print(f"  loss components       : total={a1_parts['total'].item():.4f}  "
          f"seg={a1_parts['seg'].item():.4f}  nce={a1_parts['nce'].item():.4f}")
    print(line)
