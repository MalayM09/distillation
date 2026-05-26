"""
src/models.py — CAISc 2026

EdgeUNet: 4-stage encoder/decoder U-Net targeting <3M parameters for
edge-deployable ultrasound segmentation. Standard double-Conv blocks
(Conv3x3 → BN → ReLU). MaxPool downsampling, ConvTranspose upsampling,
skip connections concatenated along the channel axis.

Returns raw logits (no terminal sigmoid) so the loss head can use
BCEWithLogitsLoss directly — numerically more stable than BCE+sigmoid.

Channel widths chosen empirically to fit the 3M envelope with headroom
for the contrastive projection head (Phase 2b):
    encoder/decoder = [16, 32, 64, 128], bottleneck = 256
    EdgeUNet alone           ≈ 1.94M params
    EdgeUNet + projector     ≈ 2.01M params  (still well under 3M)

ContrastiveProjector: 1x1 Conv (256→256) → bilinear upsample (16→64) →
L2-normalize along channel dim. Maps the U-Net bottleneck (256, 16, 16)
into MedSAM ViT-B encoder space (256, 64, 64). No nonlinearity, no
hidden layer — by design, to keep the alignment objective interpretable.
Adds 65,792 params.

We deliberately do NOT add a learnable projector on the teacher side.
The contrastive comparison happens directly in MedSAM's native 256-d
encoder space; the teacher features are merely L2-normalized at loss
time (no learnable transformation). This preserves the interpretability
claim: "the student bottleneck is being aligned to MedSAM's encoder
geometry," not "to some learned transformation of it."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv3x3 → BN → ReLU) × 2 — the canonical U-Net residual-free block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool(2) → DoubleConv. Halves spatial, expands channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d(2x2, stride 2) → concat(skip) → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Defensive: ConvT can mismatch by 1px on odd dims. Align to skip.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# ContrastiveProjector — Phase 2b head
# ---------------------------------------------------------------------------

class ContrastiveProjector(nn.Module):
    """
    Maps EdgeUNet bottleneck → MedSAM ViT-B encoder space.

    Architecture: Conv1x1(in_ch → out_ch) → bilinear upsample → L2-normalize.

    No activation between the conv and the upsample — the contrastive loss
    operates on cosine similarities, so we want the projection to be a pure
    linear map of the bottleneck features. Adding a ReLU here would make
    "negative-direction alignment" impossible by construction.

    Shape contract:
        in  : (B, 256, 16, 16)   — U-Net bottleneck @ 256² input
        out : (B, 256, 64, 64)   — L2-normalized along dim=1
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256, target_size: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# EdgeUNet
# ---------------------------------------------------------------------------

class EdgeUNet(nn.Module):
    """
    Sub-3M-parameter U-Net for edge deployment on portable ultrasound probes.

    Args
    ----
    in_channels      : 3 — RGB-tiled grayscale ultrasound input
    out_channels     : 1 — single-foreground (lesion) logits
    base_ch          : 16 — encoder root width (controls overall capacity)
    with_projector   : if True, attach a ContrastiveProjector at the bottleneck.
                       Phase 2a (baseline) uses False; Phase 2b uses True.
    projector_target : spatial size of the projector output (64 → matches
                       MedSAM ViT-B encoder grid at 1024² teacher input).

    Bottleneck stride 16, so a 256×256 input has a 16×16 bottleneck —
    matches MedSAM's 1024→64 stride-16 pattern at a smaller resolution,
    keeping the receptive-field ratio comparable for distillation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_ch: int = 16,
        with_projector: bool = False,
        projector_target: int = 64,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        cb = base_ch * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.bottleneck = Down(c4, cb)

        self.up1 = Up(cb, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)

        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

        # Bottleneck channels are exposed for the contrastive head in Phase 2b
        self.bottleneck_channels = cb

        self.projector = (
            ContrastiveProjector(in_channels=cb, out_channels=cb, target_size=projector_target)
            if with_projector else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encoder + decoder. Returns (logits, bottleneck_tensor)."""
        x1 = self.inc(x)                    # (B, c1, H,    W)
        x2 = self.down1(x1)                 # (B, c2, H/2,  W/2)
        x3 = self.down2(x2)                 # (B, c3, H/4,  W/4)
        x4 = self.down3(x3)                 # (B, c4, H/8,  W/8)
        x5 = self.bottleneck(x4)            # (B, cb, H/16, W/16)

        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        logits = self.outc(u4)              # (B, out_channels, H, W) — raw logits
        return logits, x5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard segmentation forward. Drops the bottleneck handle."""
        logits, _ = self._forward_features(x)
        return logits

    def forward_distill(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward returning (logits, projected_bottleneck) for Phase 2b distillation.

        projected_bottleneck : (B, 256, 64, 64), L2-normalized along channels.
        """
        if self.projector is None:
            raise RuntimeError(
                "forward_distill() requires the model to be constructed with "
                "with_projector=True"
            )
        logits, bottleneck = self._forward_features(x)
        projected = self.projector(bottleneck)
        return logits, projected

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck feature map (pre-projection). Diagnostics only."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return self.bottleneck(x4)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BUDGET = 3_000_000

    # Vanilla (Phase 2a) configuration
    base = EdgeUNet(in_channels=3, out_channels=1, with_projector=False)
    n_base = sum(p.numel() for p in base.parameters())
    print("EdgeUNet — baseline (no projector)")
    print(f"  params    : {n_base:>10,}  ({n_base / 1e6:.3f} M)")
    print(f"  status    : {'OK' if n_base < BUDGET else 'OVER BUDGET'} (budget {BUDGET:,})")
    assert n_base < BUDGET

    base.eval()
    with torch.no_grad():
        for hw in (128, 256, 512):
            y = base(torch.randn(2, 3, hw, hw))
            assert y.shape == (2, 1, hw, hw), f"shape mismatch at {hw}: {y.shape}"
            print(f"  forward   : (2, 3, {hw}, {hw}) → (2, 1, {hw}, {hw}) ✓")

    # Distillation (Phase 2b) configuration
    distill = EdgeUNet(in_channels=3, out_channels=1, with_projector=True)
    n_distill = sum(p.numel() for p in distill.parameters())
    n_proj = sum(p.numel() for p in distill.projector.parameters())
    print("\nEdgeUNet — distillation (with projector)")
    print(f"  params    : {n_distill:>10,}  ({n_distill / 1e6:.3f} M)")
    print(f"  projector : {n_proj:>10,}  (added cost)")
    print(f"  status    : {'OK' if n_distill < BUDGET else 'OVER BUDGET'} (budget {BUDGET:,})")
    assert n_distill < BUDGET, f"EdgeUNet+Projector exceeds budget: {n_distill:,}"

    distill.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 256, 256)
        logits, projected = distill.forward_distill(x)
        assert logits.shape == (2, 1, 256, 256), logits.shape
        assert projected.shape == (2, distill.bottleneck_channels, 64, 64), projected.shape
        # L2-normalization sanity
        norms = projected.pow(2).sum(dim=1).sqrt()           # (B, 64, 64)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "projector not L2-normalized"
        print(f"  forward_distill: logits {tuple(logits.shape)} | projected {tuple(projected.shape)} (L2-norm = 1.0) ✓")
