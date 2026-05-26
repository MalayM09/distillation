"""
src/models.py — CAISc 2026

EdgeUNet (~1.94M) + ContrastiveProjector (+65K params).
Total ≈ 2.01M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
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

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class ContrastiveProjector(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, target_size=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.target_size = target_size

    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        return F.normalize(x, p=2, dim=1)


class EdgeUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=16,
                 with_projector=False, projector_target=64):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8
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
        self.bottleneck_channels = cb
        self.projector = (
            ContrastiveProjector(in_channels=cb, out_channels=cb, target_size=projector_target)
            if with_projector else None
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _forward_features(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        return self.outc(u4), x5

    def forward(self, x):
        logits, _ = self._forward_features(x)
        return logits

    def forward_distill(self, x):
        if self.projector is None:
            raise RuntimeError("forward_distill() requires with_projector=True")
        logits, bottleneck = self._forward_features(x)
        return logits, self.projector(bottleneck)

    @torch.no_grad()
    def encode(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        return self.bottleneck(x4)


if __name__ == "__main__":
    BUDGET = 3_000_000
    base = EdgeUNet(with_projector=False)
    n_base = sum(p.numel() for p in base.parameters())
    print(f"EdgeUNet baseline: {n_base:,} params ({n_base/1e6:.3f} M)")
    assert n_base < BUDGET

    distill = EdgeUNet(with_projector=True)
    n_distill = sum(p.numel() for p in distill.parameters())
    n_proj = sum(p.numel() for p in distill.projector.parameters())
    print(f"EdgeUNet+Projector: {n_distill:,} params ({n_distill/1e6:.3f} M) | proj={n_proj:,}")
    assert n_distill < BUDGET

    distill.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 256, 256)
        logits, proj = distill.forward_distill(x)
        assert logits.shape == (2, 1, 256, 256)
        assert proj.shape == (2, 256, 64, 64)
        assert torch.allclose(proj.pow(2).sum(dim=1).sqrt(), torch.ones(2, 64, 64), atol=1e-5)
        print("forward_distill OK; L2-norm OK")
