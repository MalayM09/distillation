"""
src/models.py — TAKD assistant: Hybrid Attention Network (HA-Net, ~15M)

Phase 1 of the TAKD cascade: MedSAM (91M) -> HA-Net (15M) -> U-Net (<3M).

The assistant ingests MedSAM via dense InfoNCE on Hop 1 and exposes
CNN-native distillation surfaces (logits + deep bottleneck features) for
Hop 2's MSE feature matching + KL-divergence transfer into the student.
`HANet.forward` therefore returns a (logits, bottleneck_features) tuple by
design — both surfaces are required by Hop 2's composite loss and must be
emitted in a single forward pass for gradient consistency.

Architectural summary
---------------------
    Stem            7x7 conv s2  -> 64  ch at H/2,   skip_0
    MaxPool s2                   -> 64  ch at H/4
    Layer1 [2x BasicBlock]       -> 64  ch at H/4,   skip_1
    Layer2 [2x BasicBlock] s2    -> 128 ch at H/8,   skip_2
    Layer3 [2x BasicBlock] s2    -> 256 ch at H/16,  skip_3
    Layer4 [2x BasicBlock] s2    -> 512 ch at H/32
    BottleneckRefine (1x1->3x3->1x1, residual, squeeze=256)
                                 -> 512 ch at H/32   (= bottleneck_features)
    CBAM channel+spatial gating on every encoder skip
    DecoderBlocks (bilinear up + concat + 2x conv3x3-BN-SiLU)
        up4: 512 + skip_3(256) -> 256 at H/16
        up3: 256 + skip_2(128) -> 128 at H/8
        up2: 128 + skip_1(64)  -> 64  at H/4
        up1: 64  + skip_0(64)  -> 32  at H/2
    Head: bilinear up to H, conv3x3 32->16, conv1x1 16->num_classes (logits)

AMP / FP16
----------
All learnable modules are Conv2d + BatchNorm2d + SiLU. No LayerNorm,
no GroupNorm, no manual `.float()` promotions, no Python control flow
that depends on tensor values — `torch.cuda.amp.autocast` dispatches
the full forward pass in FP16 without internal upcasts.

Parameter target: ~15M (verified in __main__).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Encoder primitives                                                          #
# --------------------------------------------------------------------------- #

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock with SiLU; identity-or-1x1 downsample on shape change."""

    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.downsample: nn.Module = nn.Sequential(
                conv1x1(in_ch, out_ch, stride),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


# --------------------------------------------------------------------------- #
# Hybrid attention: CBAM channel + spatial gating on encoder skips            #
# --------------------------------------------------------------------------- #

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# --------------------------------------------------------------------------- #
# Decoder primitives                                                          #
# --------------------------------------------------------------------------- #

class DecoderBlock(nn.Module):
    """Bilinear up to skip resolution, concat, then two conv3x3-BN-SiLU stages."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            conv3x3(in_ch + skip_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class BottleneckRefine(nn.Module):
    """Residual 1x1 -> 3x3 -> 1x1 squeeze block at encoder stride /32.

    Output is the canonical `bottleneck_features` tensor returned by HANet.forward;
    Hop-2 MSE feature matching pulls the student's deepest projected feature toward
    this map.
    """

    def __init__(self, channels: int, squeeze: int = 256) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            conv1x1(channels, squeeze),
            nn.BatchNorm2d(squeeze),
            nn.SiLU(inplace=True),
        )
        self.conv = nn.Sequential(
            conv3x3(squeeze, squeeze),
            nn.BatchNorm2d(squeeze),
            nn.SiLU(inplace=True),
        )
        self.expand = nn.Sequential(
            conv1x1(squeeze, channels),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.expand(self.conv(self.reduce(x))))


# --------------------------------------------------------------------------- #
# HANet                                                                       #
# --------------------------------------------------------------------------- #

class HANet(nn.Module):
    """Hybrid Attention Network — TAKD assistant (~15M params).

    Parameters
    ----------
    in_channels : int
        Input image channels. BUSI is grayscale; the default of 3 matches the
        MedSAM-replicated cache layout used during Hop-1 distillation.
    num_classes : int
        Output channels of the segmentation head. BUSI binary segmentation -> 1.

    Returns
    -------
    forward(x) -> (logits, bottleneck_features)
        logits             : (B, num_classes, H, W) raw, unactivated. Consumed by
                              Hop-2 KL-divergence distillation against the student.
        bottleneck_features: (B, 512, H/32, W/32) post-refinement deep features.
                              Consumed by Hop-2 MSE feature matching.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Bottleneck refinement (the exposed Hop-2 MSE target)
        self.bottleneck = BottleneckRefine(channels=512, squeeze=256)

        # Skip attention gates
        self.attn0 = CBAM(64)
        self.attn1 = CBAM(64)
        self.attn2 = CBAM(128)
        self.attn3 = CBAM(256)

        # Decoder
        self.up4 = DecoderBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.up3 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up2 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.up1 = DecoderBlock(in_ch=64,  skip_ch=64,  out_ch=32)

        # Segmentation head (logits at native input resolution)
        self.head = nn.Sequential(
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1, bias=True),
        )

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h0, w0 = x.shape[-2:]

        # Encoder
        s0 = self.stem(x)        # (B, 64,  H/2,  W/2)
        x1 = self.pool(s0)       # (B, 64,  H/4,  W/4)
        s1 = self.layer1(x1)     # (B, 64,  H/4,  W/4)
        s2 = self.layer2(s1)     # (B, 128, H/8,  W/8)
        s3 = self.layer3(s2)     # (B, 256, H/16, W/16)
        s4 = self.layer4(s3)     # (B, 512, H/32, W/32)

        bottleneck_features = self.bottleneck(s4)

        # Attention-gated skips
        a0 = self.attn0(s0)
        a1 = self.attn1(s1)
        a2 = self.attn2(s2)
        a3 = self.attn3(s3)

        # Decoder
        d4 = self.up4(bottleneck_features, a3)
        d3 = self.up3(d4, a2)
        d2 = self.up2(d3, a1)
        d1 = self.up1(d2, a0)

        # Final upsample to native input resolution and segmentation head
        d0 = F.interpolate(d1, size=(h0, w0), mode="bilinear", align_corners=False)
        logits = self.head(d0)

        return logits, bottleneck_features


# --------------------------------------------------------------------------- #
# Validation block                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    model = HANet(in_channels=3, num_classes=1).eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits, bottleneck = model(x)

    line = "=" * 68
    print(line)
    print("HA-Net (TAKD Assistant) — architectural validation")
    print(line)
    print(f"Input shape            : {tuple(x.shape)}")
    print(f"Logits shape           : {tuple(logits.shape)}    "
          f"(expected (1, 1, 256, 256))")
    print(f"Bottleneck shape       : {tuple(bottleneck.shape)}     "
          f"(expected (1, 512, 8, 8))")
    print(f"Total parameters       : {n_params:>14,d}   "
          f"(~{n_params / 1e6:.2f}M, target ~15M)")
    print(f"Trainable parameters   : {n_trainable:>14,d}")
    print(line)

    # AMP smoke test: confirm the full forward pass runs cleanly under autocast
    # without an internal fp32 promotion. Skipped if CUDA is unavailable.
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        x_cuda = x.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            logits_amp, bottleneck_amp = model_cuda(x_cuda)
        print(f"AMP forward dtype      : logits={logits_amp.dtype}, "
              f"bottleneck={bottleneck_amp.dtype}")
        print(line)
