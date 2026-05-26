"""
src/edgeunet.py — TAKD deployable student: vanilla 6-level U-Net (<3M params)

The single student architecture used by every row of the BUSI ablation
matrix (B1, A1, A2, A3). Strictly self-contained: no imports from
src/models.py — the HA-Net file is the assistant; this file is the student;
keeping them physically separate matches the "one model per file" workflow
used by the Kaggle notebooks.

Channel widths [16, 32, 64, 96, 144, 192] yield ~2.19M parameters with
~810K of headroom under the 3M edge-deployment ceiling (docs/03). The
bottleneck stride is /32 so the (B, 192, H/32, W/32) feature map is
spatially aligned with HA-Net's (B, 512, H/32, W/32) bottleneck — Hop-2
MSE matching reduces to a per-pixel distance after a single 1x1 channel
projection (192 -> 512) on the student side at training time.

Operator set
------------
    Conv2d, BatchNorm2d, ReLU, MaxPool2d, Upsample(bilinear), Concat.
    No attention, no transformer block, no LayerNorm, no GroupNorm.
    No Python-side control flow in the forward pass; static ONNX trace
    is verified in __main__ (opset 17).

Distillation surfaces
---------------------
    forward(x) -> (logits, bottleneck_features)
        logits             : (B, num_classes, H, W) raw, unactivated logits
                              at native input resolution — Hop-2 KL target.
        bottleneck_features: (B, 192, H/32, W/32) deepest encoder map,
                              captured BEFORE the first decoder upsample —
                              Hop-2 MSE target against HA-Net's bottleneck.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Primitives                                                                  #
# --------------------------------------------------------------------------- #

class DoubleConv(nn.Module):
    """Two conv3x3 -> BN -> ReLU stages. Canonical vanilla U-Net block.

    ReLU (not SiLU) is used in the student to maximize INT8 PTQ accuracy:
    ReLU is non-negative and folds cleanly into per-tensor calibration,
    whereas SiLU requires a lookup table on Hexagon / Ethos-U / CoreML
    INT8 backends and produces calibration distributions with long
    negative tails that degrade quantization fidelity. Binding per
    docs/03 (no operators outside the INT8-deployable set).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
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
    """MaxPool2x -> DoubleConv. Halves spatial, lifts channels."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Fixed-scale bilinear upsample x2 -> concat(skip) -> DoubleConv.

    `nn.Upsample(scale_factor=2)` is used rather than `F.interpolate(size=...)`
    because the fixed scale factor produces a static ONNX Resize node with no
    runtime shape dependency. Dynamic-shape Resize nodes are a recurring source
    of INT8 calibration failure on Hexagon / Ethos-U backends.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


# --------------------------------------------------------------------------- #
# EdgeUNet                                                                    #
# --------------------------------------------------------------------------- #

class EdgeUNet(nn.Module):
    """Vanilla 6-level U-Net (<3M params), ONNX-clean, INT8-deployable.

    Stride layout for a 256x256 input:
        e1 :  16 ch @ 256x256       (skip 0)
        e2 :  32 ch @ 128x128       (skip 1)
        e3 :  64 ch @  64x 64       (skip 2)
        e4 :  96 ch @  32x 32       (skip 3)
        e5 : 144 ch @  16x 16       (skip 4)
        b  : 192 ch @   8x  8       <- bottleneck_features (Hop-2 MSE target)

    Parameters
    ----------
    in_channels : int
        Input image channels. Default 3 to match the MedSAM-replicated cache.
    num_classes : int
        Output channels of the segmentation head. BUSI binary => 1.

    Returns
    -------
    forward(x) -> (logits, bottleneck_features)
    """

    BOTTLENECK_CHANNELS: int = 192

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        c = (16, 32, 64, 96, 144, 192)

        # Encoder
        self.enc1 = DoubleConv(in_channels, c[0])
        self.down2 = Down(c[0], c[1])
        self.down3 = Down(c[1], c[2])
        self.down4 = Down(c[2], c[3])
        self.down5 = Down(c[3], c[4])
        self.down6 = Down(c[4], c[5])  # produces the bottleneck at /32

        # Decoder
        self.up5 = Up(in_ch=c[5], skip_ch=c[4], out_ch=c[4])
        self.up4 = Up(in_ch=c[4], skip_ch=c[3], out_ch=c[3])
        self.up3 = Up(in_ch=c[3], skip_ch=c[2], out_ch=c[2])
        self.up2 = Up(in_ch=c[2], skip_ch=c[1], out_ch=c[1])
        self.up1 = Up(in_ch=c[1], skip_ch=c[0], out_ch=c[0])

        # 1x1 segmentation head (logits with bias for BCEWithLogitsLoss)
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)        # (B,  16, H,    W)
        e2 = self.down2(e1)      # (B,  32, H/2,  W/2)
        e3 = self.down3(e2)      # (B,  64, H/4,  W/4)
        e4 = self.down4(e3)      # (B,  96, H/8,  W/8)
        e5 = self.down5(e4)      # (B, 144, H/16, W/16)
        bottleneck_features = self.down6(e5)   # (B, 192, H/32, W/32)

        d5 = self.up5(bottleneck_features, e5)
        d4 = self.up4(d5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        logits = self.head(d1)
        return logits, bottleneck_features


# --------------------------------------------------------------------------- #
# Validation block                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    model = EdgeUNet(in_channels=3, num_classes=1).eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits, bottleneck = model(x)

    # Hard constraints from docs/03_CONSTRAINTS_AND_RULES.md
    assert n_params < 3_000_000, (
        f"EdgeUNet violates the <3M edge-deployment budget: "
        f"got {n_params:,d} parameters."
    )
    assert logits.shape == (1, 1, 256, 256), (
        f"logits shape mismatch: got {tuple(logits.shape)}, "
        f"expected (1, 1, 256, 256)."
    )
    assert bottleneck.shape == (1, EdgeUNet.BOTTLENECK_CHANNELS, 8, 8), (
        f"bottleneck shape mismatch: got {tuple(bottleneck.shape)}, "
        f"expected (1, {EdgeUNet.BOTTLENECK_CHANNELS}, 8, 8)."
    )

    line = "=" * 68
    print(line)
    print("EdgeUNet (TAKD deployable student) — architectural validation")
    print(line)
    print(f"Input shape            : {tuple(x.shape)}")
    print(f"Logits shape           : {tuple(logits.shape)}    "
          f"(expected (1, 1, 256, 256))")
    print(f"Bottleneck shape       : {tuple(bottleneck.shape)}     "
          f"(expected (1, 192, 8, 8))")
    print(f"Total parameters       : {n_params:>14,d}   "
          f"(~{n_params / 1e6:.2f}M, strict ceiling <3.00M)")
    print(f"Trainable parameters   : {n_trainable:>14,d}")
    print(f"Headroom under 3M      : {3_000_000 - n_params:>14,d}   "
          f"(~{(3_000_000 - n_params) / 1e6:.2f}M)")
    print(line)

    # AMP smoke test (skipped if CUDA is unavailable locally; will run on T4).
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        x_cuda = x.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            logits_amp, bottleneck_amp = model_cuda(x_cuda)
        print(f"AMP forward dtype      : logits={logits_amp.dtype}, "
              f"bottleneck={bottleneck_amp.dtype}")
        print(line)

    # ONNX static-trace smoke test (verifies no data-dependent control flow).
    # Uses the legacy TorchScript-based exporter (`dynamo=False`) so the only
    # required dependency is torch itself — no `onnxscript` install on Kaggle.
    try:
        import io
        buf = io.BytesIO()
        torch.onnx.export(
            model,
            x,
            buf,
            input_names=["image"],
            output_names=["logits", "bottleneck"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
        print(f"ONNX static trace      : OK ({buf.tell():,d} bytes, opset 17)")
        print(line)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"ONNX static trace      : FAILED ({type(exc).__name__}: {exc})")
        print(line)
