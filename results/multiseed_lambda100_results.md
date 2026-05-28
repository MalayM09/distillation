# TAKD Ablation Matrix — Multi-Seed Results

| Row | Ablation | n seeds | Dice (mean ± std) | IoU (mean ± std) |
|---|---|---|---|---|
| **B1** | Vanilla EdgeUNet (floor) | 3 | 0.7357 ± 0.0222 | 0.6469 ± 0.0233 |
| **B2** | HA-Net (ceiling) | 3 | 0.7653 ± 0.0240 | 0.6754 ± 0.0279 |
| **A1** | MedSAM->EdgeUNet (Capacity Gap) | 3 | 0.7353 ± 0.0063 | 0.6434 ± 0.0126 |
| **A2** | HA-Net(sup)->EdgeUNet | 3 | 0.7579 ± 0.0235 | 0.6624 ± 0.0240 |
| **A3h1** | MedSAM->HA-Net (assistant) | 3 | 0.7317 ± 0.0298 | 0.6405 ± 0.0316 |
| **A3** | TAKD: MedSAM->HA-Net->EdgeUNet | 3 | 0.7536 ± 0.0230 | 0.6623 ± 0.0272 |

## Paired Bootstrap (Dice, n=10,000)

| Comparison | Δ Dice | CI95 | p | sig |
|---|---|---|---|---|
| A1 - B1  Capacity Gap test | -0.0005 | [-0.0157, +0.0230] | 0.733 | ns |
| A2 - B1  A2 - B1 lift | +0.0221 | [+0.0203, +0.0233] | 0.000 | *** |
| A3 - B1  A3 - B1 lift | +0.0178 | [+0.0132, +0.0231] | 0.000 | *** |
| A3 - A2  TAKD decision rule | -0.0043 | [-0.0101, +0.0003] | 0.067 | ns |
| A2 - B2  A2 - B2 | -0.0074 | [-0.0190, +0.0067] | 0.300 | ns |
| A3 - B2  A3 - B2 | -0.0117 | [-0.0187, -0.0034] | 0.000 | *** |
| A3h1 - B2  A3-h1 vs B2 | -0.0335 | [-0.0439, -0.0236] | 0.000 | *** |
