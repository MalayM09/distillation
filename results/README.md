# Results — CAISc 2026 TAKD Ablation Matrix

Canonical experimental results for the BUSI TAKD study. All checkpoints
omitted (large, regeneratable); per-epoch metric histories preserved as
`training_history.json` files.

## Contents

```
results/
├── README.md                              (this file)
│
├── multiseed_lambda100_results.md         B1/B2/A1/A2/A3-h1/A3-h2 paper table (λ_NCE = 1.0)
├── multiseed_lambda100_per_seed.csv       Flat per-seed Dice/IoU for the above
├── multiseed_lambda100/                   Per-run training_history.json (18 runs)
│   ├── b1_edgeunet_seed{20260530,42,1337}/
│   ├── b2_hanet_seed{...}/
│   ├── a1_direct_seed{...}/
│   ├── a2_seed{...}/
│   ├── a3_hop1_seed{...}/
│   └── a3_hop2_seed{...}/
│
├── multiseed_lambda025_summary.md         Hop-1 λ_NCE ablation diagnostic
├── multiseed_lambda025_per_seed.csv       Flat per-seed for the ablation
└── multiseed_lambda025/                   Per-run histories (6 runs: A3-h1 + A3-h2 only)
    ├── a3_hop1_seed{...}/
    └── a3_hop2_seed{...}/
```

## How to reproduce

```bash
# Train the full λ=1.0 matrix (18 runs, ~6h on Kaggle T4):
python scripts/run_multiseed.py \
    --busi_root <BUSI_path> --medsam_cache_root <MedSAM_cache_path> \
    --output_root /kaggle/working/runs_multiseed \
    --seeds 20260530 42 1337 --epochs 40 --batch_size 16 --lr 3e-4

# Train the λ=0.25 Hop-1 ablation (6 runs, ~3h):
for seed in 20260530 42 1337; do
    python scripts/train_hop1.py --busi_root <...> --medsam_cache_root <...> \
        --output_dir runs_lambda025/a3_hop1_seed${seed} \
        --epochs 40 --batch_size 16 --lr 3e-4 --num_workers 2 \
        --lambda_nce 0.25 --seed $seed
    python scripts/train_hop2.py --busi_root <...> \
        --teacher_checkpoint runs_lambda025/a3_hop1_seed${seed}/hanet_best.pt \
        --output_dir runs_lambda025/a3_hop2_seed${seed} \
        --epochs 40 --batch_size 16 --lr 3e-4 --num_workers 2 --seed $seed
done

# Aggregate the λ=1.0 matrix:
python scripts/aggregate_multiseed.py --runs_root results/multiseed_lambda100 \
    --csv /tmp/out.csv --md /tmp/out.md
```

## Headline results

Three claims, all multi-seed validated (3 seeds, paired bootstrap, n=10,000):

1. **Capacity Gap is real.** At R ≈ 30 (MedSAM 91M → EdgeUNet 3M),
   direct InfoNCE distillation provides no mean Dice lift over
   supervised training: A1 − B1 = −0.0005, p = 0.733 (ns).

2. **Capacity-matched distillation works.** At R ≈ 5
   (HA-Net 15M → EdgeUNet 3M, CNN→CNN MSE+KL), the same student gains
   +0.022 Dice over supervised baseline: A2 − B1 = +0.0221,
   CI95 [+0.0203, +0.0233], p < 0.001.

3. **Naive TAKD does not bridge the gap, and the failure is not
   loss-weight tuning.** A3 − A2 = −0.0043 at λ_NCE = 1.0 (p = 0.067),
   −0.0050 at λ_NCE = 0.25 (p = 0.291). Dropping λ_NCE significantly
   improves the Hop-1 assistant (Δ +0.0152, p < 0.001) but the
   improvement does not propagate to the student (Δ −0.0007, p = 0.808).
   The bottleneck is the cascade's Hop-2 transfer capacity, not Hop-1
   tuning.

**Auxiliary finding (variance):** Contrastive distillation reduces seed
variance ~3.5× (A1 std 0.0063 vs B1 std 0.0222), a robust property
independent of mean Dice.
