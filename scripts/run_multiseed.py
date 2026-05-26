"""
scripts/run_multiseed.py — Multi-seed the full TAKD ablation matrix.

Runs 6 ablations × 3 seeds = 18 trainings in dependency order:

    Stage 1 (independent):  B1, B2, A1, A3-h1   for each seed
    Stage 2 (dependent):    A2  (teacher = same-seed B2),
                            A3  (teacher = same-seed A3-h1)

Seed-matched teachers are CRITICAL: pairing A3-h2's teacher to the same seed
as the student isolates the variance contribution of the cascade from the
seed contribution of the assistant. Without this, the paired bootstrap that
docs/02 demands ("A3 must beat A2 by p<0.05") would be confounded.

Resumability
------------
A run is considered done if its `training_history.json` contains >= --epochs
records. Partial runs (e.g. killed by Kaggle session timeout) are re-executed
from scratch — there is no mid-run resume because checkpoints reload weights
but not the LR schedule or seed state.

Usage on Kaggle
---------------
    !cd /kaggle/working/distillation && \\
        python scripts/run_multiseed.py \\
            --busi_root /kaggle/input/datasets/aryashah2k/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT \\
            --medsam_cache_root /kaggle/input/datasets/malaym09/caisc-v1/medsam_cache \\
            --output_root /kaggle/working/runs_multiseed \\
            --epochs 40 --batch_size 16 --num_workers 2

The default seed list [20260530, 42, 1337] matches the legacy 3-seed study
in project_phase_status.md so deltas are comparable across the pivot.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("run_multiseed")


REPO_ROOT = Path(__file__).resolve().parent.parent


def is_complete(run_dir: Path, target_epochs: int) -> bool:
    """A run is complete if its history has >= target_epochs records."""
    hp = run_dir / "training_history.json"
    if not hp.is_file():
        return False
    try:
        hist = json.loads(hp.read_text())
        return len(hist) >= target_epochs
    except Exception:
        return False


def run(cmd: List[str], label: str) -> None:
    log.info("[%s] %s", label, " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        log.error("[%s] FAILED with exit %d after %.1fs", label, proc.returncode, time.time() - t0)
        sys.exit(proc.returncode)
    log.info("[%s] OK  (%.1fs)", label, time.time() - t0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed the TAKD ablation matrix.")
    p.add_argument("--busi_root", type=Path, required=True)
    p.add_argument("--medsam_cache_root", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True,
                   help="Parent dir; each run lands in <output_root>/<ablation>_seed<seed>/.")
    p.add_argument("--seeds", type=int, nargs="+", default=[20260530, 42, 1337],
                   help="Default matches the legacy 3-seed study.")
    p.add_argument("--epochs", type=int, default=40,
                   help="40 covers the typical peak epoch (37-40 for slow movers) per kagglerun2.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--skip_a3h1", action="store_true",
                   help="Skip A3-hop-1 standalone runs. Cannot be combined with running A3-h2 "
                        "(A3-h2 requires a same-seed Hop-1 teacher).")
    p.add_argument("--only", nargs="+", choices=["b1", "b2", "a1", "a2", "a3h1", "a3h2"],
                   default=None, help="Optional: run only a subset of ablations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    common = [
        "--busi_root", str(args.busi_root),
        "--output_dir", "<filled-per-run>",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--seed", "<filled-per-run>",
    ]

    def fill(template: List[str], output_dir: Path, seed: int) -> List[str]:
        out = list(template)
        out[out.index("<filled-per-run>")] = str(output_dir)
        out[out.index("<filled-per-run>")] = str(seed)
        return out

    selected = set(args.only) if args.only else {"b1", "b2", "a1", "a2", "a3h1", "a3h2"}
    if args.skip_a3h1:
        selected.discard("a3h1")

    t_total = time.time()

    # =====================================================================
    # Stage 1 — independent runs (B1, B2, A1, A3-h1) for every seed
    # =====================================================================
    for seed in args.seeds:
        # ---- B1 (vanilla EdgeUNet) ----
        if "b1" in selected:
            rd = args.output_root / f"b1_edgeunet_seed{seed}"
            if is_complete(rd, args.epochs):
                log.info("[skip] B1 seed=%d already done at %s", seed, rd)
            else:
                cmd = ["python", "scripts/train_baseline.py", "--model", "edgeunet"] + \
                      fill(common, rd, seed)
                run(cmd, f"B1.s{seed}")

        # ---- B2 (HA-Net supervised; teacher for A2) ----
        if "b2" in selected:
            rd = args.output_root / f"b2_hanet_seed{seed}"
            if is_complete(rd, args.epochs):
                log.info("[skip] B2 seed=%d already done at %s", seed, rd)
            else:
                cmd = ["python", "scripts/train_baseline.py", "--model", "hanet"] + \
                      fill(common, rd, seed)
                run(cmd, f"B2.s{seed}")

        # ---- A1 (direct MedSAM->EdgeUNet) ----
        if "a1" in selected:
            rd = args.output_root / f"a1_direct_seed{seed}"
            if is_complete(rd, args.epochs):
                log.info("[skip] A1 seed=%d already done at %s", seed, rd)
            else:
                cmd = ["python", "scripts/train_a1_direct.py",
                       "--medsam_cache_root", str(args.medsam_cache_root)] + \
                      fill(common, rd, seed)
                run(cmd, f"A1.s{seed}")

        # ---- A3-h1 (MedSAM->HA-Net; teacher for A3-h2) ----
        if "a3h1" in selected:
            rd = args.output_root / f"a3_hop1_seed{seed}"
            if is_complete(rd, args.epochs):
                log.info("[skip] A3-h1 seed=%d already done at %s", seed, rd)
            else:
                cmd = ["python", "scripts/train_hop1.py",
                       "--medsam_cache_root", str(args.medsam_cache_root)] + \
                      fill(common, rd, seed)
                run(cmd, f"A3h1.s{seed}")

    # =====================================================================
    # Stage 2 — dependent runs (A2 needs B2; A3-h2 needs A3-h1)
    # =====================================================================
    for seed in args.seeds:
        # ---- A2 (HA-Net(sup) -> EdgeUNet) ----
        if "a2" in selected:
            rd = args.output_root / f"a2_seed{seed}"
            teacher = args.output_root / f"b2_hanet_seed{seed}" / "hanet_baseline_best.pt"
            if is_complete(rd, args.epochs):
                log.info("[skip] A2 seed=%d already done at %s", seed, rd)
            elif not teacher.is_file():
                log.error("[A2.s%d] teacher missing at %s; skipping. Run B2 first.", seed, teacher)
            else:
                cmd = ["python", "scripts/train_hop2.py",
                       "--teacher_checkpoint", str(teacher)] + fill(common, rd, seed)
                run(cmd, f"A2.s{seed}")

        # ---- A3-h2 (TAKD: HA-Net(distilled) -> EdgeUNet) ----
        if "a3h2" in selected:
            rd = args.output_root / f"a3_hop2_seed{seed}"
            teacher = args.output_root / f"a3_hop1_seed{seed}" / "hanet_best.pt"
            if is_complete(rd, args.epochs):
                log.info("[skip] A3-h2 seed=%d already done at %s", seed, rd)
            elif not teacher.is_file():
                log.error("[A3h2.s%d] teacher missing at %s; skipping. Run A3-h1 first.",
                          seed, teacher)
            else:
                cmd = ["python", "scripts/train_hop2.py",
                       "--teacher_checkpoint", str(teacher)] + fill(common, rd, seed)
                run(cmd, f"A3h2.s{seed}")

    log.info("=" * 64)
    log.info("multi-seed orchestrator complete in %.1f min", (time.time() - t_total) / 60.0)
    log.info("output root: %s", args.output_root)
    log.info("next step  : python scripts/aggregate_multiseed.py --runs_root %s", args.output_root)


if __name__ == "__main__":
    main()
