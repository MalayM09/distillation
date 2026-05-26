"""
scripts/aggregate_multiseed.py — Aggregate the multi-seed matrix into final numbers.

Reads every `training_history.json` under --runs_root, computes per-ablation
mean ± std of best val Dice / IoU, and runs paired bootstrap tests on the
decision rules of docs/02:

    A1 - B1   : Capacity Gap test          (expect Δ ≈ 0 if claim holds)
    A2 - B1   : direct CNN-CNN distillation lift over floor
    A3 - B1   : full TAKD lift over floor
    A3 - A2   : *the* TAKD decision rule    (must be > 0 with p<0.05)
    A2 - B2   : student's approach to ceiling (A2 path)
    A3 - B2   : student's approach to ceiling (A3 path)

The bootstrap is paired-by-seed: a delta is computed within each seed first
(so per-seed run-to-run noise cancels out), and the bootstrap is over the
distribution of those per-seed deltas. This is the same protocol cited in
the docs/02 decision rules.

Outputs
-------
- stdout : human-readable summary + bootstrap table
- --csv  : per-seed best Dice/IoU as a flat table (one row per seed × ablation)
- --md   : final paper-ready Markdown table
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ABLATIONS = [
    ("B1",   "b1_edgeunet_seed",  "Vanilla EdgeUNet (floor)"),
    ("B2",   "b2_hanet_seed",     "HA-Net (ceiling)"),
    ("A1",   "a1_direct_seed",    "MedSAM->EdgeUNet (Capacity Gap)"),
    ("A2",   "a2_seed",           "HA-Net(sup)->EdgeUNet"),
    ("A3h1", "a3_hop1_seed",      "MedSAM->HA-Net (assistant)"),
    ("A3",   "a3_hop2_seed",      "TAKD: MedSAM->HA-Net->EdgeUNet"),
]


def best_metrics(hist: list) -> Tuple[float, float, int]:
    best = max(hist, key=lambda r: r["val_dice_mean"])
    return best["val_dice_mean"], best["val_iou_mean"], int(best["epoch"])


def gather(runs_root: Path) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for label, prefix, _ in ABLATIONS:
        rows = []
        for d in sorted(runs_root.glob(f"{prefix}*")):
            m = re.search(r"seed(\d+)$", d.name)
            if not m:
                continue
            seed = int(m.group(1))
            hp = d / "training_history.json"
            if not hp.is_file():
                continue
            hist = json.loads(hp.read_text())
            dice, iou, ep = best_metrics(hist)
            rows.append({"seed": seed, "dice": dice, "iou": iou, "best_epoch": ep,
                         "n_epochs": len(hist), "run_dir": str(d)})
        rows.sort(key=lambda r: r["seed"])
        if rows:
            out[label] = rows
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> Dict:
    """Two-sided paired bootstrap of (a - b) over seed-matched samples."""
    rng = np.random.default_rng(seed)
    deltas = np.asarray(a) - np.asarray(b)
    obs = float(deltas.mean())
    boot = np.empty(n)
    K = len(deltas)
    for i in range(n):
        idx = rng.integers(0, K, K)
        boot[i] = deltas[idx].mean()
    # Two-sided p: probability of bootstrap mean as extreme as 0 under null.
    p_two_sided = float(2.0 * min((boot >= 0).mean(), (boot <= 0).mean()))
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {
        "obs_delta": obs,
        "p_two_sided": p_two_sided,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n_pairs": K,
    }


def sig_marker(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def paired_arrays(results: Dict[str, List[Dict]], a_lbl: str, b_lbl: str
                  ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    rows_a = {r["seed"]: r["dice"] for r in results.get(a_lbl, [])}
    rows_b = {r["seed"]: r["dice"] for r in results.get(b_lbl, [])}
    common = sorted(set(rows_a) & set(rows_b))
    a = np.array([rows_a[s] for s in common])
    b = np.array([rows_b[s] for s in common])
    return a, b, common


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate multi-seed TAKD ablation matrix.")
    ap.add_argument("--runs_root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional: write per-seed flat table here.")
    ap.add_argument("--md", type=Path, default=None,
                    help="Optional: write paper-ready Markdown table here.")
    ap.add_argument("--bootstrap_n", type=int, default=10000)
    args = ap.parse_args()

    results = gather(args.runs_root)
    if not results:
        raise RuntimeError(f"No training_history.json files found under {args.runs_root}")

    # ---------------------------------------------------------------- Summary
    print()
    print(f"{'Ablation':<6s} {'desc':<35s} {'n':>3s} {'Dice mean ± std':>18s} "
          f"{'IoU  mean ± std':>18s}  per-seed Dice")
    print("-" * 120)
    summary: Dict[str, Dict] = {}
    for label, _, descr in ABLATIONS:
        if label not in results:
            print(f"{label:<6s} {descr:<35s} {'-':>3s}   (no runs found)")
            continue
        rows = results[label]
        dices = np.array([r["dice"] for r in rows])
        ious = np.array([r["iou"] for r in rows])
        summary[label] = {"dices": dices, "ious": ious, "rows": rows}
        per_seed = "  ".join(f"s{r['seed']}={r['dice']:.4f}(ep{r['best_epoch']})" for r in rows)
        print(f"{label:<6s} {descr:<35s} {len(rows):>3d} "
              f"{dices.mean():>10.4f} ± {dices.std(ddof=0):.4f} "
              f"{ious.mean():>11.4f} ± {ious.std(ddof=0):.4f}  {per_seed}")

    # ---------------------------------------------------------- Paired tests
    print()
    print(f"Paired bootstrap on Dice (n={args.bootstrap_n} resamples; pairs are seed-matched):")
    print("-" * 90)
    pair_tests = [
        ("A1", "B1", "Capacity Gap test         (docs/01: expect A1 ≈ B1)"),
        ("A2", "B1", "A2 - B1 lift"),
        ("A3", "B1", "A3 - B1 lift"),
        ("A3", "A2", "TAKD decision rule        (docs/02: A3 must beat A2 with p<0.05)"),
        ("A2", "B2", "A2 - B2  (student vs ceiling)"),
        ("A3", "B2", "A3 - B2  (student vs ceiling)"),
        ("A3h1", "B2", "A3-h1 vs B2  (assistant quality)"),
    ]
    for a, b, descr in pair_tests:
        if a not in summary or b not in summary:
            print(f"  {descr:<60s} skipped (missing {a} or {b})")
            continue
        xa, xb, seeds = paired_arrays(results, a, b)
        if len(xa) < 2:
            print(f"  {descr:<60s} skipped (need ≥2 paired seeds; got {len(xa)})")
            continue
        res = paired_bootstrap(xa, xb, n=args.bootstrap_n)
        print(f"  {descr:<60s}")
        print(f"    Δ = {res['obs_delta']:+.4f}  CI95=[{res['ci95_lo']:+.4f}, {res['ci95_hi']:+.4f}]"
              f"  p={res['p_two_sided']:.3f}  {sig_marker(res['p_two_sided'])}"
              f"  (n_pairs={res['n_pairs']}, seeds={seeds})")

    # ---------------------------------------------------------- CSV export
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ablation", "seed", "best_dice", "best_iou", "best_epoch",
                        "n_epochs", "run_dir"])
            for label, _, _ in ABLATIONS:
                for r in results.get(label, []):
                    w.writerow([label, r["seed"], f"{r['dice']:.6f}", f"{r['iou']:.6f}",
                                r["best_epoch"], r["n_epochs"], r["run_dir"]])
        print(f"\nwrote per-seed CSV to {args.csv}")

    # ---------------------------------------------------------- Markdown export
    if args.md:
        lines = []
        lines.append("# TAKD Ablation Matrix — Multi-Seed Results\n")
        lines.append("| Row | Ablation | n seeds | Dice (mean ± std) | IoU (mean ± std) |")
        lines.append("|---|---|---|---|---|")
        for label, _, descr in ABLATIONS:
            if label not in summary:
                continue
            d = summary[label]["dices"]
            i = summary[label]["ious"]
            lines.append(f"| **{label}** | {descr} | {len(d)} | "
                         f"{d.mean():.4f} ± {d.std(ddof=0):.4f} | "
                         f"{i.mean():.4f} ± {i.std(ddof=0):.4f} |")
        lines.append("\n## Paired Bootstrap (Dice, n=10,000)\n")
        lines.append("| Comparison | Δ Dice | CI95 | p | sig |")
        lines.append("|---|---|---|---|---|")
        for a, b, descr in pair_tests:
            if a not in summary or b not in summary:
                continue
            xa, xb, _ = paired_arrays(results, a, b)
            if len(xa) < 2:
                continue
            r = paired_bootstrap(xa, xb, n=args.bootstrap_n)
            lines.append(f"| {a} - {b}  {descr.split('(')[0].strip()} | "
                         f"{r['obs_delta']:+.4f} | "
                         f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | "
                         f"{r['p_two_sided']:.3f} | {sig_marker(r['p_two_sided'])} |")
        args.md.write_text("\n".join(lines) + "\n")
        print(f"wrote Markdown table to {args.md}")


if __name__ == "__main__":
    main()
