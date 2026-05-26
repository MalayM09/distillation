"""
scripts/train_hop2.py — Hop-2 distillation trainer (HA-Net -> EdgeUNet)

Trains the EdgeUNet student against a frozen HA-Net teacher under the Hop-2
loss (BCE + Dice + MSE feature matching + KL logit distillation, per Rule 2
of docs/02). This script is the engine for ablation A2 (HA-Net -> U-Net)
and for the second hop of A3 (TAKD).

Wiring
------
- Teacher  : HA-Net (frozen, eval mode, no_grad forward).
- Student  : EdgeUNet (trainable).
- Loss     : src.loss_hop2.Hop2DistillationLoss
             (contains a trainable 192->512 1x1 projector).
- Optimizer: AdamW over EdgeUNet params + projector params.

Resolution-faithful evaluation
------------------------------
Training is at 256x256. Validation Dice and IoU are computed at native
ultrasound resolution: student logits are bilinearly upsampled to the
original (H, W) and re-thresholded against the native-size GT mask.
This is the only metric we report.

Kaggle T4 memory hygiene
------------------------
- Teacher forward under torch.no_grad() + AMP autocast.
- Per-batch `del` of teacher tensors and `torch.cuda.empty_cache()` (kept
  defensively per the project's stated rules; if profiling shows it costs
  more wall-time than VRAM headroom buys, tune `--empty_cache_every`).
- AMP / FP16 throughout via `torch.cuda.amp.autocast` + `GradScaler`.

Usage
-----
    python scripts/train_hop2.py \\
        --busi_root /kaggle/input/busi/Dataset_BUSI_with_GT \\
        --teacher_checkpoint /kaggle/working/hanet_phaseB2.pt \\
        --output_dir /kaggle/working/outputs_hop2 \\
        --epochs 60 --batch_size 16 --lr 3e-4 --seed 20260530

The teacher checkpoint must be a state_dict produced by training a
src.models.HANet to convergence under the B2 supervised-only protocol (or
under Hop-1 InfoNCE distillation for the A3 cascade — both produce a
compatible HANet state_dict).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import HANet
from src.edgeunet import EdgeUNet
from src.loss_hop2 import Hop2DistillationLoss


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("train_hop2")


# --------------------------------------------------------------------------- #
# BUSI dataset                                                                #
# --------------------------------------------------------------------------- #

_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)
BUSI_CLASSES = ("benign", "malignant")


def discover_busi_pairs(root: Path) -> List[Tuple[Path, Tuple[Path, ...], str]]:
    """Walk a BUSI directory and return (image_path, mask_paths, class) tuples.

    Multi-component lesions have multiple mask files (`*_mask.png`, `*_mask_1.png`,
    ...); they are OR-merged at __getitem__ time into a single binary mask.
    """
    pairs: List[Tuple[Path, Tuple[Path, ...], str]] = []
    for cls in BUSI_CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            if _MASK_RE.search(img_path.stem):
                continue
            mask_paths = tuple(sorted(cls_dir.glob(f"{img_path.stem}_mask*.png")))
            if not mask_paths:
                continue
            pairs.append((img_path, mask_paths, cls))
    return pairs


def or_merge_masks(mask_paths: Tuple[Path, ...]) -> np.ndarray:
    merged = None
    for mp in mask_paths:
        m = np.asarray(Image.open(mp).convert("L"))
        b = (m > 127).astype(np.uint8)
        merged = b if merged is None else np.maximum(merged, b)
    return merged  # type: ignore[return-value]


class BUSITrainDataset(Dataset):
    """256x256 BUSI training set. hflip-only augmentation per docs/03."""

    def __init__(self, pairs, image_size: int = 256, augment: bool = True) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_paths, _ = self.pairs[idx]

        img = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        merged = or_merge_masks(mask_paths)
        mask_pil = Image.fromarray((merged * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST)

        img_t = TF.pil_to_tensor(img).float().div_(255.0)
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()

        if self.augment and torch.rand(1).item() < 0.5:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)

        return img_t, mask_t


class BUSIValDataset(Dataset):
    """Resolution-faithful validation set: returns (img_256, gt_native, native_hw)."""

    def __init__(self, pairs, image_size: int = 256) -> None:
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        img_path, mask_paths, _ = self.pairs[idx]

        img_native = Image.open(img_path).convert("RGB")
        native_h, native_w = img_native.height, img_native.width
        img_resized = img_native.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_t = TF.pil_to_tensor(img_resized).float().div_(255.0)

        merged_native = or_merge_masks(mask_paths)  # (H, W) uint8 in {0, 1}
        gt_native = torch.from_numpy(merged_native).to(torch.float32).unsqueeze(0)

        return img_t, gt_native, (native_h, native_w)


def val_collate(batch):
    """Custom collate: native masks have variable size, so we keep a list."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    gts = [b[1] for b in batch]
    sizes = [b[2] for b in batch]
    return imgs, gts, sizes


# --------------------------------------------------------------------------- #
# Resolution-faithful Dice / IoU at native size                               #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def upsample_logits_to_native(
    logits_256: torch.Tensor,
    native_hw: Tuple[int, int],
) -> torch.Tensor:
    """logits_256 (1, 1, 256, 256) -> probs (1, 1, H_native, W_native)."""
    up = F.interpolate(
        logits_256.float(),
        size=native_hw,
        mode="bilinear",
        align_corners=False,
    )
    return torch.sigmoid(up)


@torch.no_grad()
def dice_iou_native(
    pred_prob_native: torch.Tensor,
    gt_native: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Tuple[float, float]:
    """Per-sample Dice and IoU at native resolution. Inputs (1, H, W) or (1, 1, H, W)."""
    pred = (pred_prob_native > threshold).float().reshape(-1)
    gt = (gt_native > 0.5).float().reshape(-1)
    inter = (pred * gt).sum().item()
    p = pred.sum().item()
    g = gt.sum().item()
    dice = (2.0 * inter + smooth) / (p + g + smooth)
    iou = (inter + smooth) / (p + g - inter + smooth)
    return float(dice), float(iou)


# --------------------------------------------------------------------------- #
# Train / val loops                                                           #
# --------------------------------------------------------------------------- #

def train_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loss_fn: Hop2DistillationLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    empty_cache_every: int,
) -> Dict[str, float]:
    student.train()
    loss_fn.train()
    teacher.eval()

    sums = {k: 0.0 for k in ("total", "seg", "bce", "dice", "mse", "kl")}
    n_batches = 0

    pbar = tqdm(loader, desc=f"train ep{epoch}", leave=False)
    for step, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            # Frozen teacher forward — no grad, no parameter updates.
            with torch.no_grad():
                t_logits, t_bottleneck = teacher(imgs)

            s_logits, s_bottleneck = student(imgs)
            total, parts = loss_fn(
                student_logits=s_logits,
                teacher_logits=t_logits,
                student_bottleneck=s_bottleneck,
                teacher_bottleneck=t_bottleneck,
                ground_truth_mask=masks,
            )

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        for k in sums:
            sums[k] += float(parts[k].item())
        n_batches += 1

        pbar.set_postfix(
            total=f"{parts['total'].item():.3f}",
            seg=f"{parts['seg'].item():.3f}",
            mse=f"{parts['mse'].item():.3f}",
            kl=f"{parts['kl'].item():.3f}",
        )

        # Per-batch memory hygiene
        del imgs, masks, t_logits, t_bottleneck, s_logits, s_bottleneck, total, parts
        if empty_cache_every > 0 and (step + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

    return {k: sums[k] / max(n_batches, 1) for k in sums}


@torch.no_grad()
def validate(
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> Dict[str, float]:
    student.eval()

    dices: List[float] = []
    ious: List[float] = []

    pbar = tqdm(loader, desc=f"val ep{epoch}  ", leave=False)
    for imgs, gts_native, sizes in pbar:
        imgs = imgs.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            s_logits, _ = student(imgs)

        # Native-resolution scoring (resolution-faithful per docs/03).
        for b_idx, (gt_native, hw) in enumerate(zip(gts_native, sizes)):
            prob = upsample_logits_to_native(s_logits[b_idx : b_idx + 1], hw).cpu()
            d, i = dice_iou_native(prob, gt_native)
            dices.append(d)
            ious.append(i)

        del imgs, s_logits

    return {
        "val_dice_mean": float(np.mean(dices)),
        "val_dice_std":  float(np.std(dices)),
        "val_iou_mean":  float(np.mean(ious)),
        "val_iou_std":   float(np.std(ious)),
        "n_val":         len(dices),
    }


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # benchmark for T4 throughput


def deterministic_split(
    pairs: List, val_frac: float, seed: int,
) -> Tuple[List, List]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = int(round(len(pairs) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, p in enumerate(pairs):
        (val if i in val_idx else train).append(p)
    return train, val


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hop-2 distillation trainer (HA-Net -> EdgeUNet).")
    p.add_argument("--busi_root", type=Path, required=True,
                   help="Path to BUSI root containing benign/ and malignant/ subdirs.")
    p.add_argument("--teacher_checkpoint", type=Path, required=True,
                   help="Path to a trained HA-Net state_dict (.pt).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directory for checkpoints, history, and manifest.")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=1,
                   help="Val batch size; native masks are variable-shaped so 1 is safest.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--dice_smooth", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=20260530)
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP (default on; AMP is mandatory per docs/03 on T4).")
    p.add_argument("--empty_cache_every", type=int, default=1,
                   help="Call torch.cuda.empty_cache() every N batches. 0 disables. Default 1.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    log.info("device=%s   amp=%s   seed=%d", device, use_amp, args.seed)

    # ----- Data -----
    pairs = discover_busi_pairs(args.busi_root)
    if not pairs:
        raise RuntimeError(f"No BUSI pairs discovered under {args.busi_root}")
    train_pairs, val_pairs = deterministic_split(pairs, args.val_frac, args.seed)
    log.info("BUSI: total=%d  train=%d  val=%d", len(pairs), len(train_pairs), len(val_pairs))

    train_ds = BUSITrainDataset(train_pairs, image_size=256, augment=True)
    val_ds = BUSIValDataset(val_pairs, image_size=256)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=val_collate, persistent_workers=(args.num_workers > 0),
    )

    # ----- Models -----
    teacher = HANet(in_channels=3, num_classes=1).to(device)
    state = torch.load(args.teacher_checkpoint, map_location=device)
    # Accept either a bare state_dict or a wrapped {"model": ...} checkpoint.
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = teacher.load_state_dict(state, strict=True)
    log.info("teacher loaded from %s  (missing=%d unexpected=%d)",
             args.teacher_checkpoint, len(missing), len(unexpected))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = EdgeUNet(in_channels=3, num_classes=1).to(device)
    n_student = sum(p.numel() for p in student.parameters())
    assert n_student < 3_000_000, f"student violates <3M budget: {n_student:,d}"
    log.info("student EdgeUNet params=%s (<3M ok)", f"{n_student:,d}")

    loss_fn = Hop2DistillationLoss(
        student_bottleneck_channels=EdgeUNet.BOTTLENECK_CHANNELS,
        teacher_bottleneck_channels=512,
        temperature=args.temperature,
        lambda_mse=args.lambda_mse,
        lambda_kl=args.lambda_kl,
        dice_smooth=args.dice_smooth,
    ).to(device)
    n_proj = sum(p.numel() for p in loss_fn.projector_parameters())
    log.info("loss projector params=%s   T=%.1f  lambda_mse=%.2f  lambda_kl=%.2f",
             f"{n_proj:,d}", args.temperature, args.lambda_mse, args.lambda_kl)

    # ----- Optimizer (student + projector) -----
    trainable_params = list(student.parameters()) + list(loss_fn.projector_parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ----- Train -----
    history: List[Dict] = []
    best_dice = -1.0
    best_path = args.output_dir / "student_best.pt"
    last_path = args.output_dir / "student_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            teacher=teacher, student=student, loss_fn=loss_fn,
            loader=train_loader, optimizer=optimizer, scaler=scaler,
            device=device, epoch=epoch, use_amp=use_amp,
            empty_cache_every=args.empty_cache_every,
        )
        val_metrics = validate(
            student=student, loader=val_loader, device=device,
            use_amp=use_amp, epoch=epoch,
        )

        log.info(
            "ep%03d  train: total=%.4f seg=%.4f mse=%.4f kl=%.4f  "
            "val: dice=%.4f iou=%.4f  (%.1fs)",
            epoch,
            train_metrics["total"], train_metrics["seg"],
            train_metrics["mse"], train_metrics["kl"],
            val_metrics["val_dice_mean"], val_metrics["val_iou_mean"],
            time.time() - t0,
        )

        record = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(record)

        # ----- Checkpoint (student + projector + optimizer for resumability) -----
        ckpt = {
            "epoch": epoch,
            "student": student.state_dict(),
            "projector": loss_fn.projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val_dice": val_metrics["val_dice_mean"],
            "args": vars(args) | {
                "busi_root": str(args.busi_root),
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "output_dir": str(args.output_dir),
            },
        }
        torch.save(ckpt, last_path)
        if val_metrics["val_dice_mean"] > best_dice:
            best_dice = val_metrics["val_dice_mean"]
            torch.save(ckpt, best_path)
            log.info("  -> new best val_dice=%.4f saved to %s", best_dice, best_path.name)

        # Periodic full cache flush at epoch boundary
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----- Persist training history -----
    with open(args.output_dir / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    log.info("training history written to %s", args.output_dir / "training_history.json")
    log.info("best val_dice=%.4f", best_dice)


if __name__ == "__main__":
    main()
