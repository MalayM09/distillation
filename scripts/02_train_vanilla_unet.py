"""
scripts/02_train_vanilla_unet.py — CAISc 2026

Phase 2a: Vanilla EdgeUNet baseline on BUSI. No distillation, no teacher.
Trained purely on ground-truth masks with BCEWithLogitsLoss + soft Dice.

This script's job is to *anchor the ablation*: it establishes the empirical
floor (Dice, IoU, param count) that the Phase 2b contrastively-distilled
student must beat to justify the paper's central novelty claim.

Reads BUSI PNGs directly — deliberately not the MedSAM .pt cache, since the
baseline must not have any teacher signal in scope.

Dataset
-------
BUSI benign + malignant (647 cases). Normal class excluded — consistent with
Phase 1 and with the segmentation-only framing of the paper.

Loss
----
BCEWithLogitsLoss + soft DiceLoss, equal weight. BCE drives pixel-level
calibration; Dice rescues the small-foreground regime (BUSI lesions are
~1-15% of frame, so unweighted BCE alone collapses to all-background).

Augmentations (train only)
--------------------------
Paired random horizontal flip + ±15° random rotation. No vertical flip
(anatomy-preserving for ultrasound). No colour jitter (clinically misleading
on grayscale intensity).

Output
------
outputs/vanilla_unet_best.pth — state_dict + metadata for best val Dice.
outputs/vanilla_unet_history.json — per-epoch metrics for paper plots.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Project-root on path so `from src.models import EdgeUNet` works regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import EdgeUNet  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset discovery — replicates Phase 1's BUSI scan, normal class excluded
# ---------------------------------------------------------------------------

_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)
BUSI_CLASSES = ("benign", "malignant")


def discover_busi_pairs(root: Path) -> list[tuple[Path, tuple[Path, ...], str]]:
    pairs: list[tuple[Path, tuple[Path, ...], str]] = []
    for cls in BUSI_CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            logging.warning("missing class directory: %s", cls_dir)
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            if _MASK_RE.search(img_path.stem):
                continue
            mask_paths = tuple(sorted(cls_dir.glob(f"{img_path.stem}_mask*.png")))
            if not mask_paths:
                continue
            pairs.append((img_path, mask_paths, cls))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BUSIVanillaDataset(Dataset):
    """
    BUSI image/mask loader for the no-distillation baseline.

    Returns
    -------
    image : (3, H, W) float32 in [0, 1]
    mask  : (1, H, W) float32 in {0, 1} — OR-merge of all _mask*.png siblings
    """

    def __init__(
        self,
        pairs: list[tuple[Path, tuple[Path, ...], str]],
        image_size: int = 256,
        augment: bool = False,
        rotation_deg: float = 15.0,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.rotation_deg = rotation_deg

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_paths, _cls = self.pairs[idx]

        # Image → 3-channel RGB at target resolution
        img = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )

        # OR-merge multi-mask BUSI cases at native resolution, then resize
        # nearest-neighbour to preserve binary boundaries.
        merged: np.ndarray | None = None
        for mp in mask_paths:
            m = np.asarray(Image.open(mp).convert("L"))
            b = (m > 127).astype(np.uint8)
            merged = b if merged is None else np.maximum(merged, b)
        mask_pil = Image.fromarray((merged * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST,
        )

        img_t = TF.pil_to_tensor(img).float().div_(255.0)               # (3, H, W) ∈ [0, 1]
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()  # (1, H, W) ∈ {0, 1}

        if self.augment:
            img_t, mask_t = self._paired_augment(img_t, mask_t)
        return img_t, mask_t

    def _paired_augment(
        self, img: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img); mask = TF.hflip(mask)
        angle = (torch.rand(1).item() * 2 - 1) * self.rotation_deg
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        # Rotation can produce values slightly off {0, 1} via interpolation —
        # the NEAREST mode above prevents that for the mask. Re-binarise as
        # belt-and-braces in case of upstream torchvision changes.
        mask = (mask > 0.5).float()
        return img, mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    """1 - mean per-sample Dice. Operates on raw logits (applies sigmoid)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_iou_dice(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6,
) -> tuple[float, float]:
    preds = (torch.sigmoid(logits) > threshold).float()
    dims = (2, 3)
    inter = (preds * targets).sum(dim=dims)
    p_sum = preds.sum(dim=dims)
    t_sum = targets.sum(dim=dims)
    union = p_sum + t_sum - inter
    iou = (inter + eps) / (union + eps)
    dice = (2.0 * inter + eps) / (p_sum + t_sum + eps)
    return iou.mean().item(), dice.mean().item()


# ---------------------------------------------------------------------------
# Train / val passes
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler) -> tuple[float, float, float]:
    model.train()
    losses, ious, dices = [], [], []
    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            iou, dice = batch_iou_dice(logits.float(), mask)
        losses.append(loss.item()); ious.append(iou); dices.append(dice)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    return float(np.mean(losses)), float(np.mean(ious)), float(np.mean(dices))


@torch.inference_mode()
def validate(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    losses, ious, dices = [], [], []
    pbar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        loss = criterion(logits, mask)
        iou, dice = batch_iou_dice(logits, mask)
        losses.append(loss.item()); ious.append(iou); dices.append(dice)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")
    return float(np.mean(losses)), float(np.mean(ious)), float(np.mean(dices))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Vanilla EdgeUNet baseline on BUSI (CAISc 2026 Phase 2a).")
    parser.add_argument("--busi-root", type=Path, required=True,
                        help="Root containing benign/ malignant/ subdirs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Where vanilla_unet_best.pth + history.json land.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed-precision (default: AMP on for CUDA).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Reproducibility
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True   # input shapes are fixed → fastest kernels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable — CPU training will be very slow.")
    use_amp = (not args.no_amp) and device.type == "cuda"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "vanilla_unet_best.pth"
    history_path = args.output_dir / "vanilla_unet_history.json"

    # --- Data --------------------------------------------------------------
    logging.info("discovering BUSI under %s", args.busi_root)
    pairs = discover_busi_pairs(args.busi_root)
    if not pairs:
        logging.error("no pairs found — verify --busi-root layout")
        return 2
    by_cls = {c: sum(1 for _, _, k in pairs if k == c) for c in BUSI_CLASSES}
    logging.info("discovered %d pairs (%s)", len(pairs), by_cls)

    # Deterministic shuffle then split. We don't stratify by class because
    # benign:malignant is ~2:1 and random splits hold the ratio within ±1%
    # at the 80/20 cut — stratification would only matter at smaller splits.
    random.Random(args.seed).shuffle(pairs)
    n_val = int(round(len(pairs) * args.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    logging.info("split — train: %d | val: %d", len(train_pairs), len(val_pairs))

    train_ds = BUSIVanillaDataset(train_pairs, image_size=args.image_size, augment=True)
    val_ds = BUSIVanillaDataset(val_pairs, image_size=args.image_size, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=args.num_workers > 0,
    )

    # --- Model -------------------------------------------------------------
    model = EdgeUNet(in_channels=3, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("EdgeUNet params: %.3fM (%d)", n_params / 1e6, n_params)
    if n_params >= 3_000_000:
        logging.error("model exceeds 3M-param budget (%d) — abort", n_params)
        return 2

    # --- Optim -------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # --- Train -------------------------------------------------------------
    best_dice = 0.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou, tr_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
        )
        va_loss, va_iou, va_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        logging.info(
            "epoch %3d/%d | train loss %.4f dice %.4f iou %.4f | "
            "val loss %.4f dice %.4f iou %.4f | lr %.2e",
            epoch, args.epochs, tr_loss, tr_dice, tr_iou, va_loss, va_dice, va_iou, lr_now,
        )
        history.append({
            "epoch": epoch, "lr": lr_now,
            "train_loss": tr_loss, "train_iou": tr_iou, "train_dice": tr_dice,
            "val_loss":   va_loss, "val_iou":   va_iou, "val_dice":   va_dice,
        })
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "val_dice":    va_dice,
                "val_iou":     va_iou,
                "n_params":    n_params,
                "config":      vars(args) | {"image_size": args.image_size},
            }, ckpt_path)
            logging.info("→ new best val Dice %.4f saved to %s", va_dice, ckpt_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logging.info(
        "DONE — best val Dice %.4f | params %d (%.3fM) | ckpt %s",
        best_dice, n_params, n_params / 1e6, ckpt_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
