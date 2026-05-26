"""
scripts/train_baseline.py — Supervised baselines (B1 EdgeUNet, B2 HA-Net)

A single unified trainer for the two no-distillation rows of the ablation
matrix in docs/02_ABLATION_MATRIX.md:

    B1 (--model edgeunet) — Vanilla EdgeUNet (<3M)   establishes the *floor*.
    B2 (--model hanet)    — HA-Net (~15M)            establishes the *ceiling*.

Both models are trained under the identical supervised protocol:

    L = L_seg = BCEWithLogitsLoss + Soft Dice

No teacher, no MedSAM cache, no contrastive head, no KL term — this file
defines the "no-distillation" anchors that every distilled run (A1, A2, A3)
must be measured against. Importing any distillation module here would
silently corrupt the empirical baselines, so the file is intentionally free
of `src.loss_hop1` / `src.loss_hop2` / MedSAM-feature loading code.

Resolution-faithful evaluation
------------------------------
Validation Dice / IoU are computed at the native ultrasound resolution per
docs/03: logits are bilinearly upsampled to (H_native, W_native) and
re-thresholded against the original-size GT mask. The same protocol used
by train_hop1 and train_hop2, so all five matrix rows are directly
comparable.

Usage
-----
    # B1 — vanilla EdgeUNet floor
    python scripts/train_baseline.py \\
        --model edgeunet \\
        --busi_root /kaggle/input/busi/Dataset_BUSI_with_GT \\
        --output_dir /kaggle/working/outputs_b1_edgeunet \\
        --epochs 60 --batch_size 16 --lr 3e-4 --seed 20260530

    # B2 — HA-Net domain ceiling
    python scripts/train_baseline.py \\
        --model hanet \\
        --busi_root /kaggle/input/busi/Dataset_BUSI_with_GT \\
        --output_dir /kaggle/working/outputs_b2_hanet \\
        --epochs 60 --batch_size 16 --lr 3e-4 --seed 20260530
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


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("train_baseline")


# --------------------------------------------------------------------------- #
# BUSI dataset                                                                #
# --------------------------------------------------------------------------- #

_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)
BUSI_CLASSES = ("benign", "malignant")


def discover_busi_pairs(root: Path) -> List[Tuple[Path, Tuple[Path, ...], str]]:
    """Walk BUSI and return (image_path, mask_paths, class) tuples.

    Multi-component lesions have multiple `*_mask*.png` files which are
    OR-merged at __getitem__ time into a single binary mask.
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


class BUSIDataset(Dataset):
    """Training-time BUSI loader. 256x256, hflip-only augmentation.

    Pure supervised — no MedSAM features, no teacher tensors. The dataset
    has a single responsibility: produce (image, mask) at training resolution.
    """

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

        merged_native = or_merge_masks(mask_paths)
        gt_native = torch.from_numpy(merged_native).to(torch.float32).unsqueeze(0)

        return img_t, gt_native, (native_h, native_w)


def val_collate(batch):
    """Custom collate: native masks have variable shape, keep them as a list."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    gts = [b[1] for b in batch]
    sizes = [b[2] for b in batch]
    return imgs, gts, sizes


# --------------------------------------------------------------------------- #
# Segmentation loss (inline — no distillation modules imported)               #
# --------------------------------------------------------------------------- #

class SegmentationLoss(nn.Module):
    """L_seg = BCEWithLogits + Soft Dice. The *entire* loss for B1 and B2.

    Inlined here rather than imported so this file has no path to any
    distillation module; a future contributor can't accidentally wire
    KL / MSE / InfoNCE into the baseline runs by editing the wrong import.
    """

    def __init__(self, dice_smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_smooth = float(dice_smooth)

    @staticmethod
    def _soft_dice(logits: torch.Tensor, targets: torch.Tensor, smooth: float) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        return 1.0 - dice.mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_bce = self.bce(logits, targets)
        l_dice = self._soft_dice(logits, targets, self.dice_smooth)
        l_total = l_bce + l_dice
        return l_total, {
            "total": l_total.detach(),
            "bce":   l_bce.detach(),
            "dice":  l_dice.detach(),
        }


# --------------------------------------------------------------------------- #
# Resolution-faithful Dice / IoU at native size                               #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def upsample_logits_to_native(
    logits_256: torch.Tensor,
    native_hw: Tuple[int, int],
) -> torch.Tensor:
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
    model: nn.Module,
    loss_fn: SegmentationLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    empty_cache_every: int,
) -> Dict[str, float]:
    model.train()

    sums = {k: 0.0 for k in ("total", "bce", "dice")}
    n_batches = 0

    pbar = tqdm(loader, desc=f"train ep{epoch}", leave=False)
    for step, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            # Both HANet and EdgeUNet return (logits, bottleneck); bottleneck
            # is unused by B1 / B2 — these are supervised-only baselines.
            logits, _bottleneck = model(imgs)
            total, parts = loss_fn(logits, masks)

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        for k in sums:
            sums[k] += float(parts[k].item())
        n_batches += 1

        pbar.set_postfix(
            total=f"{parts['total'].item():.3f}",
            bce=f"{parts['bce'].item():.3f}",
            dice=f"{parts['dice'].item():.3f}",
        )

        # Per-batch memory hygiene (docs/03; tunable via --empty_cache_every).
        del imgs, masks, logits, _bottleneck, total, parts
        if empty_cache_every > 0 and (step + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

    return {k: sums[k] / max(n_batches, 1) for k in sums}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    dices: List[float] = []
    ious: List[float] = []

    pbar = tqdm(loader, desc=f"val ep{epoch}  ", leave=False)
    for imgs, gts_native, sizes in pbar:
        imgs = imgs.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            logits, _bottleneck = model(imgs)

        for b_idx, (gt_native, hw) in enumerate(zip(gts_native, sizes)):
            prob = upsample_logits_to_native(logits[b_idx : b_idx + 1], hw).cpu()
            d, i = dice_iou_native(prob, gt_native)
            dices.append(d)
            ious.append(i)

        del imgs, logits, _bottleneck

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
    torch.backends.cudnn.deterministic = False


def deterministic_split(pairs, val_frac: float, seed: int):
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
# Model factory                                                               #
# --------------------------------------------------------------------------- #

def build_model(model_name: str, in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    """Return the model implied by --model, with B1's <3M assertion enforced."""
    if model_name == "edgeunet":
        m = EdgeUNet(in_channels=in_channels, num_classes=num_classes)
        n = sum(p.numel() for p in m.parameters())
        # Hard constraint from docs/03_CONSTRAINTS_AND_RULES.md.
        assert n < 3_000_000, (
            f"EdgeUNet baseline violates the <3M edge-deployment budget: "
            f"got {n:,d} parameters."
        )
        return m
    if model_name == "hanet":
        return HANet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"unknown --model {model_name!r}; expected 'edgeunet' or 'hanet'")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised baseline trainer (B1 EdgeUNet | B2 HA-Net).")
    p.add_argument("--model", choices=["edgeunet", "hanet"], required=True,
                   help="Which baseline to train: edgeunet=B1 (floor), hanet=B2 (ceiling).")
    p.add_argument("--busi_root", type=Path, required=True,
                   help="Path to BUSI root with benign/ and malignant/ subdirs.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output dir for checkpoints, history, manifest.")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=1,
                   help="Native masks are variable-shape; 1 is safest.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dice_smooth", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=20260530)
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP (default on; AMP is mandatory per docs/03 on T4).")
    p.add_argument("--empty_cache_every", type=int, default=1,
                   help="Call torch.cuda.empty_cache() every N batches. 0 disables.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    log.info("model=%s   device=%s   amp=%s   seed=%d",
             args.model, device, use_amp, args.seed)

    # ----- Data -----
    pairs = discover_busi_pairs(args.busi_root)
    if not pairs:
        raise RuntimeError(f"No BUSI pairs discovered under {args.busi_root}")
    train_pairs, val_pairs = deterministic_split(pairs, args.val_frac, args.seed)
    log.info("BUSI: total=%d  train=%d  val=%d",
             len(pairs), len(train_pairs), len(val_pairs))

    train_ds = BUSIDataset(train_pairs, image_size=256, augment=True)
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

    # ----- Model -----
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("%s params=%s (~%.2fM)", args.model, f"{n_params:,d}", n_params / 1e6)

    # ----- Loss -----
    loss_fn = SegmentationLoss(dice_smooth=args.dice_smooth).to(device)

    # ----- Optimizer (model parameters only; no projector — there isn't one) -----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ----- Train -----
    history: List[Dict] = []
    best_dice = -1.0
    best_path = args.output_dir / f"{args.model}_baseline_best.pt"
    last_path = args.output_dir / f"{args.model}_baseline_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model, loss_fn=loss_fn,
            loader=train_loader, optimizer=optimizer, scaler=scaler,
            device=device, epoch=epoch, use_amp=use_amp,
            empty_cache_every=args.empty_cache_every,
        )
        val_metrics = validate(
            model=model, loader=val_loader, device=device,
            use_amp=use_amp, epoch=epoch,
        )

        log.info(
            "ep%03d  train: total=%.4f bce=%.4f dice=%.4f  "
            "val: dice=%.4f iou=%.4f  (%.1fs)",
            epoch,
            train_metrics["total"], train_metrics["bce"], train_metrics["dice"],
            val_metrics["val_dice_mean"], val_metrics["val_iou_mean"],
            time.time() - t0,
        )
        history.append({"epoch": epoch, **train_metrics, **val_metrics})

        ckpt = {
            "epoch": epoch,
            "model_name": args.model,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val_dice": val_metrics["val_dice_mean"],
            "args": vars(args) | {
                "busi_root":  str(args.busi_root),
                "output_dir": str(args.output_dir),
            },
        }
        torch.save(ckpt, last_path)
        if val_metrics["val_dice_mean"] > best_dice:
            best_dice = val_metrics["val_dice_mean"]
            torch.save(ckpt, best_path)
            log.info("  -> new best val_dice=%.4f saved to %s", best_dice, best_path.name)

        # Epoch-boundary cache flush
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----- Persist history -----
    with open(args.output_dir / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    log.info("training history written to %s", args.output_dir / "training_history.json")
    log.info("best val_dice=%.4f", best_dice)


if __name__ == "__main__":
    main()
