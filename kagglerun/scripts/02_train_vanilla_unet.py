"""
scripts/02_train_vanilla_unet.py — Phase 2a vanilla baseline trainer.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import EdgeUNet
from src.losses import BCEDiceLoss


_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)
BUSI_CLASSES = ("benign", "malignant")


def discover_busi_pairs(root):
    pairs = []
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


class BUSIVanillaDataset(Dataset):
    def __init__(self, pairs, image_size=256, augment=False, rotation_deg=15.0):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.rotation_deg = rotation_deg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_paths, _ = self.pairs[idx]
        img = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        merged = None
        for mp in mask_paths:
            m = np.asarray(Image.open(mp).convert("L"))
            b = (m > 127).astype(np.uint8)
            merged = b if merged is None else np.maximum(merged, b)
        mask_pil = Image.fromarray((merged * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST)
        img_t = TF.pil_to_tensor(img).float().div_(255.0)
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img_t = TF.hflip(img_t); mask_t = TF.hflip(mask_t)
            if self.rotation_deg > 0:
                angle = (torch.rand(1).item() * 2 - 1) * self.rotation_deg
                img_t = TF.rotate(img_t, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_t = TF.rotate(mask_t, angle, interpolation=TF.InterpolationMode.NEAREST)
                mask_t = (mask_t > 0.5).float()
        return img_t, mask_t


@torch.no_grad()
def batch_iou_dice(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    dims = (2, 3)
    inter = (preds * targets).sum(dim=dims)
    p_sum, t_sum = preds.sum(dim=dims), targets.sum(dim=dims)
    union = p_sum + t_sum - inter
    iou = (inter + eps) / (union + eps)
    dice = (2.0 * inter + eps) / (p_sum + t_sum + eps)
    return iou.mean().item(), dice.mean().item()


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses, ious, dices = [], [], []
    for img, mask in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(img); loss = criterion(logits, mask)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(img); loss = criterion(logits, mask)
            loss.backward(); optimizer.step()
        with torch.no_grad():
            iou, dice = batch_iou_dice(logits.float(), mask)
        losses.append(loss.item()); ious.append(iou); dices.append(dice)
    return float(np.mean(losses)), float(np.mean(ious)), float(np.mean(dices))


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    losses, ious, dices = [], [], []
    for img, mask in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        logits = model(img); loss = criterion(logits, mask)
        iou, dice = batch_iou_dice(logits, mask)
        losses.append(loss.item()); ious.append(iou); dices.append(dice)
    return float(np.mean(losses)), float(np.mean(ious)), float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--busi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "vanilla_unet_best.pth"
    history_path = args.output_dir / "vanilla_unet_history.json"

    pairs = discover_busi_pairs(args.busi_root)
    logging.info("found %d image-mask pairs", len(pairs))
    random.Random(args.seed).shuffle(pairs)
    n_val = int(round(len(pairs) * args.val_split))
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]
    logging.info("split — train %d val %d | rotation_deg=%.1f", len(train_pairs), len(val_pairs), args.rotation_deg)

    train_ds = BUSIVanillaDataset(train_pairs, image_size=args.image_size, augment=True, rotation_deg=args.rotation_deg)
    val_ds = BUSIVanillaDataset(val_pairs, image_size=args.image_size, augment=False)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)

    model = EdgeUNet(in_channels=3, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("EdgeUNet params: %.3fM", n_params / 1e6)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    best_dice = 0.0; history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou, tr_dice = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_iou, va_dice = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        logging.info("epoch %3d/%d | tr dice %.4f | val dice %.4f iou %.4f | lr %.2e",
                     epoch, args.epochs, tr_dice, va_dice, va_iou, lr_now)
        history.append({
            "epoch": epoch, "lr": lr_now,
            "train_loss": tr_loss, "train_iou": tr_iou, "train_dice": tr_dice,
            "val_loss": va_loss, "val_iou": va_iou, "val_dice": va_dice,
        })
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_dice": va_dice, "val_iou": va_iou, "n_params": n_params,
                        "config": vars(args)}, ckpt_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logging.info("DONE — best val Dice %.4f at %s", best_dice, ckpt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
