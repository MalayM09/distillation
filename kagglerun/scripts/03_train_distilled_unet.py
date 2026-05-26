"""
scripts/03_train_distilled_unet.py — Phase 2b distillation trainer.
Supports both InfoNCE (main method) and MSE-hint (ablation baseline)
via --loss-variant.
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import EdgeUNet
from src.losses import (
    HeterogeneousDistillationLoss,
    HintMSEDistillationLoss,
    alpha_for_epoch,
)


_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)
BUSI_CLASSES = ("benign", "malignant")
TEACHER_GRID = 64


def discover_pairs_with_cache(busi_root, cache_root):
    items = []
    for cls in BUSI_CLASSES:
        cls_dir = busi_root / cls
        cache_dir = cache_root / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            if _MASK_RE.search(img_path.stem):
                continue
            mask_paths = tuple(sorted(cls_dir.glob(f"{img_path.stem}_mask*.png")))
            if not mask_paths:
                continue
            pt_path = cache_dir / f"{img_path.stem}.pt"
            if not pt_path.is_file():
                continue
            items.append((img_path, mask_paths, pt_path, cls))
    return items


class DistillationDataset(Dataset):
    def __init__(self, items, image_size=256, augment=False):
        self.items = items
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_paths, pt_path, _ = self.items[idx]
        img = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        img_t = TF.pil_to_tensor(img).float().div_(255.0)
        merged = None
        for mp in mask_paths:
            m = np.asarray(Image.open(mp).convert("L"))
            b = (m > 127).astype(np.uint8)
            merged = b if merged is None else np.maximum(merged, b)
        mask_pil = Image.fromarray((merged * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST)
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        teacher_emb = payload["image_embedding"].float()
        if self.augment and torch.rand(1).item() < 0.5:
            img_t = TF.hflip(img_t); mask_t = TF.hflip(mask_t)
            teacher_emb = teacher_emb.flip(dims=[-1])
        mask_64 = F.interpolate(mask_t.unsqueeze(0),
                                size=(TEACHER_GRID, TEACHER_GRID),
                                mode="nearest").squeeze(0)
        return img_t, mask_t, mask_64, teacher_emb


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


def train_epoch(model, loader, criterion, optimizer, device, scaler, alpha):
    model.train()
    seg_l, ctr_l, tot_l, ious, dices = [], [], [], [], []
    for img, mask, mask_64, teacher_emb in tqdm(loader, desc=f"train α={alpha:.2f}",
                                                leave=False, dynamic_ncols=True):
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        mask_64 = mask_64.to(device, non_blocking=True)
        teacher_emb = teacher_emb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits, projection = model.forward_distill(img)
            out = criterion(logits.float(), mask, projection, teacher_emb, mask_64, alpha=alpha)
            scaler.scale(out.total).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits, projection = model.forward_distill(img)
            out = criterion(logits, mask, projection, teacher_emb, mask_64, alpha=alpha)
            out.total.backward(); optimizer.step()
        with torch.no_grad():
            iou, dice = batch_iou_dice(logits.float(), mask)
        seg_l.append(out.seg.item()); ctr_l.append(out.contrastive.item())
        tot_l.append(out.total.item()); ious.append(iou); dices.append(dice)
    return {"loss": float(np.mean(tot_l)), "seg": float(np.mean(seg_l)),
            "contrast": float(np.mean(ctr_l)),
            "iou": float(np.mean(ious)), "dice": float(np.mean(dices))}


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    losses, ious, dices = [], [], []
    for img, mask, _m64, _t in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        logits = model(img)
        seg = criterion.bce_w * criterion.bce(logits, mask) + criterion.dice_w * criterion.dice(logits, mask)
        iou, dice = batch_iou_dice(logits, mask)
        losses.append(seg.item()); ious.append(iou); dices.append(dice)
    return {"loss": float(np.mean(losses)), "iou": float(np.mean(ious)), "dice": float(np.mean(dices))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--busi-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--alpha-warmup-end", type=int, default=10)
    parser.add_argument("--alpha-ramp-end", type=int, default=40)
    parser.add_argument("--alpha-start", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--alpha-static", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--n-anchors", type=int, default=64)
    parser.add_argument("--n-negatives", type=int, default=256)
    parser.add_argument("--loss-variant", choices=("infonce", "mse"), default="infonce")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "distilled_unet_best.pth"
    history_path = args.output_dir / "distilled_unet_history.json"

    items = discover_pairs_with_cache(args.busi_root, args.cache_root)
    logging.info("paired %d items", len(items))
    random.Random(args.seed).shuffle(items)
    n_val = int(round(len(items) * args.val_split))
    train_items, val_items = items[n_val:], items[:n_val]
    logging.info("split — train %d val %d | loss-variant=%s", len(train_items), len(val_items), args.loss_variant)

    train_ds = DistillationDataset(train_items, image_size=args.image_size, augment=True)
    val_ds = DistillationDataset(val_items, image_size=args.image_size, augment=False)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)

    model = EdgeUNet(in_channels=3, out_channels=1, with_projector=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("EdgeUNet+Projector params: %.3fM", n_params / 1e6)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    if args.loss_variant == "infonce":
        criterion = HeterogeneousDistillationLoss(
            temperature=args.temperature, n_anchors=args.n_anchors, n_negatives=args.n_negatives)
    else:
        criterion = HintMSEDistillationLoss()

    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    best_dice = 0.0; history = []
    for epoch in range(1, args.epochs + 1):
        alpha = alpha_for_epoch(epoch, warmup_end=args.alpha_warmup_end,
                                ramp_end=args.alpha_ramp_end, alpha_start=args.alpha_start,
                                alpha_max=args.alpha_max, scale=args.alpha_scale,
                                static=args.alpha_static)
        tr = train_epoch(model, train_loader, criterion, optimizer, device, scaler, alpha)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        logging.info("epoch %3d/%d | α %.3f | tr seg %.4f ctr %.4f dice %.4f | val dice %.4f iou %.4f | lr %.2e",
                     epoch, args.epochs, alpha, tr["seg"], tr["contrast"], tr["dice"],
                     va["dice"], va["iou"], lr_now)
        history.append({"epoch": epoch, "lr": lr_now, "alpha": alpha,
                        "train_loss": tr["loss"], "train_seg": tr["seg"],
                        "train_contrast": tr["contrast"], "train_iou": tr["iou"], "train_dice": tr["dice"],
                        "val_loss": va["loss"], "val_iou": va["iou"], "val_dice": va["dice"]})
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)
        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "alpha": alpha,
                        "val_dice": va["dice"], "val_iou": va["iou"], "n_params": n_params,
                        "config": vars(args)}, ckpt_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logging.info("DONE — best val Dice %.4f", best_dice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
