"""
scripts/03_train_distilled_unet.py — CAISc 2026

Phase 2b: heterogeneous contrastive distillation. Same EdgeUNet backbone as
Phase 2a, plus a ContrastiveProjector at the bottleneck. Trains against
BUSI ground-truth masks (BCE+Dice) AND the cached MedSAM teacher features
via dense per-token InfoNCE.

The total parameter budget is ~2.01M (within the 3M edge-deployment ceiling).
This script's job is to demonstrate that the +66K-param contrastive head
delivers a meaningful Δ Dice over the Phase 2a baseline (val Dice 0.756).

α (contrastive weight) schedule is the central training-dynamics design:
    epoch 1..10:  α = 0          (learn basic edges via pure seg loss)
    epoch 11..40: α ∈ [0.1, 1.0] (linear ramp)
    epoch 41+:    α = 1.0        (held constant)
Override with --alpha-static for the {0.1, 0.5, 1.0, 2.0} ablation in the appendix.

CRITICAL augmentation note
---------------------------
The teacher embeddings were extracted from un-augmented images during Phase 1.
Geometric augmentations applied to the student input would break the token-
level spatial correspondence required by the contrastive loss (token (i, j)
on the student must align to MedSAM token (i, j) on the teacher).

This script therefore restricts augmentation to operations that can be
**symmetrically applied** to both the image and the cached teacher embedding:
horizontal flip only. We do NOT use rotation here, because rotating a ViT
feature grid is not a meaningful operation in MedSAM's representation space.
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
from src.models import EdgeUNet                                            # noqa: E402
from src.losses import HeterogeneousDistillationLoss, alpha_for_epoch      # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)
BUSI_CLASSES = ("benign", "malignant")
TEACHER_GRID = 64    # MedSAM ViT-B encoder grid at 1024² input


def discover_pairs_with_cache(
    busi_root: Path, cache_root: Path,
) -> list[tuple[Path, tuple[Path, ...], Path, str]]:
    """
    Walk BUSI benign+malignant, pair each image with its mask siblings AND
    the corresponding Phase 1 teacher .pt at <cache_root>/<cls>/<stem>.pt.
    Cases missing a cache file are skipped with a warning.
    """
    items: list[tuple[Path, tuple[Path, ...], Path, str]] = []
    for cls in BUSI_CLASSES:
        cls_dir = busi_root / cls
        cache_dir = cache_root / cls
        if not cls_dir.is_dir():
            logging.warning("missing BUSI class dir: %s", cls_dir)
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            if _MASK_RE.search(img_path.stem):
                continue
            mask_paths = tuple(sorted(cls_dir.glob(f"{img_path.stem}_mask*.png")))
            if not mask_paths:
                continue
            pt_path = cache_dir / f"{img_path.stem}.pt"
            if not pt_path.is_file():
                logging.warning("no teacher cache for %s — skipping", img_path.name)
                continue
            items.append((img_path, mask_paths, pt_path, cls))
    return items


class DistillationDataset(Dataset):
    """
    Returns per-case:
        image       : (3, S, S) float32 in [0, 1]
        mask        : (1, S, S) float32 binary
        mask_64     : (1, 64, 64) float32 binary  — for contrastive token selection
        teacher_emb : (256, 64, 64) float32       — MedSAM ViT encoder output, fp16→fp32

    Only horizontal flip is used for augmentation (see CRITICAL note in the
    module docstring). Both the student input AND the teacher embedding are
    flipped together to preserve token-level spatial correspondence.
    """

    def __init__(
        self,
        items: list[tuple[Path, tuple[Path, ...], Path, str]],
        image_size: int = 256,
        augment: bool = False,
    ):
        self.items = items
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_paths, pt_path, _cls = self.items[idx]

        # Image → RGB, target resolution
        img = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        img_t = TF.pil_to_tensor(img).float().div_(255.0)                  # (3, S, S)

        # OR-merge multi-mask cases at native res, nearest-resize to target
        merged: np.ndarray | None = None
        for mp in mask_paths:
            m = np.asarray(Image.open(mp).convert("L"))
            b = (m > 127).astype(np.uint8)
            merged = b if merged is None else np.maximum(merged, b)
        mask_pil = Image.fromarray((merged * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST,
        )
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()    # (1, S, S)

        # Teacher embedding (fp16 on disk → fp32 in RAM for the contrastive head)
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        teacher_emb = payload["image_embedding"].float()                       # (256, 64, 64)

        # Horizontal flip applied symmetrically to image, mask, AND teacher
        if self.augment and torch.rand(1).item() < 0.5:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)
            teacher_emb = teacher_emb.flip(dims=[-1])

        # Downsample GT mask to teacher grid (64×64) AFTER the flip
        mask_64 = F.interpolate(
            mask_t.unsqueeze(0), size=(TEACHER_GRID, TEACHER_GRID), mode="nearest",
        ).squeeze(0)                                                          # (1, 64, 64)

        return img_t, mask_t, mask_64, teacher_emb


# ---------------------------------------------------------------------------
# Metrics — identical to Phase 2a so the comparison is apples-to-apples
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

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, alpha):
    model.train()
    seg_losses, contrast_losses, total_losses, ious, dices = [], [], [], [], []
    pbar = tqdm(loader, desc=f"train α={alpha:.2f}", leave=False, dynamic_ncols=True)
    for img, mask, mask_64, teacher_emb in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        mask_64 = mask_64.to(device, non_blocking=True)
        teacher_emb = teacher_emb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits, projection = model.forward_distill(img)
            # Loss in fp32 (contrastive needs precise cosine sims)
            out = criterion(logits.float(), mask, projection, teacher_emb, mask_64, alpha=alpha)
            scaler.scale(out.total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, projection = model.forward_distill(img)
            out = criterion(logits, mask, projection, teacher_emb, mask_64, alpha=alpha)
            out.total.backward()
            optimizer.step()

        with torch.no_grad():
            iou, dice = batch_iou_dice(logits.float(), mask)
        seg_losses.append(out.seg.item())
        contrast_losses.append(out.contrastive.item())
        total_losses.append(out.total.item())
        ious.append(iou); dices.append(dice)
        pbar.set_postfix(
            seg=f"{out.seg.item():.3f}",
            ctr=f"{out.contrastive.item():.3f}",
            dice=f"{dice:.3f}",
        )

    return {
        "loss":       float(np.mean(total_losses)),
        "seg":        float(np.mean(seg_losses)),
        "contrast":   float(np.mean(contrast_losses)),
        "iou":        float(np.mean(ious)),
        "dice":       float(np.mean(dices)),
    }


@torch.inference_mode()
def validate(model, loader, criterion, device, alpha):
    """
    Validation passes only the segmentation half of the loss — the contrastive
    term is *training-time regularization* and including it in the val score
    would conflate generalization with teacher-alignment. Dice/IoU remain the
    headline numbers for the paper.
    """
    model.eval()
    losses, ious, dices = [], [], []
    pbar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for img, mask, _mask_64, _teacher in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)                                  # plain seg forward, skip projector
        # Reuse the criterion's seg components by hand to keep things uniform.
        seg_loss = criterion.bce_w * criterion.bce(logits, mask) + criterion.dice_w * criterion.dice(logits, mask)
        iou, dice = batch_iou_dice(logits, mask)
        losses.append(seg_loss.item()); ious.append(iou); dices.append(dice)
        pbar.set_postfix(loss=f"{seg_loss.item():.4f}", dice=f"{dice:.4f}")
    return {
        "loss": float(np.mean(losses)),
        "iou":  float(np.mean(ious)),
        "dice": float(np.mean(dices)),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2b — contrastive distillation training (CAISc 2026).")
    parser.add_argument("--busi-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Phase 1 .pt cache root (contains benign/ malignant/ subdirs).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable AMP. AMP is on by default but contrastive sims "
                             "use an explicit fp32 cast for stability.")

    # Alpha schedule controls
    parser.add_argument("--alpha-warmup-end", type=int, default=10,
                        help="Epochs of pure-seg warmup (α=0).")
    parser.add_argument("--alpha-ramp-end", type=int, default=40,
                        help="Last epoch of linear α ramp.")
    parser.add_argument("--alpha-start", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=1.0,
                        help="Global multiplier on the scheduled α (for ablation).")
    parser.add_argument("--alpha-static", type=float, default=None,
                        help="If set, bypass the schedule and hold α constant "
                             "(use for the {0.1, 0.5, 1.0, 2.0} appendix ablation).")

    # Contrastive head hyperparams
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--n-anchors", type=int, default=64)
    parser.add_argument("--n-negatives", type=int, default=256)

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
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable — CPU training will be very slow.")
    use_amp = (not args.no_amp) and device.type == "cuda"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "distilled_unet_best.pth"
    history_path = args.output_dir / "distilled_unet_history.json"

    # --- Data --------------------------------------------------------------
    logging.info("discovering BUSI+cache under %s | %s", args.busi_root, args.cache_root)
    items = discover_pairs_with_cache(args.busi_root, args.cache_root)
    if not items:
        logging.error("no image/cache pairs discovered — verify paths")
        return 2
    by_cls = {c: sum(1 for _, _, _, k in items if k == c) for c in BUSI_CLASSES}
    logging.info("paired %d items (%s)", len(items), by_cls)

    random.Random(args.seed).shuffle(items)
    n_val = int(round(len(items) * args.val_split))
    val_items = items[:n_val]
    train_items = items[n_val:]
    logging.info("split — train: %d | val: %d", len(train_items), len(val_items))

    train_ds = DistillationDataset(train_items, image_size=args.image_size, augment=True)
    val_ds = DistillationDataset(val_items, image_size=args.image_size, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    # --- Model -------------------------------------------------------------
    model = EdgeUNet(in_channels=3, out_channels=1, with_projector=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_proj = sum(p.numel() for p in model.projector.parameters())
    logging.info("EdgeUNet+Projector params: %.3fM (%d, projector contributes %d)",
                 n_params / 1e6, n_params, n_proj)
    if n_params >= 3_000_000:
        logging.error("model exceeds 3M-param budget (%d) — abort", n_params)
        return 2

    # --- Optim / loss ------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    criterion = HeterogeneousDistillationLoss(
        temperature=args.temperature,
        n_anchors=args.n_anchors,
        n_negatives=args.n_negatives,
    )
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # --- Train -------------------------------------------------------------
    best_dice = 0.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        alpha = alpha_for_epoch(
            epoch,
            warmup_end=args.alpha_warmup_end,
            ramp_end=args.alpha_ramp_end,
            alpha_start=args.alpha_start,
            alpha_max=args.alpha_max,
            scale=args.alpha_scale,
            static=args.alpha_static,
        )

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, alpha)
        val_m = validate(model, val_loader, criterion, device, alpha)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        logging.info(
            "epoch %3d/%d | α %.3f | train loss %.4f seg %.4f ctr %.4f dice %.4f iou %.4f | "
            "val loss %.4f dice %.4f iou %.4f | lr %.2e",
            epoch, args.epochs, alpha,
            train_m["loss"], train_m["seg"], train_m["contrast"], train_m["dice"], train_m["iou"],
            val_m["loss"], val_m["dice"], val_m["iou"], lr_now,
        )
        history.append({
            "epoch": epoch, "lr": lr_now, "alpha": alpha,
            "train_loss":     train_m["loss"],
            "train_seg":      train_m["seg"],
            "train_contrast": train_m["contrast"],
            "train_iou":      train_m["iou"],
            "train_dice":     train_m["dice"],
            "val_loss":       val_m["loss"],
            "val_iou":        val_m["iou"],
            "val_dice":       val_m["dice"],
        })
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "alpha":       alpha,
                "val_dice":    val_m["dice"],
                "val_iou":     val_m["iou"],
                "n_params":    n_params,
                "config":      vars(args),
            }, ckpt_path)
            logging.info("→ new best val Dice %.4f (Δ vs baseline 0.756 = %+.4f) saved to %s",
                         val_m["dice"], val_m["dice"] - 0.756, ckpt_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logging.info(
        "DONE — best val Dice %.4f | params %d (%.3fM) | ckpt %s | Δ vs baseline %+.4f",
        best_dice, n_params, n_params / 1e6, ckpt_path, best_dice - 0.756,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
