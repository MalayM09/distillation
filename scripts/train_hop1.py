"""
scripts/train_hop1.py — Hop-1 distillation trainer (MedSAM -> HA-Net)

Trains the HA-Net assistant against MedSAM's cached image-encoder features
under the Hop-1 dense InfoNCE loss (Rule 1 of docs/02_ABLATION_MATRIX.md).
This is the engine for the *first* hop of the A3 TAKD cascade. After
convergence, the resulting HA-Net checkpoint is consumed by scripts/train_hop2.py
as the frozen teacher in the second hop (HA-Net -> EdgeUNet).

Wiring
------
- Teacher  : MedSAM, *not loaded in the training loop* — its features are
             read from disk per-case as cached (256, 64, 64) FP16 tensors.
- Student  : HA-Net (15M, fully trainable).
- Loss     : src.loss_hop1.Hop1ContrastiveLoss(student_channels=512),
             whose internal ContrastiveProjector (512 -> 256 + bilinear
             8x8 -> 64x64) is also trainable.
- Optimizer: AdamW over HA-Net params + projector params (single optimizer
             group; same LR / WD applied to both).

MedSAM cache contract
---------------------
Per-case payload at `<cache_root>/<class>/<stem>.pt` is a dict with key
    "image_embedding" : (256, 64, 64) FP16 — MedSAM ViT-B encoder output,
                        the dense feature map against which InfoNCE positives
                        are scored.
Other payload keys (`teacher_logits`, `teacher_mask`, `meta`) are *not used*
by this trainer — Hop-1 is feature-only contrastive distillation.

Synchronized hflip
------------------
hflip augmentation must be applied identically to the image, the 256x256
training mask, AND the spatial dim of the MedSAM feature tensor (its W axis,
position -1 of the (C, H, W) tensor). Without synchronization, the InfoNCE
positive at student spatial location (i, j) would target a *flipped* MedSAM
token at (i, W-1-j), and the contrastive objective would collapse.

Resolution-faithful evaluation
------------------------------
Validation Dice / IoU are computed at the *native* ultrasound resolution per
docs/03: student logits are bilinearly upsampled to (H_native, W_native) and
re-thresholded against the original-size GT mask. MedSAM features are NOT
loaded at val time — Hop-1 distillation is a train-only objective.

Usage
-----
    python scripts/train_hop1.py \\
        --busi_root /kaggle/input/busi/Dataset_BUSI_with_GT \\
        --medsam_cache_root /kaggle/input/caisc-v1/medsam_cache \\
        --output_dir /kaggle/working/outputs_hop1 \\
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
from src.loss_hop1 import Hop1ContrastiveLoss


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("train_hop1")


# --------------------------------------------------------------------------- #
# BUSI + MedSAM-cache pairing                                                 #
# --------------------------------------------------------------------------- #

_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)
BUSI_CLASSES = ("benign", "malignant")


class Case:
    __slots__ = ("img_path", "mask_paths", "cls", "cache_path")

    def __init__(self, img_path, mask_paths, cls, cache_path) -> None:
        self.img_path = img_path
        self.mask_paths = mask_paths
        self.cls = cls
        self.cache_path = cache_path


def discover_cases(busi_root: Path, cache_root: Path) -> List[Case]:
    """Walk BUSI and pair each image with its MedSAM cache file.

    Cases without a corresponding `<cache_root>/<cls>/<stem>.pt` are dropped
    with a warning — Hop-1 cannot proceed without the teacher embedding.
    """
    cases: List[Case] = []
    n_missing = 0
    for cls in BUSI_CLASSES:
        cls_dir = busi_root / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            if _MASK_RE.search(img_path.stem):
                continue
            mask_paths = tuple(sorted(cls_dir.glob(f"{img_path.stem}_mask*.png")))
            if not mask_paths:
                continue
            cache_path = cache_root / cls / f"{img_path.stem}.pt"
            if not cache_path.is_file():
                n_missing += 1
                continue
            cases.append(Case(img_path, mask_paths, cls, cache_path))
    if n_missing:
        log.warning("dropped %d BUSI cases with missing MedSAM cache under %s",
                    n_missing, cache_root)
    return cases


def or_merge_masks(mask_paths: Tuple[Path, ...]) -> np.ndarray:
    merged = None
    for mp in mask_paths:
        m = np.asarray(Image.open(mp).convert("L"))
        b = (m > 127).astype(np.uint8)
        merged = b if merged is None else np.maximum(merged, b)
    return merged  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# DistillationDataset (train) + ValDataset (resolution-faithful eval)         #
# --------------------------------------------------------------------------- #

class DistillationDataset(Dataset):
    """Training-time dataset: returns (image_256, mask_256, medsam_features).

    Augmentation is hflip-only per docs/03; rotation would invalidate the
    spatial correspondence between the student bottleneck and the cached
    MedSAM tokens, breaking the InfoNCE positive-pair construction.

    The hflip is applied synchronously to:
        - image_256                    : (3, 256, 256), flipped along W=dim-1
        - mask_256                     : (1, 256, 256), flipped along W=dim-1
        - medsam_features (C, 64, 64)  : flipped along W=dim-1
    so the (i, j) anchor / (i, j) positive correspondence is preserved.
    """

    MEDSAM_EMBEDDING_KEY = "image_embedding"

    def __init__(self, cases: List[Case], image_size: int = 256, augment: bool = True) -> None:
        self.cases = cases
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.cases)

    def _load_medsam_features(self, cache_path: Path) -> torch.Tensor:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        feat = payload[self.MEDSAM_EMBEDDING_KEY]
        # Cache is stored FP16; keep as FP16 — the loss casts to FP32 internally.
        if feat.dtype != torch.float16:
            feat = feat.to(torch.float16)
        # Shape contract: (256, 64, 64). Fail loudly if the cache is corrupt.
        if feat.dim() != 3 or feat.shape[0] != 256 or feat.shape[1] != 64 or feat.shape[2] != 64:
            raise ValueError(
                f"MedSAM cache at {cache_path} has unexpected shape "
                f"{tuple(feat.shape)}; expected (256, 64, 64)."
            )
        return feat

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case = self.cases[idx]

        img = Image.open(case.img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        merged = or_merge_masks(case.mask_paths)
        mask_pil = Image.fromarray((merged * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST)

        img_t = TF.pil_to_tensor(img).float().div_(255.0)
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()

        medsam_feat = self._load_medsam_features(case.cache_path)  # (256, 64, 64)

        # Synchronized hflip: image/mask along W (last spatial axis), MedSAM
        # feature along its W axis (dim=-1 of a (C, H, W) tensor). This keeps
        # the student-to-MedSAM spatial correspondence intact after augmentation.
        if self.augment and torch.rand(1).item() < 0.5:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)
            medsam_feat = torch.flip(medsam_feat, dims=[-1])

        return img_t, mask_t, medsam_feat


class ValDataset(Dataset):
    """Validation dataset: returns (image_256, gt_native, native_hw).

    MedSAM features are not loaded — Hop-1 InfoNCE is not evaluated at val
    time; only Dice / IoU at native resolution are reported.
    """

    def __init__(self, cases: List[Case], image_size: int = 256) -> None:
        self.cases = cases
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        case = self.cases[idx]

        img_native = Image.open(case.img_path).convert("RGB")
        native_h, native_w = img_native.height, img_native.width
        img_resized = img_native.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_t = TF.pil_to_tensor(img_resized).float().div_(255.0)

        merged_native = or_merge_masks(case.mask_paths)  # (H, W) uint8 in {0,1}
        gt_native = torch.from_numpy(merged_native).to(torch.float32).unsqueeze(0)

        return img_t, gt_native, (native_h, native_w)


def val_collate(batch):
    """Custom collate: native masks are variable-shape, keep them as a list."""
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
    student: nn.Module,
    loss_fn: Hop1ContrastiveLoss,
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

    sums = {k: 0.0 for k in ("total", "seg", "bce", "dice", "nce")}
    n_batches = 0

    pbar = tqdm(loader, desc=f"train ep{epoch}", leave=False)
    for step, (imgs, masks, medsam_feats) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        medsam_feats = medsam_feats.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            s_logits, s_bottleneck = student(imgs)
            total, parts = loss_fn(
                student_logits=s_logits,
                student_bottleneck=s_bottleneck,
                medsam_features=medsam_feats,
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
            nce=f"{parts['nce'].item():.3f}",
        )

        # Per-batch memory hygiene
        del imgs, masks, medsam_feats, s_logits, s_bottleneck, total, parts
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

        # Resolution-faithful scoring at native ultrasound size (docs/03).
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
    cases: List[Case], val_frac: float, seed: int,
) -> Tuple[List[Case], List[Case]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)
    n_val = int(round(len(cases) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, c in enumerate(cases):
        (val if i in val_idx else train).append(c)
    return train, val


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hop-1 distillation trainer (MedSAM -> HA-Net).")
    p.add_argument("--busi_root", type=Path, required=True,
                   help="Path to BUSI root with benign/ and malignant/ subdirs.")
    p.add_argument("--medsam_cache_root", type=Path, required=True,
                   help="Root of MedSAM .pt cache: <root>/<class>/<stem>.pt.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output dir for checkpoints, history, manifest.")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=1,
                   help="Native-resolution masks are variable-shape; 1 is safest.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda_nce", type=float, default=1.0)
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
    log.info("device=%s   amp=%s   seed=%d", device, use_amp, args.seed)

    # ----- Data -----
    cases = discover_cases(args.busi_root, args.medsam_cache_root)
    if not cases:
        raise RuntimeError(
            f"No BUSI/MedSAM-cache pairs discovered under "
            f"{args.busi_root} + {args.medsam_cache_root}"
        )
    train_cases, val_cases = deterministic_split(cases, args.val_frac, args.seed)
    log.info("cases: total=%d  train=%d  val=%d",
             len(cases), len(train_cases), len(val_cases))

    train_ds = DistillationDataset(train_cases, image_size=256, augment=True)
    val_ds = ValDataset(val_cases, image_size=256)

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

    # ----- Student (HA-Net, fully trainable) -----
    student = HANet(in_channels=3, num_classes=1).to(device)
    n_student = sum(p.numel() for p in student.parameters())
    log.info("student HA-Net params=%s (~%.2fM)", f"{n_student:,d}", n_student / 1e6)

    # ----- Loss + projector -----
    loss_fn = Hop1ContrastiveLoss(
        student_channels=512,
        teacher_channels=256,
        target_grid=64,
        temperature=args.temperature,
        lambda_nce=args.lambda_nce,
        dice_smooth=args.dice_smooth,
    ).to(device)
    n_proj = sum(p.numel() for p in loss_fn.projector_parameters())
    log.info("contrastive projector params=%s   tau=%.3f  lambda_nce=%.2f",
             f"{n_proj:,d}", args.temperature, args.lambda_nce)

    # ----- Optimizer (student + projector) -----
    trainable_params = list(student.parameters()) + list(loss_fn.projector_parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ----- Train -----
    history: List[Dict] = []
    best_dice = -1.0
    best_path = args.output_dir / "hanet_best.pt"
    last_path = args.output_dir / "hanet_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            student=student, loss_fn=loss_fn,
            loader=train_loader, optimizer=optimizer, scaler=scaler,
            device=device, epoch=epoch, use_amp=use_amp,
            empty_cache_every=args.empty_cache_every,
        )
        val_metrics = validate(
            student=student, loader=val_loader, device=device,
            use_amp=use_amp, epoch=epoch,
        )

        log.info(
            "ep%03d  train: total=%.4f seg=%.4f nce=%.4f  "
            "val: dice=%.4f iou=%.4f  (%.1fs)",
            epoch,
            train_metrics["total"], train_metrics["seg"], train_metrics["nce"],
            val_metrics["val_dice_mean"], val_metrics["val_iou_mean"],
            time.time() - t0,
        )
        history.append({"epoch": epoch, **train_metrics, **val_metrics})

        # ----- Checkpoint (HA-Net + projector + optimizer for resumability) -----
        ckpt = {
            "epoch": epoch,
            "model": student.state_dict(),
            "projector": loss_fn.projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val_dice": val_metrics["val_dice_mean"],
            "args": vars(args) | {
                "busi_root":         str(args.busi_root),
                "medsam_cache_root": str(args.medsam_cache_root),
                "output_dir":        str(args.output_dir),
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

    # ----- Persist training history -----
    with open(args.output_dir / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    log.info("training history written to %s", args.output_dir / "training_history.json")
    log.info("best val_dice=%.4f", best_dice)


if __name__ == "__main__":
    main()
