"""
scripts/train_critical_path.py — Cross-dataset critical-path trainer (§5.4)

Runs the *critical path* of the ablation matrix (B1, B2, A1, A2) on any
dataset given a Kvasir-SEG-style flat layout. Skips A3-h1 and A3 because
the cross-dataset story is "does capacity-matched CNN→CNN distillation
beat direct ViT→CNN distillation on a second dataset?" — which is fully
answered by (B1, A1, A2). The 3-seed multi-seed bootstrap is computed
internally; no separate aggregator step is needed.

Dataset layout
--------------
    <data_root>/
        images/   *.jpg|*.png      (RGB images, any aspect)
        masks/    *.jpg|*.png      (binary masks, same stem as images)

Sequencing
----------
    For each seed in --seeds:
        B1 → vanilla EdgeUNet                    (supervised only)
        B2 → vanilla HA-Net                      (supervised only)
              ↳ produces seed-matched teacher for A2
        A1 → direct MedSAM→EdgeUNet              (Hop-1 InfoNCE)
        A2 → HA-Net→EdgeUNet                     (Hop-2 MSE+KL)

Each (ablation, seed) pair writes to:
    <output_root>/<ablation>_seed<seed>/training_history.json
    <output_root>/<ablation>_seed<seed>/<checkpoint>_best.pt

Resumable: any run whose training_history.json already contains
≥ --epochs records is skipped.

Usage on Kaggle
---------------
    !python scripts/train_critical_path.py \\
        --data_root /kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG \\
        --medsam_cache_root /kaggle/working/medsam_cache_kvasir \\
        --output_root /kaggle/working/runs_kvasir_critical \\
        --epochs 40 --batch_size 16 --num_workers 2 \\
        --seeds 20260530 42 1337
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
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
from src.loss_hop1 import Hop1ContrastiveLoss
from src.loss_hop2 import Hop2DistillationLoss


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("train_critical_path")


# --------------------------------------------------------------------------- #
# Kvasir-SEG dataset (flat layout)                                            #
# --------------------------------------------------------------------------- #

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class Case:
    __slots__ = ("img_path", "mask_path", "cache_path")

    def __init__(self, img_path, mask_path, cache_path) -> None:
        self.img_path = img_path
        self.mask_path = mask_path
        self.cache_path = cache_path


def discover_dataset(data_root: Path, cache_root: Path | None) -> List[Case]:
    """Walk <data_root>/images and pair each image with its <data_root>/masks
    sibling and (optionally) its MedSAM cache file at <cache_root>/<stem>.pt.

    Returns a list of Case objects; cache_path is None if MedSAM cache wasn't
    requested or is missing for that case.
    """
    img_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(
            f"Expected {data_root}/images/ and {data_root}/masks/, found neither.")

    cases: List[Case] = []
    n_no_mask = 0
    n_no_cache = 0
    for ext in IMG_EXTS:
        for img_path in sorted(img_dir.glob(f"*{ext}")):
            mask_match = None
            for mext in IMG_EXTS:
                cand = mask_dir / f"{img_path.stem}{mext}"
                if cand.is_file():
                    mask_match = cand
                    break
            if mask_match is None:
                n_no_mask += 1
                continue

            cache_path = None
            if cache_root is not None:
                cand = cache_root / f"{img_path.stem}.pt"
                if cand.is_file():
                    cache_path = cand
                else:
                    n_no_cache += 1
            cases.append(Case(img_path, mask_match, cache_path))
    if n_no_mask:
        log.warning("dropped %d images with no mask sibling", n_no_mask)
    if n_no_cache:
        log.warning("%d cases have no MedSAM cache (A1 will skip these)", n_no_cache)
    return cases


def deterministic_split(cases: List[Case], val_frac: float, seed: int
                        ) -> Tuple[List[Case], List[Case]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)
    n_val = int(round(len(cases) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train = [c for i, c in enumerate(cases) if i not in val_idx]
    val = [c for i, c in enumerate(cases) if i in val_idx]
    return train, val


# --------------------------------------------------------------------------- #
# Dataset classes (training: 256x256; val: native-resolution mask)            #
# --------------------------------------------------------------------------- #

class SupTrainDataset(Dataset):
    """Supervised training set (B1, B2): (image_256, mask_256). hflip-only aug."""

    def __init__(self, cases: List[Case], image_size: int = 256, augment: bool = True) -> None:
        self.cases = cases
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.cases[idx]
        img = Image.open(c.img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        mask_native = np.asarray(Image.open(c.mask_path).convert("L"))
        mask_bin = (mask_native > 127).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_bin).resize(
            (self.image_size, self.image_size), Image.NEAREST)
        img_t = TF.pil_to_tensor(img).float().div_(255.0)
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()
        if self.augment and torch.rand(1).item() < 0.5:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)
        return img_t, mask_t


class DistillTrainDataset(Dataset):
    """A1 training set: (image_256, mask_256, medsam_features). Synchronised hflip."""

    def __init__(self, cases: List[Case], image_size: int = 256, augment: bool = True) -> None:
        self.cases = [c for c in cases if c.cache_path is not None]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.cases)

    def _load_medsam(self, p: Path) -> torch.Tensor:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        f = payload["image_embedding"]
        if f.dtype != torch.float16:
            f = f.to(torch.float16)
        if f.shape != (256, 64, 64):
            raise ValueError(f"bad embedding shape {tuple(f.shape)} at {p}")
        return f

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cases[idx]
        img = Image.open(c.img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        mask_native = np.asarray(Image.open(c.mask_path).convert("L"))
        mask_bin = (mask_native > 127).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_bin).resize(
            (self.image_size, self.image_size), Image.NEAREST)
        img_t = TF.pil_to_tensor(img).float().div_(255.0)
        mask_t = (TF.pil_to_tensor(mask_pil).float() / 255.0 > 0.5).float()
        med = self._load_medsam(c.cache_path)
        if self.augment and torch.rand(1).item() < 0.5:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)
            med = torch.flip(med, dims=[-1])
        return img_t, mask_t, med


class ValDataset(Dataset):
    """Resolution-faithful val: (image_256, gt_native, native_hw)."""

    def __init__(self, cases: List[Case], image_size: int = 256) -> None:
        self.cases = cases
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        c = self.cases[idx]
        img_native = Image.open(c.img_path).convert("RGB")
        nh, nw = img_native.height, img_native.width
        img = img_native.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_t = TF.pil_to_tensor(img).float().div_(255.0)
        mask_native = np.asarray(Image.open(c.mask_path).convert("L"))
        gt = torch.from_numpy((mask_native > 127).astype(np.float32)).unsqueeze(0)
        return img_t, gt, (nh, nw)


def val_collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    gts = [b[1] for b in batch]
    sizes = [b[2] for b in batch]
    return imgs, gts, sizes


# --------------------------------------------------------------------------- #
# Supervised seg loss (inline; no distillation imports)                       #
# --------------------------------------------------------------------------- #

class SegLoss(nn.Module):
    def __init__(self, dice_smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_smooth = float(dice_smooth)

    @staticmethod
    def _soft_dice(logits, targets, smooth):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        l_bce = self.bce(logits, targets)
        l_dice = self._soft_dice(logits, targets, self.dice_smooth)
        total = l_bce + l_dice
        return total, {"total": total.detach(), "bce": l_bce.detach(), "dice": l_dice.detach()}


# --------------------------------------------------------------------------- #
# Resolution-faithful metrics                                                 #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def dice_iou_native(prob_native, gt_native, threshold=0.5, smooth=1.0):
    pred = (prob_native > threshold).float().reshape(-1)
    gt = (gt_native > 0.5).float().reshape(-1)
    inter = (pred * gt).sum().item()
    p, g = pred.sum().item(), gt.sum().item()
    dice = (2.0 * inter + smooth) / (p + g + smooth)
    iou = (inter + smooth) / (p + g - inter + smooth)
    return float(dice), float(iou)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    dices, ious = [], []
    for imgs, gts, sizes in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            logits, _ = model(imgs)
        for b, (gt, hw) in enumerate(zip(gts, sizes)):
            up = F.interpolate(logits[b:b+1].float(), size=hw, mode="bilinear",
                               align_corners=False)
            prob = torch.sigmoid(up).cpu()
            d, i = dice_iou_native(prob, gt)
            dices.append(d); ious.append(i)
    return {"val_dice_mean": float(np.mean(dices)), "val_dice_std": float(np.std(dices)),
            "val_iou_mean": float(np.mean(ious)), "val_iou_std": float(np.std(ious)),
            "n_val": len(dices)}


# --------------------------------------------------------------------------- #
# Generic training loops (sup / Hop-1 / Hop-2)                                #
# --------------------------------------------------------------------------- #

def train_supervised(model, loader, loss_fn, optimizer, scaler, device, epoch, use_amp,
                     empty_cache_every):
    model.train()
    sums = {"total": 0.0, "bce": 0.0, "dice": 0.0}; n = 0
    pbar = tqdm(loader, desc=f"sup ep{epoch}", leave=False)
    for step, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            logits, _ = model(imgs)
            total, parts = loss_fn(logits, masks)
        scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
        for k in sums: sums[k] += float(parts[k].item())
        n += 1
        del imgs, masks, logits, total, parts
        if empty_cache_every > 0 and (step + 1) % empty_cache_every == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    return {k: sums[k] / max(n, 1) for k in sums}


def train_hop1(model, loader, loss_fn, optimizer, scaler, device, epoch, use_amp,
               empty_cache_every):
    model.train(); loss_fn.train()
    sums = {"total": 0.0, "seg": 0.0, "bce": 0.0, "dice": 0.0, "nce": 0.0}; n = 0
    pbar = tqdm(loader, desc=f"hop1 ep{epoch}", leave=False)
    for step, (imgs, masks, med) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
        med = med.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            s_logits, s_bot = model(imgs)
            total, parts = loss_fn(student_logits=s_logits, student_bottleneck=s_bot,
                                   medsam_features=med, ground_truth_mask=masks)
        scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
        for k in sums: sums[k] += float(parts[k].item())
        n += 1
        del imgs, masks, med, s_logits, s_bot, total, parts
        if empty_cache_every > 0 and (step + 1) % empty_cache_every == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    return {k: sums[k] / max(n, 1) for k in sums}


def train_hop2(teacher, student, loader, loss_fn, optimizer, scaler, device, epoch, use_amp,
               empty_cache_every):
    student.train(); loss_fn.train(); teacher.eval()
    sums = {"total": 0.0, "seg": 0.0, "bce": 0.0, "dice": 0.0, "mse": 0.0, "kl": 0.0}; n = 0
    pbar = tqdm(loader, desc=f"hop2 ep{epoch}", leave=False)
    for step, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            with torch.no_grad():
                t_logits, t_bot = teacher(imgs)
            s_logits, s_bot = student(imgs)
            total, parts = loss_fn(student_logits=s_logits, teacher_logits=t_logits,
                                   student_bottleneck=s_bot, teacher_bottleneck=t_bot,
                                   ground_truth_mask=masks)
        scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
        for k in sums: sums[k] += float(parts[k].item())
        n += 1
        del imgs, masks, t_logits, t_bot, s_logits, s_bot, total, parts
        if empty_cache_every > 0 and (step + 1) % empty_cache_every == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    return {k: sums[k] / max(n, 1) for k in sums}


# --------------------------------------------------------------------------- #
# Per-ablation runners                                                        #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def is_complete(run_dir: Path, target_epochs: int) -> bool:
    hp = run_dir / "training_history.json"
    if not hp.is_file(): return False
    try:
        return len(json.loads(hp.read_text())) >= target_epochs
    except Exception:
        return False


def run_supervised(ablation: str, cases_train: List[Case], cases_val: List[Case],
                   model_factory, ckpt_name: str, args, seed: int) -> Path:
    rd = args.output_root / f"{ablation}_seed{seed}"
    if is_complete(rd, args.epochs):
        log.info("[skip] %s seed=%d already complete", ablation, seed)
        return rd / ckpt_name

    rd.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_ds = SupTrainDataset(cases_train, image_size=256, augment=True)
    val_ds = ValDataset(cases_val, image_size=256)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp,
                              drop_last=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_amp,
                            collate_fn=val_collate,
                            persistent_workers=(args.num_workers > 0))

    model = model_factory().to(device)
    loss_fn = SegLoss(dice_smooth=args.dice_smooth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    best_dice = -1.0
    best_path = rd / ckpt_name
    last_path = rd / ckpt_name.replace("best", "last")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_supervised(model, train_loader, loss_fn, optimizer, scaler, device,
                              epoch, use_amp, args.empty_cache_every)
        vm = validate(model, val_loader, device, use_amp)
        log.info("[%s.s%d] ep%03d  train=%.4f val_dice=%.4f val_iou=%.4f (%.1fs)",
                 ablation, seed, epoch, tm["total"], vm["val_dice_mean"], vm["val_iou_mean"],
                 time.time() - t0)
        history.append({"epoch": epoch, **tm, **vm})
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "val_dice": vm["val_dice_mean"], "ablation": ablation, "seed": seed}
        torch.save(ckpt, last_path)
        if vm["val_dice_mean"] > best_dice:
            best_dice = vm["val_dice_mean"]; torch.save(ckpt, best_path)
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    (rd / "training_history.json").write_text(json.dumps(history, indent=2))
    log.info("[%s.s%d] DONE  best_val_dice=%.4f", ablation, seed, best_dice)
    return best_path


def run_a1(cases_train: List[Case], cases_val: List[Case], args, seed: int) -> Path:
    rd = args.output_root / f"a1_seed{seed}"
    if is_complete(rd, args.epochs):
        log.info("[skip] A1 seed=%d already complete", seed)
        return rd / "edgeunet_a1_best.pt"

    rd.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_ds = DistillTrainDataset(cases_train, image_size=256, augment=True)
    val_ds = ValDataset(cases_val, image_size=256)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp,
                              drop_last=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_amp,
                            collate_fn=val_collate,
                            persistent_workers=(args.num_workers > 0))

    student = EdgeUNet(in_channels=3, num_classes=1).to(device)
    loss_fn = Hop1ContrastiveLoss(student_channels=EdgeUNet.BOTTLENECK_CHANNELS,
                                  teacher_channels=256, target_grid=64,
                                  temperature=args.temperature,
                                  lambda_nce=args.lambda_nce,
                                  dice_smooth=args.dice_smooth).to(device)
    params = list(student.parameters()) + list(loss_fn.projector_parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []; best_dice = -1.0
    best_path = rd / "edgeunet_a1_best.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_hop1(student, train_loader, loss_fn, optimizer, scaler, device,
                        epoch, use_amp, args.empty_cache_every)
        vm = validate(student, val_loader, device, use_amp)
        log.info("[A1.s%d] ep%03d  total=%.4f nce=%.4f val_dice=%.4f (%.1fs)",
                 seed, epoch, tm["total"], tm["nce"], vm["val_dice_mean"], time.time() - t0)
        history.append({"epoch": epoch, **tm, **vm})
        ckpt = {"epoch": epoch, "model": student.state_dict(),
                "projector": loss_fn.projector.state_dict(),
                "val_dice": vm["val_dice_mean"], "ablation": "A1_direct", "seed": seed}
        if vm["val_dice_mean"] > best_dice:
            best_dice = vm["val_dice_mean"]; torch.save(ckpt, best_path)
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    (rd / "training_history.json").write_text(json.dumps(history, indent=2))
    log.info("[A1.s%d] DONE  best_val_dice=%.4f", seed, best_dice)
    return best_path


def run_a2(cases_train: List[Case], cases_val: List[Case],
           teacher_ckpt: Path, args, seed: int) -> Path:
    rd = args.output_root / f"a2_seed{seed}"
    if is_complete(rd, args.epochs):
        log.info("[skip] A2 seed=%d already complete", seed)
        return rd / "student_best.pt"

    if not teacher_ckpt.is_file():
        log.error("[A2.s%d] teacher missing at %s — skip", seed, teacher_ckpt)
        return rd / "student_best.pt"

    rd.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_ds = SupTrainDataset(cases_train, image_size=256, augment=True)
    val_ds = ValDataset(cases_val, image_size=256)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_amp,
                              drop_last=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_amp,
                            collate_fn=val_collate,
                            persistent_workers=(args.num_workers > 0))

    teacher = HANet(in_channels=3, num_classes=1).to(device)
    state = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    log.info("[A2.s%d] teacher loaded from %s", seed, teacher_ckpt)

    student = EdgeUNet(in_channels=3, num_classes=1).to(device)
    loss_fn = Hop2DistillationLoss(
        student_bottleneck_channels=EdgeUNet.BOTTLENECK_CHANNELS,
        teacher_bottleneck_channels=512,
        temperature=args.kd_temperature,
        lambda_mse=args.lambda_mse,
        lambda_kl=args.lambda_kl,
        dice_smooth=args.dice_smooth).to(device)
    params = list(student.parameters()) + list(loss_fn.projector_parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []; best_dice = -1.0
    best_path = rd / "student_best.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_hop2(teacher, student, train_loader, loss_fn, optimizer, scaler, device,
                        epoch, use_amp, args.empty_cache_every)
        vm = validate(student, val_loader, device, use_amp)
        log.info("[A2.s%d] ep%03d  total=%.4f mse=%.4f kl=%.4f val_dice=%.4f (%.1fs)",
                 seed, epoch, tm["total"], tm["mse"], tm["kl"],
                 vm["val_dice_mean"], time.time() - t0)
        history.append({"epoch": epoch, **tm, **vm})
        ckpt = {"epoch": epoch, "student": student.state_dict(),
                "projector": loss_fn.projector.state_dict(),
                "val_dice": vm["val_dice_mean"], "ablation": "A2", "seed": seed}
        if vm["val_dice_mean"] > best_dice:
            best_dice = vm["val_dice_mean"]; torch.save(ckpt, best_path)
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    (rd / "training_history.json").write_text(json.dumps(history, indent=2))
    log.info("[A2.s%d] DONE  best_val_dice=%.4f", seed, best_dice)
    return best_path


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Critical-path B1/B2/A1/A2 trainer.")
    p.add_argument("--data_root", type=Path, required=True,
                   help="<data_root>/images and <data_root>/masks must exist.")
    p.add_argument("--medsam_cache_root", type=Path, required=True,
                   help="Directory of <stem>.pt MedSAM cache files (flat).")
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260530, 42, 1337])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dice_smooth", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Hop-1 InfoNCE temperature.")
    p.add_argument("--lambda_nce", type=float, default=1.0,
                   help="Hop-1 InfoNCE loss weight (paper default).")
    p.add_argument("--kd_temperature", type=float, default=2.0,
                   help="Hop-2 KL temperature.")
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--empty_cache_every", type=int, default=1)
    p.add_argument("--only", nargs="+", choices=["b1", "b2", "a1", "a2"], default=None,
                   help="Optional: run only a subset of the 4 ablations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only) if args.only else {"b1", "b2", "a1", "a2"}
    log.info("device=%s   seeds=%s   ablations=%s",
             "cuda" if torch.cuda.is_available() else "cpu",
             args.seeds, sorted(selected))

    cases = discover_dataset(args.data_root, args.medsam_cache_root)
    log.info("dataset: %d total cases", len(cases))

    t_total = time.time()
    for seed in args.seeds:
        train_cases, val_cases = deterministic_split(cases, args.val_frac, seed)
        log.info("=== SEED %d   train=%d  val=%d ===", seed, len(train_cases), len(val_cases))

        if "b1" in selected:
            run_supervised("b1_edgeunet", train_cases, val_cases,
                           lambda: EdgeUNet(in_channels=3, num_classes=1),
                           "edgeunet_baseline_best.pt", args, seed)

        b2_ckpt = None
        if "b2" in selected:
            b2_ckpt = run_supervised("b2_hanet", train_cases, val_cases,
                                     lambda: HANet(in_channels=3, num_classes=1),
                                     "hanet_baseline_best.pt", args, seed)

        if "a1" in selected:
            run_a1(train_cases, val_cases, args, seed)

        if "a2" in selected:
            if b2_ckpt is None:
                b2_ckpt = args.output_root / f"b2_hanet_seed{seed}" / "hanet_baseline_best.pt"
            run_a2(train_cases, val_cases, b2_ckpt, args, seed)

    log.info("=" * 60)
    log.info("critical-path trainer complete in %.1f min",
             (time.time() - t_total) / 60.0)
    log.info("output root: %s", args.output_root)


if __name__ == "__main__":
    main()
