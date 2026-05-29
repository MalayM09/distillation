"""
scripts/extract_medsam_kvasir.py — MedSAM image-encoder feature cache for Kvasir-SEG

Cross-dataset replication target (§5.4 of the paper). Mirrors the BUSI
extractor in scripts/01_extract_medsam_features.py but adapted to Kvasir-SEG's
flat layout:

    <kvasir_root>/
        images/   *.jpg|*.png    — RGB endoscopy frames
        masks/    *.jpg|*.png    — binary polyp masks at matching filenames

Per cached `.pt` we store ONLY the ViT image embedding:
    image_embedding : (256, 64, 64) fp16

For the critical-path replication (B1 / A1 / A2) we do not need bbox-prompted
decoder logits — only the encoder feature map that Hop-1 InfoNCE consumes.
This keeps the cache compact (~270 KB/sample × 1000 samples ≈ 270 MB).

Resumable: per-image `.pt` files written atomically; existing files are skipped.

Usage on Kaggle
---------------
    !pip install -q git+https://github.com/facebookresearch/segment-anything.git
    !python scripts/extract_medsam_kvasir.py \\
        --kvasir_root /kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG \\
        --medsam_ckpt flaviagiammarino/medsam-vit-base \\
        --out_dir /kaggle/working/medsam_cache_kvasir

The --medsam_ckpt argument accepts either a local `.pth` (loaded via
segment_anything) or a HuggingFace repo id (loaded via transformers).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("extract_medsam_kvasir")


MEDSAM_INPUT_SIZE = 1024
EMBED_CHANNELS = 256
EMBED_SPATIAL = MEDSAM_INPUT_SIZE // 16    # 64
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Backend loading (shared between local .pth and HF repo)                     #
# --------------------------------------------------------------------------- #

class _Backend:
    @torch.inference_mode()
    def encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _LocalBackend(_Backend):
    name = "segment_anything"

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        from segment_anything import sam_model_registry
        self.device = device
        self.model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.image_encoder(img_tensor)


class _HFBackend(_Backend):
    name = "hf-transformers"

    def __init__(self, repo_id: str, device: torch.device) -> None:
        from transformers import SamModel
        self.device = device
        self.model = SamModel.from_pretrained(repo_id)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_embeddings(img_tensor)


def build_backend(ref: str, device: torch.device) -> _Backend:
    p = Path(ref)
    if p.is_file():
        log.info("MedSAM backend: segment_anything from %s", p)
        return _LocalBackend(p, device)
    log.info("MedSAM backend: HuggingFace repo %s", ref)
    return _HFBackend(ref, device)


# --------------------------------------------------------------------------- #
# Kvasir-SEG discovery                                                        #
# --------------------------------------------------------------------------- #

def discover_kvasir(kvasir_root: Path, out_dir: Path) -> List[Path]:
    """Return list of image paths whose mask sibling exists."""
    img_dir = kvasir_root / "images"
    mask_dir = kvasir_root / "masks"
    if not img_dir.is_dir():
        raise SystemExit(f"images/ subdir missing under {kvasir_root}")
    if not mask_dir.is_dir():
        raise SystemExit(f"masks/ subdir missing under {kvasir_root}")
    pairs: List[Path] = []
    missing_mask = 0
    for ext in IMG_EXTS:
        for img_path in sorted(img_dir.glob(f"*{ext}")):
            # Mask may have the same stem but possibly different extension.
            mask_match = None
            for mext in IMG_EXTS:
                cand = mask_dir / f"{img_path.stem}{mext}"
                if cand.is_file():
                    mask_match = cand
                    break
            if mask_match is None:
                missing_mask += 1
                continue
            pairs.append(img_path)
    if missing_mask:
        log.warning("dropped %d images with no mask sibling", missing_mask)
    log.info("found %d paired Kvasir-SEG cases", len(pairs))
    return pairs


# --------------------------------------------------------------------------- #
# MedSAM preprocessing (identical to BUSI extractor)                          #
# --------------------------------------------------------------------------- #

def load_and_preprocess(img_path: Path, device: torch.device) -> torch.Tensor:
    """Read image -> (1, 3, 1024, 1024) float32 on device, min-max [0,1]."""
    img_rgb = np.asarray(Image.open(img_path).convert("RGB"))   # (H, W, 3) uint8
    pil = Image.fromarray(img_rgb).resize(
        (MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE), Image.BICUBIC,
    )
    img = np.asarray(pil).astype(np.float32, copy=False)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / max(hi - lo, 1e-8)
    t = (torch.from_numpy(img).permute(2, 0, 1).contiguous()
                .float().unsqueeze(0).to(device, non_blocking=True))
    assert t.shape == (1, 3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE)
    return t


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MedSAM image-encoder feature cache for Kvasir-SEG.")
    p.add_argument("--kvasir_root", type=Path, required=True,
                   help="Path to Kvasir-SEG root containing images/ and masks/.")
    p.add_argument("--medsam_ckpt", type=str, required=True,
                   help="Local medsam_vit_b.pth path OR HF repo id "
                        "(e.g. flaviagiammarino/medsam-vit-base).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Cache output dir; each <stem>.pt receives "
                        "{'image_embedding': (256, 64, 64) FP16}.")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu. Auto if None.")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if cache already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("device=%s", device)

    cases = discover_kvasir(args.kvasir_root, args.out_dir)
    pending = []
    for p in cases:
        out_p = args.out_dir / f"{p.stem}.pt"
        if not args.force and out_p.is_file():
            continue
        pending.append((p, out_p))
    log.info("%d to extract (%d already cached)", len(pending), len(cases) - len(pending))
    if not pending:
        log.info("nothing to do.")
        return

    backend = build_backend(args.medsam_ckpt, device)

    for img_path, out_path in tqdm(pending, desc="MedSAM encode"):
        try:
            x = load_and_preprocess(img_path, device)
            emb = backend.encode(x)  # (1, 256, 64, 64) fp32 on device
        except Exception as exc:
            log.error("[%s] FAILED: %s", img_path.name, exc)
            continue

        # Shape contract.
        if emb.shape != (1, EMBED_CHANNELS, EMBED_SPATIAL, EMBED_SPATIAL):
            log.error("[%s] unexpected embedding shape %s — skipping",
                      img_path.name, tuple(emb.shape))
            continue

        payload = {
            "image_embedding": emb.squeeze(0).to(torch.float16).cpu().contiguous(),
            "meta": {"source": str(img_path), "dataset": "kvasir-seg"},
        }
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_done = sum(1 for p in cases if (args.out_dir / f"{p.stem}.pt").is_file())
    log.info("done — %d/%d cache files present under %s",
             n_done, len(cases), args.out_dir)


if __name__ == "__main__":
    main()
