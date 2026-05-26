"""
scripts/01_extract_medsam_features.py — CAISc 2026

Phase 1 of the heterogeneous distillation pipeline. Runs the frozen MedSAM
(ViT-B/16) over the BUSI corpus and serialises per-image teacher artifacts
to disk so Phase 2 can train the student U-Net *without* MedSAM resident in
VRAM. Joint residency is infeasible on Kaggle T4 (16 GB) at 1024² inference.

Per cached .pt we store:
    image_embedding : ViT encoder output, fp16, (256, 64, 64)
    teacher_logits  : decoder logits at 256x256 PRE-sigmoid, fp16
    teacher_mask    : binarised decoder mask at ORIGINAL (H0, W0), uint8
    meta            : original size, scaled+padded bbox, source path

We persist logits (not sigmoids) so the student can apply KL / soft-Dice at
arbitrary temperature without information loss.

Preprocessing contract (locked to the MedSAM evaluation protocol):
    1. Read image → 3-channel uint8 RGB (grayscale tiled to 3 channels).
    2. OR-merge all `*_mask*.png` siblings via np.maximum  ← multi-mask cases.
    3. Tight bbox of the merged mask, then symmetric +15 px margin, clamped.
    4. Bicubic resize to 1024×1024 (no aspect preservation — MedSAM stretches).
    5. Per-image min-max normalisation to [0, 1] AFTER resize. We deliberately
       do NOT use SAM's ImageNet mean/std — MedSAM's authors report it
       suppresses contrast on grayscale medical inputs.
    6. Tensorise → [1, 3, 1024, 1024], float32, on device. Strict assertion.
    7. Scale bbox xyxy by (1024 / W0, 1024 / H0) to align with the resized
       canvas before handing to the prompt encoder.

Resumable: each .pt is written atomically; existing cache files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from segment_anything import sam_model_registry
    _HAS_SEGMENT_ANYTHING = True
except ImportError:
    sam_model_registry = None
    _HAS_SEGMENT_ANYTHING = False

try:
    from transformers import SamModel
    _HAS_TRANSFORMERS = True
except ImportError:
    SamModel = None
    _HAS_TRANSFORMERS = False

if not (_HAS_SEGMENT_ANYTHING or _HAS_TRANSFORMERS):
    raise SystemExit(
        "Need at least one MedSAM loader. Install one of:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        "  pip install transformers"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEDSAM_INPUT_SIZE = 1024
BBOX_MARGIN_PX = 15
DECODER_LOWRES = 256
EMBED_CHANNELS = 256
EMBED_SPATIAL = MEDSAM_INPUT_SIZE // 16   # ViT-B/16 patch grid → 64
BUSI_CLASSES = ("benign", "malignant", "normal")

_MASK_RE = re.compile(r"_mask(?:_\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Case:
    image_path: Path
    mask_paths: tuple[Path, ...]
    cls: str
    cache_path: Path


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_busi(root: Path, cache_root: Path) -> list[Case]:
    cases: list[Case] = []
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
                logging.warning("no mask siblings for %s; skipping", img_path.name)
                continue
            cache_path = cache_root / cls / f"{img_path.stem}.pt"
            cases.append(Case(img_path, mask_paths, cls, cache_path))
    return cases


# ---------------------------------------------------------------------------
# Mask + bbox preprocessing
# ---------------------------------------------------------------------------

def merge_masks(mask_paths: tuple[Path, ...]) -> np.ndarray:
    """OR-merge every `*_mask*.png` sibling. Returns binary uint8 (H0, W0)."""
    acc: np.ndarray | None = None
    for p in mask_paths:
        m = np.asarray(Image.open(p).convert("L"))
        b = (m > 127).astype(np.uint8)
        acc = b if acc is None else np.maximum(acc, b)
    assert acc is not None and acc.ndim == 2
    return acc


def bbox_from_mask(mask: np.ndarray, margin: int = BBOX_MARGIN_PX) -> np.ndarray | None:
    """
    Tight bbox of `mask>0`, expanded by `margin` px per side and clamped.
    Returns xyxy float32 in ORIGINAL coords, or None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    h, w = mask.shape
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin, w - 1)
    y1 = min(int(ys.max()) + margin, h - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def scale_bbox_to_1024(bbox_xyxy: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    """Affine-scale xyxy from (H0, W0) into the 1024×1024 canvas (no padding)."""
    h0, w0 = orig_hw
    sx = MEDSAM_INPUT_SIZE / w0
    sy = MEDSAM_INPUT_SIZE / h0
    b = bbox_xyxy.copy()
    b[[0, 2]] *= sx
    b[[1, 3]] *= sy
    return b


# ---------------------------------------------------------------------------
# Image preprocessing — the heart of the contract
# ---------------------------------------------------------------------------

def _load_image_rgb(path: Path) -> np.ndarray:
    """
    Load and force 3-channel uint8 RGB. PIL's .convert('RGB') already tiles
    grayscale into 3 channels, but we explicitly handle the L-mode source so
    BUSI PNGs (which arrive as L, LA, or RGB depending on the export tool)
    are normalised to a single canonical layout.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    assert arr.ndim == 3 and arr.shape[-1] == 3, f"bad image layout for {path}: {arr.shape}"
    return arr.astype(np.uint8, copy=False)


def preprocess_image(
    image_path: Path,
    mask_paths: tuple[Path, ...],
    device: torch.device,
) -> dict:
    """
    Returns a dict carrying every artifact downstream needs:
        image_tensor   : torch.float32 [1, 3, 1024, 1024] on device, in [0, 1]
        bbox_1024      : torch.float32 [1, 1, 4] on device, MedSAM-space xyxy
        bbox_orig      : np.ndarray (4,) float32, xyxy in original coords
        merged_mask    : np.ndarray (H0, W0) uint8 — for downstream eval / sanity
        orig_hw        : (H0, W0)
    Raises ValueError if the merged mask is empty (caller should skip).
    """
    # --- (1) Image → 3-channel uint8 RGB
    img_rgb = _load_image_rgb(image_path)          # (H0, W0, 3) uint8
    h0, w0 = img_rgb.shape[:2]

    # --- (2) OR-merged GT mask at original resolution
    merged = merge_masks(mask_paths)               # (H0, W0) uint8 {0,1}
    if merged.shape != (h0, w0):
        # BUSI ships matched dims, but guard against rare aspect-mismatched masks
        merged = np.asarray(
            Image.fromarray(merged * 255).resize((w0, h0), Image.NEAREST)
        )
        merged = (merged > 127).astype(np.uint8)

    # --- (3) Bbox with +15px margin in ORIGINAL coords
    bbox_orig = bbox_from_mask(merged, margin=BBOX_MARGIN_PX)
    if bbox_orig is None:
        raise ValueError("empty merged mask — no foreground pixels")

    # --- (4) Bicubic resize to 1024×1024 (MedSAM convention: stretch, no pad)
    pil = Image.fromarray(img_rgb).resize(
        (MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE), Image.BICUBIC,
    )
    img_1024 = np.asarray(pil).astype(np.float32, copy=False)   # (1024,1024,3)

    # --- (5) Per-image min-max → [0, 1]. Epsilon-clamped for degenerate tiles.
    lo = img_1024.min()
    hi = img_1024.max()
    img_1024 = (img_1024 - lo) / max(float(hi - lo), 1e-8)

    # --- (6) → tensor [1, 3, 1024, 1024], float32, contiguous, on device
    image_tensor = (
        torch.from_numpy(img_1024)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .unsqueeze(0)
            .to(device, non_blocking=True)
    )
    assert image_tensor.shape == (1, 3, MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE), (
        f"tensor shape contract violated: {tuple(image_tensor.shape)}"
    )
    assert image_tensor.dtype == torch.float32

    # --- (7) Scale bbox into 1024 canvas
    bbox_1024 = scale_bbox_to_1024(bbox_orig, (h0, w0))
    bbox_tensor = (
        torch.from_numpy(bbox_1024)
            .float()
            .view(1, 1, 4)
            .to(device, non_blocking=True)
    )

    return {
        "image_tensor": image_tensor,
        "bbox_1024": bbox_tensor,
        "bbox_orig": bbox_orig,
        "merged_mask": merged,
        "orig_hw": (h0, w0),
    }


# ---------------------------------------------------------------------------
# Teacher mask un-projection
# ---------------------------------------------------------------------------

def unproject_logits_to_original(
    low_res_logits: torch.Tensor,        # (1, 1, 256, 256) fp32
    orig_hw: tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    """1024-stretched canvas → original H0×W0 (no padding to strip)."""
    m = F.interpolate(
        low_res_logits, size=(MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE),
        mode="bilinear", align_corners=False,
    )
    m = F.interpolate(m, size=orig_hw, mode="bilinear", align_corners=False)
    prob = torch.sigmoid(m).squeeze().to(torch.float32).cpu().numpy()
    return (prob > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Backends — unified MedSAM forward across segment_anything (.pth) and
# HuggingFace Hub (transformers.SamModel). Both produce the same shapes:
#   encode  : (1, 3, 1024, 1024) → (1, 256, 64, 64)
#   decode  : embedding + bbox_1024 (1, 1, 4) → (1, 1, 256, 256) pre-sigmoid
# Preprocessing (min-max [0, 1], stretch-resize) is identical for both —
# the HF port uses the same MedSAM weights and the same input distribution.
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO = "flaviagiammarino/medsam-vit-base"


class _Backend:
    name: str

    @torch.inference_mode()
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def decode(self, image_embedding: torch.Tensor, box_1024: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _LocalBackend(_Backend):
    """segment_anything ViT-B with a local medsam_vit_b.pth checkpoint."""
    name = "segment_anything"

    def __init__(self, checkpoint: Path, device: torch.device):
        if not _HAS_SEGMENT_ANYTHING:
            raise SystemExit(
                "segment_anything required for local .pth: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        self.device = device
        self.model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.image_encoder(image_tensor)

    @torch.inference_mode()
    def decode(self, image_embedding: torch.Tensor, box_1024: torch.Tensor) -> torch.Tensor:
        sparse_emb, dense_emb = self.model.prompt_encoder(
            points=None, boxes=box_1024, masks=None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        return low_res_logits


class _HFBackend(_Backend):
    """transformers.SamModel loaded from a HuggingFace Hub repo id."""
    name = "hf-transformers"

    def __init__(self, repo_id: str, device: torch.device, revision: str | None = None):
        if not _HAS_TRANSFORMERS:
            raise SystemExit(
                "transformers required for HF loader: pip install transformers"
            )
        self.device = device
        self.repo_id = repo_id
        self.model = SamModel.from_pretrained(repo_id, revision=revision)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Bypasses SamProcessor (which would apply SAM's ImageNet normalisation);
        # our pixel_values are already MedSAM-correct (min-max → [0, 1]).
        return self.model.get_image_embeddings(image_tensor)

    @torch.inference_mode()
    def decode(self, image_embedding: torch.Tensor, box_1024: torch.Tensor) -> torch.Tensor:
        # HF SAM expects input_boxes shape (B, num_objects, 4). One object → reshape.
        input_boxes = box_1024.view(1, 1, 4)
        outputs = self.model(
            pixel_values=None,
            image_embeddings=image_embedding,
            input_boxes=input_boxes,
            multimask_output=False,
        )
        # pred_masks: (B, num_objects, num_masks=1, 256, 256). Squeeze num_masks dim
        # so the tensor matches segment_anything's (1, 1, 256, 256) contract.
        pred = outputs.pred_masks
        if pred.dim() == 5:
            pred = pred.squeeze(2)
        return pred


def build_backend(
    ref: Path, device: torch.device, hf_revision: str | None = None,
) -> _Backend:
    """Local .pth file → segment_anything; otherwise treat as HF repo id."""
    if ref.is_file():
        logging.info("MedSAM backend = segment_anything (%s)", ref)
        return _LocalBackend(ref, device)
    logging.info("MedSAM backend = HF transformers (%s)", ref)
    return _HFBackend(str(ref), device, revision=hf_revision)


def main() -> int:
    parser = argparse.ArgumentParser(description="MedSAM Phase-1 feature/mask cache for BUSI.")
    parser.add_argument("--busi-root", type=Path, required=True,
                        help="Root containing benign/ malignant/ normal/ subdirs.")
    parser.add_argument(
        "--medsam-ckpt", type=Path, required=True,
        help=(
            "Either a local medsam_vit_b.pth path (loaded via segment_anything) "
            f"OR a HuggingFace Hub repo id, e.g. '{DEFAULT_HF_REPO}' "
            "(loaded via transformers.SamModel). Auto-detected by file existence."
        ),
    )
    parser.add_argument(
        "--hf-revision", type=str, default=None,
        help="Optional commit/branch pin when --medsam-ckpt is a HF repo id.",
    )
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Cache output root.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="CSV manifest path (default: <out-dir>/manifest.csv).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache exists.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable — CPU run will be very slow.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cls in BUSI_CLASSES:
        (args.out_dir / cls).mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out_dir / "manifest.csv")

    logging.info("discovering BUSI under %s", args.busi_root)
    cases = discover_busi(args.busi_root, args.out_dir)
    if not cases:
        logging.error("no cases discovered — verify --busi-root layout")
        return 2
    by_cls = {c: sum(1 for x in cases if x.cls == c) for c in BUSI_CLASSES}
    logging.info("discovered %d cases (%s)", len(cases), by_cls)

    logging.info("loading MedSAM from %s", args.medsam_ckpt)
    backend = build_backend(args.medsam_ckpt, device, hf_revision=args.hf_revision)

    pending = [c for c in cases if args.force or not c.cache_path.exists()]
    logging.info("%d to process (%d already cached)", len(pending), len(cases) - len(pending))

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    pbar = tqdm(pending, desc="MedSAM extract", unit="img", dynamic_ncols=True)
    for case in pbar:
        try:
            with torch.no_grad():
                batch = preprocess_image(case.image_path, case.mask_paths, device)
                image_embedding = backend.encode(batch["image_tensor"])
                low_res_logits = backend.decode(image_embedding, batch["bbox_1024"])
                teacher_mask = unproject_logits_to_original(
                    low_res_logits.float(), batch["orig_hw"],
                )

            payload = {
                "image_embedding": image_embedding.squeeze(0).to(torch.float16).cpu().contiguous(),
                "teacher_logits":  low_res_logits.squeeze(0).squeeze(0).to(torch.float16).cpu().contiguous(),
                "teacher_mask":    torch.from_numpy(teacher_mask),
                "meta": {
                    "source":         str(case.image_path),
                    "cls":            case.cls,
                    "orig_hw":        batch["orig_hw"],
                    "bbox_orig_xyxy": batch["bbox_orig"].tolist(),
                    "bbox_1024_xyxy": batch["bbox_1024"].squeeze().cpu().tolist(),
                    "n_masks":        len(case.mask_paths),
                },
            }

            tmp = case.cache_path.with_suffix(".pt.tmp")
            torch.save(payload, tmp, _use_new_zipfile_serialization=True)
            tmp.replace(case.cache_path)                       # atomic

            rows.append({
                "image_path": str(case.image_path),
                "cache_path": str(case.cache_path),
                "cls":        case.cls,
                "orig_h":     batch["orig_hw"][0],
                "orig_w":     batch["orig_hw"][1],
                "bbox_orig":  ";".join(f"{v:.2f}" for v in batch["bbox_orig"].tolist()),
            })

        except Exception as e:  # noqa: BLE001 — keep the kernel alive, report at end
            logging.exception("failed on %s", case.image_path)
            failures.append((str(case.image_path), repr(e)))

        finally:
            # Per-iteration memory hygiene — Kaggle T4 is unforgiving at 1024².
            for name in ("batch", "image_embedding", "low_res_logits", "teacher_mask", "payload"):
                if name in locals():
                    del locals()[name]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # Merge with prior manifest (resumed runs)
    prior: list[dict] = []
    if manifest_path.exists():
        with manifest_path.open("r", newline="") as f:
            prior = list(csv.DictReader(f))
        new_caches = {r["cache_path"] for r in rows}
        prior = [r for r in prior if r["cache_path"] not in new_caches]
    final_rows = prior + rows
    if final_rows:
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            w.writeheader()
            w.writerows(final_rows)
        logging.info("manifest: %d rows → %s", len(final_rows), manifest_path)

    if failures:
        logging.error("%d failures; first 5: %s", len(failures), failures[:5])
        return 1
    logging.info("done — cached %d new tensors under %s", len(rows), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
