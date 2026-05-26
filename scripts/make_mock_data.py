"""
scripts/make_mock_data.py — Synthetic BUSI + MedSAM cache for local smoke tests.

Creates `test_data/` with the directory structure expected by every trainer in
this repo:

    test_data/
        busi/
            benign/       *.png         (3-channel ultrasound, random native res)
                          *_mask.png    (1-channel binary, matching native res)
            malignant/    ...
        medsam_cache/
            benign/       *.pt          {"image_embedding": (256, 64, 64) FP16}
            malignant/    ...

Image stems match across BUSI and the MedSAM cache, so `discover_cases` in
train_hop1.py / train_a1_direct.py pairs them deterministically. Every mask
contains exactly one rectangular foreground blob so InfoNCE has non-degenerate
foreground anchors on every case.

Deterministic — seeded with 42 — so reruns produce byte-identical synthetic data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "test_data"
N_PER_CLASS = 15
CLASSES = ("benign", "malignant")


def main() -> None:
    rng = np.random.RandomState(42)
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    busi_root = OUTPUT_ROOT / "busi"
    cache_root = OUTPUT_ROOT / "medsam_cache"
    for cls in CLASSES:
        (busi_root / cls).mkdir(parents=True)
        (cache_root / cls).mkdir(parents=True)

    n_total = 0
    for cls in CLASSES:
        for i in range(N_PER_CLASS):
            stem = f"{cls}_{i:03d}"

            # Random native ultrasound resolution.
            h = int(rng.randint(320, 600))
            w = int(rng.randint(320, 600))

            # 3-channel RGB ultrasound (noise; the smoke test doesn't care about realism).
            img_arr = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
            Image.fromarray(img_arr, mode="RGB").save(busi_root / cls / f"{stem}.png")

            # Binary mask with exactly one rectangular foreground blob.
            mask_arr = np.zeros((h, w), dtype=np.uint8)
            y0 = int(rng.randint(h // 4, h // 2))
            y1 = y0 + int(rng.randint(h // 8, h // 3))
            x0 = int(rng.randint(w // 4, w // 2))
            x1 = x0 + int(rng.randint(w // 8, w // 3))
            mask_arr[y0:y1, x0:x1] = 255
            Image.fromarray(mask_arr, mode="L").save(busi_root / cls / f"{stem}_mask.png")

            # MedSAM cache: (256, 64, 64) FP16 random embedding.
            gen = torch.Generator().manual_seed(abs(hash(stem)) % (2**31))
            emb = torch.randn(256, 64, 64, generator=gen).to(torch.float16)
            torch.save(
                {"image_embedding": emb},
                cache_root / cls / f"{stem}.pt",
            )
            n_total += 1

    print(f"[make_mock_data] generated {n_total} synthetic cases under {OUTPUT_ROOT}")
    print(f"[make_mock_data] busi  : {busi_root}")
    print(f"[make_mock_data] cache : {cache_root}")


if __name__ == "__main__":
    main()
