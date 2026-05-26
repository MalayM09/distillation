# 03 — Engineering Constraints and Rules

These are non-negotiable constraints for the CAISc 2026 submission. Any code, experiment, or design decision that violates a rule below is invalid and must be reworked before review.

## Hardware

- **Training target:** single Kaggle T4 (16 GB VRAM, 13 GB RAM headroom after PyTorch). All training and ablation runs must fit and complete within Kaggle's 9-hour session budget per fold.
- **Mixed precision:** AMP/FP16 is mandatory for HA-Net training; FP32 master weights only where numerically required (e.g., loss scaling).
- **Feature cache:** MedSAM is *never* loaded in the training loop. Its embeddings and logits are precomputed once to disk (`.pt`) and streamed as offline targets. This keeps the active VRAM budget for student and assistant only.

## Edge Deployment Constraint (Student U-Net)

- **Hard parameter budget:** strictly **< 3.0M parameters** for the deployable student. Verified via `sum(p.numel() for p in model.parameters())` before any training run; runs that exceed the budget are aborted, not patched.
- **ONNX export:** the student must export cleanly via `torch.onnx.export` to opset ≥ 17 with no custom autograd functions and no Python-side control flow.
- **Operator set:** no attention layers, no `nn.MultiheadAttention`, no custom CUDA kernels, no `einops` rearrangements that don't fold into standard ONNX ops. Permitted: Conv2d, BN/GN, ReLU/SiLU, bilinear upsample, concat, residual add. This guarantees INT8 deployability on Hexagon, Ethos-U, and CoreML NPUs without per-op fallback to FP32.
- **Quantization-aware sanity check:** every committed student checkpoint must pass a dry-run INT8 PTQ pass (ONNX Runtime static quantization) without unsupported-op errors. This is a CI-style gate, not a final accuracy report.

## Evaluation (Resolution-Faithful)

- **All reported Dice and IoU are computed at the original ultrasound resolution**, not at the 256×256 training resolution. Predicted logits are bilinearly upsampled and re-thresholded at native size before metric computation.
- Rationale: BUSI images vary in native size; metrics at 256×256 systematically overestimate boundary quality because both the prediction and the ground truth have been downsampled through the same smoothing kernel, hiding sub-pixel boundary error that is clinically meaningful.
- This rule applies uniformly to B1, B2, A1, A2, and A3. Any table or plot using 256-resolution metrics is for *training diagnostics only* and must be labeled as such.

## Reproducibility

- Seed fixed per fold; seed list committed to the repo.
- All random sources seeded: `torch`, `numpy`, `random`, CUDA, and DataLoader workers.
- Each run writes a single `run_manifest.json` recording: git SHA, seed, fold, model param count, peak VRAM, wall-clock, final metrics, and the SHA-256 of the MedSAM feature cache used.

## Scope Discipline

- The student architecture is **vanilla U-Net only**. No attention, no transformer blocks, no learned upsampling beyond bilinear + conv. This is what makes the edge claim defensible; any deviation invalidates the paper's deployment story.
- The assistant is **HA-Net (~15M)**. Its hybrid-attention blocks are permitted *only* because the assistant is never deployed — it exists solely as a distillation source.
- No third teacher, no ensemble teachers, no self-distillation on the student. The thesis is a clean three-rung cascade.
