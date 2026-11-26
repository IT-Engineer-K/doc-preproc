# doc-preproc

Lightweight, zero-dependency, non-destructive document image quality diagnosis, adaptive preprocessing, and exact 2D coordinate mapping for Python (`>=3.10`).

## Design Principles

- **Pure Python core, zero required runtime dependencies**: Operates on a lightweight `GrayImage` 2D grid (`0..255`) with seamless duck-typed support for 2D grayscale or 3D RGB `numpy` arrays when passed in.
- **Non-destructive by default**: Never mutates input images and never applies destructive hard binarization.
- **No-op fast path**: Clean images with negligible skew, healthy contrast, and even lighting incur zero pixel transforms (`is_noop=True`, identity `AffineTransform2D`).
- **Exact reversible geometry**: Every geometric transform returns an exact 2x3 `AffineTransform2D` with `forward(x, y)` and `inverse(x', y')` mapping between original and preprocessed coordinates.
- **Strict safety boundary**: Performs deterministic signal processing only—no OCR inference, no generative super-resolution/image synthesis, and no network calls.

## Installation

Install in editable mode from a local checkout:

```bash
pip install -e .
```

## Quick Start

```python
from doc_preproc import GrayImage, compute_crop_plan, diagnose, preprocess

# Construct from nested lists (or pass a 2D/3D numpy array directly)
image = GrayImage.filled(120, 80, value=250.0)

# 1. Inspect image quality
report = diagnose(image)
print(report.skew_angle_deg, report.blur_score, report.contrast_score)

# 2. Run adaptive, non-destructive preprocessing
result = preprocess(image, profile="receipt")
print(result.is_noop, result.applied_transforms)

# Map a point from preprocessed coordinates back to original image coordinates
orig_x, orig_y = result.transform.inverse(50.0, 30.0)

# 3. Compute an optimal local re-read crop plan with exact coordinate transform
plan = compute_crop_plan(
    bbox=(10.0, 20.0, 90.0, 45.0),
    image_size=(image.width, image.height),
    estimated_char_height_px=14.0,
    target_char_height_px=28.0,
)
print(plan.scale_factor, plan.output_size)
```

## Public API

- `GrayImage`: Row-major 2D grayscale container with `from_nested`, `from_any`, `to_nested`, `copy`, and bilinear sampling.
- `AffineTransform2D`: Composable 2x3 affine matrix (`identity`, `translation`, `scale`, `rotation`, `forward`, `inverse`, `invert`, `compose`).
- `diagnose(image, config=None) -> QualityReport`: Computes `skew_angle_deg`, `blur_score`, `contrast_score`, `illumination_unevenness`, `estimated_char_height_px`, `aspect_ratio`, and advisory `unreadable_flag` / `unreadable_reasons`.
- `preprocess(image, profile="receipt"|"invoice"|"crop", config=None) -> PreprocessResult`: Runs diagnosis and applies only needed transforms (`deskew`, `illumination_normalization`, `contrast_normalization`), returning `(image, report, transform, applied_transforms, is_noop)`.
- `compute_crop_plan(...) -> CropPlan` & `apply_crop_plan(image, plan) -> GrayImage`: Computes padded crop bounds, optimal scale factor, and exact coordinate transform for local re-read regions.

## Running Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
