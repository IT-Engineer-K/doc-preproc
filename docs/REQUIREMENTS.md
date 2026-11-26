# Requirements & Architecture Specification

## 1. Scope & Safety Boundary

`doc-preproc` is a deterministic, domain-agnostic document image preprocessing library.

- **In scope**:
  - Image quality diagnosis (micro-skew, blur, contrast, illumination unevenness, character/line height estimation, aspect ratio, advisory unreadability flags).
  - Adaptive, non-destructive image preprocessing (micro-deskew via bilinear interpolation, smooth local background division, percentile contrast stretching).
  - Exact 2x3 affine coordinate transforms (`forward`, `inverse`, `compose`) preserving correspondence between original and transformed pixel spaces.
  - Local re-read crop planning (padding, target character-height scaling, exact transform).
- **Out of scope (strictly prohibited)**:
  - OCR or text recognition inference.
  - Generative image synthesis or AI super-resolution.
  - Destructive hard binarization by default.
  - Rejecting documents (unreadability flags are strictly advisory).
  - Network I/O or external service dependencies.

## 2. Runtime & Packaging Requirements

- Python `>=3.10`.
- Zero required runtime dependencies (standard library only).
- PEP 561 typed package (`py.typed`).
- Seamless duck-typed acceptance of 2D grayscale and 3D RGB `numpy`-like arrays (`.shape`, `.tolist()`) without requiring `numpy` at install or import time.

## 3. Functional Requirements

### 3.1 Quality Diagnosis (`diagnose`)

Returns an immutable `QualityReport`:

| Metric | Definition |
| --- | --- |
| `skew_angle_deg` | Projection-profile variance maximization across `[-max_skew_search_deg, +max_skew_search_deg]` with sub-degree refinement. |
| `blur_score` | Variance of the discrete 2D Laplacian operator over interior pixels. |
| `contrast_score` | Standard deviation of grayscale pixel intensities. |
| `illumination_unevenness` | Range (`max - min`) of local 85th-percentile background estimates across a spatial block grid. |
| `estimated_char_height_px` | Median height of contiguous text-active horizontal projection runs. |
| `aspect_ratio` | `height / width`. |
| `unreadable_flag` / `unreadable_reasons` | Advisory flag set for `crushed_blacks`, `blown_highlights`, `flat_image`, or `extreme_blur`. |

### 3.2 Coordinate Transform (`AffineTransform2D`)

- Exact 2x3 matrix `(a, b, tx, c, d, ty)` mapping `(x, y) -> (x', y')`.
- Exact analytical inversion (`inverse(x', y')`, `invert()`) with round-trip error `< 1e-6`.
- Associative composition via `t1.compose(t2)` applying `t1` then `t2`.

### 3.3 Adaptive Preprocessing (`preprocess`)

- Supports profiles `"receipt"`, `"invoice"`, and `"crop"`, plus optional `PreprocessConfig` overrides.
- **Non-destructive**: Input images are never mutated.
- **No-op fast path**: If no quality metric breaches its threshold, returns `is_noop=True`, `applied_transforms=()`, and an identity `AffineTransform2D`.
- **Conditional deskew**: Triggered only when `abs(skew_angle_deg) > skew_threshold_deg`.
- **Conditional illumination normalization**: Smooth coarse-grid background division triggered only when `illumination_unevenness > illumination_threshold`.
- **Conditional contrast stretching**: Linear percentile stretch (`2%..98%` -> `10..250`) triggered only when `contrast_score < contrast_threshold` on non-flat images.

### 3.4 Crop Planning (`compute_crop_plan`)

- Adds configurable padding around `bbox`, clamps to image bounds, computes bounded scale factor `target_char_height_px / estimated_char_height_px`, and returns exact `AffineTransform2D` from original image coordinates to cropped output coordinates.
