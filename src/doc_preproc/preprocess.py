"""Adaptive, non-destructive document preprocessing and local crop planning."""

from __future__ import annotations

import math
from typing import Any

from .diagnose import _percentile, compute_contrast_score, diagnose
from .geometry import AffineTransform2D
from .models import (
    CropPlan,
    GrayImage,
    PreprocessConfig,
    PreprocessResult,
    ProfileName,
    _clamp_pixel,
)


def deskew_image(
    image: GrayImage,
    angle_deg: float,
    fill_value: float = 255.0,
) -> tuple[GrayImage, AffineTransform2D]:
    """Deskew an image by rotating ``-angle_deg`` around its center using bilinear interpolation.

    Returns ``(deskewed_image, transform)`` where ``transform`` maps original
    image coordinates ``(x, y)`` to deskewed coordinates ``(x', y')``, and
    ``transform.inverse(x', y')`` maps back to original coordinates.
    """
    w, h = image.width, image.height
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    transform = AffineTransform2D.rotation(-angle_deg, cx, cy)
    inv = transform.invert()

    out_rows: list[list[float]] = []
    for y_out in range(h):
        row: list[float] = []
        base_x = inv.b * y_out + inv.tx
        base_y = inv.d * y_out + inv.ty
        for x_out in range(w):
            src_x = inv.a * x_out + base_x
            src_y = inv.c * x_out + base_y
            row.append(image.sample_bilinear(src_x, src_y, fill=fill_value))
        out_rows.append(row)

    return GrayImage(out_rows), transform


def normalize_illumination(
    image: GrayImage,
    grid_rows: int = 6,
    grid_cols: int = 6,
    target_background: float = 245.0,
) -> GrayImage:
    """Remove uneven lighting/shadows via smooth local background division.

    Non-destructive: preserves continuous grayscale tone without binarization.
    """
    w, h = image.width, image.height
    rows = max(2, min(grid_rows, max(2, h // 8)))
    cols = max(2, min(grid_cols, max(2, w // 8)))
    if w < 4 or h < 4:
        return image.copy()

    pixels = image.pixels
    bg_grid: list[list[float]] = []
    row_centers: list[float] = []
    col_centers: list[float] = []

    for r in range(rows):
        y0 = (r * h) // rows
        y1 = ((r + 1) * h) // rows
        row_centers.append(0.5 * (y0 + y1 - 1))
        grid_row: list[float] = []
        for c in range(cols):
            x0 = (c * w) // cols
            x1 = ((c + 1) * w) // cols
            if r == 0:
                col_centers.append(0.5 * (x0 + x1 - 1))
            vals: list[float] = []
            for y in range(y0, max(y0 + 1, y1)):
                vals.extend(pixels[y][x0 : max(x0 + 1, x1)])
            vals.sort()
            grid_row.append(max(1.0, _percentile(vals, 0.85)))
        bg_grid.append(grid_row)

    def _interp_1d(coord: float, centers: list[float]) -> tuple[int, int, float]:
        if coord <= centers[0]:
            return 0, 0, 0.0
        if coord >= centers[-1]:
            last = len(centers) - 1
            return last, last, 0.0
        for idx in range(len(centers) - 1):
            c0 = centers[idx]
            c1 = centers[idx + 1]
            if c0 <= coord <= c1:
                span = c1 - c0
                frac = (coord - c0) / span if span > 1e-9 else 0.0
                return idx, idx + 1, frac
        last = len(centers) - 1
        return last, last, 0.0

    x_interp = [_interp_1d(float(x), col_centers) for x in range(w)]
    out_rows: list[list[float]] = []

    for y in range(h):
        r0, r1, fy = _interp_1d(float(y), row_centers)
        bg_r0 = bg_grid[r0]
        bg_r1 = bg_grid[r1]
        src_row = pixels[y]
        out_row: list[float] = []
        for x in range(w):
            c0, c1, fx = x_interp[x]
            top = bg_r0[c0] * (1.0 - fx) + bg_r0[c1] * fx
            bottom = bg_r1[c0] * (1.0 - fx) + bg_r1[c1] * fx
            local_bg = max(1.0, top * (1.0 - fy) + bottom * fy)
            out_row.append(_clamp_pixel(src_row[x] * (target_background / local_bg)))
        out_rows.append(out_row)

    return GrayImage(out_rows)


def stretch_contrast(
    image: GrayImage,
    low_quantile: float = 0.02,
    high_quantile: float = 0.98,
    out_low: float = 10.0,
    out_high: float = 250.0,
) -> GrayImage:
    """Apply non-destructive linear percentile contrast stretching."""
    flat: list[float] = []
    for row in image.pixels:
        flat.extend(row)
    flat.sort()

    p_low = _percentile(flat, low_quantile)
    p_high = _percentile(flat, high_quantile)
    span = p_high - p_low
    if span < 1e-6:
        return image.copy()

    scale = (out_high - out_low) / span
    out_rows = [
        [_clamp_pixel((v - p_low) * scale + out_low) for v in row]
        for row in image.pixels
    ]
    return GrayImage(out_rows)


def preprocess(
    image: GrayImage | Any,
    profile: ProfileName | str = "receipt",
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Run quality diagnosis and adaptive, non-destructive preprocessing.

    - Never mutates the input image.
    - Clean images take a zero-transform no-op fast path (``is_noop=True``).
    - Never performs OCR, generative synthesis, or hard binarization.
    """
    gray = GrayImage.from_any(image)
    cfg = config if config is not None else PreprocessConfig.for_profile(profile)
    report = diagnose(gray, config=cfg)

    is_flat = "flat_image" in report.unreadable_reasons
    need_deskew = (not is_flat) and (abs(report.skew_angle_deg) > cfg.skew_threshold_deg)
    need_illum = (not is_flat) and (
        report.illumination_unevenness > cfg.illumination_threshold
    )
    need_contrast = (not is_flat) and (report.contrast_score < cfg.contrast_threshold)

    if not (need_deskew or need_illum or need_contrast):
        return PreprocessResult(
            image=gray.copy(),
            report=report,
            transform=AffineTransform2D.identity(),
            applied_transforms=(),
            is_noop=True,
        )

    current = gray.copy()
    total_transform = AffineTransform2D.identity()
    applied: list[str] = []

    if need_deskew:
        current, deskew_tf = deskew_image(
            current,
            report.skew_angle_deg,
            fill_value=cfg.fill_value,
        )
        total_transform = total_transform.compose(deskew_tf)
        applied.append("deskew")

    if need_illum:
        current = normalize_illumination(current)
        applied.append("illumination_normalization")
        if not need_contrast:
            need_contrast = compute_contrast_score(current) < cfg.contrast_threshold

    if need_contrast:
        current = stretch_contrast(current)
        applied.append("contrast_normalization")

    return PreprocessResult(
        image=current,
        report=report,
        transform=total_transform,
        applied_transforms=tuple(applied),
        is_noop=False,
    )


def compute_crop_plan(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    estimated_char_height_px: float,
    target_char_height_px: float = 28.0,
    padding_ratio: float = 0.15,
    min_scale: float = 0.5,
    max_scale: float = 4.0,
) -> CropPlan:
    """Compute an optimal padded crop box, scale factor, and exact coordinate transform.

    Args:
        bbox: ``(x0, y0, x1, y1)`` region of interest in original image coordinates.
        image_size: ``(width, height)`` of the source image.
        estimated_char_height_px: Estimated character/line height in the source image.
        target_char_height_px: Desired character height in the cropped output.
        padding_ratio: Fractional padding added around ``bbox`` on each side.
        min_scale: Lower bound for the computed scale factor.
        max_scale: Upper bound for the computed scale factor.

    Returns:
        ``CropPlan`` whose ``transform`` maps original image coordinates to
        cropped-and-scaled output coordinates, and ``transform.inverse`` maps
        output coordinates back to the original image.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    width, height = int(image_size[0]), int(image_size[1])

    if width <= 0 or height <= 0:
        raise ValueError("image_size dimensions must be positive.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox must satisfy x1 > x0 and y1 > y0.")
    if target_char_height_px <= 0.0:
        raise ValueError("target_char_height_px must be positive.")

    bw = x1 - x0
    bh = y1 - y0
    pad_x = max(2.0, bw * padding_ratio)
    pad_y = max(2.0, bh * padding_ratio)

    px0 = max(0.0, x0 - pad_x)
    py0 = max(0.0, y0 - pad_y)
    px1 = min(float(width), x1 + pad_x)
    py1 = min(float(height), y1 + pad_y)

    if px1 <= px0 or py1 <= py0:
        raise ValueError("Clamped crop bounding box is empty.")

    if estimated_char_height_px > 0.0:
        raw_scale = target_char_height_px / estimated_char_height_px
    else:
        raw_scale = 1.0
    scale = max(min_scale, min(max_scale, raw_scale))

    out_w = max(1, int(math.ceil((px1 - px0) * scale)))
    out_h = max(1, int(math.ceil((py1 - py0) * scale)))

    transform = AffineTransform2D.translation(-px0, -py0).compose(
        AffineTransform2D.scale(scale, scale)
    )

    return CropPlan(
        bbox=(px0, py0, px1, py1),
        scale_factor=scale,
        output_size=(out_w, out_h),
        transform=transform,
    )


def apply_crop_plan(
    image: GrayImage | Any,
    plan: CropPlan,
    fill_value: float = 255.0,
) -> GrayImage:
    """Render a ``CropPlan`` using bilinear interpolation."""
    gray = GrayImage.from_any(image)
    out_w, out_h = plan.output_size
    inv = plan.transform.invert()

    out_rows: list[list[float]] = []
    for y_out in range(out_h):
        row: list[float] = []
        base_x = inv.b * y_out + inv.tx
        base_y = inv.d * y_out + inv.ty
        for x_out in range(out_w):
            src_x = inv.a * x_out + base_x
            src_y = inv.c * x_out + base_y
            row.append(gray.sample_bilinear(src_x, src_y, fill=fill_value))
        out_rows.append(row)

    return GrayImage(out_rows)
