"""Document image quality diagnosis metrics."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .models import GrayImage, PreprocessConfig, QualityReport


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Compute linear-interpolated percentile from a pre-sorted sequence."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _image_stats(gray: GrayImage) -> tuple[float, float, float, float, float, float]:
    """Return (mean, std, min_val, max_val, p5, p95) for pixel intensities."""
    flat: list[float] = []
    for row in gray.pixels:
        flat.extend(row)
    n = len(flat)
    mean = sum(flat) / n
    var = sum((v - mean) * (v - mean) for v in flat) / n
    std = math.sqrt(var)
    flat.sort()
    return (
        mean,
        std,
        flat[0],
        flat[-1],
        _percentile(flat, 0.05),
        _percentile(flat, 0.95),
    )


def compute_blur_score(image: GrayImage | Any) -> float:
    """Estimate sharpness via variance of the discrete 2D Laplacian.

    Higher values indicate sharper edges; values near zero indicate
    severe blur or flat/featureless content.
    """
    gray = GrayImage.from_any(image)
    w, h = gray.width, gray.height
    if w < 3 or h < 3:
        return 0.0

    pixels = gray.pixels
    count = (w - 2) * (h - 2)
    sum_l = 0.0
    sum_l2 = 0.0
    for y in range(1, h - 1):
        row_prev = pixels[y - 1]
        row_curr = pixels[y]
        row_next = pixels[y + 1]
        for x in range(1, w - 1):
            lap = (
                row_prev[x]
                + row_next[x]
                + row_curr[x - 1]
                + row_curr[x + 1]
                - 4.0 * row_curr[x]
            )
            sum_l += lap
            sum_l2 += lap * lap

    mean_l = sum_l / count
    var_l = (sum_l2 / count) - (mean_l * mean_l)
    return max(0.0, var_l)


def compute_contrast_score(image: GrayImage | Any) -> float:
    """Compute contrast as the standard deviation of grayscale intensities."""
    gray = GrayImage.from_any(image)
    _, std, _, _, _, _ = _image_stats(gray)
    return std


def compute_illumination_unevenness(
    image: GrayImage | Any,
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> float:
    """Measure spatial illumination gradient across a grid of blocks.

    Uses the 85th percentile inside each block as a robust local background
    estimate so that text density differences do not masquerade as shadows.
    Returns ``max(block_bg) - min(block_bg)``.
    """
    gray = GrayImage.from_any(image)
    w, h = gray.width, gray.height
    rows = min(grid_rows, h)
    cols = min(grid_cols, w)
    if rows <= 1 and cols <= 1:
        return 0.0

    block_bgs: list[float] = []
    pixels = gray.pixels
    for r in range(rows):
        y0 = (r * h) // rows
        y1 = ((r + 1) * h) // rows
        if y1 <= y0:
            continue
        for c in range(cols):
            x0 = (c * w) // cols
            x1 = ((c + 1) * w) // cols
            if x1 <= x0:
                continue
            vals: list[float] = []
            for y in range(y0, y1):
                vals.extend(pixels[y][x0:x1])
            vals.sort()
            block_bgs.append(_percentile(vals, 0.85))

    if len(block_bgs) < 2:
        return 0.0
    return max(block_bgs) - min(block_bgs)


def estimate_skew_angle(
    image: GrayImage | Any,
    max_angle_deg: float = 10.0,
    step_deg: float = 0.5,
) -> float:
    """Estimate document micro-skew angle in degrees via projection-profile variance.

    Positive angle means horizontal text lines slope downward to the right
    (clockwise tilt in top-left image coordinates). Deskewing rotates by
    ``-skew_angle_deg``.
    """
    gray = GrayImage.from_any(image)
    w, h = gray.width, gray.height
    if w < 8 or h < 8:
        return 0.0

    mean, std, _, _, _, _ = _image_stats(gray)
    if std < 1.0:
        return 0.0

    # Uniform downsampling for speed on large images while preserving aspect ratio.
    stride = max(1, max(w, h) // 160)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    thresh = mean - 0.35 * std
    fg_points: list[tuple[float, float, float]] = []
    pixels = gray.pixels
    for y in range(0, h, stride):
        row = pixels[y]
        dy = (y - cy) / stride
        for x in range(0, w, stride):
            p = row[x]
            if p < thresh:
                fg_points.append(((x - cx) / stride, dy, thresh - p))

    if len(fg_points) < 10:
        fg_points = []
        for y in range(0, h, stride):
            row = pixels[y]
            dy = (y - cy) / stride
            for x in range(0, w, stride):
                p = row[x]
                if p < mean:
                    fg_points.append(((x - cx) / stride, dy, mean - p))

    if len(fg_points) < 10:
        return 0.0

    scaled_w = w / stride
    scaled_h = h / stride
    half_diag = int(math.ceil(math.hypot(scaled_w, scaled_h) / 2.0)) + 2
    num_bins = 2 * half_diag + 4

    def _projection_energy(angle_deg: float) -> float:
        rad = math.radians(angle_deg)
        sin_a = math.sin(rad)
        cos_a = math.cos(rad)
        bins = [0.0] * num_bins
        offset = float(half_diag + 1)
        for dx, dy, weight in fg_points:
            # Project onto rotated y-axis for deskew angle (-angle_deg).
            y_rot = -dx * sin_a + dy * cos_a + offset
            idx = int(math.floor(y_rot))
            if 0 <= idx < num_bins - 1:
                frac = y_rot - idx
                bins[idx] += weight * (1.0 - frac)
                bins[idx + 1] += weight * frac
        return sum(b * b for b in bins)

    # Coarse search
    step = max(0.1, float(step_deg))
    num_steps = int(round((2.0 * max_angle_deg) / step))
    best_angle = 0.0
    best_energy = _projection_energy(0.0)
    zero_energy = best_energy

    for i in range(num_steps + 1):
        angle = -max_angle_deg + i * step
        energy = _projection_energy(angle)
        if energy > best_energy:
            best_energy = energy
            best_angle = angle

    # Fine refinement around best coarse angle
    fine_step = min(0.1, step / 4.0)
    fine_start = max(-max_angle_deg, best_angle - step)
    fine_end = min(max_angle_deg, best_angle + step)
    fine_steps = int(round((fine_end - fine_start) / fine_step))
    for i in range(fine_steps + 1):
        angle = fine_start + i * fine_step
        energy = _projection_energy(angle)
        if energy > best_energy:
            best_energy = energy
            best_angle = angle

    # Prefer exact 0.0 if improvement over 0 deg is negligible.
    if best_energy <= zero_energy * 1.001:
        return 0.0

    return round(best_angle, 3)


def estimate_char_height_px(image: GrayImage | Any) -> float:
    """Estimate typical character/line stroke height in pixels from horizontal runs."""
    gray = GrayImage.from_any(image)
    w, h = gray.width, gray.height
    if w < 4 or h < 4:
        return 0.0

    flat: list[float] = []
    for row in gray.pixels:
        flat.extend(row)
    n = len(flat)
    mean = sum(flat) / n
    std = math.sqrt(sum((v - mean) * (v - mean) for v in flat) / n)
    if std < 2.0:
        return 0.0

    flat.sort()
    bg_est = _percentile(flat, 0.85)
    ink_thresh = bg_est - max(15.0, 0.5 * std)
    min_ink_per_row = max(2, int(0.02 * w))

    runs: list[int] = []
    current_run = 0
    for row in gray.pixels:
        ink_count = sum(1 for v in row if v < ink_thresh)
        if ink_count >= min_ink_per_row:
            current_run += 1
        elif current_run > 0:
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    if not runs:
        return 0.0

    runs.sort()
    mid = len(runs) // 2
    if len(runs) % 2 == 1:
        return float(runs[mid])
    return 0.5 * (runs[mid - 1] + runs[mid])


def diagnose(
    image: GrayImage | Any,
    config: PreprocessConfig | None = None,
) -> QualityReport:
    """Run all lightweight quality diagnostics on a document image.

    Unreadable flags are strictly advisory and never reject an image.
    """
    gray = GrayImage.from_any(image)
    cfg = config if config is not None else PreprocessConfig()

    mean, std, min_val, max_val, p5, p95 = _image_stats(gray)
    blur_score = compute_blur_score(gray)
    contrast_score = std
    illumination_unevenness = compute_illumination_unevenness(gray)
    skew_angle_deg = estimate_skew_angle(
        gray,
        max_angle_deg=cfg.max_skew_search_deg,
        step_deg=cfg.skew_step_deg,
    )
    estimated_char_height_px = estimate_char_height_px(gray)
    aspect_ratio = gray.height / float(gray.width)

    reasons: list[str] = []
    if mean < 15.0 and p95 < 30.0:
        reasons.append("crushed_blacks")
    if mean > 245.0 and p5 > 235.0:
        reasons.append("blown_highlights")
    if contrast_score < 2.0 or (max_val - min_val) < cfg.min_dynamic_range:
        reasons.append("flat_image")
    if blur_score < cfg.blur_unreadable_threshold:
        reasons.append("extreme_blur")

    return QualityReport(
        width=gray.width,
        height=gray.height,
        skew_angle_deg=skew_angle_deg,
        blur_score=blur_score,
        contrast_score=contrast_score,
        illumination_unevenness=illumination_unevenness,
        estimated_char_height_px=estimated_char_height_px,
        aspect_ratio=aspect_ratio,
        unreadable_flag=bool(reasons),
        unreadable_reasons=tuple(reasons),
    )
