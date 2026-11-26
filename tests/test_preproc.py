"""Unit tests for doc_preproc quality diagnosis, preprocessing, and geometry."""

from __future__ import annotations

import math
import unittest

from doc_preproc import (
    AffineTransform2D,
    GrayImage,
    PreprocessConfig,
    apply_crop_plan,
    compute_contrast_score,
    compute_crop_plan,
    compute_illumination_unevenness,
    diagnose,
    preprocess,
)


def _make_clean_text_image(
    width: int = 160,
    height: int = 120,
    bg: float = 248.0,
    ink: float = 25.0,
    line_height: int = 8,
    line_spacing: int = 18,
) -> GrayImage:
    """Create a synthetic document image with crisp horizontal text-like bands."""
    rows = [[bg for _ in range(width)] for _ in range(height)]
    y = 14
    while y + line_height < height - 10:
        for yy in range(y, y + line_height):
            for x in range(14, width - 14):
                # Simulated character strokes with word gaps
                if (x % 22) < 17 and ((x % 5) != 0 or (yy - y) in (0, line_height - 1)):
                    rows[yy][x] = ink
        y += line_spacing
    return GrayImage(rows)


def _rotate_image(
    image: GrayImage,
    angle_deg: float,
    fill: float = 248.0,
) -> tuple[GrayImage, AffineTransform2D]:
    """Rotate an image clockwise by ``angle_deg`` around its center."""
    cx = (image.width - 1) / 2.0
    cy = (image.height - 1) / 2.0
    tf = AffineTransform2D.rotation(angle_deg, cx, cy)
    inv = tf.invert()
    out = []
    for y in range(image.height):
        row = []
        for x in range(image.width):
            sx, sy = inv.forward(float(x), float(y))
            row.append(image.sample_bilinear(sx, sy, fill=fill))
        out.append(row)
    return GrayImage(out), tf


class _DummyArray:
    """Minimal duck-typed numpy-like array for zero-dependency compatibility tests."""

    def __init__(self, data: list, shape: tuple[int, ...]) -> None:
        self._data = data
        self.shape = shape

    def tolist(self) -> list:
        return self._data


class TestDocPreproc(unittest.TestCase):
    def test_clean_image_takes_noop_fast_path_and_is_non_destructive(self) -> None:
        clean = _make_clean_text_image()
        original_snapshot = clean.to_nested()

        result = preprocess(clean, profile="invoice")

        self.assertTrue(result.is_noop)
        self.assertEqual(result.applied_transforms, ())
        self.assertTrue(result.transform.is_identity())
        self.assertFalse(result.report.unreadable_flag)
        self.assertEqual(result.report.unreadable_reasons, ())
        self.assertLessEqual(abs(result.report.skew_angle_deg), 0.5)
        self.assertGreater(result.report.estimated_char_height_px, 4.0)

        # Input image must remain untouched and decoupled from output.
        result.image.pixels[0][0] = 0.0
        self.assertEqual(clean.to_nested(), original_snapshot)

    def test_skewed_image_deskew_and_coordinate_roundtrip(self) -> None:
        base = _make_clean_text_image(width=180, height=130)
        true_skew_deg = 5.0
        skewed, rot_tf = _rotate_image(base, true_skew_deg)
        skewed_snapshot = skewed.to_nested()

        report = diagnose(skewed)
        self.assertAlmostEqual(report.skew_angle_deg, true_skew_deg, delta=0.6)

        result = preprocess(skewed, profile="receipt")
        self.assertFalse(result.is_noop)
        self.assertIn("deskew", result.applied_transforms)
        self.assertEqual(skewed.to_nested(), skewed_snapshot)

        # Exact algebraic round-trip error < 1e-6 across multiple coordinates.
        for x, y in [(0.0, 0.0), (42.5, 19.25), (90.0, 65.0), (179.0, 129.0)]:
            fx, fy = result.transform.forward(x, y)
            rx, ry = result.transform.inverse(fx, fy)
            self.assertLess(math.hypot(rx - x, ry - y), 1e-6)

        # Verify a synthetic landmark rotated into the skewed image maps back
        # close to its original unrotated position after deskew.
        landmark_orig = (40.0, 35.0)
        landmark_in_skewed = rot_tf.forward(*landmark_orig)
        landmark_deskewed = result.transform.forward(*landmark_in_skewed)
        err_px = math.hypot(
            landmark_deskewed[0] - landmark_orig[0],
            landmark_deskewed[1] - landmark_orig[1],
        )
        self.assertLess(err_px, 1.0)

        # Re-diagnosing the deskewed image should show near-zero residual skew.
        post_report = diagnose(result.image)
        self.assertLess(abs(post_report.skew_angle_deg), 1.0)

    def test_shadowed_low_contrast_image_triggers_non_destructive_normalization(self) -> None:
        base = _make_clean_text_image(width=140, height=100, bg=200.0, ink=145.0)
        # Apply a strong horizontal shadow gradient (dark left, brighter right).
        shadowed_rows = []
        for y in range(base.height):
            row = []
            for x in range(base.width):
                shade = 0.45 + 0.55 * (x / float(base.width - 1))
                row.append(base.pixels[y][x] * shade)
            shadowed_rows.append(row)
        shadowed = GrayImage(shadowed_rows)
        before_snapshot = shadowed.to_nested()

        pre_contrast = compute_contrast_score(shadowed)
        pre_uneven = compute_illumination_unevenness(shadowed)
        self.assertGreater(pre_uneven, 40.0)

        result = preprocess(shadowed, profile="receipt")

        self.assertFalse(result.is_noop)
        self.assertIn("illumination_normalization", result.applied_transforms)
        self.assertIn("contrast_normalization", result.applied_transforms)
        self.assertEqual(shadowed.to_nested(), before_snapshot)

        post_contrast = compute_contrast_score(result.image)
        post_uneven = compute_illumination_unevenness(result.image)
        self.assertLess(post_uneven, pre_uneven * 0.5)
        self.assertGreater(post_contrast, pre_contrast)

        # Non-destructive check: image must retain intermediate grayscale tones,
        # not hard-binarized 0/255 values only.
        unique_buckets = {
            int(v // 10) for row in result.image.pixels for v in row
        }
        self.assertGreater(len(unique_buckets), 3)

    def test_low_contrast_even_lighting_triggers_contrast_only(self) -> None:
        low_contrast = _make_clean_text_image(width=120, height=90, bg=190.0, ink=145.0)
        pre_contrast = compute_contrast_score(low_contrast)
        self.assertLess(pre_contrast, 25.0)

        result = preprocess(low_contrast, profile="invoice")
        self.assertFalse(result.is_noop)
        self.assertEqual(result.applied_transforms, ("contrast_normalization",))
        self.assertTrue(result.transform.is_identity())
        self.assertGreater(compute_contrast_score(result.image), pre_contrast * 2.0)

    def test_tall_receipt_aspect_ratio_and_crop_plan_mapping(self) -> None:
        tall_receipt = _make_clean_text_image(
            width=80,
            height=240,
            line_height=10,
            line_spacing=20,
        )
        report = diagnose(tall_receipt)
        self.assertAlmostEqual(report.aspect_ratio, 3.0, places=5)

        bbox = (10.0, 50.0, 70.0, 110.0)
        plan = compute_crop_plan(
            bbox=bbox,
            image_size=(tall_receipt.width, tall_receipt.height),
            estimated_char_height_px=14.0,
            target_char_height_px=28.0,
        )

        self.assertAlmostEqual(plan.scale_factor, 2.0, places=6)
        px0, py0, px1, py1 = plan.bbox
        self.assertLess(px0, bbox[0])
        self.assertLess(py0, bbox[1])
        self.assertGreater(px1, bbox[2])
        self.assertGreater(py1, bbox[3])
        self.assertGreaterEqual(px0, 0.0)
        self.assertLessEqual(px1, float(tall_receipt.width))

        # Exact inverse mapping at crop corners and interior point.
        for pt in [(px0, py0), (px1, py1), (35.5, 82.25)]:
            cx, cy = plan.transform.forward(*pt)
            rx, ry = plan.transform.inverse(cx, cy)
            self.assertLess(math.hypot(rx - pt[0], ry - pt[1]), 1e-9)

        self.assertEqual(plan.transform.forward(px0, py0), (0.0, 0.0))

        cropped = apply_crop_plan(tall_receipt, plan)
        self.assertEqual((cropped.width, cropped.height), plan.output_size)

    def test_unreadable_advisory_flags_on_extreme_images(self) -> None:
        black = GrayImage.filled(64, 64, value=2.0)
        rep_black = diagnose(black)
        self.assertTrue(rep_black.unreadable_flag)
        self.assertIn("crushed_blacks", rep_black.unreadable_reasons)
        self.assertIn("flat_image", rep_black.unreadable_reasons)

        white = GrayImage.filled(64, 64, value=253.0)
        rep_white = diagnose(white)
        self.assertTrue(rep_white.unreadable_flag)
        self.assertIn("blown_highlights", rep_white.unreadable_reasons)
        self.assertIn("flat_image", rep_white.unreadable_reasons)

        flat_gray = GrayImage.filled(64, 64, value=128.0)
        rep_flat = diagnose(flat_gray)
        self.assertTrue(rep_flat.unreadable_flag)
        self.assertIn("flat_image", rep_flat.unreadable_reasons)
        self.assertIn("extreme_blur", rep_flat.unreadable_reasons)

        # Advisory only: preprocess must never raise on unreadable images.
        res_black = preprocess(black)
        self.assertTrue(res_black.report.unreadable_flag)
        self.assertTrue(res_black.is_noop)

    def test_duck_typed_array_and_rgb_support_and_compose(self) -> None:
        arr_2d = _DummyArray([[10, 20], [30, 40]], shape=(2, 2))
        g2 = GrayImage.from_any(arr_2d)
        self.assertEqual((g2.width, g2.height), (2, 2))
        self.assertEqual(g2.at(1, 0), 20.0)

        rgb_3d = _DummyArray(
            [[[100, 100, 100], [200, 200, 200]]],
            shape=(1, 2, 3),
        )
        g3 = GrayImage.from_any(rgb_3d)
        self.assertAlmostEqual(g3.at(0, 0), 100.0, places=5)
        self.assertAlmostEqual(g3.at(1, 0), 200.0, places=5)

        t1 = AffineTransform2D.translation(5.0, -3.0)
        t2 = AffineTransform2D.rotation(15.0, cx=10.0, cy=10.0)
        t3 = AffineTransform2D.scale(2.0, 0.5)
        composed = t1.compose(t2).compose(t3)

        p = (12.0, 7.0)
        step_by_step = t3.forward(*t2.forward(*t1.forward(*p)))
        direct = composed.forward(*p)
        self.assertLess(math.hypot(direct[0] - step_by_step[0], direct[1] - step_by_step[1]), 1e-9)

        recovered = composed.inverse(*direct)
        self.assertLess(math.hypot(recovered[0] - p[0], recovered[1] - p[1]), 1e-9)

        with self.assertRaises(ValueError):
            PreprocessConfig.for_profile("unknown")


if __name__ == "__main__":
    unittest.main()
