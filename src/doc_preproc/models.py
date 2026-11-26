"""Core data models for lightweight document image representation and results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

from .geometry import AffineTransform2D

ProfileName = Literal["receipt", "invoice", "crop"]


def _clamp_pixel(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return float(value)


def _rgb_to_luma(pixel: Sequence[float | int]) -> float:
    if len(pixel) == 1:
        return _clamp_pixel(float(pixel[0]))
    if len(pixel) >= 3:
        r = float(pixel[0])
        g = float(pixel[1])
        b = float(pixel[2])
        return _clamp_pixel(0.299 * r + 0.587 * g + 0.114 * b)
    raise ValueError(f"Unsupported channel count: {len(pixel)}")


class GrayImage:
    """Lightweight pure-Python 2D grayscale image container.

    Pixel intensities are stored in row-major order ``pixels[y][x]`` as floats
    in ``[0.0, 255.0]``.
    """

    __slots__ = ("width", "height", "pixels")

    def __init__(self, pixels: Sequence[Sequence[float | int]]) -> None:
        if not pixels:
            raise ValueError("Image must have at least one row.")
        first_row_len = len(pixels[0])
        if first_row_len == 0:
            raise ValueError("Image rows must have at least one column.")

        normalized: list[list[float]] = []
        for row in pixels:
            if len(row) != first_row_len:
                raise ValueError("All image rows must have equal length.")
            normalized.append([_clamp_pixel(float(v)) for v in row])

        self.height: int = len(normalized)
        self.width: int = first_row_len
        self.pixels: list[list[float]] = normalized

    @classmethod
    def from_nested(cls, rows: Sequence[Sequence[Any]]) -> GrayImage:
        """Create a ``GrayImage`` from a 2D grayscale or 3D RGB nested sequence."""
        if not rows:
            raise ValueError("Image must have at least one row.")
        first_row = rows[0]
        if not first_row:
            raise ValueError("Image rows must have at least one column.")

        first_cell = first_row[0]
        if isinstance(first_cell, (list, tuple)):
            gray_rows = [
                [_rgb_to_luma(cell) for cell in row]
                for row in rows
            ]
            return cls(gray_rows)
        return cls(rows)

    @classmethod
    def from_any(cls, image: GrayImage | Any) -> GrayImage:
        """Accept a ``GrayImage``, 2D/3D numpy-like array, or nested sequence."""
        if isinstance(image, cls):
            return image

        # Duck-typed support for numpy arrays without importing numpy.
        if hasattr(image, "tolist") and hasattr(image, "shape"):
            shape = getattr(image, "shape")
            if len(shape) not in (2, 3):
                raise ValueError(
                    f"Expected a 2D grayscale or 3D RGB array, got shape {shape!r}."
                )
            nested = image.tolist()
            return cls.from_nested(nested)

        if isinstance(image, Sequence):
            return cls.from_nested(image)

        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            "Expected GrayImage, 2D/3D numpy array, or nested sequence."
        )

    @classmethod
    def filled(cls, width: int, height: int, value: float = 255.0) -> GrayImage:
        """Create a constant-intensity image of size ``(width, height)``."""
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive.")
        v = _clamp_pixel(value)
        return cls([[v] * width for _ in range(height)])

    def to_nested(self) -> list[list[float]]:
        """Return a deep copy of pixel data as nested Python lists."""
        return [row[:] for row in self.pixels]

    def copy(self) -> GrayImage:
        """Return an independent deep copy of this image."""
        return GrayImage(self.to_nested())

    def at(self, x: int, y: int) -> float:
        """Return the pixel value at integer coordinates ``(x, y)``."""
        return self.pixels[y][x]

    def sample_bilinear(self, x: float, y: float, fill: float = 255.0) -> float:
        """Sample the image at floating-point coordinates ``(x, y)`` using bilinear interpolation."""
        if x < 0.0 or y < 0.0 or x > self.width - 1 or y > self.height - 1:
            return fill

        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)

        fx = x - x0
        fy = y - y0

        row0 = self.pixels[y0]
        row1 = self.pixels[y1]

        top = row0[x0] * (1.0 - fx) + row0[x1] * fx
        bottom = row1[x0] * (1.0 - fx) + row1[x1] * fx
        return top * (1.0 - fy) + bottom * fy

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GrayImage):
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.pixels == other.pixels
        )

    def __repr__(self) -> str:
        return f"GrayImage(width={self.width}, height={self.height})"


@dataclass(frozen=True)
class QualityReport:
    """Diagnostic quality metrics for a document image."""

    width: int
    height: int
    skew_angle_deg: float
    blur_score: float
    contrast_score: float
    illumination_unevenness: float
    estimated_char_height_px: float
    aspect_ratio: float
    unreadable_flag: bool
    unreadable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessConfig:
    """Thresholds and parameters controlling quality diagnosis and preprocessing."""

    skew_threshold_deg: float = 1.0
    max_skew_search_deg: float = 10.0
    skew_step_deg: float = 0.5
    contrast_threshold: float = 28.0
    illumination_threshold: float = 35.0
    blur_unreadable_threshold: float = 5.0
    min_dynamic_range: float = 12.0
    fill_value: float = 255.0

    @classmethod
    def for_profile(cls, profile: str) -> PreprocessConfig:
        """Return tuned default configuration for a named document profile."""
        if profile == "receipt":
            return cls(
                skew_threshold_deg=1.0,
                max_skew_search_deg=10.0,
                skew_step_deg=0.5,
                contrast_threshold=28.0,
                illumination_threshold=30.0,
            )
        if profile == "invoice":
            return cls(
                skew_threshold_deg=0.8,
                max_skew_search_deg=10.0,
                skew_step_deg=0.5,
                contrast_threshold=26.0,
                illumination_threshold=35.0,
            )
        if profile == "crop":
            return cls(
                skew_threshold_deg=0.75,
                max_skew_search_deg=10.0,
                skew_step_deg=0.5,
                contrast_threshold=25.0,
                illumination_threshold=32.0,
            )
        raise ValueError(
            f"Unknown profile {profile!r}. Expected 'receipt', 'invoice', or 'crop'."
        )


@dataclass(frozen=True)
class CropPlan:
    """Computed local re-read crop geometry and exact coordinate transform."""

    bbox: tuple[float, float, float, float]
    scale_factor: float
    output_size: tuple[int, int]
    transform: AffineTransform2D


@dataclass(frozen=True)
class PreprocessResult:
    """Output of adaptive document preprocessing."""

    image: GrayImage
    report: QualityReport
    transform: AffineTransform2D
    applied_transforms: tuple[str, ...]
    is_noop: bool
