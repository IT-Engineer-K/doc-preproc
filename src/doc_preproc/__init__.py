"""doc-preproc: lightweight, non-destructive document image preprocessing."""

from .diagnose import (
    compute_blur_score,
    compute_contrast_score,
    compute_illumination_unevenness,
    diagnose,
    estimate_char_height_px,
    estimate_skew_angle,
)
from .geometry import AffineTransform2D
from .models import (
    CropPlan,
    GrayImage,
    PreprocessConfig,
    PreprocessResult,
    ProfileName,
    QualityReport,
)
from .preprocess import (
    apply_crop_plan,
    compute_crop_plan,
    deskew_image,
    normalize_illumination,
    preprocess,
    stretch_contrast,
)

__all__ = [
    "AffineTransform2D",
    "CropPlan",
    "GrayImage",
    "PreprocessConfig",
    "PreprocessResult",
    "ProfileName",
    "QualityReport",
    "apply_crop_plan",
    "compute_blur_score",
    "compute_contrast_score",
    "compute_crop_plan",
    "compute_illumination_unevenness",
    "deskew_image",
    "diagnose",
    "estimate_char_height_px",
    "estimate_skew_angle",
    "normalize_illumination",
    "preprocess",
    "stretch_contrast",
]

__version__ = "0.1.0"
