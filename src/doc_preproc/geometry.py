"""Exact 2D affine coordinate transformations for document preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AffineTransform2D:
    """Immutable 2x3 affine transformation matrix.

    Maps a point ``(x, y)`` to ``(x', y')`` via::

        x' = a * x + b * y + tx
        y' = c * x + d * y + ty

    Image coordinates follow the standard top-left origin convention:
    ``x`` increases to the right and ``y`` increases downward.
    """

    a: float = 1.0
    b: float = 0.0
    tx: float = 0.0
    c: float = 0.0
    d: float = 1.0
    ty: float = 0.0

    @classmethod
    def identity(cls) -> AffineTransform2D:
        """Return the identity transform."""
        return cls(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @classmethod
    def translation(cls, dx: float, dy: float) -> AffineTransform2D:
        """Return a pure translation by ``(dx, dy)``."""
        return cls(1.0, 0.0, float(dx), 0.0, 1.0, float(dy))

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> AffineTransform2D:
        """Return a scaling transform about the origin."""
        sy_val = float(sx if sy is None else sy)
        return cls(float(sx), 0.0, 0.0, 0.0, sy_val, 0.0)

    @classmethod
    def rotation(
        cls,
        angle_deg: float,
        cx: float = 0.0,
        cy: float = 0.0,
    ) -> AffineTransform2D:
        """Return a rotation by ``angle_deg`` around ``(cx, cy)``.

        In image coordinates (``x`` right, ``y`` down), a positive angle
        rotates clockwise: ``(1, 0) -> (cos(a), sin(a))``.
        """
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        tx = cx - (cos_a * cx - sin_a * cy)
        ty = cy - (sin_a * cx + cos_a * cy)
        return cls(cos_a, -sin_a, tx, sin_a, cos_a, ty)

    def determinant(self) -> float:
        """Return the determinant of the 2x2 linear component."""
        return self.a * self.d - self.b * self.c

    def forward(self, x: float, y: float) -> tuple[float, float]:
        """Map a point ``(x, y)`` from source coordinates to target coordinates."""
        return (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )

    def invert(self) -> AffineTransform2D:
        """Return the exact inverse ``AffineTransform2D``."""
        det = self.determinant()
        if abs(det) < 1e-12:
            raise ValueError("Affine transform is singular and cannot be inverted.")
        inv_a = self.d / det
        inv_b = -self.b / det
        inv_c = -self.c / det
        inv_d = self.a / det
        inv_tx = -(inv_a * self.tx + inv_b * self.ty)
        inv_ty = -(inv_c * self.tx + inv_d * self.ty)
        return AffineTransform2D(inv_a, inv_b, inv_tx, inv_c, inv_d, inv_ty)

    def inverse(self, x_prime: float, y_prime: float) -> tuple[float, float]:
        """Map a point ``(x', y')`` from target coordinates back to source coordinates."""
        det = self.determinant()
        if abs(det) < 1e-12:
            raise ValueError("Affine transform is singular and cannot be inverted.")
        dx = x_prime - self.tx
        dy = y_prime - self.ty
        return (
            (self.d * dx - self.b * dy) / det,
            (-self.c * dx + self.a * dy) / det,
        )

    def compose(self, other: AffineTransform2D) -> AffineTransform2D:
        """Compose ``self`` followed by ``other``.

        The resulting transform satisfies::

            composed.forward(x, y) == other.forward(*self.forward(x, y))
        """
        return AffineTransform2D(
            a=other.a * self.a + other.b * self.c,
            b=other.a * self.b + other.b * self.d,
            tx=other.a * self.tx + other.b * self.ty + other.tx,
            c=other.c * self.a + other.d * self.c,
            d=other.c * self.b + other.d * self.d,
            ty=other.c * self.tx + other.d * self.ty + other.ty,
        )

    def is_identity(self, tol: float = 1e-12) -> bool:
        """Return True if the matrix matches the identity within ``tol``."""
        return (
            abs(self.a - 1.0) <= tol
            and abs(self.b) <= tol
            and abs(self.tx) <= tol
            and abs(self.c) <= tol
            and abs(self.d - 1.0) <= tol
            and abs(self.ty) <= tol
        )

    def as_matrix(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return the 2x3 matrix as nested tuples."""
        return (
            (self.a, self.b, self.tx),
            (self.c, self.d, self.ty),
        )
