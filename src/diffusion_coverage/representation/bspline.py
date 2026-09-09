from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class BSplineApproximation:
    control_points: np.ndarray
    knots: np.ndarray
    degree: int

    def evaluate(self, parameter: np.ndarray) -> np.ndarray:
        values = np.asarray(parameter, dtype=np.float64)
        return BSpline(self.knots, self.control_points, self.degree, extrapolate=False)(values)


def fit_fixed_control_bspline(
    points: np.ndarray,
    normalized_arclength: np.ndarray,
    *,
    num_control_points: int,
    degree: int = 3,
) -> BSplineApproximation:
    """Least-squares clamped B-spline with exact open-path endpoints."""

    samples = np.asarray(points, dtype=np.float64)
    parameter = np.asarray(normalized_arclength, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or parameter.shape != (len(samples),):
        raise ValueError("points and normalized_arclength have incompatible shapes")
    if degree < 1 or num_control_points < degree + 1 or num_control_points > len(samples):
        raise ValueError("invalid number of B-spline control points")
    internal_count = num_control_points - degree - 1
    internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
    design = BSpline.design_matrix(parameter, knots, degree).toarray()
    control = np.empty((num_control_points, 3), dtype=np.float64)
    control[0] = samples[0]
    control[-1] = samples[-1]
    right_hand_side = samples - design[:, :1] * control[0] - design[:, -1:] * control[-1]
    control[1:-1], _, _, _ = np.linalg.lstsq(design[:, 1:-1], right_hand_side, rcond=None)
    return BSplineApproximation(control_points=control, knots=knots, degree=degree)
