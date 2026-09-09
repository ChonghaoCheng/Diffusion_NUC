from __future__ import annotations

import torch
from torch import nn

from diffusion_coverage.models.path_noise import sample_path_noise


@torch.no_grad()
def heun_sample(
    vector_field: nn.Module,
    surface: torch.Tensor,
    condition: torch.Tensor,
    *,
    num_waypoints: int,
    num_steps: int = 32,
    initial_noise: torch.Tensor | None = None,
    noise_smoothing_sigma: float = 0.0,
    noise_smoothing_fraction: float = 0.0,
    path_mask: torch.Tensor | None = None,
    path_arclength: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the learned ODE from Gaussian noise with Heun's method."""

    if num_steps < 1 or num_waypoints < 2:
        raise ValueError("num_steps and num_waypoints must be positive")
    batch_size = surface.shape[0]
    path_dim = int(getattr(getattr(vector_field, "config", None), "path_dim", 3))
    if path_mask is not None and path_mask.shape != (batch_size, num_waypoints):
        raise ValueError("path_mask must have shape [B, M]")
    if path_arclength is not None and path_arclength.shape != (batch_size, num_waypoints):
        raise ValueError("path_arclength must have shape [B, M]")
    x = (
        sample_path_noise(
            torch.empty(
                batch_size, num_waypoints, path_dim,
                device=surface.device, dtype=surface.dtype,
            ),
            smoothing_sigma=noise_smoothing_sigma,
            smoothing_fraction=noise_smoothing_fraction,
            path_mask=path_mask,
        )
        if initial_noise is None
        else initial_noise.clone()
    )
    if x.shape != (batch_size, num_waypoints, path_dim):
        raise ValueError("initial_noise has the wrong shape")
    if path_mask is not None:
        x = x.masked_fill(~path_mask[..., None], 0.0)
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        t0 = torch.full((batch_size,), step * step_size, device=x.device, dtype=x.dtype)
        t1 = torch.full((batch_size,), (step + 1) * step_size, device=x.device, dtype=x.dtype)
        velocity0 = vector_field(
            x, t0, surface, condition,
            path_mask=path_mask, path_arclength=path_arclength,
        )
        predictor = x + step_size * velocity0
        if path_mask is not None:
            predictor = predictor.masked_fill(~path_mask[..., None], 0.0)
        velocity1 = vector_field(
            predictor, t1, surface, condition,
            path_mask=path_mask, path_arclength=path_arclength,
        )
        x = x + 0.5 * step_size * (velocity0 + velocity1)
        if path_mask is not None:
            x = x.masked_fill(~path_mask[..., None], 0.0)
    return x
