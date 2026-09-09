from __future__ import annotations

import torch
from torch.nn import functional as functional


def sample_path_noise(
    reference: torch.Tensor,
    *,
    smoothing_sigma: float = 0.0,
    smoothing_fraction: float = 0.0,
    path_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample unit-scale Gaussian path noise with optional waypoint correlation."""

    noise = torch.randn_like(reference)
    if reference.ndim != 3 or reference.shape[-1] < 1:
        raise ValueError("reference must have shape [B, M, D] with D >= 1")
    if path_mask is not None:
        if path_mask.shape != reference.shape[:2]:
            raise ValueError("path_mask must have shape [B, M]")
        _validate_prefix_mask(path_mask)
    if smoothing_sigma > 0.0 and smoothing_fraction > 0.0:
        raise ValueError("specify smoothing_sigma or smoothing_fraction, not both")
    if smoothing_fraction < 0.0:
        raise ValueError("smoothing_fraction must be nonnegative")
    if smoothing_sigma <= 0.0 and smoothing_fraction <= 0.0:
        return noise if path_mask is None else noise.masked_fill(~path_mask[..., None], 0.0)
    if path_mask is not None:
        smoothed = torch.zeros_like(noise)
        for batch_index, count_tensor in enumerate(path_mask.sum(dim=1)):
            count = int(count_tensor)
            sigma = smoothing_fraction * count if smoothing_fraction > 0.0 else smoothing_sigma
            smoothed[batch_index : batch_index + 1, :count] = _smooth_noise(
                noise[batch_index : batch_index + 1, :count], sigma
            )
        return smoothed
    sigma = smoothing_fraction * reference.shape[1] if smoothing_fraction > 0.0 else smoothing_sigma
    return _smooth_noise(noise, sigma)


def _smooth_noise(noise: torch.Tensor, smoothing_sigma: float) -> torch.Tensor:
    radius = min(noise.shape[1] - 1, max(1, int(round(3.0 * smoothing_sigma))))
    coordinates = torch.arange(-radius, radius + 1, device=noise.device, dtype=noise.dtype)
    kernel = torch.exp(-0.5 * (coordinates / smoothing_sigma).square())
    kernel = kernel / kernel.square().sum().sqrt()
    channels = noise.shape[-1]
    weights = kernel.view(1, 1, -1).repeat(channels, 1, 1)
    values = noise.transpose(1, 2)
    padding_mode = "reflect" if noise.shape[1] > radius else "replicate"
    values = functional.pad(values, (radius, radius), mode=padding_mode)
    return functional.conv1d(values, weights, groups=channels).transpose(1, 2)


def _validate_prefix_mask(path_mask: torch.Tensor) -> None:
    if path_mask.dtype != torch.bool:
        raise ValueError("path_mask must be boolean")
    if (path_mask[:, 1:] & ~path_mask[:, :-1]).any():
        raise ValueError("path_mask must contain one contiguous valid prefix")
    if (path_mask.sum(dim=1) < 2).any():
        raise ValueError("every path must contain at least two valid tokens")
