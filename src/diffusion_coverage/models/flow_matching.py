from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from diffusion_coverage.models.path_noise import sample_path_noise


@dataclass(frozen=True)
class FlowMatchingLoss:
    total: torch.Tensor
    velocity: torch.Tensor
    smoothness: torch.Tensor
    coverage: torch.Tensor
    excess_length: torch.Tensor
    tangent: torch.Tensor
    surface_consistency: torch.Tensor


class ConditionalFlowMatcher(nn.Module):
    """Conditional flow matching on straight probability paths."""

    def __init__(
        self,
        vector_field: nn.Module,
        *,
        smoothness_weight: float = 0.0,
        coverage_weight: float = 0.0,
        excess_length_weight: float = 0.0,
        tangent_weight: float = 0.0,
        surface_consistency_weight: float = 0.0,
        coverage_max_path_points: int = 256,
        noise_smoothing_sigma: float = 0.0,
        noise_smoothing_fraction: float = 0.0,
        coupling: Literal["independent", "minibatch_ot"] = "independent",
    ) -> None:
        super().__init__()
        if min(
            smoothness_weight,
            coverage_weight,
            excess_length_weight,
            tangent_weight,
            surface_consistency_weight,
        ) < 0.0:
            raise ValueError("loss weights must be nonnegative")
        if coverage_max_path_points < 2:
            raise ValueError("coverage_max_path_points must be at least two")
        if coupling not in {"independent", "minibatch_ot"}:
            raise ValueError("unsupported flow coupling")
        self.vector_field = vector_field
        self.smoothness_weight = smoothness_weight
        self.coverage_weight = coverage_weight
        self.excess_length_weight = excess_length_weight
        self.tangent_weight = tangent_weight
        self.surface_consistency_weight = surface_consistency_weight
        self.coverage_max_path_points = coverage_max_path_points
        self.noise_smoothing_sigma = noise_smoothing_sigma
        self.noise_smoothing_fraction = noise_smoothing_fraction
        self.coupling = coupling

    def loss(
        self,
        target_path: torch.Tensor,
        surface: torch.Tensor,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        path_mask: torch.Tensor | None = None,
        path_arclength: torch.Tensor | None = None,
    ) -> FlowMatchingLoss:
        batch_size = target_path.shape[0]
        x0 = (
            sample_path_noise(
                target_path,
                smoothing_sigma=self.noise_smoothing_sigma,
                smoothing_fraction=self.noise_smoothing_fraction,
                path_mask=path_mask,
            )
            if noise is None else noise
        )
        if path_mask is not None:
            if path_mask.shape != target_path.shape[:2]:
                raise ValueError("path_mask must have shape [B, M]")
            x0 = x0.masked_fill(~path_mask[..., None], 0.0)
        if self.coupling == "minibatch_ot" and batch_size > 1:
            x0 = minibatch_ot_noise(target_path, x0, condition, path_mask)
        t = torch.rand(batch_size, device=target_path.device, dtype=target_path.dtype) if time is None else time
        interpolation = t[:, None, None]
        xt = (1.0 - interpolation) * x0 + interpolation * target_path
        if path_mask is not None:
            xt = xt.masked_fill(~path_mask[..., None], 0.0)
        target_velocity = target_path - x0
        predicted_velocity = self.vector_field(
            xt, t, surface, condition,
            path_mask=path_mask, path_arclength=path_arclength,
        )
        velocity_error = (predicted_velocity.float() - target_velocity.float()).square()
        if path_mask is None:
            velocity_loss = velocity_error.mean()
        else:
            valid = path_mask[..., None].expand_as(velocity_error)
            velocity_loss = velocity_error.masked_select(valid).mean()
        predicted_clean = xt + (1.0 - interpolation) * predicted_velocity
        if predicted_clean.shape[1] >= 3:
            second_difference = predicted_clean[:, 2:] - 2.0 * predicted_clean[:, 1:-1] + predicted_clean[:, :-2]
            smoothness_error = second_difference.float().square()
            if path_mask is None:
                smoothness_loss = smoothness_error.mean()
            else:
                valid_triples = path_mask[:, 2:] & path_mask[:, 1:-1] & path_mask[:, :-2]
                if valid_triples.any():
                    valid = valid_triples[..., None].expand_as(smoothness_error)
                    smoothness_loss = smoothness_error.masked_select(valid).mean()
                else:
                    smoothness_loss = velocity_loss.new_zeros(())
        else:
            smoothness_loss = velocity_loss.new_zeros(())
        coverage_loss = self._coverage_loss(
            predicted_clean, surface, condition, path_mask
        )
        excess_length_loss = self._excess_length_loss(
            predicted_clean, target_path, path_mask
        )
        tangent_loss = self._tangent_loss(predicted_clean, target_path, path_mask)
        surface_consistency_loss = self._surface_consistency_loss(
            predicted_clean, surface, condition, path_mask
        )
        total = (
            velocity_loss
            + self.smoothness_weight * smoothness_loss
            + self.coverage_weight * coverage_loss
            + self.excess_length_weight * excess_length_loss
            + self.tangent_weight * tangent_loss
            + self.surface_consistency_weight * surface_consistency_loss
        )
        return FlowMatchingLoss(
            total=total,
            velocity=velocity_loss,
            smoothness=smoothness_loss,
            coverage=coverage_loss,
            excess_length=excess_length_loss,
            tangent=tangent_loss,
            surface_consistency=surface_consistency_loss,
        )
    def _coverage_loss(
        self,
        predicted_clean: torch.Tensor,
        surface: torch.Tensor,
        condition: torch.Tensor,
        path_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.coverage_weight == 0.0:
            return predicted_clean.new_zeros(())
        losses = []
        for batch_index in range(predicted_clean.shape[0]):
            count = (
                predicted_clean.shape[1]
                if path_mask is None
                else int(path_mask[batch_index].sum())
            )
            path = predicted_clean[batch_index, :count].float()
            if count > self.coverage_max_path_points:
                indices = torch.linspace(
                    0, count - 1, self.coverage_max_path_points,
                    device=path.device,
                ).round().long()
                path = path[indices]
            distances = torch.cdist(surface[batch_index, :, :3].float(), path).amin(dim=1)
            radius = condition[batch_index, 0].float().clamp_min(1e-4)
            tolerance = condition[batch_index, 1].float()
            temperature = (0.1 * radius).clamp_min(1e-3)
            soft_covered = torch.sigmoid((radius - distances) / temperature)
            soft_missed = 1.0 - soft_covered.mean()
            losses.append(torch.relu(soft_missed - tolerance).square())
        return torch.stack(losses).mean()


    def _excess_length_loss(
        self,
        predicted_clean: torch.Tensor,
        target_path: torch.Tensor,
        path_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.excess_length_weight == 0.0:
            return predicted_clean.new_zeros(())
        predicted_segments = torch.linalg.vector_norm(
            predicted_clean[:, 1:].float() - predicted_clean[:, :-1].float(), dim=-1
        )
        target_segments = torch.linalg.vector_norm(
            target_path[:, 1:].float() - target_path[:, :-1].float(), dim=-1
        )
        if path_mask is not None:
            valid = path_mask[:, 1:] & path_mask[:, :-1]
            predicted_segments = predicted_segments.masked_fill(~valid, 0.0)
            target_segments = target_segments.masked_fill(~valid, 0.0)
        predicted_length = predicted_segments.sum(dim=1)
        target_length = target_segments.sum(dim=1).clamp_min(1e-6)
        relative_excess = torch.relu(predicted_length / target_length - 1.0)
        return relative_excess.square().mean()

    def _tangent_loss(
        self,
        predicted_clean: torch.Tensor,
        target_path: torch.Tensor,
        path_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.tangent_weight == 0.0:
            return predicted_clean.new_zeros(())
        predicted = predicted_clean[:, 1:].float() - predicted_clean[:, :-1].float()
        target = target_path[:, 1:].float() - target_path[:, :-1].float()
        target_norm = torch.linalg.vector_norm(target, dim=-1)
        predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
        valid = target_norm > 1e-6
        if path_mask is not None:
            valid = valid & path_mask[:, 1:] & path_mask[:, :-1]
        if not valid.any():
            return predicted_clean.new_zeros(())
        cosine = (predicted * target).sum(dim=-1) / (
            predicted_norm.clamp_min(1e-6) * target_norm.clamp_min(1e-6)
        )
        return (1.0 - cosine.clamp(-1.0, 1.0)).masked_select(valid).mean()

    def _surface_consistency_loss(
        self,
        predicted_clean: torch.Tensor,
        surface: torch.Tensor,
        condition: torch.Tensor,
        path_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.surface_consistency_weight == 0.0:
            return predicted_clean.new_zeros(())
        losses = []
        for batch_index in range(predicted_clean.shape[0]):
            count = (
                predicted_clean.shape[1]
                if path_mask is None
                else int(path_mask[batch_index].sum())
            )
            path = predicted_clean[batch_index, :count].float()
            if count > self.coverage_max_path_points:
                indices = torch.linspace(
                    0, count - 1, self.coverage_max_path_points,
                    device=path.device,
                ).round().long()
                path = path[indices]
            points = surface[batch_index, :, :3].float()
            normals = surface[batch_index, :, 3:].float()
            nearest = torch.cdist(path, points).argmin(dim=1)
            normal_offset = ((path - points[nearest]) * normals[nearest]).sum(dim=-1)
            radius = condition[batch_index, 0].float().clamp_min(1e-4)
            losses.append((normal_offset / radius).square().mean())
        return torch.stack(losses).mean()


def minibatch_ot_noise(
    target_path: torch.Tensor,
    noise: torch.Tensor,
    condition: torch.Tensor,
    path_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Pair noise and targets by exact OT within matching mode/token groups."""
    if target_path.shape != noise.shape:
        raise ValueError("target_path and noise must have the same shape")
    batch_size, num_tokens, _ = target_path.shape
    if path_mask is None:
        token_counts = torch.full(
            (batch_size,), num_tokens, device=target_path.device, dtype=torch.long
        )
    else:
        if path_mask.shape != target_path.shape[:2]:
            raise ValueError("path_mask must have shape [B, M]")
        token_counts = path_mask.sum(dim=1)
    mode_ids = (
        condition[:, 2:].argmax(dim=1)
        if condition.shape[1] > 2
        else torch.zeros(batch_size, device=condition.device, dtype=torch.long)
    )
    assigned = noise.clone()
    keys = torch.stack((token_counts, mode_ids), dim=1).detach().cpu()
    for key in torch.unique(keys, dim=0):
        group = torch.nonzero((keys == key).all(dim=1), as_tuple=False).flatten()
        if group.numel() < 2:
            continue
        count = int(key[0])
        device_group = group.to(target_path.device)
        targets = target_path[device_group, :count].float().flatten(start_dim=1)
        sources = noise[device_group, :count].float().flatten(start_dim=1)
        costs = torch.cdist(targets, sources).square()
        from scipy.optimize import linear_sum_assignment

        target_rows, source_columns = linear_sum_assignment(costs.detach().cpu().numpy())
        destination = device_group[torch.as_tensor(target_rows, device=target_path.device)]
        source = device_group[torch.as_tensor(source_columns, device=target_path.device)]
        assigned[destination] = noise[source]
    if path_mask is not None:
        assigned = assigned.masked_fill(~path_mask[..., None], 0.0)
    return assigned
