from __future__ import annotations

import torch

from diffusion_coverage.models import (
    ConditionalFlowMatcher,
    PathVectorField,
    PathVectorFieldConfig,
    heun_sample,
    minibatch_ot_noise,
)
from diffusion_coverage.models.path_noise import sample_path_noise


def test_minibatch_ot_coupling_reduces_cost_within_mode_and_token_groups():
    target = torch.tensor(
        [[[0.0], [0.0]], [[10.0], [10.0]], [[100.0], [100.0]]]
    )
    noise = torch.tensor(
        [[[9.0], [9.0]], [[1.0], [1.0]], [[99.0], [99.0]]]
    )
    condition = torch.tensor(
        [[0.1, 0.05, 1.0, 0.0], [0.2, 0.05, 1.0, 0.0], [0.1, 0.05, 0.0, 1.0]]
    )
    mask = torch.ones(3, 2, dtype=torch.bool)
    coupled = minibatch_ot_noise(target, noise, condition, mask)
    independent_cost = (target - noise).square().sum()
    coupled_cost = (target - coupled).square().sum()
    assert coupled_cost < independent_cost
    assert torch.equal(coupled[2], noise[2])


def test_minibatch_ot_coupling_preserves_padding_and_noise_multiset():
    torch.manual_seed(2)
    target = torch.randn(4, 5, 2)
    noise = torch.randn_like(target)
    mask = torch.tensor([[True] * 3 + [False] * 2] * 4)
    target = target.masked_fill(~mask[..., None], 0.0)
    noise = noise.masked_fill(~mask[..., None], 0.0)
    condition = torch.tensor([[0.1, 0.05, 1.0, 0.0]] * 4)
    coupled = minibatch_ot_noise(target, noise, condition, mask)
    assert torch.count_nonzero(coupled[:, 3:]) == 0
    original_rows = sorted(tuple(row.tolist()) for row in noise.flatten(start_dim=1))
    coupled_rows = sorted(tuple(row.tolist()) for row in coupled.flatten(start_dim=1))
    assert coupled_rows == original_rows


def test_smoothed_path_noise_has_lower_local_variation_than_white_noise():
    torch.manual_seed(9)
    reference = torch.empty(32, 128, 3)
    white = sample_path_noise(reference, smoothing_sigma=0.0)
    torch.manual_seed(9)
    smooth = sample_path_noise(reference, smoothing_sigma=6.0)
    white_variation = (white[:, 1:] - white[:, :-1]).square().mean()
    smooth_variation = (smooth[:, 1:] - smooth[:, :-1]).square().mean()
    assert smooth.std() > 0.5
    assert smooth_variation < 0.1 * white_variation


def test_vector_field_flow_loss_and_sampler_shapes_and_gradients():
    torch.manual_seed(3)
    config = PathVectorFieldConfig(hidden_dim=32, num_layers=1, num_heads=4)
    vector_field = PathVectorField(config)
    matcher = ConditionalFlowMatcher(vector_field, smoothness_weight=0.01)
    surface = torch.randn(2, 20, 6)
    surface[..., 3:] = torch.nn.functional.normalize(surface[..., 3:], dim=-1)
    target = torch.randn(2, 12, 3)
    condition = torch.tensor([[0.1, 0.05], [0.2, 0.08]])
    loss = matcher.loss(target, surface, condition)
    assert loss.total.ndim == 0
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in vector_field.parameters())

    sampled = heun_sample(
        vector_field,
        surface,
        condition,
        num_waypoints=12,
        num_steps=2,
        initial_noise=torch.zeros_like(target),
    )
    assert sampled.shape == target.shape
    assert torch.isfinite(sampled).all()


def test_objective_aware_losses_reward_coverage_and_only_penalize_excess_length():
    matcher = ConditionalFlowMatcher(
        torch.nn.Identity(), coverage_weight=1.0, excess_length_weight=1.0
    )
    surface = torch.zeros(1, 6, 6)
    surface[0, :, 0] = torch.linspace(0.0, 1.0, 6)
    surface[0, :, 5] = 1.0
    condition = torch.tensor([[0.15, 0.05]])
    near_path = torch.zeros(1, 6, 3)
    near_path[0, :, 0] = torch.linspace(0.0, 1.0, 6)
    far_path = near_path + torch.tensor([0.0, 1.0, 0.0])
    near_coverage = matcher._coverage_loss(near_path, surface, condition, None)
    far_coverage = matcher._coverage_loss(far_path, surface, condition, None)
    assert near_coverage < far_coverage

    target = near_path
    longer = 2.0 * near_path
    shorter = 0.5 * near_path
    assert matcher._excess_length_loss(longer, target, None) > 0.0
    assert matcher._excess_length_loss(shorter, target, None) == 0.0

    tangent_matcher = ConditionalFlowMatcher(torch.nn.Identity(), tangent_weight=1.0)
    assert tangent_matcher._tangent_loss(target, target, None) < 1e-6
    assert tangent_matcher._tangent_loss(-target, target, None) > 1.9

    surface_matcher = ConditionalFlowMatcher(
        torch.nn.Identity(), surface_consistency_weight=1.0
    )
    assert surface_matcher._surface_consistency_loss(
        near_path, surface, condition, None
    ) < 1e-8
    assert surface_matcher._surface_consistency_loss(
        near_path + torch.tensor([0.0, 0.0, 0.15]), surface, condition, None
    ) > 0.9


def test_padding_does_not_change_valid_velocity_or_loss():
    torch.manual_seed(14)
    model = PathVectorField(
        PathVectorFieldConfig(
            hidden_dim=32, num_layers=2, num_heads=4, path_self_attention=True
        )
    ).eval()
    matcher = ConditionalFlowMatcher(model, smoothness_weight=0.1, noise_smoothing_sigma=2.0)
    surface = torch.randn(1, 16, 6)
    surface[..., 3:] = torch.nn.functional.normalize(surface[..., 3:], dim=-1)
    condition = torch.tensor([[0.15, 0.05]])
    path = torch.randn(1, 8, 3)
    arclength = torch.linspace(0.0, 1.0, 8).unsqueeze(0)
    mask = torch.ones(1, 8, dtype=torch.bool)
    padded_path = torch.zeros(1, 13, 3)
    padded_path[:, :8] = path
    padded_arclength = torch.zeros(1, 13)
    padded_arclength[:, :8] = arclength
    padded_mask = torch.zeros(1, 13, dtype=torch.bool)
    padded_mask[:, :8] = True
    time = torch.tensor([0.37])
    noise = torch.randn_like(path)
    padded_noise = torch.zeros_like(padded_path)
    padded_noise[:, :8] = noise

    velocity = model(path, time, surface, condition, path_mask=mask, path_arclength=arclength)
    padded_velocity = model(
        padded_path, time, surface, condition,
        path_mask=padded_mask, path_arclength=padded_arclength,
    )
    assert torch.allclose(velocity, padded_velocity[:, :8], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(padded_velocity[:, 8:]) == 0

    loss = matcher.loss(
        path, surface, condition, noise=noise, time=time,
        path_mask=mask, path_arclength=arclength,
    )
    padded_loss = matcher.loss(
        padded_path, surface, condition, noise=padded_noise, time=time,
        path_mask=padded_mask, path_arclength=padded_arclength,
    )
    assert torch.allclose(loss.total, padded_loss.total, atol=1e-6, rtol=1e-6)


def test_masked_heun_sampling_keeps_padding_zero():
    torch.manual_seed(22)
    model = PathVectorField(PathVectorFieldConfig(hidden_dim=32, num_layers=1, num_heads=4)).eval()
    surface = torch.randn(2, 12, 6)
    surface[..., 3:] = torch.nn.functional.normalize(surface[..., 3:], dim=-1)
    condition = torch.tensor([[0.1, 0.05], [0.2, 0.05]])
    mask = torch.tensor([[True] * 7 + [False] * 4, [True] * 11])
    arclength = torch.zeros(2, 11)
    arclength[0, :7] = torch.linspace(0.0, 1.0, 7)
    arclength[1] = torch.linspace(0.0, 1.0, 11)
    sampled = heun_sample(
        model, surface, condition,
        num_waypoints=11, num_steps=2,
        path_mask=mask, path_arclength=arclength,
        noise_smoothing_sigma=2.0,
    )
    assert torch.isfinite(sampled).all()
    assert torch.count_nonzero(sampled[0, 7:]) == 0
    assert torch.count_nonzero(sampled[1]) > 0


def test_two_dimensional_chart_vector_field_and_sampler():
    model = PathVectorField(
        PathVectorFieldConfig(path_dim=2, hidden_dim=32, num_layers=1, num_heads=4)
    ).eval()
    surface = torch.randn(2, 16, 6)
    condition = torch.tensor([[0.1, 0.05], [0.2, 0.05]])
    sampled = heun_sample(model, surface, condition, num_waypoints=9, num_steps=2)
    assert sampled.shape == (2, 9, 2)
    assert torch.isfinite(sampled).all()


def test_flow_matching_supports_six_dof_configuration_tokens():
    model = PathVectorField(
        PathVectorFieldConfig(
            path_dim=6,
            condition_dim=4,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
        )
    ).eval()
    matcher = ConditionalFlowMatcher(model)
    target = torch.randn(2, 11, 6)
    surface = torch.randn(2, 16, 6)
    condition = torch.randn(2, 4)
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[1, 8:] = False

    losses = matcher.loss(target, surface, condition, path_mask=mask)
    sampled = heun_sample(
        model,
        surface,
        condition,
        num_waypoints=11,
        num_steps=2,
        path_mask=mask,
    )

    assert torch.isfinite(losses.total)
    assert sampled.shape == target.shape
    assert torch.count_nonzero(sampled[1, 8:]) == 0


def test_mode_conditioned_chart_vector_field_accepts_extended_condition():
    model = PathVectorField(
        PathVectorFieldConfig(
            path_dim=2,
            condition_dim=7,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
        )
    ).eval()
    surface = torch.randn(2, 16, 6)
    condition = torch.zeros(2, 7)
    condition[:, :2] = torch.tensor([[0.1, 0.05], [0.2, 0.05]])
    condition[0, 2] = 1.0
    condition[1, 5] = 1.0
    sampled = heun_sample(model, surface, condition, num_waypoints=9, num_steps=2)
    assert sampled.shape == (2, 9, 2)
    assert torch.isfinite(sampled).all()
