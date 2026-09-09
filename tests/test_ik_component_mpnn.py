import torch

from diffusion_coverage.models import IKComponentMPNN


def test_ik_component_mpnn_returns_one_logit_per_component():
    model = IKComponentMPNN(input_dim=17, hidden_dim=16, layers=2)
    logits = model(
        torch.randn(5, 17),
        torch.tensor([[0, 1, 3], [1, 2, 4]]),
        torch.tensor([0, 0, 1, 2, 2]),
        3,
        torch.tensor([0, 0, 0, 1, 1]),
        torch.tensor([0, 0, 1]),
        2,
        torch.tensor([0.25, 0.25, 0.5]),
    )
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
