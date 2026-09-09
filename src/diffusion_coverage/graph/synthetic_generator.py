from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_coverage.problem import GraphInstance


@dataclass(frozen=True)
class SyntheticGeneratorConfig:
    num_nodes: int = 12
    num_colours: int = 4
    nq: int = 6
    edge_probability: float = 0.25
    valid_colour_probability: float = 0.75
    boundary_fraction: float = 0.35
    compatibility_noise: float = 0.08
    q_noise: float = 0.25


def generate_synthetic_instance(
    seed: int | None = None,
    config: SyntheticGeneratorConfig | None = None,
) -> GraphInstance:
    cfg = config or SyntheticGeneratorConfig()
    if cfg.num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    rng = np.random.default_rng(seed)

    edge_index = _connected_undirected_edges(cfg.num_nodes, cfg.edge_probability, rng)
    degrees = np.bincount(edge_index.reshape(-1), minlength=cfg.num_nodes)

    valid_mask = rng.random((cfg.num_nodes, cfg.num_colours)) < cfg.valid_colour_probability
    for i in range(cfg.num_nodes):
        if not np.any(valid_mask[i]):
            valid_mask[i, rng.integers(cfg.num_colours)] = True

    num_boundary = max(1, int(round(cfg.boundary_fraction * cfg.num_nodes)))
    boundary_nodes = rng.choice(cfg.num_nodes, size=num_boundary, replace=False)
    boundary_mask = np.zeros(cfg.num_nodes, dtype=bool)
    boundary_mask[boundary_nodes] = True

    latent_sheet_q = rng.normal(size=(cfg.num_colours, cfg.nq))
    node_offsets = rng.normal(scale=0.15, size=(cfg.num_nodes, cfg.nq))
    q_candidates = (
        latent_sheet_q[None, :, :]
        + node_offsets[:, None, :]
        + rng.normal(scale=cfg.q_noise, size=(cfg.num_nodes, cfg.num_colours, cfg.nq))
    )

    compatibility = np.zeros((edge_index.shape[1], cfg.num_colours, cfg.num_colours), dtype=bool)
    for e, (src, dst) in enumerate(edge_index.T):
        for c in range(cfg.num_colours):
            for d in range(cfg.num_colours):
                if not valid_mask[src, c] or not valid_mask[dst, d]:
                    continue
                same_sheet = c == d
                noisy_bridge = rng.random() < cfg.compatibility_noise
                compatibility[e, c, d] = same_sheet or noisy_bridge

    node_features = np.column_stack(
        [
            boundary_mask.astype(float),
            valid_mask.sum(axis=1) / cfg.num_colours,
            degrees / max(1, degrees.max()),
        ]
    )
    edge_features = np.ones((edge_index.shape[1], 1), dtype=float)
    manipulability = rng.uniform(0.2, 1.0, size=(cfg.num_nodes, cfg.num_colours))
    joint_limit_margin = rng.uniform(0.1, 1.0, size=(cfg.num_nodes, cfg.num_colours))
    manipulability[~valid_mask] = 0.0
    joint_limit_margin[~valid_mask] = 0.0

    return GraphInstance(
        node_features=node_features,
        edge_index=edge_index.astype(np.int64),
        edge_features=edge_features,
        valid_colour_mask=valid_mask,
        boundary_mask=boundary_mask,
        q_candidates=q_candidates.astype(float),
        colour_compatibility=compatibility,
        manipulability=manipulability,
        joint_limit_margin=joint_limit_margin,
        graph_id=f"synthetic-{seed}",
        robot_id="synthetic",
        metadata={"seed": seed, "config": cfg.__dict__},
    )


def _connected_undirected_edges(num_nodes: int, edge_probability: float, rng: np.random.Generator) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    order = rng.permutation(num_nodes)
    for idx in range(1, num_nodes):
        a = int(order[idx])
        b = int(order[rng.integers(idx)])
        edges.add((min(a, b), max(a, b)))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_probability:
                edges.add((i, j))
    sorted_edges = sorted(edges)
    return np.asarray(sorted_edges, dtype=np.int64).T
