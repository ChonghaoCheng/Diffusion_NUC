from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffusion_coverage.nuc import (
    generate_nuc_skeleton,
    generate_nuc_skeleton_variants,
    run_upstream_reference,
    validate_nuc_skeleton,
)
from diffusion_coverage.surface import make_saddle


UPSTREAM_BUILD = Path("/data/chocheng/Code/NUC_upstream/build-audit")


def two_triangle_mesh():
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2], [2, 1, 3]])
    return vertices, faces


def test_upstream_first_matches_compiled_reference():
    if not list(UPSTREAM_BUILD.glob("nuc_tmech23*.so")):
        pytest.skip("separately built upstream NUC extension is unavailable")
    vertices, faces = two_triangle_mesh()
    reference_topology, reference_geometry = run_upstream_reference(
        vertices, faces, build_directory=UPSTREAM_BUILD
    )
    adapted = generate_nuc_skeleton((vertices, faces), policy="upstream_first")
    assert np.array_equal(adapted.topological_path, reference_topology)
    assert np.allclose(adapted.waypoints, reference_geometry, atol=1e-12)


def test_all_policies_preserve_structural_contract():
    surface = make_saddle(nx=4, ny=4, samples_per_face=1)
    for policy, seed in (
        ("upstream_first", None), ("reverse_order", None),
        ("seeded_random", 7), ("frontier_random", 7),
    ):
        skeleton = generate_nuc_skeleton(surface, policy=policy, seed=seed)
        validate_nuc_skeleton(surface.vertices, surface.faces, skeleton)
        assert len(skeleton.visited_faces) == surface.num_faces
        assert len(skeleton.tree_edges) == surface.num_faces - 1


def test_seeded_random_policy_is_deterministic_and_records_decisions():
    surface = make_saddle(nx=5, ny=5, samples_per_face=1)
    first = generate_nuc_skeleton(surface, policy="seeded_random", seed=19)
    second = generate_nuc_skeleton(surface, policy="seeded_random", seed=19)
    assert np.array_equal(first.topological_path, second.topological_path)
    assert first.expansion_decisions == second.expansion_decisions
    assert first.seed == 19


def test_frontier_random_policy_is_deterministic():
    surface = make_saddle(nx=5, ny=5, samples_per_face=1)
    first = generate_nuc_skeleton(surface, policy="frontier_random", seed=23)
    second = generate_nuc_skeleton(surface, policy="frontier_random", seed=23)
    assert np.array_equal(first.topological_path, second.topological_path)
    assert first.expansion_decisions == second.expansion_decisions


def test_variant_generator_returns_unique_valid_skeletons():
    surface = make_saddle(nx=5, ny=5, samples_per_face=1)
    variants = generate_nuc_skeleton_variants(surface, 6, seed=31)
    assert len(variants) == 6
    assert len({variant.topological_path.tobytes() for variant in variants}) == 6
    assert variants[0].policy == "upstream_first"
    assert variants[1].policy == "reverse_order"
