from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from diffusion_coverage.surface.surface_instance import SurfaceInstance


ExpansionPolicy = Literal["upstream_first", "reverse_order", "seeded_random"]


@dataclass(frozen=True)
class ExpansionDecision:
    parent_face: int
    available_edges: tuple[int, ...]
    selected_edge_order: tuple[int, ...]
    added_faces: tuple[int, ...]


@dataclass(frozen=True)
class NUCSkeleton:
    topological_path: np.ndarray
    waypoints: np.ndarray
    visited_faces: np.ndarray
    tree_edges: np.ndarray
    policy: str
    seed: int | None
    expansion_decisions: tuple[ExpansionDecision, ...]
    root_face: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FacetNode:
    face: int
    subvertex: int
    child_slot: int
    children: list["_FacetNode"] = field(default_factory=list)


def generate_nuc_skeleton(
    mesh: SurfaceInstance | tuple[np.ndarray, np.ndarray] | Any,
    *,
    policy: ExpansionPolicy = "upstream_first",
    seed: int | None = None,
    root_face: int | None = None,
) -> NUCSkeleton:
    """Generate a NUC facet-tree skeleton with controlled legal expansion order.

    ``upstream_first`` is a compatibility implementation of upstream commit
    f28c9a0b182d3e7b6b6223972ce7682e7e3b1300. It does not reproduce the
    contact model from the associated paper.
    """

    if policy not in {"upstream_first", "reverse_order", "seeded_random"}:
        raise ValueError(f"unknown NUC expansion policy: {policy}")
    if policy == "seeded_random" and seed is None:
        raise ValueError("seeded_random requires an explicit seed")
    vertices, faces = _extract_mesh(mesh)
    valid_faces = np.flatnonzero(np.all(faces >= 0, axis=1))
    if not len(valid_faces):
        raise ValueError("NUC requires at least one valid triangular facet")
    root = int(valid_faces[0] if root_face is None else root_face)
    if root not in set(valid_faces.tolist()):
        raise ValueError("root_face must identify a valid facet")

    adjacency = _directed_adjacency(faces, len(vertices))
    covered = np.any(faces < 0, axis=1)
    root_node = _FacetNode(root, _root_subvertex(faces[root], adjacency, len(vertices)), -1)
    covered[root] = True
    queue: deque[_FacetNode] = deque((root_node,))
    rng = np.random.default_rng(seed)
    decisions: list[ExpansionDecision] = []
    tree_edges: list[tuple[int, int, int]] = []

    while queue:
        node = queue.popleft()
        edge_candidates = _uncovered_neighbours(
            faces, node.face, adjacency, covered, len(vertices)
        )
        available_edges = tuple(edge for edge, _ in edge_candidates)
        edge_order = _ordered_edges(available_edges, policy, rng)
        by_edge = {edge: neighbour for edge, neighbour in edge_candidates}
        added: list[int] = []
        for edge in edge_order:
            neighbour = by_edge[edge]
            if covered[neighbour]:
                continue
            first = int(faces[node.face, edge])
            second = int(faces[node.face, (edge + 1) % 3])
            child = _FacetNode(
                neighbour,
                _child_subvertex(faces[neighbour], second, first),
                edge,
            )
            node.children.append(child)
            covered[neighbour] = True
            queue.append(child)
            added.append(neighbour)
            tree_edges.append((node.face, neighbour, edge))
        decisions.append(
            ExpansionDecision(
                parent_face=node.face,
                available_edges=available_edges,
                selected_edge_order=edge_order,
                added_faces=tuple(added),
            )
        )

    topological: list[int] = []
    geometric: list[np.ndarray] = []
    _traverse(root_node, vertices, faces, topological, geometric, 0)
    skeleton = NUCSkeleton(
        topological_path=np.asarray(topological, dtype=np.int64),
        waypoints=np.asarray(geometric, dtype=np.float64),
        visited_faces=np.flatnonzero(covered & np.all(faces >= 0, axis=1)),
        tree_edges=np.asarray(tree_edges, dtype=np.int64).reshape(-1, 3),
        policy=policy,
        seed=seed,
        expansion_decisions=tuple(decisions),
        root_face=root,
        metadata={
            "upstream_commit": "f28c9a0b182d3e7b6b6223972ce7682e7e3b1300",
            "variant_degree_of_freedom": "per-parent legal facet-edge expansion order",
        },
    )
    validate_nuc_skeleton(vertices, faces, skeleton)
    return skeleton


def generate_nuc_skeleton_variants(
    mesh: SurfaceInstance | tuple[np.ndarray, np.ndarray] | Any,
    num_variants: int,
    seed: int,
    *,
    root_face: int | None = None,
    require_unique: bool = True,
) -> tuple[NUCSkeleton, ...]:
    if num_variants < 1:
        raise ValueError("num_variants must be positive")
    variants = [
        generate_nuc_skeleton(mesh, policy="upstream_first", root_face=root_face)
    ]
    if num_variants > 1:
        variants.append(
            generate_nuc_skeleton(mesh, policy="reverse_order", root_face=root_face)
        )
    fingerprints = {_fingerprint(item) for item in variants}
    next_seed = int(seed)
    maximum_attempts = max(100, 50 * num_variants)
    attempts = 0
    while len(variants) < num_variants and attempts < maximum_attempts:
        candidate = generate_nuc_skeleton(
            mesh,
            policy="seeded_random",
            seed=next_seed,
            root_face=root_face,
        )
        next_seed += 1
        attempts += 1
        fingerprint = _fingerprint(candidate)
        if require_unique and fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        variants.append(candidate)
    if len(variants) != num_variants:
        raise RuntimeError(
            f"only {len(variants)} unique NUC skeletons found after {attempts} random policies"
        )
    return tuple(variants)


def validate_nuc_skeleton(
    vertices: np.ndarray, faces: np.ndarray, skeleton: NUCSkeleton
) -> None:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    valid_faces = np.flatnonzero(np.all(faces >= 0, axis=1))
    expected_codes = np.concatenate(
        [3 * valid_faces[:, None] + np.arange(3)[None, :]], axis=1
    ).reshape(-1)
    if len(skeleton.topological_path) != 3 * len(valid_faces):
        raise ValueError("NUC path must visit three subfacets per valid facet")
    if not np.array_equal(np.sort(skeleton.topological_path), np.sort(expected_codes)):
        raise ValueError("NUC path does not visit every required subfacet exactly once")
    if skeleton.waypoints.shape != (len(skeleton.topological_path), 3):
        raise ValueError("NUC geometric path must align with the topological path")
    if not np.all(np.isfinite(skeleton.waypoints)):
        raise ValueError("NUC path contains non-finite waypoints")
    if not np.array_equal(np.sort(skeleton.visited_faces), valid_faces):
        raise ValueError("NUC expansion did not visit every valid facet")
    if len(valid_faces) > 1 and len(skeleton.tree_edges) != len(valid_faces) - 1:
        raise ValueError("NUC expansion must form a facet spanning tree")
    adjacency = _directed_adjacency(faces, len(vertices))
    for parent, child, edge in skeleton.tree_edges:
        first = int(faces[parent, edge])
        second = int(faces[parent, (edge + 1) % 3])
        if adjacency.get(second + len(vertices) * first) != child:
            raise ValueError("NUC tree contains a non-adjacent facet expansion")


def _extract_mesh(
    mesh: SurfaceInstance | tuple[np.ndarray, np.ndarray] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, tuple) and len(mesh) == 2:
        vertices, faces = mesh
    else:
        vertices, faces = mesh.vertices, mesh.faces
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh faces must have shape [F, 3]")
    return vertices, faces


def _directed_adjacency(faces: np.ndarray, vertex_count: int) -> dict[int, int]:
    adjacency: dict[int, int] = {}
    for face_index, face in enumerate(faces):
        if face[0] < 0:
            continue
        for edge in range(3):
            first = int(face[edge])
            second = int(face[(edge + 1) % 3])
            adjacency[first + vertex_count * second] = face_index
    return adjacency


def _root_subvertex(face: np.ndarray, adjacency: dict[int, int], vertex_count: int) -> int:
    for edge in range(3):
        first = int(face[edge])
        second = int(face[(edge + 1) % 3])
        if second + vertex_count * first not in adjacency:
            return edge
    return 0


def _uncovered_neighbours(
    faces: np.ndarray,
    face_index: int,
    adjacency: dict[int, int],
    covered: np.ndarray,
    vertex_count: int,
) -> list[tuple[int, int]]:
    result = []
    face = faces[face_index]
    for edge in range(3):
        first = int(face[edge])
        second = int(face[(edge + 1) % 3])
        neighbour = adjacency.get(second + vertex_count * first)
        if neighbour is not None and not covered[neighbour]:
            result.append((edge, neighbour))
    return result


def _ordered_edges(
    available: tuple[int, ...], policy: ExpansionPolicy, rng: np.random.Generator
) -> tuple[int, ...]:
    if policy == "upstream_first":
        return available
    if policy == "reverse_order":
        return tuple(reversed(available))
    if not available:
        return ()
    return tuple(int(value) for value in rng.permutation(available))


def _child_subvertex(face: np.ndarray, second: int, first: int) -> int:
    for edge in range(3):
        if int(face[edge]) == second and int(face[(edge + 1) % 3]) == first:
            return edge
    return 0


def _facet_cycles(
    vertices: np.ndarray, faces: np.ndarray, face_index: int, subvertex: int
) -> tuple[list[int], list[np.ndarray]]:
    v0, v1, v2 = vertices[faces[face_index]]
    m01 = 0.5 * (v0 + v1)
    m12 = 0.5 * (v1 + v2)
    m20 = 0.5 * (v2 + v0)
    center = (v0 + v1 + v2) / 3.0
    polygon_centers = [
        (v1 + m12 + center + m01) / 4.0,
        (v2 + m20 + center + m12) / 4.0,
        (v0 + m01 + center + m20) / 4.0,
    ]
    offset = subvertex % 3
    order = [(offset + index) % 3 for index in range(3)]
    return [3 * face_index + index for index in order], [polygon_centers[index] for index in order]


def _traverse(
    node: _FacetNode,
    vertices: np.ndarray,
    faces: np.ndarray,
    topological: list[int],
    geometric: list[np.ndarray],
    insert_at: int,
) -> None:
    codes, points = _facet_cycles(vertices, faces, node.face, node.subvertex)
    topological[insert_at:insert_at] = codes
    geometric[insert_at:insert_at] = points
    parent_codes = (3 * node.face + 2, 3 * node.face, 3 * node.face + 1)
    for child in node.children:
        location = topological.index(parent_codes[child.child_slot])
        _traverse(child, vertices, faces, topological, geometric, location + 1)


def _fingerprint(skeleton: NUCSkeleton) -> bytes:
    return skeleton.topological_path.tobytes()
