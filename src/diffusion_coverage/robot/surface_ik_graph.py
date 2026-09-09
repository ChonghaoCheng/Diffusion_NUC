from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_coverage.coverage.patterns import map_surface_parameters
from diffusion_coverage.robot.ur5e_mujoco import (
    IKCandidate,
    UR5eKinematics,
    interpolate_vertex_normals,
    transform_surface_pose_path,
)
from diffusion_coverage.surface.projection import project_points
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class SurfaceIKGraph:
    """Numerically enumerated UR5e IK sheets over a coarse 3D surface grid."""

    uv: np.ndarray
    positions: np.ndarray
    axes: np.ndarray
    candidates: tuple[tuple[IKCandidate, ...], ...]
    edge_index: np.ndarray
    edge_compatibility: tuple[np.ndarray, ...]
    component_labels: tuple[np.ndarray, ...]
    grid_shape: tuple[int, int]
    edge_witnesses: tuple[dict[tuple[int, int], np.ndarray], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_count = len(self.uv)
        if self.uv.shape != (node_count, 2):
            raise ValueError("uv must have shape [N, 2]")
        if self.positions.shape != (node_count, 3) or self.axes.shape != (node_count, 3):
            raise ValueError("positions and axes must have shape [N, 3]")
        if len(self.candidates) != node_count or len(self.component_labels) != node_count:
            raise ValueError("candidate and component layers must match node count")
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if len(self.edge_compatibility) != self.edge_index.shape[1]:
            raise ValueError("one compatibility matrix is required per edge")
        if not self.edge_witnesses:
            object.__setattr__(
                self, "edge_witnesses", tuple({} for _ in range(self.edge_index.shape[1]))
            )
        if len(self.edge_witnesses) != self.edge_index.shape[1]:
            raise ValueError("one witness mapping is required per edge")
        for edge, compatibility in enumerate(self.edge_compatibility):
            source, target = self.edge_index[:, edge]
            if compatibility.shape != (
                len(self.candidates[int(source)]), len(self.candidates[int(target)])
            ):
                raise ValueError("edge compatibility shape does not match candidate layers")
            for pair, q_path in self.edge_witnesses[edge].items():
                if not compatibility[pair]:
                    raise ValueError("edge witness must correspond to a compatible pair")
                if np.asarray(q_path).ndim != 2 or np.asarray(q_path).shape[1] != 6:
                    raise ValueError("edge witness q paths must have shape [T, 6]")
        for layer, labels in zip(self.candidates, self.component_labels):
            if labels.shape != (len(layer),):
                raise ValueError("component labels must match candidate layers")

    @property
    def num_nodes(self) -> int:
        return len(self.candidates)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_candidates(self) -> int:
        return sum(len(layer) for layer in self.candidates)

    @property
    def reachable_mask(self) -> np.ndarray:
        return np.asarray([bool(layer) for layer in self.candidates])

    @property
    def num_components(self) -> int:
        labels = np.concatenate(
            [layer for layer in self.component_labels if len(layer)], axis=0
        ) if any(len(layer) for layer in self.component_labels) else np.empty(0, dtype=int)
        return 0 if not len(labels) else int(labels.max()) + 1

    def component_node_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_components, dtype=np.int64)
        for labels in self.component_labels:
            if len(labels):
                counts[np.unique(labels)] += 1
        return counts

    def summary(self) -> dict[str, Any]:
        component_nodes = self.component_node_counts()
        return {
            "grid_shape": list(self.grid_shape),
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "ik_candidates": self.num_candidates,
            "reachable_nodes": int(self.reachable_mask.sum()),
            "reachable_fraction": float(self.reachable_mask.mean()),
            "components": self.num_components,
            "components_spanning_multiple_nodes": int(np.sum(component_nodes > 1)),
            "largest_component_nodes": int(component_nodes.max(initial=0)),
            "largest_component_fraction": float(
                component_nodes.max(initial=0) / self.num_nodes
            ),
            "mean_candidates_per_reachable_node": (
                0.0
                if not self.reachable_mask.any()
                else float(
                    np.mean(
                        [len(layer) for layer in self.candidates if len(layer)]
                    )
                )
            ),
            **self.metadata,
        }


def build_surface_ik_graph(
    robot: UR5eKinematics,
    surface: SurfaceInstance,
    transform_base_from_surface: np.ndarray,
    *,
    grid_shape: tuple[int, int] = (8, 8),
    axis_tolerance: float = np.deg2rad(3.0),
    random_restarts: int = 24,
    max_candidates: int = 12,
    orientation_cone_samples: int = 1,
    inner_cone_tolerance: float | None = None,
    inner_cone_samples: int = 1,
    inner_max_candidates: int | None = None,
    maximum_joint_step: float = 0.8,
    minimum_manipulability: float = 1e-5,
    task_edge_samples: int = 5,
    task_position_tolerance: float = 3e-3,
    task_transition_mode: str = "ik_continuation",
    candidate_match_tolerance: float = 0.25,
    max_target_matches: int | None = 4,
    seed: int = 0,
) -> SurfaceIKGraph:
    """Enumerate local IK candidates and compatibility-connected components."""

    nu, nv = grid_shape
    if nu < 2 or nv < 2:
        raise ValueError("surface IK grid dimensions must be at least two")
    if random_restarts < 1 or max_candidates < 1:
        raise ValueError("IK enumeration counts must be positive")
    if task_edge_samples < 2 or task_position_tolerance <= 0.0:
        raise ValueError("task-edge tracking settings are invalid")
    if task_transition_mode not in {"linear_tracking", "ik_continuation"}:
        raise ValueError("unknown task transition mode")
    if candidate_match_tolerance <= 0.0:
        raise ValueError("candidate match tolerance must be positive")
    if max_target_matches is not None and max_target_matches < 1:
        raise ValueError("max_target_matches must be positive when provided")
    periodic_u = surface.surface_id in {"cylinder", "hemisphere", "torus"}
    u = (np.arange(nu, dtype=np.float64) + 0.5) / nu
    v = (np.arange(nv, dtype=np.float64) + 0.5) / nv
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uv = np.column_stack((uu.reshape(-1), vv.reshape(-1)))
    analytic_points = map_surface_parameters(surface, uv[:, 0], uv[:, 1])
    projection = project_points(surface, analytic_points)
    normals = interpolate_vertex_normals(
        surface.vertices,
        surface.faces,
        surface.face_normals,
        surface.face_areas,
        projection.face_indices,
        projection.barycentric,
    )
    positions, axes = transform_surface_pose_path(
        projection.points, normals, transform_base_from_surface
    )
    rng = np.random.default_rng(seed)
    candidate_layers = tuple(
        tuple(
            robot.enumerate_ik(
                position,
                axis,
                random_restarts=random_restarts,
                rng=rng,
                axis_tolerance=axis_tolerance,
                minimum_manipulability=minimum_manipulability,
                max_candidates=max_candidates,
                orientation_cone_samples=orientation_cone_samples,
                inner_cone_tolerance=inner_cone_tolerance,
                inner_cone_samples=inner_cone_samples,
                inner_max_candidates=inner_max_candidates,
            )
        )
        for position, axis in zip(positions, axes)
    )
    edge_index = surface_grid_edges(nu, nv, periodic_u=periodic_u)
    compatibilities = []
    witnesses = []
    for source, target in edge_index.T:
        source_layer = candidate_layers[int(source)]
        target_layer = candidate_layers[int(target)]
        edge_positions, edge_axes = surface_edge_target_poses(
            surface,
            transform_base_from_surface,
            uv[int(source)],
            uv[int(target)],
            samples=task_edge_samples,
            periodic_u=periodic_u,
        )
        compatibility = np.zeros((len(source_layer), len(target_layer)), dtype=bool)
        edge_witnesses: dict[tuple[int, int], np.ndarray] = {}
        if task_transition_mode == "linear_tracking":
            for source_index, source_candidate in enumerate(source_layer):
                for target_index, target_candidate in enumerate(target_layer):
                    transition = robot.check_task_transition(
                        source_candidate.q,
                        target_candidate.q,
                        edge_positions,
                        edge_axes,
                        maximum_joint_step=maximum_joint_step,
                        minimum_manipulability=minimum_manipulability,
                        position_tolerance=task_position_tolerance,
                        axis_tolerance=axis_tolerance,
                        allow_equivalent_end=False,
                    )
                    compatibility[source_index, target_index] = transition.feasible
        else:
            _match_continuation_endpoints(
                robot,
                source_layer,
                target_layer,
                edge_positions,
                edge_axes,
                compatibility,
                maximum_joint_step=maximum_joint_step,
                minimum_manipulability=minimum_manipulability,
                position_tolerance=task_position_tolerance,
                axis_tolerance=axis_tolerance,
                candidate_match_tolerance=candidate_match_tolerance,
                max_target_matches=max_target_matches,
                witnesses=edge_witnesses,
                reverse=False,
            )
            _match_continuation_endpoints(
                robot,
                target_layer,
                source_layer,
                edge_positions[::-1],
                edge_axes[::-1],
                compatibility.T,
                maximum_joint_step=maximum_joint_step,
                minimum_manipulability=minimum_manipulability,
                position_tolerance=task_position_tolerance,
                axis_tolerance=axis_tolerance,
                candidate_match_tolerance=candidate_match_tolerance,
                max_target_matches=max_target_matches,
                witnesses=edge_witnesses,
                reverse=True,
            )
        compatibilities.append(compatibility)
        witnesses.append(edge_witnesses)
    component_labels = connected_component_labels(
        tuple(len(layer) for layer in candidate_layers),
        edge_index,
        tuple(compatibilities),
    )
    return SurfaceIKGraph(
        uv=uv,
        positions=positions,
        axes=axes,
        candidates=candidate_layers,
        edge_index=edge_index,
        edge_compatibility=tuple(compatibilities),
        component_labels=component_labels,
        grid_shape=grid_shape,
        edge_witnesses=tuple(witnesses),
        metadata={
            "surface_id": surface.surface_id,
            "axis_tolerance_degrees": float(np.rad2deg(axis_tolerance)),
            "random_restarts": random_restarts,
            "max_candidates": max_candidates,
            "orientation_cone_samples": orientation_cone_samples,
            "inner_cone_tolerance_degrees": (
                None
                if inner_cone_tolerance is None
                else float(np.rad2deg(inner_cone_tolerance))
            ),
            "inner_cone_samples": inner_cone_samples,
            "inner_max_candidates": inner_max_candidates,
            "maximum_joint_step": maximum_joint_step,
            "minimum_manipulability": minimum_manipulability,
            "task_edge_samples": task_edge_samples,
            "task_position_tolerance": task_position_tolerance,
            "task_transition_mode": task_transition_mode,
            "candidate_match_tolerance": candidate_match_tolerance,
            "max_target_matches": max_target_matches,
            "periodic_u": periodic_u,
        },
    )


def _match_continuation_endpoints(
    robot: UR5eKinematics,
    source_layer: tuple[IKCandidate, ...],
    target_layer: tuple[IKCandidate, ...],
    positions: np.ndarray,
    axes: np.ndarray,
    compatibility: np.ndarray,
    *,
    maximum_joint_step: float,
    minimum_manipulability: float,
    position_tolerance: float,
    axis_tolerance: float,
    candidate_match_tolerance: float,
    max_target_matches: int | None,
    witnesses: dict[tuple[int, int], np.ndarray],
    reverse: bool,
) -> None:
    for source_index, source_candidate in enumerate(source_layer):
        prefix = robot.continue_task_transition(
            source_candidate.q,
            positions[:-1],
            axes[:-1],
            maximum_joint_step=maximum_joint_step,
            minimum_manipulability=minimum_manipulability,
            position_tolerance=position_tolerance,
            axis_tolerance=axis_tolerance,
        )
        if not prefix.feasible:
            continue
        fractions = np.linspace(0.0, 1.0, 3)
        final_positions = (
            (1.0 - fractions[:, None]) * positions[-2]
            + fractions[:, None] * positions[-1]
        )
        final_axes = (
            (1.0 - fractions[:, None]) * axes[-2]
            + fractions[:, None] * axes[-1]
        )
        final_axes /= np.maximum(
            np.linalg.norm(final_axes, axis=1, keepdims=True), 1e-12
        )
        differences = np.asarray(
            [
                (candidate.q - prefix.q_path[-1] + np.pi) % (2.0 * np.pi) - np.pi
                for candidate in target_layer
            ]
        )
        distances = np.max(np.abs(differences), axis=1) if len(differences) else np.empty(0)
        order = np.argsort(distances, kind="stable")
        if max_target_matches is not None:
            order = order[:max_target_matches]
        for target_index in order:
            if distances[target_index] > candidate_match_tolerance:
                break
            target_candidate = target_layer[int(target_index)]
            transition = robot.check_task_transition(
                prefix.q_path[-1],
                target_candidate.q,
                final_positions,
                final_axes,
                maximum_joint_step=maximum_joint_step,
                minimum_manipulability=minimum_manipulability,
                position_tolerance=position_tolerance,
                axis_tolerance=axis_tolerance,
                allow_equivalent_end=False,
                check_endpoints=False,
            )
            if transition.feasible:
                compatibility[source_index, int(target_index)] = True
                q_path = np.concatenate(
                    (prefix.q_path, transition.q_path[1:]), axis=0
                )
                key = (
                    (int(target_index), source_index)
                    if reverse
                    else (source_index, int(target_index))
                )
                witnesses.setdefault(key, q_path[::-1].copy() if reverse else q_path.copy())


def surface_edge_target_poses(
    surface: SurfaceInstance,
    transform_base_from_surface: np.ndarray,
    source_uv: np.ndarray,
    target_uv: np.ndarray,
    *,
    samples: int,
    periodic_u: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the shortest analytic-chart edge between neighbouring grid nodes."""

    if samples < 2:
        raise ValueError("surface edge requires at least two samples")
    source = np.asarray(source_uv, dtype=np.float64)
    target = np.asarray(target_uv, dtype=np.float64)
    if source.shape != (2,) or target.shape != (2,):
        raise ValueError("surface edge endpoints must be two-dimensional")
    delta = target - source
    if periodic_u and abs(delta[0]) > 0.5:
        delta[0] -= np.sign(delta[0])
    fractions = np.linspace(0.0, 1.0, samples)
    parameters = source[None, :] + fractions[:, None] * delta[None, :]
    if periodic_u:
        parameters[:, 0] = np.mod(parameters[:, 0], 1.0)
    analytic_points = map_surface_parameters(
        surface, parameters[:, 0], parameters[:, 1]
    )
    projection = project_points(surface, analytic_points)
    normals = interpolate_vertex_normals(
        surface.vertices,
        surface.faces,
        surface.face_normals,
        surface.face_areas,
        projection.face_indices,
        projection.barycentric,
    )
    return transform_surface_pose_path(
        projection.points, normals, transform_base_from_surface
    )


def surface_grid_edges(nu: int, nv: int, *, periodic_u: bool) -> np.ndarray:
    edges = []
    for u_index in range(nu):
        for v_index in range(nv):
            node = u_index * nv + v_index
            if v_index + 1 < nv:
                edges.append((node, node + 1))
            if u_index + 1 < nu:
                edges.append((node, node + nv))
            elif periodic_u:
                edges.append((node, v_index))
    return np.asarray(edges, dtype=np.int64).T


def connected_component_labels(
    candidate_counts: tuple[int, ...],
    edge_index: np.ndarray,
    compatibilities: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    offsets = np.cumsum((0, *candidate_counts))
    parent = np.arange(offsets[-1], dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge, compatibility in enumerate(compatibilities):
        source, target = (int(value) for value in edge_index[:, edge])
        for source_candidate, target_candidate in np.argwhere(compatibility):
            union(
                int(offsets[source] + source_candidate),
                int(offsets[target] + target_candidate),
            )
    root_to_component = {}
    labels = []
    for node, count in enumerate(candidate_counts):
        node_labels = np.empty(count, dtype=np.int64)
        for candidate in range(count):
            root = find(int(offsets[node] + candidate))
            node_labels[candidate] = root_to_component.setdefault(
                root, len(root_to_component)
            )
        labels.append(node_labels)
    return tuple(labels)


def save_surface_ik_graph(path: str | Path, graph: SurfaceIKGraph) -> None:
    """Serialize a padded numerical IK graph for routing and learning reuse."""

    maximum = max((len(layer) for layer in graph.candidates), default=0)
    q = np.zeros((graph.num_nodes, maximum, 6), dtype=np.float64)
    candidate_mask = np.zeros((graph.num_nodes, maximum), dtype=bool)
    components = np.full((graph.num_nodes, maximum), -1, dtype=np.int32)
    manipulability = np.zeros((graph.num_nodes, maximum), dtype=np.float32)
    joint_margin = np.zeros((graph.num_nodes, maximum), dtype=np.float32)
    for node, (layer, labels) in enumerate(zip(graph.candidates, graph.component_labels)):
        for candidate, value in enumerate(layer):
            q[node, candidate] = value.q
            candidate_mask[node, candidate] = True
            components[node, candidate] = labels[candidate]
            manipulability[node, candidate] = value.manipulability
            joint_margin[node, candidate] = value.joint_limit_margin
    edge_compatibility = np.zeros(
        (graph.num_edges, maximum, maximum), dtype=bool
    )
    for edge, values in enumerate(graph.edge_compatibility):
        edge_compatibility[edge, : values.shape[0], : values.shape[1]] = values
    witness_edges = []
    witness_sources = []
    witness_targets = []
    witness_offsets = [0]
    witness_q = []
    for edge, mapping in enumerate(graph.edge_witnesses):
        for (source, target), q_path in sorted(mapping.items()):
            witness_edges.append(edge)
            witness_sources.append(source)
            witness_targets.append(target)
            witness_q.extend(np.asarray(q_path, dtype=np.float64))
            witness_offsets.append(len(witness_q))
    np.savez_compressed(
        Path(path),
        uv=graph.uv,
        positions=graph.positions,
        axes=graph.axes,
        q_candidates=q,
        candidate_mask=candidate_mask,
        component_labels=components,
        manipulability=manipulability,
        joint_limit_margin=joint_margin,
        edge_index=graph.edge_index,
        edge_compatibility=edge_compatibility,
        witness_edges=np.asarray(witness_edges, dtype=np.int32),
        witness_sources=np.asarray(witness_sources, dtype=np.int32),
        witness_targets=np.asarray(witness_targets, dtype=np.int32),
        witness_offsets=np.asarray(witness_offsets, dtype=np.int64),
        witness_q=np.asarray(witness_q, dtype=np.float64).reshape(-1, 6),
        grid_shape=np.asarray(graph.grid_shape),
        metadata_json=np.asarray(json.dumps(graph.metadata, sort_keys=True)),
    )


def load_surface_ik_graph(path: str | Path) -> SurfaceIKGraph:
    """Restore a graph written by :func:`save_surface_ik_graph`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        mask = archive["candidate_mask"]
        candidates = []
        labels = []
        for node in range(mask.shape[0]):
            count = int(mask[node].sum())
            candidates.append(
                tuple(
                    IKCandidate(
                        q=np.asarray(archive["q_candidates"][node, index], dtype=np.float64),
                        position_error=0.0,
                        axis_error=0.0,
                        manipulability=float(archive["manipulability"][node, index]),
                        joint_limit_margin=float(archive["joint_limit_margin"][node, index]),
                        collision_free=True,
                    )
                    for index in range(count)
                )
            )
            labels.append(np.asarray(archive["component_labels"][node, :count], dtype=np.int64))
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        compatibility = []
        for edge, (source, target) in enumerate(edge_index.T):
            compatibility.append(
                np.asarray(
                    archive["edge_compatibility"][
                        edge, : len(candidates[int(source)]), : len(candidates[int(target)])
                    ],
                    dtype=bool,
                )
            )
        witnesses: list[dict[tuple[int, int], np.ndarray]] = [
            {} for _ in range(edge_index.shape[1])
        ]
        if "witness_edges" in archive:
            offsets = archive["witness_offsets"]
            q_values = archive["witness_q"]
            for index, edge in enumerate(archive["witness_edges"]):
                witnesses[int(edge)][
                    (int(archive["witness_sources"][index]), int(archive["witness_targets"][index]))
                ] = np.asarray(q_values[offsets[index] : offsets[index + 1]], dtype=np.float64)
        return SurfaceIKGraph(
            uv=np.asarray(archive["uv"], dtype=np.float64),
            positions=np.asarray(archive["positions"], dtype=np.float64),
            axes=np.asarray(archive["axes"], dtype=np.float64),
            candidates=tuple(candidates),
            edge_index=edge_index,
            edge_compatibility=tuple(compatibility),
            component_labels=tuple(labels),
            grid_shape=tuple(int(value) for value in archive["grid_shape"]),
            edge_witnesses=tuple(witnesses),
            metadata=json.loads(str(archive["metadata_json"])),
        )
