from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from diffusion_coverage.coverage.teacher import TeacherPlannerConfig, TeacherResult
from diffusion_coverage.surface.surface_instance import SurfaceInstance
from diffusion_coverage.surface.projection import project_points


class TeacherDatasetWriter:
    """Write numeric-only NPZ instances plus a JSONL manifest for later PyTorch loading."""

    def __init__(self, output_dir: str | Path, *, resume: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.instances_dir = self.output_dir / "instances"
        manifest = self.output_dir / "manifest.jsonl"
        if manifest.exists() and not resume:
            raise FileExistsError(f"dataset manifest already exists: {manifest}")
        self.instances_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest

    @property
    def completed_instance_ids(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        return {
            str(json.loads(line)["instance_id"])
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        }

    def write(
        self,
        instance_id: str,
        surface: SurfaceInstance,
        config: TeacherPlannerConfig,
        result: TeacherResult,
    ) -> Path:
        selected_candidates = result.feasible_candidates
        if not selected_candidates:
            raise ValueError("teacher result has no feasible candidates; refusing to create invalid targets")
        output_path = self.instances_dir / f"{instance_id}.npz"
        if output_path.exists():
            raise FileExistsError(f"instance already exists: {output_path}")

        num_candidates = len(selected_candidates)
        max_segments = max(candidate.plan.waypoints.shape[0] for candidate in selected_candidates)
        max_waypoints = max(candidate.plan.waypoints.shape[1] for candidate in selected_candidates)
        candidate_waypoints = np.zeros((num_candidates, max_segments, max_waypoints, 3), dtype=np.float64)
        candidate_segment_mask = np.zeros((num_candidates, max_segments), dtype=bool)
        candidate_waypoint_mask = np.zeros((num_candidates, max_segments, max_waypoints), dtype=bool)
        candidate_metrics = np.zeros((num_candidates, 5), dtype=np.float64)
        proposal_names: list[str] = []
        for index, candidate in enumerate(selected_candidates):
            segments, waypoints, _ = candidate.plan.waypoints.shape
            candidate_waypoints[index, :segments, :waypoints] = candidate.plan.waypoints
            candidate_segment_mask[index, :segments] = candidate.plan.segment_mask
            candidate_waypoint_mask[index, :segments, :waypoints] = candidate.plan.waypoint_mask
            candidate_metrics[index] = (
                candidate.metrics.missed_fraction,
                candidate.metrics.path_length,
                candidate.metrics.coverage_efficiency,
                candidate.smoothness_cost,
                float(candidate.feasible),
            )
            proposal_names.append(candidate.proposal_name)

        metadata = {
            "instance_id": instance_id,
            "surface_id": surface.surface_id,
            "surface_metadata": surface.metadata,
            "teacher_config": asdict(config),
            "teacher_runtime": {
                "proposal_time": result.proposal_time,
                "refinement_time": result.refinement_time,
                "total_solve_time": result.total_solve_time,
                "evaluated_plans": result.evaluated_plans,
            },
            "metric_columns": [
                "missed_fraction",
                "path_length",
                "coverage_efficiency",
                "smoothness_cost",
                "feasible",
            ],
        }
        np.savez_compressed(
            output_path,
            vertices=surface.vertices.astype(np.float32),
            faces=surface.faces.astype(np.int64),
            sample_points=surface.sample_points.astype(np.float32),
            sample_normals=surface.sample_normals.astype(np.float32),
            area_weights=surface.area_weights.astype(np.float64),
            sample_face_indices=surface.sample_face_indices.astype(np.int64),
            sample_barycentric=surface.sample_barycentric.astype(np.float64),
            candidate_waypoints=candidate_waypoints,
            candidate_segment_mask=candidate_segment_mask,
            candidate_waypoint_mask=candidate_waypoint_mask,
            candidate_metrics=candidate_metrics,
            proposal_names=np.asarray(proposal_names, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
        manifest_row = {
            "instance_id": instance_id,
            "path": str(output_path.relative_to(self.output_dir)),
            "surface_id": surface.surface_id,
            "num_vertices": surface.num_vertices,
            "num_faces": surface.num_faces,
            "num_candidates": num_candidates,
            "num_feasible_candidates": num_candidates,
            "best_missed_fraction": result.best.metrics.missed_fraction,
            "best_path_length": result.best.metrics.path_length,
        }
        with self.manifest_path.open("a", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(manifest_row, sort_keys=True) + "\n")
        return output_path


def load_teacher_instance(path: str | Path) -> dict[str, np.ndarray | dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        result: dict[str, np.ndarray | dict[str, object]] = {key: archive[key] for key in archive.files}
    metadata_raw = result.pop("metadata_json")
    result["metadata"] = json.loads(str(np.asarray(metadata_raw).item()))
    return result


def surface_from_teacher_archive(
    archive: dict[str, np.ndarray | dict[str, object]],
    *,
    surface_id: str | None = None,
) -> SurfaceInstance:
    """Restore the exact quadrature contract used to compute archived metrics."""

    vertices = np.asarray(archive["vertices"], dtype=np.float64)
    faces = np.asarray(archive["faces"], dtype=np.int64)
    sample_points = np.asarray(archive["sample_points"], dtype=np.float64)
    metadata = archive["metadata"]
    assert isinstance(metadata, dict)
    if "sample_face_indices" in archive and "sample_barycentric" in archive:
        face_indices = np.asarray(archive["sample_face_indices"])
        barycentric = np.asarray(archive["sample_barycentric"])
        restored_points = np.einsum(
            "pi,pij->pj", barycentric, vertices[faces[face_indices]]
        )
    else:
        temporary = SurfaceInstance.from_mesh(vertices, faces, samples_per_face=1)
        projection = project_points(temporary, sample_points)
        if float(projection.distances.max(initial=0.0)) > 1e-5:
            raise ValueError("archived quadrature samples do not lie on the stored mesh")
        face_indices = projection.face_indices
        barycentric = projection.barycentric
        restored_points = projection.points
    weights = np.asarray(archive["area_weights"], dtype=np.float64).copy()
    triangles = vertices[faces]
    reconstructed_area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ).sum()
    weights *= reconstructed_area / weights.sum()
    return SurfaceInstance(
        vertices=vertices,
        faces=faces,
        sample_points=restored_points,
        sample_normals=np.asarray(archive["sample_normals"]),
        area_weights=weights,
        sample_face_indices=face_indices,
        sample_barycentric=barycentric,
        surface_id=surface_id,
        metadata=dict(metadata.get("surface_metadata", {})),
    )
