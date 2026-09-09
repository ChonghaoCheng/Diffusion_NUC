from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from diffusion_coverage.coverage.coverage_plan import CoveragePlan
from diffusion_coverage.coverage.dataset import load_teacher_instance, surface_from_teacher_archive
from diffusion_coverage.coverage.patterns import (
    extract_structured_parameter_controls,
    inverse_surface_parameters,
    parse_pattern_mode,
    PATTERN_MODE_NAMES,
    simplify_parameter_polyline,
    structured_controls_to_residual,
)
from diffusion_coverage.coverage.resampling import resample_plan_fixed_waypoints
from diffusion_coverage.representation import canonicalize_surface_path, suggested_token_count
from diffusion_coverage.surface.surface_instance import SurfaceInstance


@dataclass(frozen=True)
class TeacherSampleIndex:
    instance_index: int
    candidate_index: int


def load_manifest(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    rows = [json.loads(line) for line in (dataset_path / "manifest.jsonl").read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"empty dataset manifest: {dataset_path}")
    return rows


def canonical_candidate_name(name: str) -> str:
    """Remove sanitizer provenance without merging path family or phase."""

    return name.removesuffix("_robustness_repair")


def candidate_indices_by_name(
    archive: dict[str, Any],
    candidate_name: str | None,
    *,
    allow_repaired: bool = True,
) -> list[int]:
    count = len(archive["candidate_metrics"])
    if candidate_name is None:
        if allow_repaired:
            return list(range(count))
        return [
            index
            for index, name in enumerate(archive["proposal_names"])
            if not str(name).endswith("_robustness_repair")
        ]
    return [
        index
        for index, name in enumerate(archive["proposal_names"])
        if canonical_candidate_name(str(name)) == candidate_name
        and (allow_repaired or not str(name).endswith("_robustness_repair"))
    ]


def filter_manifest_by_candidate_name(
    dataset_dir: str | Path,
    manifest_rows: Sequence[dict[str, Any]],
    candidate_name: str | None,
    *,
    allow_repaired: bool = True,
) -> list[dict[str, Any]]:
    if candidate_name is None:
        return list(manifest_rows)
    root = Path(dataset_dir)
    return [
        row
        for row in manifest_rows
        if candidate_indices_by_name(
            load_teacher_instance(root / str(row["path"])),
            candidate_name,
            allow_repaired=allow_repaired,
        )
    ]


def create_instance_split(
    manifest_rows: Sequence[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for row in manifest_rows:
        groups.setdefault(str(row.get("surface_id", "unknown")), []).append(str(row["instance_id"]))
    validation: set[str] = set()
    for instance_ids in groups.values():
        order = rng.permutation(len(instance_ids))
        validation_count = int(round(validation_fraction * len(instance_ids)))
        if validation_fraction > 0.0 and len(instance_ids) > 1:
            validation_count = min(max(1, validation_count), len(instance_ids) - 1)
        validation.update(instance_ids[index] for index in order[:validation_count])
    all_ids = [str(row["instance_id"]) for row in manifest_rows]
    train_ids = [instance_id for instance_id in all_ids if instance_id not in validation]
    validation_ids = [instance_id for instance_id in all_ids if instance_id in validation]
    return train_ids, validation_ids


class TeacherPathDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    """One canonical fixed- or variable-resolution target per feasible candidate."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        instance_ids: Sequence[str] | None = None,
        num_surface_points: int = 512,
        num_path_waypoints: int | None = 256,
        tokens_per_footprint_area: float = 1.0,
        minimum_path_tokens: int = 32,
        maximum_path_tokens: int = 2048,
        candidate_policy: str = "all",
        candidate_name: str | None = None,
        allow_repaired_candidates: bool = True,
        path_coordinate_system: str = "xyz",
        include_mode_conditioning: bool = False,
        preserve_source_waypoints: bool = False,
        seed: int = 0,
    ) -> None:
        if num_surface_points < 1 or (num_path_waypoints is not None and num_path_waypoints < 2):
            raise ValueError("surface and path point counts must be positive")
        if tokens_per_footprint_area <= 0.0 or minimum_path_tokens < 2 or maximum_path_tokens < minimum_path_tokens:
            raise ValueError("invalid variable-token settings")
        if candidate_policy not in {"all", "best"}:
            raise ValueError("candidate_policy must be 'all' or 'best'")
        if path_coordinate_system not in {
            "xyz", "analytic_uv", "analytic_uv_control", "analytic_uv_structured",
            "analytic_uv_structured_residual",
        }:
            raise ValueError(
                "unsupported path coordinate system"
            )
        if include_mode_conditioning and path_coordinate_system not in {
            "analytic_uv_structured", "analytic_uv_structured_residual"
        }:
            raise ValueError("mode conditioning requires structured analytic paths")
        self.dataset_dir = Path(dataset_dir)
        selected_ids = None if instance_ids is None else set(instance_ids)
        self.rows = [
            row for row in load_manifest(self.dataset_dir)
            if selected_ids is None or str(row["instance_id"]) in selected_ids
        ]
        if not self.rows:
            raise ValueError("no manifest rows match the requested instance IDs")
        self.num_surface_points = num_surface_points
        self.num_path_waypoints = num_path_waypoints
        self.tokens_per_footprint_area = tokens_per_footprint_area
        self.minimum_path_tokens = minimum_path_tokens
        self.maximum_path_tokens = maximum_path_tokens
        self.candidate_policy = candidate_policy
        self.candidate_name = candidate_name
        self.allow_repaired_candidates = allow_repaired_candidates
        self.path_coordinate_system = path_coordinate_system
        self.include_mode_conditioning = include_mode_conditioning
        self.preserve_source_waypoints = preserve_source_waypoints
        self.seed = seed
        self._cache: dict[int, dict[str, torch.Tensor | str | int]] = {}
        self.sample_index = []
        eligible_rows = []
        for row in self.rows:
            archive = load_teacher_instance(self.dataset_dir / str(row["path"]))
            matching = candidate_indices_by_name(
                archive,
                candidate_name,
                allow_repaired=allow_repaired_candidates,
            )
            if not matching:
                continue
            instance_index = len(eligible_rows)
            eligible_rows.append(row)
            selected = matching if candidate_policy == "all" else matching[:1]
            self.sample_index.extend(
                TeacherSampleIndex(instance_index, candidate_index)
                for candidate_index in selected
            )
        self.rows = eligible_rows
        if not self.sample_index:
            selector = "all candidates" if candidate_name is None else repr(candidate_name)
            raise ValueError(f"no samples match candidate selector {selector}")

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        if index in self._cache:
            return self._cache[index]
        sample_ref = self.sample_index[index]
        row = self.rows[sample_ref.instance_index]
        archive = load_teacher_instance(self.dataset_dir / str(row["path"]))
        metadata = archive["metadata"]
        assert isinstance(metadata, dict)
        surface = surface_from_teacher_archive(archive, surface_id=str(row["surface_id"]))
        candidate = sample_ref.candidate_index
        plan = CoveragePlan(
            archive["candidate_waypoints"][candidate],
            archive["candidate_segment_mask"][candidate],
            archive["candidate_waypoint_mask"][candidate],
        )
        config = metadata["teacher_config"]
        radius = float(config["footprint_radius"])
        proposal_name = str(archive["proposal_names"][candidate])
        mode = parse_pattern_mode(proposal_name)
        num_tokens = self.num_path_waypoints
        if self.path_coordinate_system in {
            "analytic_uv_control", "analytic_uv_structured",
            "analytic_uv_structured_residual",
        }:
            world_path = plan.active_paths()[0]
            if (
                self.path_coordinate_system in {
                    "analytic_uv_structured", "analytic_uv_structured_residual"
                }
                and "candidate_controls" in archive
                and "candidate_control_mask" in archive
            ):
                control_mask = np.asarray(archive["candidate_control_mask"])[candidate]
                path = np.asarray(archive["candidate_controls"])[candidate][control_mask]
            else:
                path = extract_structured_parameter_controls(
                    surface,
                    world_path,
                    mode_name=mode.name,
                )
            if self.path_coordinate_system == "analytic_uv_structured_residual":
                path = structured_controls_to_residual(
                    surface,
                    path,
                    footprint_radius=radius,
                    overlap=float(config["overlap"]),
                    mode_name=mode.name,
                )
            num_tokens = len(path)
            path_arclength = np.linspace(0.0, 1.0, num_tokens)
        elif num_tokens is None:
            num_tokens = suggested_token_count(
                surface.total_area,
                radius,
                tokens_per_footprint_area=self.tokens_per_footprint_area,
                minimum=self.minimum_path_tokens,
                maximum=self.maximum_path_tokens,
            )
            canonical = canonicalize_surface_path(
                surface,
                plan.active_paths()[0],
                num_tokens=num_tokens,
                preserve_source_waypoints=self.preserve_source_waypoints,
            )
            world_path = canonical.points
            path_arclength = canonical.normalized_arclength
            num_tokens = len(world_path)
        else:
            fixed_plan = resample_plan_fixed_waypoints(
                surface, plan, num_waypoints=num_tokens
            )
            world_path = fixed_plan.active_paths()[0]
            path_arclength = np.linspace(0.0, 1.0, num_tokens)

        points = np.asarray(archive["sample_points"], dtype=np.float64)
        normals = np.asarray(archive["sample_normals"], dtype=np.float64)
        weights = np.asarray(archive["area_weights"], dtype=np.float64)
        center = np.average(points, axis=0, weights=weights)
        scale = float(np.max(np.linalg.norm(points - center, axis=1)))
        if scale <= 1e-12:
            raise ValueError("surface normalization scale is zero")
        selected = self._surface_indices(len(points), sample_ref.instance_index)
        surface_features = np.concatenate(((points[selected] - center) / scale, normals[selected]), axis=1)
        if self.path_coordinate_system == "xyz":
            path = (world_path - center) / scale
        elif self.path_coordinate_system == "analytic_uv":
            path = inverse_surface_parameters(surface, world_path, unwrap_periodic=True)
        condition = np.asarray(
            [float(config["footprint_radius"]) / scale, float(config["missed_tolerance"])],
            dtype=np.float32,
        )
        if self.include_mode_conditioning:
            mode_one_hot = np.zeros(len(PATTERN_MODE_NAMES), dtype=np.float32)
            mode_one_hot[mode.mode_id] = 1.0
            condition = np.concatenate((condition, mode_one_hot))
        metrics = np.asarray(archive["candidate_metrics"][candidate], dtype=np.float32)
        sample: dict[str, torch.Tensor | str | int] = {
            "surface": torch.from_numpy(surface_features.astype(np.float32)),
            "path": torch.from_numpy(path.astype(np.float32)),
            "path_mask": torch.ones(num_tokens, dtype=torch.bool),
            "path_arclength": torch.from_numpy(path_arclength.astype(np.float32)),
            "condition": torch.from_numpy(condition),
            # These values define the task-frame coordinate transform rather
            # than model inputs. Keep them in float64: a float32 round trip can
            # change the selected mesh face at seams and alter hard geodesics.
            "center": torch.from_numpy(center),
            "scale": torch.tensor(scale, dtype=torch.float64),
            "teacher_metrics": torch.from_numpy(metrics),
            "instance_id": str(row["instance_id"]),
            "candidate_index": candidate,
            "mode_id": mode.mode_id,
            "mode_name": mode.name,
            "path_coordinate_system": self.path_coordinate_system,
        }
        self._cache[index] = sample
        return sample

    def _surface_indices(self, count: int, instance_index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 104729 * instance_index)
        return rng.choice(count, size=self.num_surface_points, replace=count < self.num_surface_points)


def collate_teacher_paths(
    samples: Sequence[dict[str, torch.Tensor | str | int]],
) -> dict[str, torch.Tensor | list[str]]:
    """Pad path tensors while preserving an explicit valid-token mask."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    maximum_tokens = max(int(sample["path"].shape[0]) for sample in samples)  # type: ignore[union-attr]
    batch_size = len(samples)
    first_path = samples[0]["path"]
    assert isinstance(first_path, torch.Tensor)
    path_dim = int(first_path.shape[-1])
    if any(int(sample["path"].shape[-1]) != path_dim for sample in samples):  # type: ignore[union-attr]
        raise ValueError("all paths in a batch must use the same coordinate dimension")
    paths = torch.zeros(batch_size, maximum_tokens, path_dim, dtype=torch.float32)
    masks = torch.zeros(batch_size, maximum_tokens, dtype=torch.bool)
    arclength = torch.zeros(batch_size, maximum_tokens, dtype=torch.float32)
    for index, sample in enumerate(samples):
        path = sample["path"]
        sample_mask = sample["path_mask"]
        sample_arclength = sample["path_arclength"]
        assert isinstance(path, torch.Tensor)
        assert isinstance(sample_mask, torch.Tensor)
        assert isinstance(sample_arclength, torch.Tensor)
        count = path.shape[0]
        paths[index, :count] = path
        masks[index, :count] = sample_mask
        arclength[index, :count] = sample_arclength
    result: dict[str, torch.Tensor | list[str]] = {
        "path": paths,
        "path_mask": masks,
        "path_arclength": arclength,
    }
    for key in samples[0]:
        if key in result:
            continue
        result[key] = default_collate([sample[key] for sample in samples])
    return result
