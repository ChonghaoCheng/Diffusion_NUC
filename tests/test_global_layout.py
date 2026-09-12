from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusion_coverage.diagnostics.global_layout import (
    coverage_equivalent,
    finite_verified_oracle,
    generate_layout,
    generate_physical_roots,
    make_planning_remesh,
    make_reference_surface,
    map_physical_root,
    remesh_statistics,
    require_g1_authorized,
    select_geometry_baseline,
    surface_hash,
    validate_remesh_library,
)
from diffusion_coverage.nuc.adapter import generate_nuc_skeleton


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/global_layout_capacity_gate_v1.json").read_text())


def test_canonical_root_and_remesh_reproduce_current_nuc_contract():
    cfg = config(); reference = make_reference_surface("saddle", cfg); mesh = make_planning_remesh("saddle", "M00", cfg)
    roots = generate_physical_roots(reference, mesh, cfg["root_count"])
    mapped = map_physical_root(mesh, roots[0]["position"])
    default = generate_nuc_skeleton(mesh, policy="upstream_first")
    explicit = generate_nuc_skeleton(mesh, policy="upstream_first", root_face=mapped["mapped_face"])
    assert mapped["mapped_face"] == 0
    assert np.array_equal(default.topological_path, explicit.topological_path)
    assert np.array_equal(default.waypoints, explicit.waypoints)


def test_physical_roots_are_remesh_independent_and_mapping_is_deterministic():
    cfg = config(); reference = make_reference_surface("saddle", cfg); canonical = make_planning_remesh("saddle", "M00", cfg)
    roots = generate_physical_roots(reference, canonical, cfg["root_count"])
    for root in roots:
        physical = np.asarray(root["position"]).copy()
        mappings = [map_physical_root(make_planning_remesh("saddle", remesh, cfg), root["position"]) for remesh in cfg["remesh_ids"]]
        assert all(mapping["mapping_distance_m"] <= cfg["root_mapping_tolerance_m"] for mapping in mappings)
        assert np.array_equal(root["position"], physical)
        assert map_physical_root(canonical, root["position"]) == map_physical_root(canonical, root["position"])


def test_remeshes_share_physical_surface_and_resolution_contract():
    cfg = config()
    for surface_id in ("saddle", "hemisphere"):
        reference = make_reference_surface(surface_id, cfg)
        records = [{"remesh_id": remesh, **remesh_statistics(surface_id, make_planning_remesh(surface_id, remesh, cfg), reference, cfg)} for remesh in cfg["remesh_ids"]]
        validate_remesh_library(records, cfg)
        assert len({(row["vertex_count"], row["face_count"]) for row in records}) == 1


def test_layouts_use_common_reference_and_fixed_policy():
    cfg = config(); reference = make_reference_surface("saddle", cfg); canonical = make_planning_remesh("saddle", "M00", cfg)
    root = generate_physical_roots(reference, canonical, 1)[0]
    hashes = set()
    for remesh_id in cfg["remesh_ids"]:
        row, skeleton, _ = generate_layout(reference, make_planning_remesh("saddle", remesh_id, cfg), root, cfg)
        hashes.add(row["evaluation_surface_hash"])
        assert skeleton.policy == cfg["nuc"]["expansion_policy"] == "upstream_first"
    assert hashes == {surface_hash(reference)}


def test_geometry_baseline_is_placement_independent_and_has_no_robot_inputs():
    rows = [
        {"layout_id": "a", "root_id": "R00", "remesh_id": "M00", "E_NUC": .2, "L_S": 2.0, "J_q": 100.0},
        {"layout_id": "b", "root_id": "R01", "remesh_id": "M01", "E_NUC": .1, "L_S": 3.0, "J_q": 1.0},
    ]
    assert select_geometry_baseline(rows)["layout_id"] == "b"
    changed = [{**row, "J_q": -row["J_q"], "placement_id": "different"} for row in rows]
    assert select_geometry_baseline(changed)["layout_id"] == "b"


def test_coverage_equivalence_and_verified_oracle_are_deterministic():
    baseline = {"E_NUC": .10}
    rows = [
        {"layout_id": "failed", "E_NUC": .11, "overall_pass": False, "J_q": .1},
        {"layout_id": "valid", "E_NUC": .12, "overall_pass": True, "J_q": 2.0},
        {"layout_id": "outside", "E_NUC": .20, "overall_pass": True, "J_q": 1.0},
    ]
    equivalent = coverage_equivalent(rows, baseline, .03)
    assert [row["layout_id"] for row in equivalent] == ["failed", "valid"]
    assert finite_verified_oracle(equivalent)["layout_id"] == "valid"


def test_g1_gate_contract_is_explicitly_conditional():
    cfg = config()
    assert cfg["gate"]["minimum_scenes_with_10pct_spread"] == 4
    assert cfg["gate"]["minimum_median_geometry_oracle_gain"] == .08
    try:
        require_g1_authorized({"decision": "NO-GO", "G1_authorized": False})
    except RuntimeError:
        pass
    else:
        raise AssertionError("G1 must refuse a G0 NO-GO")
