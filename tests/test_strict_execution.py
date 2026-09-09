from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from diffusion_coverage.robot.strict_execution import check_strict_coverage_execution
from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
from diffusion_coverage.surface import make_plane


MODEL = Path(
    "/data/chocheng/Code/ur_contact_motion_sb3/third_party/"
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)


def home_contract(
    *, sigma_safe: float = 0.05, footprint_radius: float = 0.2,
    nuc_error_tolerance: float | None = None,
):
    robot = UR5eKinematics(MODEL)
    q = np.repeat(robot.home[None, :], 3, axis=0)
    position, axis = robot.forward(robot.home)
    positions = np.repeat(position[None, :], 3, axis=0)
    axes = np.repeat(axis[None, :], 3, axis=0)
    surface = make_plane(width=0.1, height=0.1, nx=4, ny=4, samples_per_face=1)
    transform = np.eye(4)
    transform[:3, 3] = position
    result = check_strict_coverage_execution(
        robot,
        (q,),
        (positions,),
        (axes,),
        surface,
        transform,
        footprint_radius=footprint_radius,
        position_tolerance=1e-5,
        axis_tolerance=1e-5,
        characteristic_length=0.1,
        sigma_safe=sigma_safe,
        missed_tolerance=0.01,
        repeat_tolerance=0.01,
        interpolation_joint_step=0.02,
        nuc_error_tolerance=nuc_error_tolerance,
    )
    return result


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_strict_checker_reports_separate_pass_fields():
    result = home_contract()
    assert result.kinematics_pass
    assert result.coverage_pass
    assert result.timing_pass is None
    assert result.overall_pass
    assert result.execution_cost is not None
    assert result.coverage_metrics is not None
    assert result.min_sigma_min_5 > 0.0


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_strict_checker_uses_5d_singular_value_threshold():
    result = home_contract(sigma_safe=10.0)
    assert not result.kinematics_pass
    assert "task_singularity" in result.failure_reasons


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_strict_checker_reports_coverage_failure_independently():
    result = home_contract(footprint_radius=1e-4)
    assert result.kinematics_pass
    assert not result.coverage_pass
    assert not result.overall_pass
    assert "coverage_miss" in result.failure_reasons


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_strict_checker_enforces_combined_nuc_threshold():
    result = home_contract(nuc_error_tolerance=-1.0)
    assert result.kinematics_pass
    assert not result.coverage_pass
    assert "coverage_nuc" in result.failure_reasons


@pytest.mark.skipif(not MODEL.exists(), reason="local UR5e Menagerie model is unavailable")
def test_strict_checker_rejects_missing_surface_path():
    robot = UR5eKinematics(MODEL)
    surface = make_plane(nx=2, ny=2, samples_per_face=1)
    result = check_strict_coverage_execution(
        robot, (), (), (), surface, np.eye(4),
        footprint_radius=0.1, position_tolerance=1e-3, axis_tolerance=0.1,
        characteristic_length=0.1, sigma_safe=0.01, missed_tolerance=0.1,
        repeat_tolerance=0.1, interpolation_joint_step=0.05,
    )
    assert not result.kinematics_pass
    assert result.failure_reason == "missing_surface_path"
