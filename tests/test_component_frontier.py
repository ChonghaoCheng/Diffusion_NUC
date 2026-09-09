from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ur5e_component_frontier.py"
SPEC = importlib.util.spec_from_file_location("component_frontier", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_component_frontier_on_overlapping_components():
    labels = np.asarray(
        [
            [0, -1],
            [0, 1],
            [1, -1],
            [2, -1],
        ]
    )
    coverage, reachable = MODULE.component_coverage(labels)
    assert MODULE.maximum_covered_nodes(coverage, reachable, 1) == 2
    assert MODULE.maximum_covered_nodes(coverage, reachable, 2) == 3
    assert MODULE.maximum_covered_nodes(coverage, reachable, 3) == 4
    assert MODULE.minimum_cover_segments(coverage, reachable) == 3
