from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def run_upstream_reference(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    build_directory: str | Path,
    root_face: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the separately built upstream NUC pybind11 module without vendoring it."""

    build = Path(build_directory)
    libraries = sorted(build.glob("nuc_tmech23*.so"))
    if len(libraries) != 1:
        raise FileNotFoundError(
            f"expected one built nuc_tmech23 extension under {build}, found {len(libraries)}"
        )
    spec = importlib.util.spec_from_file_location("nuc_tmech23", libraries[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream NUC extension from {libraries[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    face_values = np.asarray(faces, dtype=np.int64).reshape(-1).tolist()
    vertex_values = np.asarray(vertices, dtype=np.float64).reshape(-1).tolist()
    result = (
        module.run(face_values, vertex_values)
        if root_face is None
        else module.run(face_values, vertex_values, int(root_face))
    )
    return (
        np.asarray(result[0], dtype=np.int64),
        np.asarray(result[1], dtype=np.float64).reshape(-1, 3),
    )
