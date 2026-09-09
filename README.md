# Diffusion-Guided Manipulator Coverage Planning

Research code for finite-footprint 3D surface coverage and robot-liftable coverage planning.
The repository currently contains:

- exact, greedy, and greedy-guided exact maximal-continuity graph solvers;
- synthetic graph and colour-liftability benchmarks;
- 3D surface generation, projection, coverage evaluation, and classical teachers;
- conditional surface-path Flow Matching models and training scripts;
- UR5e inverse-kinematics continuation graphs with serialized edge witnesses;
- strict q-space coverage teachers and continuous hard checking.

Direct configuration-space Flow Matching, physical execution, ROS 2 adapters, and hardware
interfaces are not implemented yet.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,learning]"
```

Install the optional robotics dependencies when running the UR5e numerical experiments:

```bash
pip install -e ".[robotics]"
```

## Tests

```bash
pytest -q
```

## Layout

- `src/diffusion_coverage/`: solver, coverage, learning, and robot modules
- `scripts/`: dataset generation, training, evaluation, and plotting entry points
- `tests/`: unit and regression tests
Large generated datasets, checkpoints, raw experiment outputs, and the Agent-Native Research
Artifact are maintained separately and excluded from this code repository.
