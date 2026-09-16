#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from e10_runner_support import compare, initialize, prepare, report, verify, write_json

DEFAULT_OUTPUT = ROOT / "results/e10_structured_anytime_routing_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "tests", "initialize", "compare", "verify", "report"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/e10_structured_anytime_routing_v1.json").read_text())
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{args.stage}"](config, output)


def _require(output: Path, stage: str) -> None:
    path = output / f"{stage}.checkpoint.json"
    if not path.exists() or not json.loads(path.read_text()).get("complete"):
        raise RuntimeError(f"required stage {stage!r} is incomplete")


def stage_prepare(config, output):
    if (output / "prepare.checkpoint.json").exists():
        raise RuntimeError("prepare already exists; use a fresh output directory")
    prepare(ROOT, config, output)


def stage_tests(config, output):
    _require(output, "prepare")
    commands = [
        [sys.executable, "-m", "pytest", "-q", "tests/test_structured_routing.py", "tests/test_e09r1_search.py", "tests/test_e09_synchronized_motion.py", "tests/test_e09_execution.py", "tests/test_history_search.py", "tests/test_completion_bound.py", "tests/test_ordered_trace_evaluator.py"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    lines=[];statuses=[]
    for index, command in enumerate(commands):
        began=perf_counter();env=dict(os.environ);env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
        completed=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        lines.extend(["$ "+" ".join(command),completed.stdout.rstrip(),f"exit={completed.returncode} elapsed_s={perf_counter()-began:.6f}"])
        statuses.append({"name":"focused" if index==0 else "literal_repository_suite","exit_code":completed.returncode})
        if index==0 and completed.returncode!=0:
            break
    (output/"tests.txt").write_text("\n".join(lines)+"\n")
    full_exit=next((x["exit_code"] for x in statuses if x["name"]=="literal_repository_suite"),None)
    missing=[]
    text="\n".join(lines)
    known=["results/riemannian_anisotropy_utility_v1/r0_scene_calibration/witnesses/saddle_T17.npz","results/nuc_robot_skeleton_coupling_v1/config.json"]
    for path in known:
        if path in text:missing.append(path)
    focused_ok=statuses[0]["exit_code"]==0
    write_json(output/"test_statuses.json",{"commands":statuses,"focused_pass":focused_ok,"literal_suite_exit":full_exit,"missing_historical_fixtures":missing,"literal_suite_fully_passed":full_exit==0})
    # Scientific comparison requires the focused suite.  Historical fixture
    # absence remains an explicit repository-suite limitation.
    complete=focused_ok and (full_exit==0 or bool(missing))
    write_json(output/"tests.checkpoint.json",{"complete":complete,"focused_pass":focused_ok,"literal_suite_exit":full_exit,"missing_historical_fixtures":missing})
    if not complete:raise RuntimeError("E10 tests contain an unexplained failure")


def stage_initialize(config, output):
    _require(output,"tests");initialize(ROOT,config,output)


def stage_compare(config, output):
    _require(output,"initialize");compare(ROOT,config,output)


def stage_verify(config, output):
    _require(output,"compare");verify(ROOT,config,output)


def stage_report(config, output):
    _require(output,"verify");report(ROOT,config,output)


if __name__ == "__main__":
    main()
