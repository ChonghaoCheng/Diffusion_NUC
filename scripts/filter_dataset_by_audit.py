#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from diffusion_coverage.learning import load_manifest


CANDIDATE_KEYS = {
    "candidate_waypoints",
    "candidate_segment_mask",
    "candidate_waypoint_mask",
    "candidate_metrics",
    "candidate_controls",
    "candidate_control_mask",
    "proposal_names",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter candidates using a complete hard audit")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    records = audit.get("instances")
    if not isinstance(records, list):
        raise ValueError("audit must contain per-candidate instances")
    decisions = {
        (str(row["instance_id"]), int(row["candidate_index"])): bool(row["feasible"])
        for row in records
    }
    source_rows = load_manifest(args.dataset)
    expected = sum(int(row["num_candidates"]) for row in source_rows)
    if len(decisions) != expected:
        raise ValueError(f"audit covers {len(decisions)} candidates, expected {expected}")
    (args.output / "instances").mkdir(parents=True)
    output_rows = []
    rejected = []
    for row in source_rows:
        instance_id = str(row["instance_id"])
        source_path = args.dataset / str(row["path"])
        with np.load(source_path, allow_pickle=False) as archive:
            names = np.asarray(archive["proposal_names"])
            keep = np.asarray(
                [decisions[(instance_id, index)] for index in range(len(names))],
                dtype=bool,
            )
            if not keep.any():
                raise RuntimeError(f"audit rejected every candidate for {instance_id}")
            for mode in np.unique(names):
                if not np.any(keep & (names == mode)):
                    raise RuntimeError(f"audit removed mode {mode} from {instance_id}")
            rejected.extend(
                {"instance_id": instance_id, "candidate_index": int(index), "mode": str(names[index])}
                for index in np.flatnonzero(~keep)
            )
            arrays = {}
            for key in archive.files:
                value = np.asarray(archive[key])
                arrays[key] = value[keep] if key in CANDIDATE_KEYS else value
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["audit_filter"] = {
                "source_dataset": str(args.dataset),
                "audit": str(args.audit),
                "candidate_contract": "analytic_uv_structured_residual_float32",
                "rejected_candidates": int((~keep).sum()),
            }
            arrays["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True), dtype=np.str_
            )
        output_path = args.output / "instances" / f"{instance_id}.npz"
        np.savez_compressed(output_path, **arrays)
        metrics = np.asarray(arrays["candidate_metrics"])
        output_row = dict(row)
        output_row["path"] = str(output_path.relative_to(args.output))
        output_row["num_candidates"] = int(keep.sum())
        output_row["num_feasible_candidates"] = int(keep.sum())
        output_row["best_missed_fraction"] = float(metrics[:, 0].min())
        output_row["best_path_length"] = float(metrics[:, 1].min())
        output_rows.append(output_row)
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "instances": len(output_rows),
        "source_candidates": expected,
        "admitted_candidates": expected - len(rejected),
        "rejected_candidates": rejected,
        "audit_summary": audit.get("summary"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
