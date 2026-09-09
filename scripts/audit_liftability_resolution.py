#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare synthetic lift segment counts across sampling resolutions")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", action="append", required=True, metavar="LABEL=RESULT_DIR")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_counts(path: Path) -> dict[tuple[object, ...], int]:
    rows = [json.loads(line) for line in (path / "candidate_results.jsonl").read_text().splitlines() if line]
    return {
        (row["instance_id"], row["difficulty"], row["field_index"], row["candidate_index"]): int(row["min_segments"])
        for row in rows
    }


def main() -> None:
    args = parse_args()
    reference = load_counts(args.reference)
    comparisons = []
    for specification in args.comparison:
        label, raw_path = specification.split("=", 1)
        values = load_counts(Path(raw_path))
        if values.keys() != reference.keys():
            raise ValueError(f"comparison key set differs for {label}")
        left = np.asarray([reference[key] for key in reference], dtype=np.int64)
        right = np.asarray([values[key] for key in reference], dtype=np.int64)
        difference = right - left
        comparisons.append(
            {
                "label": label,
                "candidate_field_pairs": len(left),
                "exact_agreement_rate": float(np.mean(difference == 0)),
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "maximum_absolute_difference": int(np.max(np.abs(difference), initial=0)),
                "comparison_underestimates": int(np.sum(difference < 0)),
                "comparison_overestimates": int(np.sum(difference > 0)),
            }
        )
    document = {"reference": str(args.reference), "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
