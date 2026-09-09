#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge disjoint teacher dataset shards")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.output / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"output dataset already exists: {args.output}")
    instances_dir = args.output / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for input_dir in args.inputs:
        for line in (input_dir / "manifest.jsonl").read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            instance_id = str(row["instance_id"])
            if instance_id in seen:
                raise ValueError(f"duplicate instance ID: {instance_id}")
            seen.add(instance_id)
            source = input_dir / str(row["path"])
            destination = instances_dir / f"{instance_id}.npz"
            shutil.copy2(source, destination)
            row["path"] = str(destination.relative_to(args.output))
            rows.append(row)
    rows.sort(key=lambda row: str(row["instance_id"]))
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"merged {len(rows)} instances into {args.output}")


if __name__ == "__main__":
    main()
